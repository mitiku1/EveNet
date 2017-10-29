# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Training script for the EveNet network.

....

"""

from __future__ import print_function

import argparse
import json
import os
import threading
import sys
import numpy as np
from termcolor import colored

from wavenet import WaveNetModel, CsvReader, optimizer_factory

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants as sig_constants

tf.logging.set_verbosity(tf.logging.INFO)


## TODO: Implement Evaluation Run Hook to generate samples from phonemes and plot them as image to Tensorboard.
##       See Github for reference implementation with seperate tf.graph object for this.

class ValidationRunHook(tf.train.SessionRunHook):
    """ValidationRunHook calculates continuous validation of the model.

    Args:
      checkpoint_dir (string): Dir to store model checkpoints
      metric_dir (string): Dir to store metrics like accuracy and auroc
      graph (tf.Graph): Evaluation graph
      eval_frequency (int): Frequency of evaluation every n checkpoint steps
      eval_steps (int): Evaluation steps to be performed
    """
    def __init__(self,
                 graph,
                 prediction_tensor,
                 checkpoint_dir,
                 eval_frequency=100,
                 **kwargs):

        self._kwargs = kwargs
        self._eval_every = eval_frequency
        self._steps_since_eval = 0
        self._graph = graph
        self._session = None

        # Get Config
        config = {}
        for cfg_tensor in tf.get_collection("config"):
            config[cfg_tensor.name.split(":")[0]] = tf.Session(graph=self._graph).run(cfg_tensor)

        # Parameters
        self._data_dim = config['data_dim']

        # Load Validation testset into RAM and split into pieces...
        with open("./reader_config.json") as json_file:
            mapping_config = json.load(json_file)

        # Data Seed file
        validation_filename = "./data/anger-validation-split/anger_validation.dat" # TODO: take from args...

        dat_seed = np.genfromtxt(validation_filename, delimiter=",")

        validation_sample_size = config['receptive_field_size'] + 1
        
        cutoff_samples = dat_seed.shape[0] % validation_sample_size
        validation_samples = dat_seed.shape[0] / (validation_sample_size - cutoff_samples)
        
        self._validation_samples = dat_seed[:-cutoff_samples].reshape(-1,validation_sample_size,self._data_dim)

        # Global conditioning on emotion categories
        gc_lut = mapping_config['emotion_categories']
        gc_feed_str = np.genfromtxt(validation_filename.replace(".dat", ".emo"), delimiter=",", dtype=str)
        gc_feed = np.array([gc_lut.index(emo) for emo in gc_feed_str])
        self._validation_samples_gc = gc_feed[:-cutoff_samples].reshape(-1,validation_sample_size,1)

        # Local conditioning on Phonemes.
        lc_lut = mapping_config['phoneme_categories']
        lc_feed_str = np.genfromtxt(validation_filename.replace(".dat", ".pho"), delimiter=",", dtype=str)
        lc_feed = np.array([lc_lut.index(pho) for pho in lc_feed_str])
        self._validation_samples_lc = lc_feed[:-cutoff_samples].reshape(-1,validation_sample_size,1)

        self._nr_validation_samples = self._validation_samples.shape[0]
        print ("Number of validation samples %d" % self._nr_validation_samples)

        # Tensor setup
        self._prediction_tensor = tf.get_collection("predict_proba")[0]
        self._target_placeholder = tf.placeholder(tf.float32, shape=(self._data_dim), name="target")
        self._validation_score_tensor = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self._target_placeholder, self._prediction_tensor))))

        self._summary_tensor = tf.summary.scalar("validation_score", self._validation_score_tensor)

        # Creates a global step to contain a counter for
        # the global training step
        self._gs = tf.contrib.framework.get_or_create_global_step()

        # MonitoredTrainingSession runs hooks in background threads
        # and it doesn't wait for the thread from the last session.run()
        # call to terminate to invoke the next hook, hence locks.
        self._eval_lock = threading.Lock()
        self._file_writer = tf.summary.FileWriter(checkpoint_dir+"/eval")




    def after_create_session(self, session, coord):
        self._session = session

    def after_run(self, run_context, run_values):
        # Always check for new checkpoints in case a single evaluation
        # takes longer than checkpoint frequency and _eval_every is >1
        self._update_latest_checkpoint()

        if self._eval_lock.acquire(False):
            try:
                if self._steps_since_eval > self._eval_every:
                    self._steps_since_eval = 0
                    self._run_eval()
            finally:
                self._eval_lock.release()

    def _update_latest_checkpoint(self):
        self._steps_since_eval += 1

    def end(self, session):
        """Called at then end of session to make sure we always evaluate."""
        self._update_latest_checkpoint()

        with self._eval_lock:
            self._run_eval()

    def _run_eval(self):
        """Run model evaluation and generate summaries."""
        print("Running model validation...")

        if self._session:
        #    # Restores previously saved variables from latest checkpoint
        #    self._saver.restore(session, self._latest_checkpoint)

            for i in range(self._nr_validation_samples):
                data = self._validation_samples[i]
                gc = self._validation_samples_gc[i]
                lc = self._validation_samples_lc[i]
                
                data_feed = data[:-1,:]
                gc_feed = gc[:-1,0]
                lc_feed = lc[:-1,0]

                target = data[-1,:]

                summaries, validation_score, train_step = self._session.run([self._summary_tensor, 
                                                                              self._validation_score_tensor, 
                                                                              self._gs], feed_dict={'samples:0': data_feed, 
                                                                                                    'gc:0': gc_feed, 
                                                                                                    'lc:0': lc_feed, 
                                                                                                    'target:0': target})

                # Write the summaries
                self._file_writer.add_summary(summaries, global_step=train_step)
                self._file_writer.flush()
                tf.logging.info(validation_score)


def run(target,
        train_steps,
        job_dir,
        train_files,
        reader_config,
        run_validation,
        batch_size,
        learning_rate,
        residual_channels,
        dilation_channels,
        skip_channels,
        dilations,
        use_biases,
        gc_channels,
        lc_channels,
        filter_width,
        sample_size,
        initial_filter_width,
        l2_regularization_strength,
        momentum,
        optimizer):

    # Create a new graph and specify that as default
    default_graph = tf.Graph()
    with default_graph.as_default():
        file_writer = tf.summary.FileWriter(job_dir, graph=default_graph)


        # Initially empty, get's filled up below
        hooks = []

        # Placement of ops on devices using replica device setter
        # which automatically places the parameters on the `ps` server
        # and the `ops` on the workers
        #
        # See:
        # https://www.tensorflow.org/api_docs/python/tf/train/replica_device_setter
        with tf.device(tf.train.replica_device_setter()):

            with open(reader_config) as json_file:
                reader_config = json.load(json_file)

            # Reader
            receptive_field_size = WaveNetModel.calculate_receptive_field(filter_width,
                                                                          dilations,
                                                                          False,
                                                                          initial_filter_width)

            reader = CsvReader(
                train_files,
                batch_size=batch_size,
                receptive_field=receptive_field_size,
                sample_size=sample_size,
                config=reader_config
            )

            # Create network.
            net = WaveNetModel(
                batch_size=batch_size,
                dilations=dilations,
                filter_width=filter_width,
                residual_channels=residual_channels,
                dilation_channels=dilation_channels,
                skip_channels=skip_channels,
                quantization_channels=reader.data_dim,
                use_biases=use_biases,
                scalar_input=False,
                initial_filter_width=initial_filter_width,
                histograms=False,
                global_channels=gc_channels,
                local_channels=lc_channels)

            global_step_tensor = tf.contrib.framework.get_or_create_global_step()

            if l2_regularization_strength == 0:
                l2_regularization_strength = None

            loss = net.loss(input_batch=reader.data_batch,
                            global_condition=reader.gc_batch,
                            local_condition=reader.lc_batch,
                            l2_regularization_strength=l2_regularization_strength)

            summary_tensor = tf.summary.scalar("loss", loss)

            optimizer = optimizer_factory[optimizer](learning_rate=learning_rate, momentum=momentum)

            trainable = tf.trainable_variables()

            train_op = optimizer.minimize(loss, var_list=trainable, global_step=global_step_tensor)

            # Add Generation operator to graph for later use in generate.py
            tf.add_to_collection("config", tf.constant(reader.data_dim, name='data_dim'))
            tf.add_to_collection("config", tf.constant(receptive_field_size, name='receptive_field_size'))
            tf.add_to_collection("config", tf.constant(sample_size, name='sample_size'))

            samples = tf.placeholder(tf.float32, shape=(receptive_field_size, reader.data_dim), name="samples")
            gc = tf.placeholder(tf.int32, shape=(receptive_field_size), name="gc")
            lc = tf.placeholder(tf.int32, shape=(receptive_field_size), name="lc")  # TODO set to one

            gc = tf.one_hot(gc, gc_channels)
            lc = tf.one_hot(lc, lc_channels / 1)  # TODO set to one...

            prediction_tensor = net.predict_proba(samples, gc, lc)
            tf.add_to_collection("predict_proba", prediction_tensor)

            # Validation Hook
            if run_validation:
              hooks = [ValidationRunHook(
                  default_graph,
                  prediction_tensor,
                  job_dir,
                  eval_frequency=100
              )]


            # TODO: Implement fast generation
            """
            if filter_width <= 2:
                samples_fast = tf.placeholder(tf.float32, shape=(1, reader.data_dim), name="samples_fast")
                gc_fast = tf.placeholder(tf.int32, shape=(1), name="gc_fast")
                lc_fast = tf.placeholder(tf.int32, shape=(1), name="lc_fast")

                gc_fast = tf.one_hot(gc_fast, gc_channels)
                lc_fast = tf.one_hot(lc_fast, lc_channels)

                tf.add_to_collection("predict_proba_incremental", net.predict_proba_incremental(samples_fast, gc_fast, lc_fast))
                tf.add_to_collection("push_ops", net.push_ops)
            """

        # Creates a MonitoredSession for training
        # MonitoredSession is a Session-like object that handles
        # initialization, recovery and hooks
        # https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession
        with tf.train.MonitoredTrainingSession(master=target,
                                               checkpoint_dir=job_dir,
                                               hooks=hooks,
                                               save_checkpoint_secs=120,
                                               save_summaries_steps=None,
                                               save_summaries_secs=None) as session:  # TODO: SUMMARIES HERE

            # Global step to keep track of global number of steps particularly in
            # distributed setting
            step = global_step_tensor.eval(session=session)
            # Run the training graph which returns the step number as tracked by
            # the global step tensor.
            # When train epochs is reached, session.should_stop() will be true.
            try:
                while (train_steps is None or
                       step < train_steps) and not session.should_stop():

                    step, _, loss_val, summaries = session.run([global_step_tensor, train_op, loss, summary_tensor])

                    file_writer.add_summary(summaries, global_step=step)
                    file_writer.flush()

                    print("step %d loss %.4f" % (step, loss_val), end='\r')
                    sys.stdout.flush()

                    # For debugging
                    # dat, gc, lc = session.run([reader.data_batch, reader.gc_batch, reader.lc_batch])
                    # print(colored(str(dat.shape), 'red', 'on_grey'))
                    # for field in dat:
                    #     print(colored(str(field), 'red'))
                    # print(colored(str(lc.shape), 'red', 'on_grey'))
                    # for i in lc[0, -1, :]:
                    #     print("%1d" % i, end='')
                    #     sys.stdout.flush()
                    # print(colored(str(gc.shape), 'red', 'on_grey'))
                    # for i in gc[0, -1, :]:
                    #     print("%1d" % i, end='')
                    #     sys.stdout.flush()
                    # for field in lc:
                    #     print(colored(str(field), 'blue'))
                    # print(colored(str(gc.shape), 'red', 'on_grey'))
                    # for field in gc:
                    #     print(colored(str(field), 'green'))

            except KeyboardInterrupt:
                pass


def dispatch(*args, **kwargs):
    """Parse TF_CONFIG to cluster_spec and call run() method
    TF_CONFIG environment variable is available when running using
    gcloud either locally or on cloud. It has all the information required
    to create a ClusterSpec which is important for running distributed code.
    """

    tf_config = os.environ.get('TF_CONFIG')

    # If TF_CONFIG is not available run local
    if not tf_config:
        return run('', *args, **kwargs)

    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    # If cluster information is empty run local
    if job_name is None or task_index is None:
        return run('', True, *args, **kwargs)

    cluster_spec = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster_spec,
                             job_name=job_name,
                             task_index=task_index)

    # Wait for incoming connections forever
    # Worker ships the graph to the ps server
    # The ps server manages the parameters of the model.
    #
    # See a detailed video on distributed TensorFlow
    # https://www.youtube.com/watch?v=la_M6bCV91M
    if job_name == 'ps':
        server.join()
        return
    elif job_name in ['master', 'worker']:
        return run(server.target, *args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TASK.PY
    parser.add_argument('--train-files',
                        required=True,
                        type=str,
                        help='Training files local or GCS', nargs="+")
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help="""\
                        GCS or local dir for checkpoints, exports, and
                        summaries. Use an existing directory to load a
                        trained model, or a new directory to retrain""")

    parser.add_argument('--train-steps',
                        type=int,
                        help='Maximum number of training steps to perform.')

    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help='Batch size for training steps, recommended 1')

    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for SGD')

    # WAVENET.PY
    parser.add_argument('--sample_size',
                        type=int,
                        default=1000,
                        help='Concatenate and cut audio samples to this many '
                        'samples. Default: 1000')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=0,
                        help='Coefficient in the L2 regularization. '
                        'Default: False')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer. Default: 0.9.')

    parser.add_argument('--reader_config', type=str,
                        default="reader_config.json", help='Specify the path to the config file.')

    parser.add_argument('--run-validation',
                        type=bool,
                        default=True,
                        help='Run validation and safe to Tensorboard.')

    # Wavenet Params
    parser.add_argument('--filter_width',
                        type=int,
                        default=8,
                        help='Part of Wavenet Params')
    parser.add_argument('--dilations',
                        type=list,
                        default=[1, 2, 4, 8, 16, 32, 64, 128,
                                 1, 2, 4, 8, 16, 32, 64, 128],
                        help='Part of Wavenet Params')
    parser.add_argument('--residual_channels',
                        type=int,
                        default=64,
                        help='Part of Wavenet Params')
    parser.add_argument('--dilation_channels',
                        type=int,
                        default=64,
                        help='Part of Wavenet Params')
    parser.add_argument('--skip_channels',
                        type=int,
                        default=512,
                        help='Part of Wavenet Params')
    parser.add_argument('--initial_filter_width',
                        type=int,
                        default=32,
                        help='Part of Wavenet Params')
    parser.add_argument('--use_biases',
                        type=bool,
                        default=True,
                        help='Part of Wavenet Params')
    parser.add_argument('--gc_channels',
                        type=int,
                        default=64,
                        help='Part of Wavenet Params')
    parser.add_argument('--lc_channels',
                        type=int,
                        default=64,
                        help='Part of Wavenet Params')

    parser.add_argument('--verbosity',
                        choices=[
                            'DEBUG',
                            'ERROR',
                            'FATAL',
                            'INFO',
                            'WARN'
                        ],
                        default='INFO',
                        help='Set logging verbosity')
    parse_args, unknown = parser.parse_known_args()
    # Set python level verbosity
    tf.logging.set_verbosity(parse_args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[parse_args.verbosity] / 10)
    del parse_args.verbosity

    if unknown:
        tf.logging.warn('Unknown arguments: {}'.format(unknown))

    dispatch(**parse_args.__dict__)
