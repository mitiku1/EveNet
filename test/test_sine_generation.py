from __future__ import print_function

import json
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime
from termcolor import colored
from wavenet import WaveNetModel, CsvReader, optimizer_factory
import matplotlib.pyplot as plt

TEST_DATA = "./test/generation_sine_test_data"

GC_CHANNELS = 128
LC_CHANNELS = 128

SAMPLE_SIZE = 2
L2 = 0

LAYERS = [1, 2, 4, 8, 1, 2, 4, 8]


class TestGeneration(tf.test.TestCase):

    def testGenerateSimple(self):
        # Reader config
        with open(TEST_DATA + "/config.json") as json_file:
            self.reader_config = json.load(json_file)

        # Initialize the reader
        receptive_field_size = WaveNetModel.calculate_receptive_field(2, LAYERS, False, 8)

        self.reader = CsvReader(
            [TEST_DATA + "/test.dat", TEST_DATA + "/test.emo", TEST_DATA + "/test.pho"],
            batch_size=1,
            receptive_field=receptive_field_size,
            sample_size=SAMPLE_SIZE,
            config=self.reader_config
        )

        # WaveNet model
        self.net = WaveNetModel(batch_size=1,
                                dilations=LAYERS,
                                filter_width=2,
                                residual_channels=8,
                                dilation_channels=8,
                                skip_channels=8,
                                quantization_channels=2,
                                use_biases=True,
                                scalar_input=False,
                                initial_filter_width=8,
                                histograms=False,
                                global_channels=GC_CHANNELS,
                                local_channels=LC_CHANNELS)

        loss = self.net.loss(input_batch=self.reader.data_batch,
                             global_condition=self.reader.gc_batch,
                             local_condition=self.reader.lc_batch,
                             l2_regularization_strength=L2)

        optimizer = optimizer_factory['adam'](learning_rate=0.003, momentum=0.9)
        trainable = tf.trainable_variables()
        train_op = optimizer.minimize(loss, var_list=trainable)

        samples = tf.placeholder(tf.float32, shape=(receptive_field_size, self.reader.data_dim), name="samples")
        gc = tf.placeholder(tf.int32, shape=(receptive_field_size), name="gc")
        lc = tf.placeholder(tf.int32, shape=(receptive_field_size), name="lc")

        gc = tf.one_hot(gc, GC_CHANNELS)
        lc = tf.one_hot(lc, LC_CHANNELS)

        predict = self.net.predict_proba(samples, gc, lc)

        '''does nothing'''
        with self.test_session() as session:
            session.run([
                tf.local_variables_initializer(),
                tf.global_variables_initializer(),
                tf.tables_initializer(),
            ])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            for ITER in range(1):

                for i in range(1000):
                    _, loss_val = session.run([train_op, loss])
                    print("step %d loss %.4f" % (i, loss_val), end='\r')
                    sys.stdout.flush()
                print()

                data_samples = np.random.random((receptive_field_size, self.reader.data_dim))
                gc_samples = np.zeros((receptive_field_size))
                lc_samples = np.zeros((receptive_field_size))

                output = []

                for EMO in range(3):
                    for PHO in range(3):
                        for _ in range(100):
                            prediction = session.run(predict, feed_dict={'samples:0': data_samples, 'gc:0': gc_samples, 'lc:0': lc_samples})
                            data_samples = data_samples[1:, :]
                            data_samples = np.append(data_samples, prediction, axis=0)

                            gc_samples = gc_samples[1:]
                            gc_samples = np.append(gc_samples, [EMO], axis=0)
                            lc_samples = lc_samples[1:]
                            lc_samples = np.append(lc_samples, [PHO], axis=0)

                            output.append(prediction[0])

                output = np.array(output)
                print("ITER %d" % ITER)
                plt.imsave("./test/SINE_test_%d.png" % ITER, np.kron(output[:, :], np.ones([1, 500])), vmin=0.0, vmax=1.0)

if __name__ == '__main__':
    tf.test.main()
