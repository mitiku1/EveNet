from __future__ import print_function

import json
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime
from termcolor import colored
from wavenet import WaveNetModel, CsvReader, optimizer_factory

TEST_DATA = "./test/generation_test_data"

GC_CHANNELS = 2
LC_CHANNELS = 16

SAMPLE_SIZE = 2


class TestGeneration(tf.test.TestCase):

    def testGenerateSimple(self):
        # Reader config
        with open(TEST_DATA + "/config.json") as json_file:
            self.reader_config = json.load(json_file)

        # Initialize the reader
        receptive_field_size = WaveNetModel.calculate_receptive_field(2, [1, 1], False, 8)

        self.reader = CsvReader(
            [TEST_DATA + "/test.dat", TEST_DATA + "/test.emo", TEST_DATA + "/test.pho"],
            batch_size=1,
            receptive_field=receptive_field_size,
            sample_size=SAMPLE_SIZE,
            config=self.reader_config
        )

        # WaveNet model
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 1],
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
                             l2_regularization_strength=0)

        optimizer = optimizer_factory['adam'](learning_rate=0.003, momentum=0.9)
        trainable = tf.trainable_variables()
        train_op = optimizer.minimize(loss, var_list=trainable)

        samples = tf.placeholder(tf.float32, shape=(receptive_field_size, self.reader.data_dim), name="samples")
        gc = tf.placeholder(tf.int32, shape=(receptive_field_size), name="gc")
        lc = tf.placeholder(tf.int32, shape=(receptive_field_size, 4), name="lc")

        gc = tf.one_hot(gc, GC_CHANNELS)
        lc = tf.one_hot(lc, int(LC_CHANNELS / 4))

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

            for i in range(3000):
                _, loss_val = session.run([train_op, loss])
                print("step %d loss %.4f" % (i, loss_val), end='\r')
                sys.stdout.flush()
            print()

            data_samples = np.random.random((receptive_field_size, self.reader.data_dim))
            gc_samples = np.zeros((receptive_field_size))
            lc_samples = np.zeros((receptive_field_size, 4))

            # WITH CONDITIONING.
            error = 0.0
            i = 0.0
            for p in range(3):
                for q in range(3):
                    gc_samples[:] = p
                    lc_samples[:, :] = q
                    for _ in range(64):
                        prediction = session.run(predict, feed_dict={'samples:0': data_samples, 'gc:0': gc_samples, 'lc:0': lc_samples})
                        data_samples = data_samples[1:, :]
                        data_samples = np.append(data_samples, prediction, axis=0)
                    print("G%d L%d - %.2f vs %.2f ERR %.2f" % (p, q, i, np.average(prediction), np.abs(i - np.average(prediction))))
                    error += np.abs(i - np.average(prediction))
                    data_samples = np.random.random((receptive_field_size, self.reader.data_dim))
                    i += 0.1

            print("TOTAL ERROR CONDITIONING: %.5f" % error)
            # WITHOUT CONDITIONING.

            data_samples = np.random.random((receptive_field_size, self.reader.data_dim))

            errorNo = 0.0
            i = 0.0
            for p in range(3):
                for q in range(3):
                    gc_samples[:] = 0
                    lc_samples[:, :] = 0
                    for _ in range(32):
                        prediction = session.run(predict, feed_dict={'samples:0': data_samples, 'gc:0': gc_samples, 'lc:0': lc_samples})
                        data_samples = data_samples[1:, :]
                        data_samples = np.append(data_samples, prediction, axis=0)
                    print("G%d L%d - %.2f vs %.2f ERR %.2f" % (p, q, i, np.average(prediction), (i - np.average(prediction))))
                    errorNo += np.abs(i - np.average(prediction))
                    data_samples = np.random.random((receptive_field_size, self.reader.data_dim))
                    i += 0.1

            print("TOTAL ERROR NO CONDITIONING: %.5f" % errorNo)
            self.assertTrue(error < 0.5)
            self.assertTrue(errorNo > 0.05)

if __name__ == '__main__':
    tf.test.main()
