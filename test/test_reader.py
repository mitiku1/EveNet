import numpy as np
import tensorflow as tf
from wavenet import CsvReader
import numpy as np
import json
import sys

TEST_DATA = "./test/reader_test_data"
# TODO: Test GC and LC with lookup


class TestReader(tf.test.TestCase):

    def setUp(self):
        with open(TEST_DATA + "/config.json") as json_file:
            self.reader_config = json.load(json_file)

        self.reader = CsvReader([TEST_DATA + "/test.dat", TEST_DATA + "/test.emo", TEST_DATA + "/test.pho"],
                                batch_size=1,
                                receptive_field=20,
                                sample_size=0,
                                config=self.reader_config)

    def testReaderSimple(self):
        '''test np vs tf reading'''
        with tf.Session() as sess:
            sess.run([
                tf.local_variables_initializer(),
                tf.global_variables_initializer(),
                tf.tables_initializer(),
            ])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                dat, gc, lc = sess.run([self.reader.data_batch, self.reader.gc_batch, self.reader.lc_batch])
                dat2, g2c, lc2 = sess.run([self.reader.data_batch, self.reader.gc_batch, self.reader.lc_batch])

            finally:
                coord.request_stop()
                coord.join(threads)

            np_data = np.genfromtxt(TEST_DATA + "/test.dat", delimiter=",")
            ref_dat = np.kron(np_data[:, :], np.ones([2, 1]))
            self.assertAllEqual(dat[0], ref_dat)
            self.assertAllEqual(dat2[0], ref_dat)


class TestReaderPartial(tf.test.TestCase):

    def setUp(self):
        with open(TEST_DATA + "/config.json") as json_file:
            self.reader_config = json.load(json_file)

        self.reader = CsvReader([TEST_DATA + "/test.dat", TEST_DATA + "/test.emo", TEST_DATA + "/test.pho"],
                                batch_size=1,
                                receptive_field=5,
                                sample_size=0,
                                config=self.reader_config)

    def testReaderSimple(self):
        '''test np vs tf reading'''
        with tf.Session() as sess:
            sess.run([
                tf.local_variables_initializer(),
                tf.global_variables_initializer(),
                tf.tables_initializer(),
            ])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                dat, gc, lc = sess.run([self.reader.data_batch, self.reader.gc_batch, self.reader.lc_batch])
                dat2, g2c, lc2 = sess.run([self.reader.data_batch, self.reader.gc_batch, self.reader.lc_batch])

            finally:
                coord.request_stop()
                coord.join(threads)

            np_data = np.genfromtxt(TEST_DATA + "/test.dat", delimiter=",")
            self.assertAllEqual(dat[0], np_data[:5, :])
            self.assertAllEqual(dat2[0], np_data[5:, :])


if __name__ == '__main__':
    tf.test.main()
