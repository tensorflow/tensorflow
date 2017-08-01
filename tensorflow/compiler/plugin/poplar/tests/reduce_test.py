# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util

import numpy as np

class IpuXlaConvTest(test_util.TensorFlowTestCase):

    def testConv3x3_Pad1x1(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float32, [2,7,7,1024], name="a")
                output = tf.reduce_mean(pa, reduction_indices=[1,2])

                fd = {
                    pa: np.zeros([2,7,7,1024])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.zeros([2,1024]))

    def testAvgPoolSamePaddingWithStridesF32(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float32, [1,10,10,1], name="a")
                output = tf.nn.avg_pool(pa, ksize=[1,5,5,1], strides=[1,2,2,1],
                                        padding='SAME', name="avg")

                fd = {
                    pa: np.ones([1,10,10,1])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,5,5,1]))

    def testAvgPoolSamePaddingWithStridesF16(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float16, [1,10,10,1], name="a")
                output = tf.nn.avg_pool(pa, ksize=[1,5,5,1], strides=[1,2,2,1],
                                        padding='SAME')

                fd = {
                    pa: np.ones([1,10,10,1])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,5,5,1]))

    def testAvgPoolValidPaddingWithStridesF32(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float32, [1,10,10,1], name="a")
                output = tf.nn.avg_pool(pa, ksize=[1,5,5,1], strides=[1,2,2,1],
                                        padding='VALID')

                fd = {
                    pa: np.ones([1,10,10,1])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,3,3,1]))

    def testAvgPoolValidPaddingWithStridesF16(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float16, [1,10,10,1], name="a")
                output = tf.nn.avg_pool(pa, ksize=[1,5,5,1], strides=[1,2,2,1],
                                        padding='VALID')

                fd = {
                    pa: np.ones([1,10,10,1])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,3,3,1]))

if __name__ == "__main__":
    import time
    time.sleep(20)
    googletest.main()
