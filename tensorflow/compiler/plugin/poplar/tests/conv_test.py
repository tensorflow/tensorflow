# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops

import numpy as np

class IpuXlaConvTest(test_util.TensorFlowTestCase):

    def testConv3x3_Pad1x1(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float32, [1,112,112,64], name="a")
                pb = tf.placeholder(tf.float32, [3,3,64,128], name="b")
                output = nn_ops.convolution(pa, pb, padding="SAME")

                fd = {
                    pa: np.zeros([1,112,112,64]),
                    pb: np.zeros([3,3,64,128])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.zeros([1,112,112,128]))

    def testConv3x3_WithBias(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float32, [1,112,112,64], name="a")
                pb = tf.placeholder(tf.float32, [3,3,64,128], name="b")
                bi = tf.placeholder(tf.float32, [128], name="b")
                output = nn_ops.convolution(pa, pb, padding="SAME")
                output = nn_ops.bias_add(output, bi)

                fd = {
                    pa: np.zeros([1,112,112,64]),
                    pb: np.zeros([3,3,64,128]),
                    bi: np.zeros([128]),
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.zeros([1,112,112,128]))


if __name__ == "__main__":
    googletest.main()
