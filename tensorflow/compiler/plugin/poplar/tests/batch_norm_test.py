# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util

import numpy as np

class IpuXlaBatchNormTest(test_util.TensorFlowTestCase):

    def testBatchNormalize1(self):

        vscope = tf.get_variable_scope()
        vscope.set_use_resource(True)

        with tf.device("/device:IPU:0"):
            with tf.Session() as sess:
                x = tf.placeholder(tf.float32, [1,64,64,4], name="a")

                beta = tf.get_variable("x", shape=[4], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.0))
                gamma = tf.get_variable("y", shape=[4], dtype=tf.float32,
                        initializer=tf.constant_initializer(1.0))

                b_mean, b_var = tf.nn.moments(x, [0,1,2], name='moments')

                normed = tf.nn.batch_normalization(x,
                                                   b_mean, b_var,
                                                   beta, gamma,
                                                   1e-3)

                fd = {
                    x: np.zeros([1,64,64,4])
                }

                sess.run(tf.global_variables_initializer())

                result = sess.run(normed, fd)
                self.assertAllClose(result,
                                    np.zeros([1,64,64,4]))

if __name__ == "__main__":
    googletest.main()
