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

if __name__ == "__main__":
    googletest.main()
