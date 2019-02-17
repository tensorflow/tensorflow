# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class IpuXlaConvTest(test_util.TensorFlowTestCase):

  def testReductionMeanDim12(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 7, 7, 32], name="a")
        output = math_ops.reduce_mean(pa, axis=[1, 2])

        fd = {
          pa: np.ones([2, 7, 7, 32])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([2, 32]))

  def testReductionMeanDim03(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 7, 7, 32], name="a")
        output = math_ops.reduce_mean(pa, axis=[0, 3])

        fd = {
          pa: np.ones([2, 7, 7, 32])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([7, 7]))

  def testReductionMeanDim13(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 7, 7, 32], name="a")
        output = math_ops.reduce_mean(pa, axis=[1, 3])

        fd = {
          pa: np.ones([2, 7, 7, 32])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([2, 7]))

  def testReductionMeanDim23(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 7, 7, 32], name="a")
        output = math_ops.reduce_mean(pa, axis=[2, 3])

        fd = {
          pa: np.ones([2, 7, 7, 32])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([2, 7]))

  def testAvgPoolSamePaddingWithStridesF32(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10], name="a")
        output = nn.avg_pool(pa, ksize=[1, 1, 5, 5], strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='SAME', name="avg")

        fd = {
          pa: np.ones([1, 1, 10, 10])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([1, 1, 5, 5]))

  def testAvgPoolSamePaddingWithStridesF16(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float16, [1, 1, 10, 10], name="a")
        output = nn.avg_pool(pa, ksize=[1, 1, 5, 5], strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='SAME')

        fd = {
          pa: np.ones([1, 1, 10, 10])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([1, 1, 5, 5]))

  def testAvgPoolValidPaddingWithStridesF32(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10], name="a")
        output = nn.avg_pool(pa, ksize=[1, 1, 5, 5], strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='VALID')

        fd = {
          pa: np.ones([1, 1, 10, 10])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([1, 1, 3, 3]))

  def testAvgPoolValidPaddingWithStridesF16(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float16, [1, 1, 10, 10], name="a")
        output = nn.avg_pool(pa, ksize=[1, 1, 5, 5], strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='VALID')

        fd = {
          pa: np.ones([1, 1, 10, 10])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([1, 1, 3, 3]))

  def testMaxPoolSamePaddingWithStridesF32(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10], name="a")
        output = nn.max_pool(pa, ksize=[1, 1, 5, 5], strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='SAME', name="max")

        fd = {
          pa: np.ones([1, 1, 10, 10])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([1, 1, 5, 5]))

  def testMaxPoolValidPaddingWithStridesF32(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10], name="a")
        output = nn.max_pool(pa, ksize=[1, 1, 5, 5], strides=[1, 1, 2, 2],
                             data_format='NCHW',
                             padding='VALID', name="max")

        fd = {
          pa: np.ones([1, 1, 10, 10])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([1, 1, 3, 3]))

  def testAvgPoolSamePaddingWithStridesF32Dim12(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [1, 10, 10, 1], name="a")
        output = nn.avg_pool(pa, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1],
                             data_format='NHWC',
                             padding='SAME', name="avg")

        fd = {
          pa: np.ones([1, 10, 10, 1])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([1, 5, 5, 1]))

  def testAvgPoolValidPaddingWithStridesF32Dim12(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [1, 10, 10, 1], name="a")
        output = nn.avg_pool(pa, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1],
                             data_format='NHWC',
                             padding='VALID', name="avg")

        fd = {
          pa: np.ones([1, 10, 10, 1])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.ones([1, 3, 3, 1]))


if __name__ == "__main__":
  googletest.main()
