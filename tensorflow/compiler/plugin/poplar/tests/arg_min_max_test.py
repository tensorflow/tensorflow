# Copyright 2019 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class ArgMinMax(test_util.TensorFlowTestCase):
  def testArgMaxBasic(self):
    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmax(a, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [3, 5, 2])
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = np.random.rand(3, 5, 2)

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result[0], np.argmax(input))

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

  def testArgMaxHalf(self):
    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmax(a, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float16, [3, 5, 2])

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      input = np.random.rand(3, 5, 2)

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result[0], np.argmax(input))

  def testArgMaxMultiDimensional(self):
    batchsize = 4
    n_categories = 1200

    def model(a, axis):
      return math_ops.argmax(a, axis=axis, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [3, 4, 5, 2, 4, 6])
      p_axis = array_ops.placeholder(np.int32, shape=())

    with ops.device("/device:IPU:0"):
      out = model(pa, p_axis)

    tu.configure_ipu_system()

    # We ignore axis=0 because np.argmax(axis=0) is not the same as tf.argmax(axis=0).
    for axis in range(1, 5):
      with tu.ipu_session() as sess:
        input = np.random.rand(3, 4, 5, 2, 4, 6)

        fd = {pa: input, p_axis: axis}
        result = sess.run(out, fd)
        self.assertAllClose(result, np.argmax(input, axis=axis))

  def testArgMinBasic(self):
    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmin(a, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [3, 5, 2])
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = np.random.rand(3, 5, 2)

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result[0], np.argmin(input))

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

  def testArgMinHalf(self):
    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmin(a, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float16, [3, 5, 2])

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      input = np.random.rand(3, 5, 2)

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result[0], np.argmin(input))

  def testArgMinMultiDimensional(self):
    batchsize = 4
    n_categories = 1200

    def model(a, axis):
      return math_ops.argmin(a, axis=axis, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [3, 4, 5, 2, 4, 6])
      p_axis = array_ops.placeholder(np.int32, shape=())

    with ops.device("/device:IPU:0"):
      out = model(pa, p_axis)

    tu.configure_ipu_system()

    # We ignore axis=0 because np.argmax(axis=0) is not the same as tf.argmax(axis=0).
    for axis in range(1, 5):
      with tu.ipu_session() as sess:
        input = np.random.rand(3, 4, 5, 2, 4, 6)

        fd = {pa: input, p_axis: axis}
        result = sess.run(out, fd)
        self.assertAllClose(result, np.argmin(input, axis=axis))


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
