# Copyright 2019 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
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

TYPES = (np.float16, np.float32, np.int32)
TESTCASES = [{"testcase_name": np.dtype(x).name, "dtype": x} for x in TYPES]


def _get_random_input(dtype, shape):
  if np.issubdtype(dtype, np.integer):
    info_fn = np.iinfo
    random_fn = np.random.random_integers
  else:
    info_fn = np.finfo
    random_fn = np.random.uniform
  return random_fn(
      info_fn(dtype).min, info_fn(dtype).max, size=shape).astype(dtype)


class ArgMinMax(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxBasic(self, dtype):
    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmax(a, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(dtype, [3, 5, 2])
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = _get_random_input(dtype, (3, 5, 2))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmax(input, axis=0))

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxHalf(self, dtype):
    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmax(a, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(dtype, [3, 5, 2])

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      input = _get_random_input(dtype, (3, 5, 2))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmax(input, axis=0))

  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxMultiDimensional(self, dtype):
    batchsize = 4
    n_categories = 1200

    def model(a, axis):
      return math_ops.argmax(a, axis=axis, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(dtype, [1, 2, 3, 4, 5, 6])
      p_axis = array_ops.placeholder(np.int32, shape=())

    with ops.device("/device:IPU:0"):
      out = model(pa, p_axis)

    tu.configure_ipu_system()

    for axis in range(6):
      with tu.ipu_session() as sess:
        input = _get_random_input(dtype, (1, 2, 3, 4, 5, 6))

        fd = {pa: input, p_axis: axis}
        result = sess.run(out, fd)
        self.assertAllClose(result, np.argmax(input, axis=axis))

  @parameterized.named_parameters(*TESTCASES)
  def testArgMinBasic(self, dtype):
    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmin(a, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(dtype, [3, 5, 2])
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = _get_random_input(dtype, (3, 5, 2))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmin(input, axis=0))

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

  @parameterized.named_parameters(*TESTCASES)
  def testArgMinHalf(self, dtype):
    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmin(a, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(dtype, [3, 5, 2])

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      input = _get_random_input(dtype, (3, 5, 2))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmin(input, axis=0))

  @parameterized.named_parameters(*TESTCASES)
  def testArgMinMultiDimensional(self, dtype):
    batchsize = 4
    n_categories = 1200

    def model(a, axis):
      return math_ops.argmin(a, axis=axis, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(dtype, [1, 2, 3, 4, 5, 6])
      p_axis = array_ops.placeholder(np.int32, shape=())

    with ops.device("/device:IPU:0"):
      out = model(pa, p_axis)

    tu.configure_ipu_system()

    for axis in range(6):
      with tu.ipu_session() as sess:
        input = _get_random_input(dtype, (1, 2, 3, 4, 5, 6))

        fd = {pa: input, p_axis: axis}
        result = sess.run(out, fd)
        self.assertAllClose(result, np.argmin(input, axis=axis))

  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxNegativeDim(self, dtype):
    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmax(a, axis=-1, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(dtype, [3, 5, 2])
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = _get_random_input(dtype, (3, 5, 2))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmax(input, axis=-1))

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxVector(self, dtype):
    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmax(a, axis=0, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(dtype, [3])
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = _get_random_input(dtype, (3))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmax(input))

      result = sess.run(report)
      self.assertTrue(len(result) == 3)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
