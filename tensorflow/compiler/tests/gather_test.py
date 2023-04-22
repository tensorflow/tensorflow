# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functional tests for XLA Gather Op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test

FLAGS = flags.FLAGS


class GatherTest(xla_test.XLATestCase):

  def _buildParams(self, data, dtype):
    data = data.astype(dtype.as_numpy_dtype)
    # For complex types, adds an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testScalar1D(self):
    with self.session() as session, self.test_scope():
      data = np.array([0, 1, 2, 3, 7, 5])
      for dtype in self.all_tf_types:
        for indices in 4, [4], [1, 2, 2, 4, 5]:
          params_np = self._buildParams(data, dtype)
          params = array_ops.placeholder(dtype=dtype)
          indices_tf = constant_op.constant(indices)
          gather_t = array_ops.gather(params, indices_tf)
          gather_val = session.run(gather_t, feed_dict={params: params_np})
          np_val = constant_op.constant(params_np[indices])
          self.assertAllEqual(np_val, gather_val)

  def testScalar2D(self):
    with self.session() as session, self.test_scope():
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                       [12, 13, 14]])
      for dtype in self.all_tf_types:
        for axis in 0, 1, -1:
          params_np = self._buildParams(data, dtype)
          params = array_ops.placeholder(dtype=dtype)
          indices = constant_op.constant(2)
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = session.run(gather_t, feed_dict={params: params_np})
          expected = constant_op.constant(
              np.take(params_np, 2, axis=axis), dtype)
          self.assertAllEqual(expected, gather_val)

  def testSimpleTwoD32(self):
    with self.session() as session, self.test_scope():
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                       [12, 13, 14]])
      for dtype in self.all_tf_types:
        for axis in 0, 1, -1:
          params_np = self._buildParams(data, dtype)
          params = array_ops.placeholder(dtype=dtype)
          # The indices must be in bounds for any axis.
          indices = constant_op.constant([0, 1, 0, 2])
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = session.run(gather_t, feed_dict={params: params_np})
          expected = constant_op.constant(
              np.take(params_np, [0, 1, 0, 2], axis=axis), dtype)
          self.assertAllEqual(expected, gather_val)

  def testSimpleTwoD32_Int64Indices(self):
    if np.int64 not in self.int_types:
      return

    with self.session() as session, self.test_scope():
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                       [12, 13, 14]])
      # The indices must be in bounds for any axis.
      indices_np = np.array([0, 1, 0, 2])
      for dtype in self.all_tf_types:
        for axis in 0, 1, -1:
          params_np = self._buildParams(data, dtype)
          params = array_ops.placeholder(dtype=dtype)
          indices = array_ops.placeholder(dtype=dtypes.int64)
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = session.run(
              gather_t, feed_dict={
                  params: params_np,
                  indices: indices_np
              })
          expected = constant_op.constant(
              np.take(params_np, [0, 1, 0, 2], axis=axis), dtype)
          self.assertAllEqual(expected, gather_val)

  def testHigherRank(self):
    """Check that scalar and empty indices shapes work as well."""
    shape = (2, 1, 3, 2)
    for indices_shape in (), (0,), (2, 0), (2, 3):
      for dtype in self.all_tf_types:
        for axis in 0, 1, 2, 3, -1, -2:
          params = self._buildParams(np.random.randn(*shape), dtype)
          indices = np.random.randint(shape[axis], size=indices_shape)
          with self.session() as sess, self.test_scope():
            tf_params = array_ops.placeholder(dtype=dtype)
            tf_indices = constant_op.constant(indices, dtype=dtypes.int32)
            gather = array_ops.gather(tf_params, tf_indices, axis=axis)
            gather_value = sess.run(gather, feed_dict={tf_params: params})
            gather_np = constant_op.constant(
                np.take(params, indices, axis=axis), dtype)
            self.assertAllEqual(gather_np, gather_value)

  def testIndicesWithDifferentDimensions(self):
    with self.session():
      for dtype in self.numeric_tf_types:
        params = array_ops.placeholder(dtype=dtype)
        indices = array_ops.placeholder(dtype=np.int32)
        with self.test_scope():
          gather = array_ops.gather(params, indices)
        self.assertAllEqual(
            7, gather.eval(feed_dict={params: [4, 7, 2], indices: 1}))
        self.assertAllEqual(
            [7], gather.eval(feed_dict={params: [4, 7, 2], indices: [1]}))
        self.assertAllEqual(
            [[7]], gather.eval(feed_dict={params: [4, 7, 2], indices: [[1]]}))

  def testGatherPrecision(self):
    with self.session() as session, self.test_scope():
      data = np.array([[0, 0, 0, 0], [0, 2 * (1 + np.exp2(-8)), 0, 0],
                       [0, 0, 0, 0], [0.015789, 0.0985, 0.55789, 0.3842]])
      indices = np.array([1, 2, 3, 1])
      dtype = dtypes.float32
      params_np = self._buildParams(data, dtype)
      params = array_ops.placeholder(dtype=dtype)
      indices_tf = constant_op.constant(indices)
      gather_t = array_ops.gather(params, indices_tf)
      gather_val = session.run(gather_t, feed_dict={params: params_np})
      np_val = params_np[indices]
      self.assertAllEqual(np_val, gather_val)


class GatherBenchmark(test.Benchmark):
  """Microbenchmarks for the gather op."""

  def _benchmarkGather(self, name, axis, gather_indices, use_xla_jit):

    def BuilderFn():
      inputs = variables.Variable(
          array_ops.zeros([100, 100, 10, 100, 50], dtype=dtypes.float32),
          dtype=dtypes.float32,
          name='input')
      indices = variables.Variable(
          gather_indices, dtype=dtypes.int32, name='indices')
      gather_t = array_ops.gather(inputs, indices, axis=axis)
      return '%s.axis%d' % (name, axis), [gather_t]

    xla_test.Benchmark(self, BuilderFn, use_xla_jit=use_xla_jit, device='cpu')

  def _benchmarkSliceGather(self, axis, use_xla_jit):
    """Benchmarks a gather op that's really a dynamic slice."""
    self._benchmarkGather('slice_gather', axis, [1], use_xla_jit)

  def _benchmarkNontrivialGather(self, axis, use_xla_jit):
    self._benchmarkGather('nontrivial_gather', axis, [9, 1, 0, 2] * 4,
                          use_xla_jit)

  def benchmarkSliceGatherAxis0(self):
    self._benchmarkSliceGather(axis=0, use_xla_jit=False)

  def benchmarkSliceGatherAxis0XLA(self):
    self._benchmarkSliceGather(axis=0, use_xla_jit=True)

  def benchmarkSliceGatherAxis1(self):
    self._benchmarkSliceGather(axis=1, use_xla_jit=False)

  def benchmarkSliceGatherAxis1XLA(self):
    self._benchmarkSliceGather(axis=1, use_xla_jit=True)

  def benchmarkSliceGatherAxis4(self):
    self._benchmarkSliceGather(axis=4, use_xla_jit=False)

  def benchmarkSliceGatherAxis4XLA(self):
    self._benchmarkSliceGather(axis=4, use_xla_jit=True)

  def benchmarkNontrivialGatherAxis0(self):
    self._benchmarkNontrivialGather(axis=0, use_xla_jit=False)

  def benchmarkNontrivialGatherAxis0XLA(self):
    self._benchmarkNontrivialGather(axis=0, use_xla_jit=True)

  def benchmarkNontrivialGatherAxis1(self):
    self._benchmarkNontrivialGather(axis=1, use_xla_jit=False)

  def benchmarkNontrivialGatherAxis1XLA(self):
    self._benchmarkNontrivialGather(axis=1, use_xla_jit=True)

  def benchmarkNontrivialGatherAxis4(self):
    self._benchmarkNontrivialGather(axis=4, use_xla_jit=False)

  def benchmarkNontrivialGatherAxis4XLA(self):
    self._benchmarkNontrivialGather(axis=4, use_xla_jit=True)


if __name__ == '__main__':
  test.main()
