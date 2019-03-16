# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.tf.scatter_nd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


def _AsType(v, vtype):
  return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)


def _FlatInnerDims(tensor, ndims=2):
  shape = list(tensor.shape)
  return tensor.reshape(
      [functools.reduce(lambda x, y: x * y, shape[:-ndims + 1], 1)] +
      shape[-ndims + 1:])


def _FlatOuterDims(tensor, ndims=2):
  shape = list(tensor.shape)
  return tensor.reshape(
      shape[:ndims - 1] +
      [functools.reduce(lambda x, y: x * y, shape[ndims - 1:], 1)])


def _NumpyScatterNd(ref, indices, updates, op):
  ixdim = indices.shape[-1]
  num_updates = indices.size // ixdim
  total_nd = len(ref.shape)
  slice_size = 1
  for i in range(ixdim, total_nd):
    slice_size *= ref.shape[i]
  flat_indices = _FlatInnerDims(indices)
  flat_updates = updates.reshape((num_updates, slice_size))
  output_flat = _FlatOuterDims(ref, ixdim + 1)
  for ix_updates, ix_output in enumerate(flat_indices):
    ix_output = tuple(ix_output)
    output_flat[ix_output] = op(output_flat[ix_output],
                                flat_updates[ix_updates])
  return output_flat.reshape(ref.shape)


def _NumpyUpdate(indices, updates, shape):
  ref = np.zeros(shape, dtype=updates.dtype)
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: u)


class ScatterNdTest(xla_test.XLATestCase):

  def _VariableRankTest(self,
                        np_scatter,
                        tf_scatter,
                        vtype,
                        itype,
                        repeat_indices=False):
    np.random.seed(8)
    ref_shapes = [(3, 6), (3, 6), (3, 6, 9), (3, 6, 9), (3, 6, 9), (3, 6, 9)]
    indices_shapes = [(2,), (2, 2), (2,), (2, 2), (2, 3), (2, 3, 3)]
    for ref_shape, indices_shape in zip(ref_shapes, indices_shapes):
      num_updates = indices_shape[0]
      ixdim = indices_shape[-1]

      indexable_area_shape = ()
      for i in range(ixdim):
        indexable_area_shape += (ref_shape[i],)
      all_indices = [
          list(coord)
          for coord, _ in np.ndenumerate(np.empty(indexable_area_shape, vtype))
      ]
      np.random.shuffle(all_indices)
      indices = np.array(all_indices[:num_updates])

      if num_updates > 1 and repeat_indices:
        indices = indices[:num_updates // 2]
        for _ in range(num_updates - num_updates // 2):
          indices = np.append(
              indices, [indices[np.random.randint(num_updates // 2)]], axis=0)
        np.random.shuffle(indices)
      indices = _AsType(indices[:num_updates], itype)

      updates_shape = (num_updates,)
      for i in range(ixdim, len(ref_shape)):
        updates_shape += (ref_shape[i],)
      updates = _AsType(np.random.randn(*(updates_shape)), vtype)

      # Scatter via numpy
      np_out = np_scatter(indices, updates, ref_shape)
      # Scatter via tensorflow
      tf_out = tf_scatter(indices, updates, ref_shape)

      self.assertAllClose(np_out, tf_out)

  def _VariableRankTests(self, np_scatter, tf_scatter):
    for vtype in self.numeric_types:
      for itype in set([np.int32, np.int64]).intersection(set(self.int_types)):
        self._VariableRankTest(np_scatter, tf_scatter, vtype, itype)

  def _runScatterNd(self, indices, updates, shape):
    with self.cached_session():
      updates_placeholder = array_ops.placeholder(updates.dtype)
      indices_placeholder = array_ops.placeholder(indices.dtype)
      with self.test_scope():
        output = array_ops.scatter_nd(indices_placeholder, updates_placeholder,
                                      shape)
      feed_dict = {updates_placeholder: updates, indices_placeholder: indices}
      return output.eval(feed_dict=feed_dict)

  def testSimple(self):
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.float32)
    expected = np.array([0, 11, 0, 10, 9, 0, 0, 12], dtype=np.int32)
    self.assertAllEqual(expected, self._runScatterNd(indices, updates, [8]))

  def testRepeatedIndices(self):
    indices = np.array([[0], [1], [0], [1]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.float32)
    expected = np.array([20, 22], dtype=np.int32)
    self.assertAllEqual(expected, self._runScatterNd(indices, updates, [2]))

  def testSimple2(self):
    indices = np.array([[1, 0], [1, 1]], dtype=np.int32)
    updates = np.array([11., 12.], dtype=np.float32)
    expected = np.array([[0., 0.], [11., 12.], [0., 0.]], dtype=np.float32)
    self.assertAllEqual(expected, self._runScatterNd(indices, updates, [3, 2]))

  def testSimple3(self):
    indices = np.array([[1]], dtype=np.int32)
    updates = np.array([[11., 12.]], dtype=np.float32)
    expected = np.array([[0., 0.], [11., 12.], [0., 0.]])
    self.assertAllEqual(expected, self._runScatterNd(indices, updates, [3, 2]))

  def testVariableRankUpdate(self):
    self._VariableRankTests(_NumpyUpdate, self._runScatterNd)

  def testExtraIndicesDimensions(self):
    indices = np.zeros([1, 1, 2], np.int32)
    updates = np.zeros([1, 1], np.int32)
    expected = np.zeros([2, 2], dtype=np.int32)
    self.assertAllEqual(expected, self._runScatterNd(indices, updates, [2, 2]))

  def testRank3InvalidShape1(self):
    indices = np.zeros([3, 2, 2], np.int32)
    updates = np.zeros([2, 2, 2], np.int32)
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                             "Must have updates.shape"):
      self._runScatterNd(indices, updates, [2, 2, 2])

  def testRank3InvalidShape2(self):
    indices = np.zeros([2, 2, 1], np.int32)
    updates = np.zeros([2, 2], np.int32)
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                             "Must have updates.shape"):
      self._runScatterNd(indices, updates, [2, 2, 2])

  def testScatterOutOfRange(self):
    updates = np.array([-3, -4, -5]).astype(np.float32)

    # Indices all in range, no problem.
    indices = np.array([[2], [0], [5]], dtype=np.int32)
    self._runScatterNd(indices, updates, [6])

    # Indices out of range should not fail. It produces implementation-defined
    # output.
    indices = np.array([[-1], [0], [5]], dtype=np.int32)
    self._runScatterNd(indices, updates, [6])
    indices = np.array([[2], [0], [6]], dtype=np.int32)
    self._runScatterNd(indices, updates, [6])


if __name__ == "__main__":
  test.main()
