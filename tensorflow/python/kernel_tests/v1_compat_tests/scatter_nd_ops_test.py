# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for scatter_nd_ops that only work in V1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
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


def _NumpyMin(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, np.minimum)


def _NumpyMax(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, np.maximum)


class StatefulScatterNdTest(test.TestCase):

  def _VariableRankTest(self,
                        np_scatter,
                        tf_scatter,
                        vtype,
                        itype,
                        repeat_indices=False):
    np.random.seed(8)
    ref_shapes = [(3, 6), (3, 6), (3, 6, 9), (3, 6, 9), (3, 6, 9), (3, 6, 9)]
    indices_shapes = [(2,), (2, 2), (2,), (2, 2), (2, 3), (2, 3, 3)]
    with test_util.device(use_gpu=True):
      for ref_shape, indices_shape in zip(ref_shapes, indices_shapes):
        num_updates = indices_shape[0]
        ixdim = indices_shape[-1]

        indexable_area_shape = ()
        for i in range(ixdim):
          indexable_area_shape += (ref_shape[i],)
        all_indices = [
            list(coord) for coord, _ in np.ndenumerate(
                np.empty(indexable_area_shape, vtype))
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
        ref = _AsType(np.random.randn(*(ref_shape)), vtype)

        # Scatter via numpy
        new = ref.copy()
        np_scatter(new, indices, updates)
        # Scatter via tensorflow
        ref_var = variables.VariableV1(ref)
        self.evaluate(ref_var.initializer)
        self.evaluate(tf_scatter(ref_var, indices, updates))

        # Compare
        self.assertAllClose(new, self.evaluate(ref_var))

  def _ScatterRepeatIndicesTest(self, np_scatter, tf_scatter):
    for vtype in (np.int32, np.float16, np.float32, np.float64):
      for itype in (np.int32, np.int64):
        self._VariableRankTest(
            np_scatter, tf_scatter, vtype, itype, repeat_indices=True)

  @test_util.run_v1_only("Don't need to test VariableV1 in TF2")
  def testScatterRepeatIndicesMinMax(self):
    """This tests scatter_add using indices that repeat."""
    self._ScatterRepeatIndicesTest(_NumpyMin, state_ops.scatter_nd_min)
    self._ScatterRepeatIndicesTest(_NumpyMax, state_ops.scatter_nd_max)

  @test_util.run_v1_only("Don't need to test VariableV1 in TF2")
  def testScatterOutOfRangeCpu(self):
    for op in (state_ops.scatter_nd_min, state_ops.scatter_nd_max):
      params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
      updates = np.array([-3, -4, -5]).astype(np.float32)
      with self.cached_session(use_gpu=False):
        ref = variables.VariableV1(params)
        self.evaluate(ref.initializer)

        # Indices all in range, no problem.
        indices = np.array([[2], [0], [5]])
        self.evaluate(op(ref, indices, updates))

        # Test some out of range errors.
        indices = np.array([[-1], [0], [5]])
        with self.assertRaisesOpError(
            r"indices\[0\] = \[-1\] does not index into shape \[6\]"):
          op(ref, indices, updates).eval()

        indices = np.array([[2], [0], [6]])
        with self.assertRaisesOpError(
            r"indices\[2\] = \[6\] does not index into shape \[6\]"):
          op(ref, indices, updates).eval()
