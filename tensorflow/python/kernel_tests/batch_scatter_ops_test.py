# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.tf.scatter."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _AsType(v, vtype):
  return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)


def _NumpyUpdate(ref, indices, updates):
  for i, indx in np.ndenumerate(indices):
    indx = i[:-1] + (indx,)
    ref[indx] = updates[i]


_TF_OPS_TO_NUMPY = {
    state_ops.batch_scatter_update: _NumpyUpdate,
}


class ScatterTest(test.TestCase):

  def _VariableRankTest(self,
                        tf_scatter,
                        vtype,
                        itype,
                        repeat_indices=False,
                        updates_are_scalar=False,
                        method=False):
    np.random.seed(8)
    with self.cached_session(use_gpu=False):
      for indices_shape in (2,), (3, 7), (3, 4, 7):
        for extra_shape in (), (5,), (5, 9):
          # Generate random indices with no duplicates for easy numpy comparison
          sparse_dim = len(indices_shape) - 1
          indices = np.random.randint(
              indices_shape[sparse_dim], size=indices_shape, dtype=itype)
          updates = _AsType(
              np.random.randn(*(indices_shape + extra_shape)), vtype)

          old = _AsType(np.random.randn(*(indices_shape + extra_shape)), vtype)

          # Scatter via numpy
          new = old.copy()
          np_scatter = _TF_OPS_TO_NUMPY[tf_scatter]
          np_scatter(new, indices, updates)
          # Scatter via tensorflow
          ref = variables.Variable(old)
          self.evaluate(variables.variables_initializer([ref]))

          if method:
            ref.batch_scatter_update(
                indexed_slices.IndexedSlices(indices, updates))
          else:
            self.evaluate(tf_scatter(ref, indices, updates))
          self.assertAllClose(ref, new)

  def testVariableRankUpdate(self):
    vtypes = [np.float32, np.float64]
    for vtype in vtypes:
      for itype in (np.int32, np.int64):
        self._VariableRankTest(
            state_ops.batch_scatter_update, vtype, itype)

  def testBooleanScatterUpdate(self):
    var = variables.Variable([True, False])
    update0 = state_ops.batch_scatter_update(var, [1], [True])
    update1 = state_ops.batch_scatter_update(
        var, constant_op.constant(
            [0], dtype=dtypes.int64), [False])
    self.evaluate(variables.variables_initializer([var]))

    self.evaluate([update0, update1])

    self.assertAllEqual([False, True], self.evaluate(var))

  def testScatterOutOfRange(self):
    params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
    updates = np.array([-3, -4, -5]).astype(np.float32)

    ref = variables.Variable(params)
    self.evaluate(variables.variables_initializer([ref]))

    # Indices all in range, no problem.
    indices = np.array([2, 0, 5])
    self.evaluate(state_ops.batch_scatter_update(ref, indices, updates))

    # Test some out of range errors.
    indices = np.array([-1, 0, 5])
    with self.assertRaisesOpError(
        r'indices\[0\] = \[-1\] does not index into shape \[6\]'):
      self.evaluate(state_ops.batch_scatter_update(ref, indices, updates))

    indices = np.array([2, 0, 6])
    with self.assertRaisesOpError(r'indices\[2\] = \[6\] does not index into '
                                  r'shape \[6\]'):
      self.evaluate(state_ops.batch_scatter_update(ref, indices, updates))

if __name__ == '__main__':
  test.main()
