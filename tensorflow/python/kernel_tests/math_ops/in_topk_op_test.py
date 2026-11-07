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
"""Tests for PrecisionOp."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class InTopKTest(test.TestCase):

  def _validateInTopK(self, predictions, target, k, expected):
    np_ans = np.array(expected, np.bool_)
    with self.cached_session():
      precision = nn_ops.in_top_k(predictions, target, k)
      out = self.evaluate(precision)
      self.assertAllClose(np_ans, out)
      self.assertShapeEqual(np_ans, precision)

  def testInTop1(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [3, 2]
    self._validateInTopK(predictions, target, 1, [True, False])

  def testInTop2(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [2, 2]
    self._validateInTopK(predictions, target, 2, [False, True])

  def testInTop2Tie(self):
    # Class 2 and 3 tie for 2nd, so both are considered in top 2.
    predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
    target = [2, 3]
    self._validateInTopK(predictions, target, 2, [True, True])

  def testInTop2_int64Target(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = np.asarray([0, 2]).astype(np.int64)
    self._validateInTopK(predictions, target, 2, [False, True])

  def testInTopNan(self):
    predictions = [[0.1, float("nan"), 0.2, 0.4], [0.1, 0.2, 0.3, float("inf")]]
    target = [1, 3]
    self._validateInTopK(predictions, target, 2, [False, False])

  def testBadTarget(self):
    predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
    target = [2, 4]  # must return False for invalid target
    self._validateInTopK(predictions, target, 2, [True, False])

  def testXlaOutOfRangeTarget(self):
    predictions = constant_op.constant(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    expected = [False, True, True, False]
    for dtype in (dtypes.int32, dtypes.int64):
      targets = constant_op.constant([-1, 0, 1, 2], dtype=dtype)

      @def_function.function(jit_compile=True)
      def in_top_k(predictions, targets):
        return nn_ops.in_top_k(predictions, targets, 1)

      self.assertAllEqual(
          self.evaluate(in_top_k(predictions, targets)), expected
      )

  def testEmpty(self):
    predictions = np.empty([0, 5])
    target = np.empty([0], np.int32)
    self._validateInTopK(predictions, target, 2, [])

  def testTensorK(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [0, 2]
    k = constant_op.constant(3)
    self._validateInTopK(predictions, target, k, [False, True])

  def testInTopKUnrelatedNaNInPredictions(self):
    # Regression test for GitHub issue #118376:
    # When any prediction in the row is NaN, the ordering is indeterminate,
    # so InTopK must return False even if the target itself is finite.
    predictions = constant_op.constant([[0.5, float("nan"), 0.7, 0.6]], dtype=dtypes.float32)
    target = constant_op.constant([2], dtype=dtypes.int32)

    # Standard eager / graph execution
    self._validateInTopK(predictions, target, 1, [False])

    # XLA-compiled execution
    @def_function.function(jit_compile=True)
    def jit_in_top_k(p, t):
      return nn_ops.in_top_k(p, t, 1)

    self.assertAllEqual(self.evaluate(jit_in_top_k(predictions, target)), [False])

  def testInTopKUnrelatedInfInPredictions(self):
    # When any prediction in the row is Inf, InTopK must return False.
    predictions = constant_op.constant([[0.5, float("inf"), 0.7, 0.6]], dtype=dtypes.float32)
    target = constant_op.constant([2], dtype=dtypes.int32)
    self._validateInTopK(predictions, target, 2, [False])

    @def_function.function(jit_compile=True)
    def jit_in_top_k(p, t):
      return nn_ops.in_top_k(p, t, 2)

    self.assertAllEqual(self.evaluate(jit_in_top_k(predictions, target)), [False])

  def testXlaInTopKNaNParity(self):
    # Comprehensive cross-backend test verifying exact eager/XLA parity across
    # batches with mixed finite and non-finite rows.
    predictions = constant_op.constant([
        [0.5, float("nan"), 0.7, 0.6],   # row 0: unrelated NaN -> False
        [0.1, 0.2, 0.3, 0.4],             # row 1: finite, target 3 is top-1 -> True
        [0.1, 0.2, 0.3, 0.4],             # row 2: finite, target 0 is not top-1 -> False
        [float("nan"), 0.2, 0.3, 0.4],   # row 3: target 0 is NaN -> False
        [0.1, float("inf"), 0.3, 0.4],   # row 4: unrelated Inf -> False
    ], dtype=dtypes.float32)
    expected = [False, True, False, False, False]

    for dtype in (dtypes.int32, dtypes.int64):
      targets = constant_op.constant([2, 3, 0, 0, 3], dtype=dtype)

      # Eager
      eager_res = nn_ops.in_top_k(predictions, targets, 1)
      self.assertAllEqual(self.evaluate(eager_res), expected)

      # XLA JIT
      @def_function.function(jit_compile=True)
      def compiled_fn(p, t):
        return nn_ops.in_top_k(p, t, 1)

      xla_res = compiled_fn(predictions, targets)
      self.assertAllEqual(self.evaluate(xla_res), expected)


if __name__ == "__main__":
  test.main()
