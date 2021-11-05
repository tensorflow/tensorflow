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
"""Tests for tensorflow.ops.clip_ops."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ClipTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testClipByValueGradient(self):
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32)
    outputs_1 = clip_ops.clip_by_value(inputs, 0.5, 3.5)
    min_val = constant_op.constant([0.5, 0.5, 0.5, 0.5], dtype=dtypes.float32)
    max_val = constant_op.constant([3.5, 3.5, 3.5, 3.5], dtype=dtypes.float32)
    outputs_2 = clip_ops.clip_by_value(inputs, min_val, max_val)
    with self.cached_session():
      error_1 = gradient_checker.compute_gradient_error(inputs, [4], outputs_1,
                                                        [4])
      self.assertLess(error_1, 1e-4)

      error_2 = gradient_checker.compute_gradient_error(inputs, [4], outputs_2,
                                                        [4])
      self.assertLess(error_2, 1e-4)

  # ClipByValue test
  def testClipByValue(self):
    with self.session():
      x = constant_op.constant([-5.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
      np_ans = [[-4.4, 2.0, 3.0], [4.0, 4.4, 4.4]]
      clip_value = 4.4
      ans = clip_ops.clip_by_value(x, -clip_value, clip_value)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Scalar, Scalar]
  def testClipByValue0Type(self):
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.bfloat16,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint8,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[2, 2, 3], [4, 4, 4]]
        clip_value_min = 2
        clip_value_max = 4
        ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Tensor, Scalar]
  def testClipByValue1Type(self):
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.bfloat16,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint8,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[2, 2, 3], [4, 4, 4]]
        clip_value_min = constant_op.constant(
            [2, 2, 2, 3, 3, 3], shape=[2, 3], dtype=dtype)
        clip_value_max = 4
        ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Scalar, Tensor]
  def testClipByValue2Type(self):
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.bfloat16,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint8,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[4, 4, 4], [4, 5, 6]]
        clip_value_min = 4
        clip_value_max = constant_op.constant(
            [6, 6, 6, 6, 6, 6], shape=[2, 3], dtype=dtype)
        ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Tensor, Tensor]
  def testClipByValue3Type(self):
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.bfloat16,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint8,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[2, 2, 3], [5, 5, 6]]
        clip_value_min = constant_op.constant(
            [2, 2, 2, 5, 5, 5], shape=[2, 3], dtype=dtype)
        clip_value_max = constant_op.constant(
            [5, 5, 5, 7, 7, 7], shape=[2, 3], dtype=dtype)
        ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  def testClipByValueBadShape(self):
    with self.session():
      x = constant_op.constant([-5.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3, 1])
      # Use a nonsensical shape.
      clip = constant_op.constant([1.0, 2.0])
      with self.assertRaises(ValueError):
        _ = clip_ops.clip_by_value(x, -clip, clip)  # pylint: disable=invalid-unary-operand-type
      with self.assertRaises(ValueError):
        _ = clip_ops.clip_by_value(x, 1.0, clip)

  def testClipByValueNonFinite(self):
    # TODO(b/78016351): Enable test on GPU once the bug is fixed.
    with self.cached_session():
      x = constant_op.constant([float('NaN'), float('Inf'), -float('Inf')])
      np_ans = [float('NaN'), 4.0, -4.0]
      clip_value = 4.0
      ans = clip_ops.clip_by_value(x, -clip_value, clip_value)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def _testClipIndexedSlicesByValue(self, values, indices, shape,
                                    clip_value_min, clip_value_max, expected):
    with self.session():
      values = constant_op.constant(values)
      indices = constant_op.constant(indices)
      shape = constant_op.constant(shape)
      # IndexedSlices mode
      indexed_slices = indexed_slices_lib.IndexedSlices(values, indices, shape)
      clipped = clip_ops.clip_by_value(indexed_slices, clip_value_min,
                                       clip_value_max)
      # clipped should be IndexedSlices
      self.assertIsInstance(clipped, indexed_slices_lib.IndexedSlices)

    self.assertAllClose(clipped.values, expected)

  def testClipByValueWithIndexedSlicesClipped(self):
    values = [[[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
              [[0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]]
    indices = [2, 6]
    shape = [10, 2, 3]
    # [-2.0, 2.0]
    self._testClipIndexedSlicesByValue(values, indices, shape, -2.0, 2.0,
                                       [[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                                        [[0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]])
    # [1.0, 2.0]
    self._testClipIndexedSlicesByValue(values, indices, shape, 1.0, 2.0,
                                       [[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                                        [[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]])
    # [-2.0, -1.0]
    self._testClipIndexedSlicesByValue(
        values, indices, shape, -2.0, -1.0,
        [[[-2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
         [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]])

  # ClipByNorm tests
  def testClipByNormClipped(self):
    # Norm clipping when clip_norm < 5
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Norm of x = sqrt(3^2 + 4^2) = 5
      np_ans = [[-2.4, 0.0, 0.0], [3.2, 0.0, 0.0]]
      clip_norm = 4.0
      ans = clip_ops.clip_by_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

      ans = clip_ops.clip_by_norm(x, clip_norm)
      tf_ans_tensor = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)
    self.assertAllClose(np_ans, tf_ans_tensor)

  @test_util.run_deprecated_v1
  def testClipByNormGradientZeros(self):
    with self.session():
      x = array_ops.zeros([3])
      b = clip_ops.clip_by_norm(x, 1.)
      grad, = gradients_impl.gradients(b, x)
      self.assertAllEqual(grad, [1., 1., 1.])

  def testClipByNormBadShape(self):
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3, 1])
      # Use a nonsensical shape.
      clip = constant_op.constant([1.0, 2.0])
      with self.assertRaises(ValueError):
        _ = clip_ops.clip_by_norm(x, clip)

  def testClipByNormNotClipped(self):
    # No norm clipping when clip_norm >= 5
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Norm of x = sqrt(3^2 + 4^2) = 5
      np_ans = [[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
      clip_norm = 6.0
      ans = clip_ops.clip_by_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByNormZero(self):
    # No norm clipping when norm = 0
    with self.session():
      x = constant_op.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
      # Norm = 0, no changes
      np_ans = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      clip_norm = 6.0
      ans = clip_ops.clip_by_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByNormClippedWithDim0(self):
    # Norm clipping when clip_norm < 5
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 3.0], shape=[2, 3])
      # Norm of x[:, 0] = sqrt(3^2 + 4^2) = 5, x[:, 2] = 3
      np_ans = [[-2.4, 0.0, 0.0], [3.2, 0.0, 3.0]]
      clip_norm = 4.0
      ans = clip_ops.clip_by_norm(x, clip_norm, [0])
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByNormClippedWithDim1(self):
    # Norm clipping when clip_norm < 5
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 3.0], shape=[2, 3])
      # Norm of x[0, :] = 3, x[1, :] = sqrt(3^2 + 4^2) = 5
      np_ans = [[-3.0, 0.0, 0.0], [3.2, 0.0, 2.4]]
      clip_norm = 4.0
      ans = clip_ops.clip_by_norm(x, clip_norm, [1])
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByNormNotClippedWithAxes(self):
    # No norm clipping when clip_norm >= 5
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 3.0], shape=[2, 3])
      # Norm of x[0, :] = 3, x[1, :] = sqrt(3^2 + 4^2) = 5
      np_ans = [[-3.0, 0.0, 0.0], [4.0, 0.0, 3.0]]
      clip_norm = 6.0
      ans = clip_ops.clip_by_norm(x, clip_norm, [1])
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  # ClipByGlobalNorm tests
  @test_util.run_deprecated_v1
  def testClipByGlobalNormClipped(self):
    # Norm clipping when clip_norm < 5
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = constant_op.constant([1.0, -2.0])
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      clip_norm = 4.0

      # Answers are the original tensors scaled by 4.0/5.0
      np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
      np_ans_1 = [0.8, -1.6]

      ans, norm = clip_ops.clip_by_global_norm((x0, x1), clip_norm)
      tf_ans_1 = ans[0].eval()
      tf_ans_2 = ans[1].eval()
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  @test_util.run_deprecated_v1
  def testClipByGlobalNormClippedTensor(self):
    # Norm clipping when clip_norm < 5
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = constant_op.constant([1.0, -2.0])
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      clip_norm = constant_op.constant(4.0)

      # Answers are the original tensors scaled by 4.0/5.0
      np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
      np_ans_1 = [0.8, -1.6]

      ans, norm = clip_ops.clip_by_global_norm((x0, x1), clip_norm)
      tf_ans_1 = ans[0].eval()
      tf_ans_2 = ans[1].eval()
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  @test_util.run_deprecated_v1
  def testClipByGlobalNormSupportsNone(self):
    # Norm clipping when clip_norm < 5
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = constant_op.constant([1.0, -2.0])
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      clip_norm = 4.0

      # Answers are the original tensors scaled by 4.0/5.0
      np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
      np_ans_1 = [0.8, -1.6]

      ans, norm = clip_ops.clip_by_global_norm((x0, None, x1, None), clip_norm)
      self.assertTrue(ans[1] is None)
      self.assertTrue(ans[3] is None)
      tf_ans_1 = ans[0].eval()
      tf_ans_2 = ans[2].eval()
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  @test_util.run_deprecated_v1
  def testClipByGlobalNormWithIndexedSlicesClipped(self):
    # Norm clipping when clip_norm < 5
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = indexed_slices_lib.IndexedSlices(
          constant_op.constant([1.0, -2.0]), constant_op.constant([3, 4]))
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      clip_norm = 4.0

      # Answers are the original tensors scaled by 4.0/5.0
      np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
      np_ans_1 = [0.8, -1.6]

      ans, norm = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
      tf_ans_1 = self.evaluate(ans[0])
      tf_ans_2 = self.evaluate(ans[1].values)
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  def testClipByGlobalNormPreservesDenseShape(self):
    dense_shape = (1,)
    slices = indexed_slices_lib.IndexedSlices(
        constant_op.constant([1.0]),
        constant_op.constant([0]),
        dense_shape=dense_shape)
    ans, _ = clip_ops.clip_by_global_norm([slices], 1.0)
    modified_slices = ans[0]
    self.assertEqual(dense_shape, slices.dense_shape)
    self.assertEqual(dense_shape, modified_slices.dense_shape)

  @test_util.run_deprecated_v1
  def testClipByGlobalNormNotClipped(self):
    # No norm clipping when clip_norm >= 5
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = constant_op.constant([1.0, -2.0])
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      np_ans_0 = [[-2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
      np_ans_1 = [1.0, -2.0]
      clip_norm = 6.0

      ans, norm = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
      tf_ans_1 = ans[0].eval()
      tf_ans_2 = ans[1].eval()
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  @test_util.run_deprecated_v1
  def testClipByGlobalNormZero(self):
    # No norm clipping when norm = 0
    with self.session():
      x0 = constant_op.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
      x1 = constant_op.constant([0.0, 0.0])
      # Norm = 0, no changes
      np_ans_0 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      np_ans_1 = [0.0, 0.0]
      clip_norm = 6.0

      ans, norm = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
      tf_ans_1 = ans[0].eval()
      tf_ans_2 = ans[1].eval()
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 0.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  @test_util.run_deprecated_v1
  def testClipByGlobalNormInf(self):
    # Expect all NaNs when global norm is inf.
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, np.inf, 4.0, 0.0, 0.0],
                                shape=[2, 3])
      x1 = constant_op.constant([1.0, -2.0])
      clip_norm = 6.0

      ans, norm = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
      tf_ans_1 = ans[0].eval()
      tf_ans_2 = ans[1].eval()
      tf_norm = self.evaluate(norm)
      self.assertAllEqual(tf_norm, float('inf'))
      self.assertAllEqual(tf_ans_1, np.full([2, 3], float('nan')))
      self.assertAllEqual(tf_ans_2, np.full([2], float('nan')))

  def testClipByAverageNormClipped(self):
    # Norm clipping when average clip_norm < 0.83333333
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Average norm of x = sqrt(3^2 + 4^2) / 6 = 0.83333333
      np_ans = [[-2.88, 0.0, 0.0], [3.84, 0.0, 0.0]]
      clip_norm = 0.8
      ans = clip_ops.clip_by_average_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByAverageNormClippedTensor(self):
    # Norm clipping when average clip_norm < 0.83333333
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Average norm of x = sqrt(3^2 + 4^2) / 6 = 0.83333333
      np_ans = [[-2.88, 0.0, 0.0], [3.84, 0.0, 0.0]]
      clip_norm = constant_op.constant(0.8)
      ans = clip_ops.clip_by_average_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByAverageNormNotClipped(self):
    # No norm clipping when average clip_norm >= 0.83333333
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Average norm of x = sqrt(3^2 + 4^2) / 6 = 0.83333333
      np_ans = [[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
      clip_norm = 0.9
      ans = clip_ops.clip_by_average_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByAverageNormZero(self):
    # No norm clipping when average clip_norm = 0
    with self.session():
      x = constant_op.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
      # Average norm = 0, no changes
      np_ans = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      clip_norm = 0.9
      ans = clip_ops.clip_by_average_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByAverageNormReplacedWithClipByNorm(self):
    # Check clip_by_average_norm(t) is the same as
    # clip_by_norm(t, clip_norm * tf.compat.v1.to_float(tf.size(t)))
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Average norm of x = sqrt(3^2 + 4^2) / 6 = 0.83333333
      # expected answer [[-2.88, 0.0, 0.0], [3.84, 0.0, 0.0]]
      clip_norm = constant_op.constant(0.8)
      with_norm = clip_ops.clip_by_average_norm(x, clip_norm)
      without_norm = clip_ops.clip_by_norm(
          x, clip_norm * math_ops.cast(array_ops.size(x), dtypes.float32))
      clip_by_average_norm_ans = self.evaluate(with_norm)
      clip_by_norm_ans = self.evaluate(without_norm)
      self.assertAllClose(clip_by_average_norm_ans, clip_by_norm_ans)

  @test_util.run_deprecated_v1
  def testClipByValueEmptyTensor(self):
    # Test case for GitHub issue 19337
    zero = array_ops.placeholder(dtype=dtypes.float32, shape=None)
    x = clip_ops.clip_by_value(zero, zero, zero)
    y = clip_ops.clip_by_value(zero, 1.0, 1.0)
    z = clip_ops.clip_by_value(zero, zero, 1.0)
    w = clip_ops.clip_by_value(zero, 1.0, zero)
    with self.session() as sess:
      sess.run([x, y, z, w], feed_dict={zero: np.zeros((7, 0))})


if __name__ == '__main__':
  test.main()
