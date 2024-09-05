# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for array_grad."""

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class ArrayGradTest(test.TestCase):

  def _testGrad(self, f, x):
    max_error = gradient_checker_v2.max_error(
        *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(max_error, 1e-4)

  def test_gather_v2_simple(self):
    x = constant_op.constant([1., 2., 3., 4., 5.], dtype=dtypes.float64)

    def f(x):
      return array_ops.gather_v2(
          x, constant_op.constant([2, 0, 2, 4], dtype=dtypes.int32))

    self._testGrad(f, x)

  def test_gather_v2_more_index_dims(self):
    x = constant_op.constant([1., 2., 3., 4., 5.], dtype=dtypes.float64)

    def f(x):
      return array_ops.gather_v2(
          x, constant_op.constant([[2, 0], [2, 4]], dtype=dtypes.int32))

    self._testGrad(f, x)

  def test_gather_v2_more_param_dims(self):
    x = constant_op.constant([[1., 2.], [3., 4.]], dtype=dtypes.float64)

    def f(x):
      return array_ops.gather_v2(
          x, constant_op.constant([1, 0], dtype=dtypes.int32))

    self._testGrad(f, x)

  def test_gather_v2_axis(self):
    x = constant_op.constant([[1., 2.], [3., 4.]], dtype=dtypes.float64)

    def f(x):
      return array_ops.gather_v2(
          x, constant_op.constant([1, 0], dtype=dtypes.int32), axis=1)

    self._testGrad(f, x)

  def test_gather_v2_batch_dims(self):
    x = constant_op.constant([[1., 2.], [3., 4.]], dtype=dtypes.float64)

    def f(x):
      return array_ops.gather_v2(
          x,
          constant_op.constant([[1, 0], [0, 0]], dtype=dtypes.int32),
          axis=1,
          batch_dims=1)

    self._testGrad(f, x)

  def test_gather_v2_2batch_dims(self):
    x = constant_op.constant([[[1., 2.], [3., 4.]]], dtype=dtypes.float64)

    def f(x):
      return array_ops.gather_v2(
          x,
          constant_op.constant([[[1, 0], [0, 0]]], dtype=dtypes.int32),
          axis=2,
          batch_dims=2)

    self._testGrad(f, x)

  def test_gather_v2_batch_dims_with_axis(self):
    x = constant_op.constant([[[1., 2.]], [[3., 4.]]], dtype=dtypes.float64)

    def f(x):
      return array_ops.gather_v2(
          x,
          constant_op.constant([[0], [0]], dtype=dtypes.int32),
          axis=2,
          batch_dims=1)

    self._testGrad(f, x)

  def test_gather_v2_zero_bsize_grad_has_matching_shapes(self):
    params = array_ops.zeros(shape=[0, 1, 8, 16], dtype=dtypes.float64)
    indices = array_ops.zeros(shape=[0, 1, 3], dtype=dtypes.int32)

    def f(params):
      return array_ops.gather_v2(params, indices, axis=2, batch_dims=2)

    grads = backprop.gradients_function(f)(params)
    self.assertLen(grads, 1)
    self.assertAllEqual(params.shape, grads[0].shape)

  def test_broadcast_to(self):
    x = constant_op.constant([1., 2., 3.], dtype=dtypes.float64)
    y = constant_op.constant([2, 3], dtype=dtypes.int32)

    def f(x):
      return array_ops.broadcast_to(
          x,
          y)

    self._testGrad(f, x)

  def test_broadcast_to_int64(self):
    x = constant_op.constant([1., 2., 3.], dtype=dtypes.float64)
    y = constant_op.constant([2, 3], dtype=dtypes.int64)

    def f(x):
      return array_ops.broadcast_to(
          x,
          y)

    self._testGrad(f, x)

  def test_slice_int64(self):
    x = constant_op.constant([1., 2., 3.], dtype=dtypes.float64)
    begin = constant_op.constant([1], dtype=dtypes.int64)
    size = constant_op.constant([1], dtype=dtypes.int64)

    def f(x):
      return array_ops.slice(x, begin, size)

    self._testGrad(f, x)

  def test_reshape_simple(self):
    x = constant_op.constant([1., 2., 3.], dtype=dtypes.float64)
    y = constant_op.constant([3, 1], dtype=dtypes.int64)

    def f(x):
      return array_ops.reshape(x, y)

    self._testGrad(f, x)

  def test_reshape_one_unknown_dim(self):
    def f(x):
      x_without_shape = array_ops.placeholder_with_default(x, shape=[None, 2])
      return array_ops.reshape(x_without_shape, [3, 2])

    x = constant_op.constant([[1., 2.], [3., 4.], [5., 6.]],
                             dtype=dtypes.float64)
    self._testGrad(f, x)

  def test_reshape_two_unknown_dims(self):
    def f(x):
      x_without_shape = array_ops.placeholder_with_default(x,
                                                           shape=[None, None])
      return array_ops.reshape(x_without_shape, [6])

    x = constant_op.constant([[1., 2., 3.], [4., 5., 6.]], dtype=dtypes.float64)
    self._testGrad(f, x)

  def test_reshape_one_unknown_dim_and_zero_elements(self):
    def f(x):
      x_without_shape = array_ops.placeholder_with_default(x, shape=[None, 0])
      return array_ops.reshape(x_without_shape, [0])

    x = constant_op.constant([], shape=[3, 0], dtype=dtypes.float64)
    self._testGrad(f, x)


if __name__ == "__main__":
  test.main()
