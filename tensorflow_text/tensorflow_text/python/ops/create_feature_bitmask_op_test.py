# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Tests for create_feature_bitmask_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.platform import test
from tensorflow_text.python.ops import create_feature_bitmask_op


@test_util.run_all_in_graph_and_eager_modes
class CreateFeatureBitmaskOpTest(test_util.TensorFlowTestCase):

  def test_docstring_example1(self):
    data = [True, False, False, True]
    result = create_feature_bitmask_op.create_feature_bitmask(data)
    self.assertAllEqual(result, 0b1001)

  def test_docstring_example2(self):
    data = [[True, False], [False, True], [True, True]]
    result = create_feature_bitmask_op.create_feature_bitmask(data)
    expected_result = constant_op.constant([0b10, 0b01, 0b11])
    self.assertAllEqual(result, expected_result)

  def test_feature_bitmask_single_dim_single_tensor(self):
    """Test that the op can reduce a single-dimension tensor to a constant."""
    data = constant_op.constant([True, False])
    result = create_feature_bitmask_op.create_feature_bitmask(data)

    expected_result = constant_op.constant(2)
    self.assertAllEqual(expected_result, result)

  def test_feature_bitmask_multiple_tensors_stack(self):
    """Test that the op can reduce a stacked list of tensors."""
    data_1 = constant_op.constant([True, False])
    data_2 = constant_op.constant([False, True])
    stack_data = array_ops_stack.stack([data_1, data_2], -1)

    expected_result = constant_op.constant([2, 1])
    result = create_feature_bitmask_op.create_feature_bitmask(stack_data)
    self.assertAllEqual(expected_result, result)

  def test_feature_bitmask_multi_dim_single_tensor(self):
    """Test that the op can reduce a multi-dimension tensor."""
    data = constant_op.constant([[True, True, False], [True, False, False]])
    result = create_feature_bitmask_op.create_feature_bitmask(data)

    expected_result = constant_op.constant([6, 4])
    self.assertAllEqual(expected_result, result)

  def test_feature_bitmask_3_dim_single_tensor(self):
    """Test that the op can reduce a 3-dimension tensor."""
    data = constant_op.constant([[[True, True, False], [True, False, False]],
                                 [[False, False, True], [True, False, True]]])
    result = create_feature_bitmask_op.create_feature_bitmask(data)

    expected_result = constant_op.constant([[6, 4], [1, 5]])
    self.assertAllEqual(expected_result, result)

  def test_feature_bitmask_multiple_tensors_multi_dim_stack(self):
    """Test that the op can reduce a stacked list of multi-dim tensors."""
    data_1 = constant_op.constant([[True, False], [False, True]])
    data_2 = constant_op.constant([[False, True], [True, True]])
    stack_data = array_ops_stack.stack([data_1, data_2], -1)

    expected_result = constant_op.constant([[2, 1], [1, 3]])
    result = create_feature_bitmask_op.create_feature_bitmask(stack_data)
    self.assertAllEqual(expected_result, result)

  def test_supports_tensors_with_unknown_shape(self):
    """Test that the op handles tensors with unknown shape."""
    data = array_ops.placeholder_with_default(
        constant_op.constant([[[True, True, False], [True, False, False]],
                              [[False, False, True], [True, False, True]]]),
        shape=None)
    result = create_feature_bitmask_op.create_feature_bitmask(data)

    expected_result = constant_op.constant([[6, 4], [1, 5]])

    self.assertAllEqual(expected_result, result)

  def test_feature_bitmask_multiple_tensors_error(self):
    """Test that the op errors when presented with a single tensor."""
    data_1 = constant_op.constant([True, False])
    data_2 = constant_op.constant([True, True])
    list_data = [data_1, data_2]
    error_message = 'CreateFeatureBitmask does not support lists of tensors.*'

    with self.assertRaisesRegex(errors.InvalidArgumentError, error_message):
      _ = create_feature_bitmask_op.create_feature_bitmask(list_data)

  def test_unsupported_dtype_type(self):
    data = constant_op.constant([True, False])
    bad_dtype = dtypes.uint32
    error_message = 'dtype must be one of: .*, was %s' % bad_dtype.name

    with self.assertRaisesRegex(errors.InvalidArgumentError, error_message):
      _ = create_feature_bitmask_op.create_feature_bitmask(
          data, dtype=bad_dtype)

  def test_unsupported_input_type(self):
    data = constant_op.constant([1.0, 0.0])
    error_message = ('Tensor conversion requested dtype bool for Tensor'
                     ' with dtype float32: .*')

    with self.assertRaisesRegex(ValueError, error_message):
      _ = create_feature_bitmask_op.create_feature_bitmask(data)

  def test_larger_than_max_shape(self):
    data = array_ops.fill([2, 64], False)
    error_message = r'data.shape\[-1\] must be less than 64, is 64.'

    with self.assertRaisesRegex(ValueError, error_message):
      _ = create_feature_bitmask_op.create_feature_bitmask(data)

  def test_larger_than_dtype_shape(self):
    data = array_ops.fill([2, 9], False)
    error_message = (r'data.shape\[-1\] is too large for %s \(was 9, cannot '
                     r'exceed 8\).*') % dtypes.uint8.name

    with self.assertRaisesRegex(ValueError, error_message):
      _ = create_feature_bitmask_op.create_feature_bitmask(
          data, dtype=dtypes.uint8)

  def test_larger_than_dtype_shape_at_runtime(self):
    data = array_ops.placeholder_with_default(
        array_ops.fill([2, 9], False), shape=None)
    error_message = (r'.*data.shape\[-1\] is too large for %s.*' %
                     dtypes.uint8.name)

    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError), error_message
    ):
      self.evaluate(
          create_feature_bitmask_op.create_feature_bitmask(
              data, dtype=dtypes.uint8))


if __name__ == '__main__':
  test.main()
