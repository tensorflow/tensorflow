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
"""Tests for ragged.to_tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorToTensorOpTest(test_util.TensorFlowTestCase,
                                 parameterized.TestCase):

  def testDocStringExamples(self):
    """Example from ragged_to_tensor.__doc__."""
    rt = ragged_factory_ops.constant([[9, 8, 7], [], [6, 5], [4]])
    dt = rt.to_tensor()
    self.assertAllEqual(dt, [[9, 8, 7], [0, 0, 0], [6, 5, 0], [4, 0, 0]])

  @parameterized.parameters(
      {
          'rt_input': [],
          'ragged_rank': 1,
          'expected': [],
          'expected_shape': [0, 0],
      },
      {
          'rt_input': [[1, 2, 3], [], [4], [5, 6]],
          'expected': [[1, 2, 3], [0, 0, 0], [4, 0, 0], [5, 6, 0]]
      },
      {
          'rt_input': [[1, 2, 3], [], [4], [5, 6]],
          'default': 9,
          'expected': [[1, 2, 3], [9, 9, 9], [4, 9, 9], [5, 6, 9]]
      },
      {
          'rt_input': [[[1], [2], [3]], [], [[4]], [[5], [6]]],
          'ragged_rank':
              1,
          'default': [9],
          'expected': [[[1], [2], [3]], [[9], [9], [9]], [[4], [9], [9]],
                       [[5], [6], [9]]]
      },
      {
          'rt_input': [[[1, 2], [], [3, 4]], [], [[5]], [[6, 7], [8]]],
          'expected': [
              [[1, 2], [0, 0], [3, 4]],  #
              [[0, 0], [0, 0], [0, 0]],  #
              [[5, 0], [0, 0], [0, 0]],  #
              [[6, 7], [8, 0], [0, 0]],  #
          ]
      },
      {
          'rt_input': [[[1, 2], [], [3, 4]], [], [[5]], [[6, 7], [8]]],
          'default':
              9,
          'expected': [
              [[1, 2], [9, 9], [3, 4]],  #
              [[9, 9], [9, 9], [9, 9]],  #
              [[5, 9], [9, 9], [9, 9]],  #
              [[6, 7], [8, 9], [9, 9]],  #
          ]
      },
      {
          'rt_input': [[[1], [2], [3]]],
          'ragged_rank': 1,
          'default': 0,
          'expected': [[[1], [2], [3]]],
      },
      {
          'rt_input': [[[[1], [2]], [], [[3]]]],
          'default': 9,
          'expected': [[[[1], [2]], [[9], [9]], [[3], [9]]]],
      },
  )
  def testRaggedTensorToTensor(self,
                               rt_input,
                               expected,
                               ragged_rank=None,
                               default=None,
                               expected_shape=None):
    rt = ragged_factory_ops.constant(rt_input, ragged_rank=ragged_rank)
    dt = rt.to_tensor(default)
    self.assertIsInstance(dt, ops.Tensor)
    self.assertEqual(rt.dtype, dt.dtype)
    self.assertTrue(dt.shape.is_compatible_with(rt.shape))
    if expected_shape is not None:
      expected = np.ndarray(expected_shape, buffer=np.array(expected))
    self.assertAllEqual(dt, expected)

  @parameterized.parameters(
      {
          'rt_input': [[1, 2, 3]],
          'default': [0],
          'error': (ValueError, r'Shape \(1,\) must have rank at most 0'),
      },
      {
          'rt_input': [[[1, 2], [3, 4]], [[5, 6]]],
          'ragged_rank': 1,
          'default': [7, 8, 9],
          'error': (ValueError, r'Shapes \(3,\) and \(2,\) are incompatible'),
      },
      {
          'rt_input': [[1, 2, 3]],
          'default': 'a',
          'error': (TypeError, '.*'),
      },
  )
  def testError(self, rt_input, default, error, ragged_rank=None):
    rt = ragged_factory_ops.constant(rt_input, ragged_rank=ragged_rank)
    with self.assertRaisesRegexp(error[0], error[1]):
      rt.to_tensor(default)


# This covers the tests above, but with the new implementation.
@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorToTensorOpNewTest(test_util.TensorFlowTestCase,
                                    parameterized.TestCase):

  def testDocStringExamples(self):
    """Example from ragged_to_tensor.__doc__."""
    rt = ragged_factory_ops.constant([[9, 8, 7], [], [6, 5], [4]])
    dt = ragged_conversion_ops.ragged_to_dense(rt)
    self.assertAllEqual(dt, [[9, 8, 7], [0, 0, 0], [6, 5, 0], [4, 0, 0]])

  @parameterized.parameters(
      {
          'rt_input': [],
          'ragged_rank': 1,
          'expected': [],
          'expected_shape': [0, 0],
      },
      {
          'rt_input': [[1, 2, 3], [], [4], [5, 6]],
          'expected': [[1, 2, 3], [0, 0, 0], [4, 0, 0], [5, 6, 0]]
      },
      {
          'rt_input': [[1, 2, 3], [], [4], [5, 6]],
          'default': 9,
          'expected': [[1, 2, 3], [9, 9, 9], [4, 9, 9], [5, 6, 9]]
      },
      {
          'rt_input': [[[1], [2], [3]], [], [[4]], [[5], [6]]],
          'ragged_rank':
              1,
          'default': [9],
          'expected': [[[1], [2], [3]], [[9], [9], [9]], [[4], [9], [9]],
                       [[5], [6], [9]]]
      },
      {
          'rt_input': [[[1, 2], [], [3, 4]], [], [[5]], [[6, 7], [8]]],
          'expected': [
              [[1, 2], [0, 0], [3, 4]],  #
              [[0, 0], [0, 0], [0, 0]],  #
              [[5, 0], [0, 0], [0, 0]],  #
              [[6, 7], [8, 0], [0, 0]],  #
          ]
      },
      {
          'rt_input': [[[1, 2], [], [3, 4]], [], [[5]], [[6, 7], [8]]],
          'default':
              9,
          'expected': [
              [[1, 2], [9, 9], [3, 4]],  #
              [[9, 9], [9, 9], [9, 9]],  #
              [[5, 9], [9, 9], [9, 9]],  #
              [[6, 7], [8, 9], [9, 9]],  #
          ]
      },
      {
          'rt_input': [[[1], [2], [3]]],
          'ragged_rank': 1,
          'default': 0,
          'expected': [[[1], [2], [3]]],
      },
      {
          'rt_input': [[[[1], [2]], [], [[3]]]],
          'default': 9,
          'expected': [[[[1], [2]], [[9], [9]], [[3], [9]]]],
      },
  )
  def testRaggedTensorToTensor(self,
                               rt_input,
                               expected,
                               ragged_rank=None,
                               default=None,
                               expected_shape=None):
    rt = ragged_factory_ops.constant(rt_input, ragged_rank=ragged_rank)
    dt = ragged_conversion_ops.ragged_to_dense(rt, default_value=default)

    self.assertIsInstance(dt, ops.Tensor)
    self.assertEqual(rt.dtype, dt.dtype)
    self.assertTrue(dt.shape.is_compatible_with(rt.shape))
    if expected_shape is not None:
      expected = np.ndarray(expected_shape, buffer=np.array(expected))
    self.assertAllEqual(dt, expected)

  @parameterized.parameters(
      {
          'rt_input': [[1, 2, 3]],
          'default': 'a',
          'error': (TypeError, '.*'),
      }, {
          'rt_input': [[1, 2, 3]],
          'default': 'b',
          'error': (TypeError, '.*'),
      })
  def testError(self, rt_input, default, error, ragged_rank=None):
    rt = ragged_factory_ops.constant(rt_input, ragged_rank=ragged_rank)
    with self.assertRaisesRegexp(error[0], error[1]):
      ragged_conversion_ops.ragged_to_dense(rt, default_value=default)


@test_util.run_all_in_graph_and_eager_modes
class RaggedToTensorOpAdditionalTests(test_util.TensorFlowTestCase):

  def _compare_to_reference(self,
                            ragged_tensor,
                            expected=None,
                            default_value=None):
    treatment = ragged_conversion_ops.ragged_to_dense(
        ragged_tensor, default_value=default_value)
    control = ragged_tensor.to_tensor(default_value=default_value)
    self.assertAllEqual(control, treatment)
    if expected is not None:
      self.assertAllEqual(expected, treatment)

  def test_already_dense_simple(self):
    """This studies a tensor initialized with value_rowids and nrows."""
    input_data = RaggedTensor.from_value_rowids(
        values=constant_op.constant([6, 7, 8, 9, 10, 11], dtype=dtypes.int64),
        value_rowids=constant_op.constant([0, 0, 0, 1, 1, 1],
                                          dtype=dtypes.int64),
        nrows=constant_op.constant(2, dtype=dtypes.int64),
        validate=True)
    self._compare_to_reference(input_data, [[6, 7, 8], [9, 10, 11]])

  def test_already_dense_with_dense_values_and_default(self):
    """This studies a tensor initialized with value_rowids and nrows."""
    input_data = RaggedTensor.from_value_rowids(
        values=constant_op.constant(
            [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17]],
            dtype=dtypes.int64),
        value_rowids=constant_op.constant([0, 0, 0, 1, 1, 1],
                                          dtype=dtypes.int64),
        nrows=constant_op.constant(2, dtype=dtypes.int64),
        validate=True)
    self._compare_to_reference(
        input_data,
        [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [16, 17]]],
        default_value=constant_op.constant([31, 32], dtype=dtypes.int64))

  def test_already_dense_with_dense_values(self):
    """This studies a tensor initialized with value_rowids and nrows."""
    input_data = RaggedTensor.from_value_rowids(
        values=constant_op.constant(
            [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17]],
            dtype=dtypes.int64),
        value_rowids=constant_op.constant([0, 0, 0, 1, 1, 1],
                                          dtype=dtypes.int64),
        nrows=constant_op.constant(2, dtype=dtypes.int64),
        validate=True)
    self._compare_to_reference(
        input_data,
        [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [16, 17]]])

  def test_ragged_with_dense_values_and_default(self):
    """This studies a tensor initialized with value_rowids and nrows."""
    input_data = RaggedTensor.from_value_rowids(
        values=constant_op.constant(
            [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]], dtype=dtypes.int64),
        value_rowids=constant_op.constant([0, 0, 0, 1, 1], dtype=dtypes.int64),
        nrows=constant_op.constant(2, dtype=dtypes.int64),
        validate=True)
    self._compare_to_reference(
        input_data, [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [2, 3]]],
        default_value=[2, 3])

  def test_ragged_with_dense_values_and_small_default(self):
    """This studies a tensor initialized with value_rowids and nrows."""
    input_data = RaggedTensor.from_value_rowids(
        values=constant_op.constant(
            [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]], dtype=dtypes.int64),
        value_rowids=constant_op.constant([0, 0, 0, 1, 1], dtype=dtypes.int64),
        nrows=constant_op.constant(2, dtype=dtypes.int64),
        validate=True)
    self._compare_to_reference(
        input_data, [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [2, 2]]],
        default_value=2)

  def test_already_dense_with_dense_values_string(self):
    """This studies a tensor initialized with value_rowids and nrows."""
    input_data = RaggedTensor.from_value_rowids(
        values=constant_op.constant(
            [[b'a', b'b'], [b'c', b'd'], [b'e', b'f'], [b'g', b'jalapeno'],
             [b'kangaroo', b'llama'], [b'manzana', b'nectar']],
            dtype=dtypes.string),
        value_rowids=constant_op.constant([0, 0, 0, 1, 1, 1],
                                          dtype=dtypes.int64),
        nrows=constant_op.constant(2, dtype=dtypes.int64),
        validate=True)
    self._compare_to_reference(input_data,
                               [[[b'a', b'b'], [b'c', b'd'], [b'e', b'f']],
                                [[b'g', b'jalapeno'], [b'kangaroo', b'llama'],
                                 [b'manzana', b'nectar']]])

  def test_already_dense_with_string(self):
    """This studies a tensor initialized with value_rowids and nrows."""
    input_data = RaggedTensor.from_value_rowids(
        values=constant_op.constant(
            ['a', 'b', 'c', 'd', 'e', 'antidisestablishmentarianism'],
            dtype=dtypes.string),
        value_rowids=constant_op.constant([0, 0, 0, 1, 1, 1],
                                          dtype=dtypes.int64),
        nrows=constant_op.constant(2, dtype=dtypes.int64),
        validate=True)
    self._compare_to_reference(
        input_data,
        [[b'a', b'b', b'c'], [b'd', b'e', b'antidisestablishmentarianism']])

  def test_already_dense(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2], [3, 4, 5]])
    self._compare_to_reference(input_data, [[0, 1, 2], [3, 4, 5]])

  def test_true_ragged(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2], [], [3]])
    self._compare_to_reference(input_data, [[0, 1, 2], [0, 0, 0], [3, 0, 0]])

  def test_true_ragged_default_3(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2], [], [3]])
    self._compare_to_reference(
        input_data, [[0, 1, 2], [3, 3, 3], [3, 3, 3]], default_value=3)

  def test_three_dimensional_ragged(self):
    input_data = ragged_factory_ops.constant([[[0, 1, 2], []], [], [[3]]])
    self._compare_to_reference(
        input_data, [[[0, 1, 2], [3, 3, 3]], [[3, 3, 3], [3, 3, 3]],
                     [[3, 3, 3], [3, 3, 3]]],
        default_value=3)

  def test_empty_tensor(self):
    input_data = RaggedTensor.from_value_rowids(
        values=constant_op.constant([], dtype=dtypes.int64),
        value_rowids=constant_op.constant([], dtype=dtypes.int64),
        nrows=constant_op.constant(2, dtype=dtypes.int64),
        validate=True)
    self._compare_to_reference(input_data, [[], []], default_value=3)

  def test_empty_last(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2], [], [3], []])
    self._compare_to_reference(input_data,
                               [[0, 1, 2], [0, 0, 0], [3, 0, 0], [0, 0, 0]])

  def test_shape_limit(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
    actual = ragged_conversion_ops.ragged_to_dense(input_data, shape=[2, 3])
    self.assertAllEqual(actual, [[0, 1, 2], [0, 0, 0]])
    self.assertEqual(actual.shape.as_list(), [2, 3])

  def test_shape_limit_tuple(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
    actual = ragged_conversion_ops.ragged_to_dense(input_data, shape=(2, 3))
    self.assertAllEqual(actual, [[0, 1, 2], [0, 0, 0]])
    self.assertEqual(actual.shape.as_list(), [2, 3])

  def test_shape_limit_tensor_shape(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
    actual = ragged_conversion_ops.ragged_to_dense(
        input_data, shape=tensor_shape.TensorShape([2, 3]))
    self.assertAllEqual(actual, [[0, 1, 2], [0, 0, 0]])
    self.assertEqual(actual.shape.as_list(), [2, 3])

  def test_shape_half_limit_tensor_shape(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
    actual = ragged_conversion_ops.ragged_to_dense(
        input_data, shape=tensor_shape.TensorShape([2, None]))
    self.assertAllEqual(actual, [[0, 1, 2, 3], [0, 0, 0, 0]])

  def test_skip_eager_shape_half_limit_tensor_shape(self):
    # Eager would produce a shape of [2, 4]
    input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
    actual = ragged_conversion_ops.ragged_to_dense(
        input_data, shape=tensor_shape.TensorShape([2, None]))
    result = actual.shape.as_list()
    # This is equal to [2, 4] in eager, or [2, None] in non-eager.
    self.assertEqual(result[0], 2)

  def test_shape_limit_shape_is_tensor_int64(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
    actual = ragged_conversion_ops.ragged_to_dense(
        input_data, shape=constant_op.constant([2, 3], dtype=dtypes.int64))
    self.assertAllEqual(actual, [[0, 1, 2], [0, 0, 0]])
    self.assertEqual(actual.shape.as_list(), [2, 3])

  def test_shape_limit_shape_is_tensor_int32(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
    actual = ragged_conversion_ops.ragged_to_dense(
        input_data, shape=constant_op.constant([2, 3], dtype=dtypes.int32))
    self.assertAllEqual(actual, [[0, 1, 2], [0, 0, 0]])
    self.assertEqual(actual.shape.as_list(), [2, 3])

  def test_shape_expand_first_dim(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2], [], [3]])
    actual = ragged_conversion_ops.ragged_to_dense(input_data, shape=[4, 4])
    self.assertAllEqual(
        actual, [[0, 1, 2, 0], [0, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]])
    self.assertEqual(actual.shape.as_list(), [4, 4])

  def test_value_transposed(self):
    # This test tries to get a tensor in columnar format, where I am uncertain
    # as to whether the underlying op, which copies data in the raw format,
    # could fail.
    my_value = array_ops.transpose(
        constant_op.constant([[0, 1, 2, 3], [4, 5, 6, 7]]))
    input_data = RaggedTensor.from_value_rowids(
        values=my_value,
        value_rowids=constant_op.constant([0, 1, 2, 3], dtype=dtypes.int64),
        nrows=constant_op.constant(4, dtype=dtypes.int64),
        validate=True)
    self._compare_to_reference(input_data,
                               [[[0, 4]], [[1, 5]], [[2, 6]], [[3, 7]]])

  # This fails on the older version of to_tensor.
  def test_broadcast_default(self):
    # This test is commented out. The functionality here is not supported.
    # The dense dimension here is 2 x 2
    input_data = ragged_factory_ops.constant([[[[1, 2], [3, 4]]], []],
                                             ragged_rank=1)
    # This placeholder has a 2 x 1 dimension.
    default_value = array_ops.placeholder_with_default([[5], [6]], shape=None)
    actual = ragged_conversion_ops.ragged_to_dense(
        input_data, default_value=default_value)
    expected = [[[[1, 2], [3, 4]]], [[[5, 5], [6, 6]]]]
    self.assertAllEqual(actual, expected)

  # This fails on the older version of to_tensor.
  def test_broadcast_default_no_placeholder(self):
    # Again, this functionality is not supported. It fails more gracefully
    # when creating the op.
    input_data = ragged_factory_ops.constant([[[[1, 2], [3, 4]]], []],
                                             ragged_rank=1)
    # default_value has a 2 x 1 dimension.
    default_value = constant_op.constant([[5], [6]], shape=None)
    actual = ragged_conversion_ops.ragged_to_dense(
        input_data, default_value=default_value)
    expected = [[[[1, 2], [3, 4]]], [[[5, 5], [6, 6]]]]
    self.assertAllEqual(actual, expected)

  def test_shape_expand_second_dim(self):
    input_data = ragged_factory_ops.constant([[0, 1, 2], [], [3], []])
    actual = ragged_conversion_ops.ragged_to_dense(input_data, shape=[3, 4])
    self.assertAllEqual(actual, [[0, 1, 2, 0], [0, 0, 0, 0], [3, 0, 0, 0]])

  def test_empty_tensor_with_shape(self):
    input_data = RaggedTensor.from_value_rowids(
        values=constant_op.constant([], dtype=dtypes.int64),
        value_rowids=constant_op.constant([], dtype=dtypes.int64),
        nrows=constant_op.constant(2, dtype=dtypes.int64),
        validate=True)
    actual = ragged_conversion_ops.ragged_to_dense(
        input_data, default_value=3, shape=[2, 3])
    self.assertAllEqual(actual, [[3, 3, 3], [3, 3, 3]])


if __name__ == '__main__':
  googletest.main()
