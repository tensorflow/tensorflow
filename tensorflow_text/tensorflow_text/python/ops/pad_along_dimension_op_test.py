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

"""Tests for pad_along_dimension_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow_text.python.ops import pad_along_dimension_op


@test_util.run_all_in_graph_and_eager_modes
class PadAlongDimensionOpTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

  def test_pads_along_positive_inner_dimension(self):
    """Test padding along the inner dimension with a positive axis integer."""
    data = constant_op.constant([[1, 1, 1], [2, 2, 1], [3, 3, 1]])
    axis = 1
    left_pad_value = [0]
    right_pad_value = [9]
    expected_result = constant_op.constant([[0, 1, 1, 1, 9], [0, 2, 2, 1, 9],
                                            [0, 3, 3, 1, 9]])

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data,
        axis=axis,
        left_pad=left_pad_value,
        right_pad=right_pad_value)

    self.assertAllEqual(expected_result, padded_result)

  def test_pads_along_positive_outer_dimension(self):
    """Test padding along the outer dimension with a positive axis integer."""
    data = constant_op.constant([[1, 1, 1], [2, 2, 1], [3, 3, 1]])
    axis = 0
    left_pad_value = [[0, 0, 0]]
    right_pad_value = [[9, 9, 9]]
    expected_result = constant_op.constant([[0, 0, 0], [1, 1, 1], [2, 2, 1],
                                            [3, 3, 1], [9, 9, 9]])

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data,
        axis=axis,
        left_pad=left_pad_value,
        right_pad=right_pad_value)

    self.assertAllEqual(expected_result, padded_result)

  def test_pads_along_negative_inner_dimension(self):
    """Test padding along the inner dimension with a negative axis integer."""
    data = constant_op.constant([[1, 1, 1], [2, 2, 1], [3, 3, 1]])
    axis = -1
    left_pad_value = [0]
    right_pad_value = [9]
    expected_result = constant_op.constant([[0, 1, 1, 1, 9], [0, 2, 2, 1, 9],
                                            [0, 3, 3, 1, 9]])

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data,
        axis=axis,
        left_pad=left_pad_value,
        right_pad=right_pad_value)

    self.assertAllEqual(expected_result, padded_result)

  def test_pads_along_negative_outer_dimension(self):
    """Test padding along the outer dimension with a negative axis integer."""
    data = constant_op.constant([[1, 1, 1], [2, 2, 1], [3, 3, 1]])
    axis = -2
    left_pad_value = [[0, 0, 0]]
    right_pad_value = [[9, 9, 9]]
    expected_result = constant_op.constant([[0, 0, 0], [1, 1, 1], [2, 2, 1],
                                            [3, 3, 1], [9, 9, 9]])

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data,
        axis=axis,
        left_pad=left_pad_value,
        right_pad=right_pad_value)

    self.assertAllEqual(expected_result, padded_result)

  def test_no_left_padding(self):
    """Test that not specifying a left pad means no left padding."""
    data = constant_op.constant([[1, 1, 1], [2, 2, 1], [3, 3, 1]])
    axis = 1
    right_pad_value = [9]
    expected_result = constant_op.constant([[1, 1, 1, 9], [2, 2, 1, 9],
                                            [3, 3, 1, 9]])

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data, axis=axis, right_pad=right_pad_value)

    self.assertAllEqual(expected_result, padded_result)

  def test_no_right_padding(self):
    """Test that not specifying a right pad means no right padding."""
    data = constant_op.constant([[1, 1, 1], [2, 2, 1], [3, 3, 1]])
    axis = 1
    left_pad_value = [0]
    expected_result = constant_op.constant([[0, 1, 1, 1], [0, 2, 2, 1],
                                            [0, 3, 3, 1]])

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data, axis=axis, left_pad=left_pad_value)

    self.assertAllEqual(expected_result, padded_result)

  def test_string_padding(self):
    """Test padding using string values."""
    data = constant_op.constant([['1', '1', '1'], ['2', '2', '2']])
    axis = 1
    left_pad_value = ['0']
    right_pad_value = ['9']
    expected_result = constant_op.constant([['0', '1', '1', '1', '9'],
                                            ['0', '2', '2', '2', '9']])

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data,
        axis=axis,
        left_pad=left_pad_value,
        right_pad=right_pad_value)

    self.assertAllEqual(expected_result, padded_result)

  def test_string_partial_no_padding(self):
    """Test padding using string values but without one padding value."""
    data = constant_op.constant([['1', '1', '1'], ['2', '2', '2']])
    axis = 1
    left_pad_value = ['0', '0']
    expected_result = constant_op.constant([['0', '0', '1', '1', '1'],
                                            ['0', '0', '2', '2', '2']])

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data, axis=axis, left_pad=left_pad_value)

    self.assertAllEqual(expected_result, padded_result)

  def test_float_padding(self):
    """Test padding using float values."""
    data = constant_op.constant([[1.0, 1.0, 1.0]])
    axis = 1
    left_pad_value = [-3.5]
    right_pad_value = [3.5]
    expected_result = constant_op.constant([[-3.5, 1.0, 1.0, 1.0, 3.5]])

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data,
        axis=axis,
        left_pad=left_pad_value,
        right_pad=right_pad_value)

    self.assertAllEqual(expected_result, padded_result)

  def test_float_partial_no_padding(self):
    """Test padding using float values."""
    data = constant_op.constant([[1.0, 1.0, 1.0]])
    axis = 1
    right_pad_value = [3.5, 3.5, 3.5]
    expected_result = constant_op.constant([[1.0, 1.0, 1.0, 3.5, 3.5, 3.5]])

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data, axis=axis, right_pad=right_pad_value)

    self.assertAllEqual(expected_result, padded_result)

  def test_padding_tensor_of_unknown_shape(self):
    """Test padding a tensor whose shape is not known at graph building time."""
    data = array_ops.placeholder_with_default(
        constant_op.constant([[1, 1, 1], [2, 2, 1], [3, 3, 1]]), shape=None)
    axis = 1
    left_pad_value = [0]
    right_pad_value = [9]
    expected_result = constant_op.constant([[0, 1, 1, 1, 9], [0, 2, 2, 1, 9],
                                            [0, 3, 3, 1, 9]])

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data,
        axis=axis,
        left_pad=left_pad_value,
        right_pad=right_pad_value)

    self.assertAllEqual(expected_result, padded_result)

  def test_no_padding(self):
    """Test padding using string values."""
    data = constant_op.constant([['1', '1', '1'], ['2', '2', '2']])
    axis = 1
    expected_result = data

    padded_result = pad_along_dimension_op.pad_along_dimension(
        data=data, axis=axis, left_pad=None, right_pad=None)

    self.assertAllEqual(expected_result, padded_result)

  def test_invalid_axis(self):
    data = constant_op.constant([[1, 1, 1], [2, 2, 1], [3, 3, 1]])
    axis = -4
    left_pad_value = [0, 0]
    right_pad_value = [9, 9, 9]

    error_msg = 'axis must be between -k <= axis <= -1 OR 0 <= axis < k'
    with self.assertRaisesRegex(errors.InvalidArgumentError, error_msg):
      _ = pad_along_dimension_op.pad_along_dimension(
          data=data,
          axis=axis,
          left_pad=left_pad_value,
          right_pad=right_pad_value)

    error_msg = 'axis must be an int'
    with self.assertRaisesRegex(TypeError, error_msg):
      _ = pad_along_dimension_op.pad_along_dimension(
          data=data,
          axis=constant_op.constant(0),
          left_pad=left_pad_value,
          right_pad=right_pad_value)

  @parameterized.parameters([
      dict(
          descr='docstring example',
          data=[['a', 'b', 'c'], ['d'], ['e', 'f']],
          axis=1,
          left_pad=['<'],
          right_pad=['>'],
          expected=[[b'<', b'a', b'b', b'c', b'>'], [b'<', b'd', b'>'],
                    [b'<', b'e', b'f', b'>']]),
      #=========================================================================
      # axis=0
      #=========================================================================
      dict(
          descr='2D data, axis=0: left padding only',
          data=[[1, 2], [3], [4, 5, 6]],
          axis=0,
          left_pad=[[0]],
          expected=[[0], [1, 2], [3], [4, 5, 6]]),
      dict(
          descr='2D data, axis=0: right padding only',
          data=[[1, 2], [3], [4, 5, 6]],
          axis=0,
          right_pad=[[9, 99], [999]],
          expected=[[1, 2], [3], [4, 5, 6], [9, 99], [999]]),
      dict(
          descr='2D data, axis=0: pad both sides',
          data=[[1, 2], [3], [4, 5, 6]],
          axis=0,
          left_pad=[[0]],
          right_pad=[[9, 99], [999]],
          expected=[[0], [1, 2], [3], [4, 5, 6], [9, 99], [999]]),
      dict(
          descr='3D data, axis=0',
          data=[[[1, 2], [3]], [[4]]],
          axis=0,
          right_pad=[[[9], [99, 999]], [[9999]]],
          expected=[[[1, 2], [3]], [[4]], [[9], [99, 999]], [[9999]]]),
      dict(
          descr='4D data, axis=0',
          data=[[[[1, 2]]], [[[4]]]],
          axis=0,
          left_pad=[[[[9, 9], [9]], [[9]]]],
          expected=[[[[9, 9], [9]], [[9]]], [[[1, 2]]], [[[4]]]]),
      dict(
          descr='2D data, axis=-2: pad both sides',
          data=[[1, 2], [3], [4, 5, 6]],
          axis=-2,
          left_pad=[[0]],
          right_pad=[[9, 99], [999]],
          expected=[[0], [1, 2], [3], [4, 5, 6], [9, 99], [999]]),
      dict(
          descr='3D data, axis=-3',
          data=[[[1, 2], [3]], [[4]]],
          axis=-3,
          right_pad=[[[9], [99, 999]], [[9999]]],
          expected=[[[1, 2], [3]], [[4]], [[9], [99, 999]], [[9999]]]),
      dict(
          descr='4D data, axis=-4',
          data=[[[[1, 2]]], [[[4]]]],
          axis=-4,
          left_pad=[[[[9, 9], [9]], [[9]]]],
          expected=[[[[9, 9], [9]], [[9]]], [[[1, 2]]], [[[4]]]]),
      #=========================================================================
      # axis=1
      #=========================================================================
      dict(
          descr='2D data, axis=1: left padding only',
          data=[[1, 2], [3]],
          axis=1,
          left_pad=[0],
          expected=[[0, 1, 2], [0, 3]]),
      dict(
          descr='2D data, axis=1: right padding only',
          data=[[1, 2], [3]],
          axis=1,
          right_pad=[9, 99],
          expected=[[1, 2, 9, 99], [3, 9, 99]]),
      dict(
          descr='2D data, axis=1: pad both sides',
          data=[[1, 2], [3]],
          axis=1,
          left_pad=[0],
          right_pad=[9, 99],
          expected=[[0, 1, 2, 9, 99], [0, 3, 9, 99]]),
      dict(
          descr='3D data, axis=1',
          data=[[[1, 2], [3]], [[4]]],
          axis=1,
          left_pad=[[0]],
          right_pad=[[9], [99, 999]],
          expected=[[[0], [1, 2], [3], [9], [99, 999]],
                    [[0], [4], [9], [99, 999]]]),
      dict(
          descr='4D data, axis=1',
          data=[[[[1, 2]]], [[[4]]]],
          axis=1,
          left_pad=[[[0]]],
          right_pad=[[[9]]],
          expected=[[[[0]], [[1, 2]], [[9]]], [[[0]], [[4]], [[9]]]]),
      dict(
          descr='2D data, axis=-1: pad both sides',
          data=[[1, 2], [3]],
          axis=-1,
          left_pad=[0],
          right_pad=[9, 99],
          expected=[[0, 1, 2, 9, 99], [0, 3, 9, 99]]),
      dict(
          descr='3D data, axis=-2',
          data=[[[1, 2], [3]], [[4]]],
          axis=-2,
          left_pad=[[0]],
          right_pad=[[9], [99, 999]],
          expected=[[[0], [1, 2], [3], [9], [99, 999]],
                    [[0], [4], [9], [99, 999]]]),
      dict(
          descr='4D data, axis=-3',
          data=[[[[1, 2]]], [[[4]]]],
          axis=-3,
          left_pad=[[[0]]],
          right_pad=[[[9]]],
          expected=[[[[0]], [[1, 2]], [[9]]], [[[0]], [[4]], [[9]]]]),
      #=========================================================================
      # axis=2
      #=========================================================================
      dict(
          descr='3D data, axis=2',
          data=[[[1, 2], [3]], [[4]]],
          axis=2,
          left_pad=[0],
          right_pad=[9],
          expected=[[[0, 1, 2, 9], [0, 3, 9]], [[0, 4, 9]]]),
      dict(
          descr='4D data, axis=2',
          data=[[[[1, 2], [3]], [[4]]], [[[5]]]],
          axis=2,
          left_pad=[[0]],
          right_pad=[[9]],
          expected=[[[[0], [1, 2], [3], [9]], [[0], [4], [9]]], [[[0], [5],
                                                                  [9]]]]),
      dict(
          descr='3D data, axis=-1',
          data=[[[1, 2], [3]], [[4]]],
          axis=-1,
          left_pad=[0],
          right_pad=[9],
          expected=[[[0, 1, 2, 9], [0, 3, 9]], [[0, 4, 9]]]),
      dict(
          descr='4D data, axis=-2',
          data=[[[[1, 2], [3]], [[4]]], [[[5]]]],
          axis=-2,
          left_pad=[[0]],
          right_pad=[[9]],
          expected=[[[[0], [1, 2], [3], [9]], [[0], [4], [9]]], [[[0], [5],
                                                                  [9]]]]),
      #=========================================================================
      # axis=3
      #=========================================================================
      dict(
          descr='4D data, axis=3',
          data=[[[[1, 2], [3]], [[4, 5, 6]]], [[[7, 8]]]],
          axis=3,
          left_pad=[0],
          right_pad=[9, 99],
          expected=[[[[0, 1, 2, 9, 99], [0, 3, 9, 99]], [[0, 4, 5, 6, 9, 99]]],
                    [[[0, 7, 8, 9, 99]]]]),
      dict(
          descr='4D data, axis=-1',
          data=[[[[1, 2], [3]], [[4, 5, 6]]], [[[7, 8]]]],
          axis=-1,
          left_pad=[0],
          right_pad=[9, 99],
          expected=[[[[0, 1, 2, 9, 99], [0, 3, 9, 99]], [[0, 4, 5, 6, 9, 99]]],
                    [[[0, 7, 8, 9, 99]]]]),
  ])
  def testRaggedPadDimension(self,
                             descr,
                             data,
                             axis,
                             expected,
                             left_pad=None,
                             right_pad=None,
                             ragged_rank=None):
    data = self._convert_ragged(data, ragged_rank)
    positive_axis = axis if axis >= 0 else axis + data.shape.ndims
    assert positive_axis >= 0
    left_pad = self._convert_ragged(left_pad, data.ragged_rank - positive_axis)
    right_pad = self._convert_ragged(right_pad,
                                     data.ragged_rank - positive_axis)
    padded = pad_along_dimension_op.pad_along_dimension(data, axis, left_pad,
                                                        right_pad)

    self.assertAllEqual(padded, expected)

  def testRaggedPadDimensionErrors(self):
    ragged_data = ragged_factory_ops.constant([[1, 2], [3, 4]])
    self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'axis must be between -k <= axis <= -1 OR 0 <= axis < k',
        pad_along_dimension_op.pad_along_dimension,
        ragged_data,
        left_pad=[0],
        axis=2,
    )
    self.assertRaisesRegex(
        ValueError,
        r'Shapes .* are incompatible',
        pad_along_dimension_op.pad_along_dimension,
        ragged_data,
        axis=1,
        left_pad=ragged_data,
    )
    if not context.executing_eagerly():
      self.assertRaisesRegex(
          ValueError,
          'axis may not be negative if data is ragged '
          'and data.ndims is not statically known.',
          pad_along_dimension_op.pad_along_dimension,
          ragged_tensor.RaggedTensor.from_tensor(
              array_ops.placeholder_with_default([[1, 2], [3, 4]], shape=None)
          ),
          left_pad=[0],
          axis=-1,
      )

  @parameterized.parameters([
      #=========================================================================
      # axis=0: pad_value is returned as-is.
      #=========================================================================
      dict(
          descr='2D data, axis=0, len(pad_value)=0',
          data=[[1, 2], [3]],
          axis=0,
          pad_value=[],
          expected=[]),
      dict(
          descr='2D data, axis=0, len(pad_value)=1',
          data=[[1, 2], [3]],
          axis=0,
          pad_value=[[9]],
          expected=[[9]]),
      dict(
          descr='2D data, axis=0, len(pad_value)=2',
          data=[[1, 2], [3]],
          axis=0,
          pad_value=[[9], [99]],
          expected=[[9], [99]]),
      dict(
          descr='3D data, axis=0',
          data=[[[1, 2], [3]], [[4]]],
          axis=0,
          pad_value=[[[9], [99, 999]], [[9999]]],
          expected=[[[9], [99, 999]], [[9999]]]),
      dict(
          descr='4D data, axis=0',
          data=[[[[1, 2]]], [[[4]]]],
          axis=0,
          pad_value=[[[[9, 9], [9]], [[9]]]],
          expected=[[[[9, 9], [9]], [[9]]]]),
      #=========================================================================
      # axis=1: pad_value is repeated for each item in 1st dimension of data
      #=========================================================================
      dict(
          descr='2D data, axis=1, len(pad_value)=1',
          data=[[1, 2], [3]],
          axis=1,
          pad_value=[9],
          expected=[[9], [9]]),
      dict(
          descr='2D data, axis=1, len(pad_value)=2',
          data=[[1, 2], [3]],
          axis=1,
          pad_value=[9, 99],
          expected=[[9, 99], [9, 99]]),
      dict(
          descr='2D data, axis=1, len(pad_value)=0',
          data=[[1, 2], [3]],
          axis=1,
          pad_value=[],
          expected=[[], []]),
      dict(
          descr='3D data, axis=1',
          data=[[[1, 2], [3]], [[4]]],
          axis=1,
          pad_value=[[9], [99, 999]],
          expected=[[[9], [99, 999]], [[9], [99, 999]]]),
      dict(
          descr='4D data, axis=1',
          data=[[[[1, 2]]], [[[4]]]],
          axis=1,
          pad_value=[[[9, 9], [9]]],
          expected=[[[[9, 9], [9]]], [[[9, 9], [9]]]]),
      #=========================================================================
      # axis=2: pad_value is repeated for each item in 2nd dimension of data
      #=========================================================================
      dict(
          descr='3D data, axis=2',
          data=[[[1, 2], [3]], [[4]]],
          axis=2,
          pad_value=[9, 99],
          expected=[[[9, 99], [9, 99]], [[9, 99]]]),
      dict(
          descr='4D data, axis=2',
          data=[[[[1, 2], [3]], [[4]]], [[[5]]]],
          axis=2,
          pad_value=[[9, 99], [999]],
          expected=[[[[9, 99], [999]], [[9, 99], [999]]], [[[9, 99], [999]]]]),
      #=========================================================================
      # axis=3: pad_value is repeated for each item in 3rd dimension of data
      #=========================================================================
      dict(
          descr='4D data, axis=3',
          data=[[[[1, 2], [3]], [[4, 5, 6]]], [[[7, 8]]]],
          axis=3,
          pad_value=[9, 99],
          expected=[[[[9, 99], [9, 99]], [[9, 99]]], [[[9, 99]]]]),
  ])
  def testPaddingForRaggedDimensionHelper(self,
                                          descr,
                                          data,
                                          axis,
                                          expected,
                                          pad_value=None,
                                          ragged_rank=None):
    data = self._convert_ragged(data, ragged_rank)
    pad_value = self._convert_ragged(pad_value, data.ragged_rank - axis)
    pad = pad_along_dimension_op._padding_for_dimension(data, axis, pad_value)

    self.assertAllEqual(pad, expected)
    self.assertEqual(data.shape.ndims, pad.shape.ndims)

  def _convert_ragged(self, value, ragged_rank):
    if value is None:
      return None
    if ragged_rank is None or ragged_rank > 0:
      return ragged_factory_ops.constant(value, ragged_rank=ragged_rank)
    else:
      return constant_op.constant(value)


if __name__ == '__main__':
  test.main()
