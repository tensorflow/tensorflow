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
"""Tests for third_party.tensorflow.python.ops.ragged_tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor

from tensorflow.python.platform import googletest


class _SliceBuilder(object):
  """Helper to construct arguments for __getitem__.

  Usage: _SliceBuilder()[<expr>] slice_spec Python generates for <expr>.
  """

  def __getitem__(self, slice_spec):
    return slice_spec


SLICE_BUILDER = _SliceBuilder()


def _make_tensor_slice_spec(slice_spec, use_constant=True):
  """Wraps all integers in an extended slice spec w/ a tensor.

  This function is used to help test slicing when the slice spec contains
  tensors, rather than integers.

  Args:
    slice_spec: The extended slice spec.
    use_constant: If true, then wrap each integer with a tf.constant.  If false,
      then wrap each integer with a tf.placeholder.

  Returns:
    A copy of slice_spec, but with each integer i replaced with tf.constant(i).
  """

  def make_piece_scalar(piece):
    if isinstance(piece, int):
      scalar = constant_op.constant(piece)
      if use_constant:
        return scalar
      else:
        return array_ops.placeholder_with_default(scalar, [])
    elif isinstance(piece, slice):
      return slice(
          make_piece_scalar(piece.start), make_piece_scalar(piece.stop),
          make_piece_scalar(piece.step))
    else:
      return piece

  if isinstance(slice_spec, tuple):
    return tuple(make_piece_scalar(piece) for piece in slice_spec)
  else:
    return make_piece_scalar(slice_spec)


# Example 2D ragged tensor value with one ragged dimension and with scalar
# values, expressed as nested python lists and as splits+values.
EXAMPLE_RAGGED_TENSOR_2D = [[b'a', b'b'], [b'c', b'd', b'e'], [b'f'], [],
                            [b'g']]
EXAMPLE_RAGGED_TENSOR_2D_SPLITS = [0, 2, 5, 6, 6, 7]
EXAMPLE_RAGGED_TENSOR_2D_VALUES = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

# Example 4D ragged tensor value, with two ragged dimensions and with values
# whose shape is [2], expressed as nested python lists and as splits+values.
EXAMPLE_RAGGED_TENSOR_4D = [
    [                                       # rt[0]
        [[1, 2], [3, 4], [5, 6]],           # rt[0][0]
        [[7, 8], [9, 10], [11, 12]]],       # rt[0][1]
    [],                                     # rt[1]
    [                                       # rt[2]
        [[13, 14], [15, 16], [17, 18]]],    # rt[2][0]
    [                                       # rt[3]
        [[19, 20]]]                         # rt[3][0]
]  # pyformat: disable
EXAMPLE_RAGGED_TENSOR_4D_SPLITS1 = [0, 2, 2, 3, 4]
EXAMPLE_RAGGED_TENSOR_4D_SPLITS2 = [0, 3, 6, 9, 10]
EXAMPLE_RAGGED_TENSOR_4D_VALUES = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                                   [11, 12], [13, 14], [15, 16], [17, 18],
                                   [19, 20]]

# Example 3D ragged tensor with uniform_row_lengths.
EXAMPLE_RAGGED_TENSOR_3D = [[[1, 2, 3], [4], [5, 6]], [[], [7, 8, 9], []]]
EXAMPLE_RAGGED_TENSOR_3D_ROWLEN = 3
EXAMPLE_RAGGED_TENSOR_3D_SPLITS = [0, 3, 4, 6, 6, 9, 9]
EXAMPLE_RAGGED_TENSOR_3D_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]


@test_util.run_all_in_graph_and_eager_modes
class RaggedGetItemTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  longMessage = True  # Property in unittest.Testcase. pylint: disable=invalid-name

  #=============================================================================
  # RaggedTensor.__getitem__
  #=============================================================================

  def _TestGetItem(self, rt, slice_spec, expected, expected_shape=None):
    """Helper function for testing RaggedTensor.__getitem__.

    Checks that calling `rt.__getitem__(slice_spec) returns the expected value.
    Checks three different configurations for each slice spec:

      * Call __getitem__ with the slice spec as-is (with int values)
      * Call __getitem__ with int values in the slice spec wrapped in
        `tf.constant()`.
      * Call __getitem__ with int values in the slice spec wrapped in
        `tf.compat.v1.placeholder()` (so value is not known at graph
        construction time).

    Args:
      rt: The RaggedTensor to test.
      slice_spec: The slice spec.
      expected: The expected value of rt.__getitem__(slice_spec), as a python
        list; or an exception class.
      expected_shape: The expected shape for `rt.__getitem__(slice_spec)`.
    """
    tensor_slice_spec1 = _make_tensor_slice_spec(slice_spec, True)
    tensor_slice_spec2 = _make_tensor_slice_spec(slice_spec, False)
    value1 = rt.__getitem__(slice_spec)
    value2 = rt.__getitem__(tensor_slice_spec1)
    value3 = rt.__getitem__(tensor_slice_spec2)
    self.assertAllEqual(value1, expected, 'slice_spec=%s' % (slice_spec,))
    self.assertAllEqual(value2, expected, 'slice_spec=%s' % (slice_spec,))
    self.assertAllEqual(value3, expected, 'slice_spec=%s' % (slice_spec,))
    if expected_shape is not None:
      value1.shape.assert_is_compatible_with(expected_shape)
      value2.shape.assert_is_compatible_with(expected_shape)
      value3.shape.assert_is_compatible_with(expected_shape)

  def _TestGetItemException(self, rt, slice_spec, expected, message):
    """Helper function for testing RaggedTensor.__getitem__ exceptions."""
    tensor_slice_spec = _make_tensor_slice_spec(slice_spec, True)
    with self.assertRaisesRegex(expected, message):
      self.evaluate(rt.__getitem__(slice_spec))
    with self.assertRaisesRegex(expected, message):
      self.evaluate(rt.__getitem__(tensor_slice_spec))

  @parameterized.parameters(
      # Tests for rt[i]
      (SLICE_BUILDER[-5], EXAMPLE_RAGGED_TENSOR_2D[-5]),
      (SLICE_BUILDER[-4], EXAMPLE_RAGGED_TENSOR_2D[-4]),
      (SLICE_BUILDER[-1], EXAMPLE_RAGGED_TENSOR_2D[-1]),
      (SLICE_BUILDER[0], EXAMPLE_RAGGED_TENSOR_2D[0]),
      (SLICE_BUILDER[1], EXAMPLE_RAGGED_TENSOR_2D[1]),
      (SLICE_BUILDER[4], EXAMPLE_RAGGED_TENSOR_2D[4]),

      # Tests for rt[i:]
      (SLICE_BUILDER[-6:], EXAMPLE_RAGGED_TENSOR_2D[-6:]),
      (SLICE_BUILDER[-3:], EXAMPLE_RAGGED_TENSOR_2D[-3:]),
      (SLICE_BUILDER[-1:], EXAMPLE_RAGGED_TENSOR_2D[-1:]),
      (SLICE_BUILDER[0:], EXAMPLE_RAGGED_TENSOR_2D[0:]),
      (SLICE_BUILDER[3:], EXAMPLE_RAGGED_TENSOR_2D[3:]),
      (SLICE_BUILDER[5:], EXAMPLE_RAGGED_TENSOR_2D[5:]),

      # Tests for rt[:j]
      (SLICE_BUILDER[:-6], EXAMPLE_RAGGED_TENSOR_2D[:-6]),
      (SLICE_BUILDER[:-3], EXAMPLE_RAGGED_TENSOR_2D[:-3]),
      (SLICE_BUILDER[:-1], EXAMPLE_RAGGED_TENSOR_2D[:-1]),
      (SLICE_BUILDER[:0], EXAMPLE_RAGGED_TENSOR_2D[:0]),
      (SLICE_BUILDER[:3], EXAMPLE_RAGGED_TENSOR_2D[:3]),
      (SLICE_BUILDER[:5], EXAMPLE_RAGGED_TENSOR_2D[:5]),

      # Tests for rt[i:j]
      (SLICE_BUILDER[0:3], EXAMPLE_RAGGED_TENSOR_2D[0:3]),
      (SLICE_BUILDER[3:5], EXAMPLE_RAGGED_TENSOR_2D[3:5]),
      (SLICE_BUILDER[-5:3], EXAMPLE_RAGGED_TENSOR_2D[-5:3]),
      (SLICE_BUILDER[3:1], EXAMPLE_RAGGED_TENSOR_2D[3:1]),
      (SLICE_BUILDER[-1:1], EXAMPLE_RAGGED_TENSOR_2D[-1:1]),
      (SLICE_BUILDER[1:-1], EXAMPLE_RAGGED_TENSOR_2D[1:-1]),

      # Tests for rt[i, j]
      (SLICE_BUILDER[0, 1], EXAMPLE_RAGGED_TENSOR_2D[0][1]),
      (SLICE_BUILDER[1, 2], EXAMPLE_RAGGED_TENSOR_2D[1][2]),
      (SLICE_BUILDER[-1, 0], EXAMPLE_RAGGED_TENSOR_2D[-1][0]),
      (SLICE_BUILDER[-3, 0], EXAMPLE_RAGGED_TENSOR_2D[-3][0]),
      (SLICE_BUILDER[:], EXAMPLE_RAGGED_TENSOR_2D),
      (SLICE_BUILDER[:, :], EXAMPLE_RAGGED_TENSOR_2D),

      # Empty slice spec.
      ([], EXAMPLE_RAGGED_TENSOR_2D),

      # Test for ellipsis
      (SLICE_BUILDER[...], EXAMPLE_RAGGED_TENSOR_2D),
      (SLICE_BUILDER[2, ...], EXAMPLE_RAGGED_TENSOR_2D[2]),
      (SLICE_BUILDER[..., :], EXAMPLE_RAGGED_TENSOR_2D),
      (SLICE_BUILDER[..., 2, 0], EXAMPLE_RAGGED_TENSOR_2D[2][0]),
      (SLICE_BUILDER[2, ..., 0], EXAMPLE_RAGGED_TENSOR_2D[2][0]),
      (SLICE_BUILDER[2, 0, ...], EXAMPLE_RAGGED_TENSOR_2D[2][0]),

      # Test for array_ops.newaxis
      (SLICE_BUILDER[array_ops.newaxis, :], [EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, array_ops.newaxis],
       [[row] for row in EXAMPLE_RAGGED_TENSOR_2D]),

      # Slicing inner ragged dimensions.
      (SLICE_BUILDER[-1:,
                     1:4], [row[1:4] for row in EXAMPLE_RAGGED_TENSOR_2D[-1:]]),
      (SLICE_BUILDER[:, 1:4], [row[1:4] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, -2:], [row[-2:] for row in EXAMPLE_RAGGED_TENSOR_2D]),

      # Strided slices
      (SLICE_BUILDER[::2], EXAMPLE_RAGGED_TENSOR_2D[::2]),
      (SLICE_BUILDER[::-1], EXAMPLE_RAGGED_TENSOR_2D[::-1]),
      (SLICE_BUILDER[::-2], EXAMPLE_RAGGED_TENSOR_2D[::-2]),
      (SLICE_BUILDER[::-3], EXAMPLE_RAGGED_TENSOR_2D[::-3]),
      (SLICE_BUILDER[:, ::2], [row[::2] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, ::-1], [row[::-1] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, ::-2], [row[::-2] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, ::-3], [row[::-3] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, 2::-1],
       [row[2::-1] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, -1::-1],
       [row[-1::-1] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[..., -1::-1],
       [row[-1::-1] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[:, 2::-2],
       [row[2::-2] for row in EXAMPLE_RAGGED_TENSOR_2D]),
      (SLICE_BUILDER[::-1, ::-1],
       [row[::-1] for row in EXAMPLE_RAGGED_TENSOR_2D[::-1]]),
  )  # pyformat: disable
  def testWithRaggedRank1(self, slice_spec, expected):
    """Test that rt.__getitem__(slice_spec) == expected."""
    # Ragged tensor
    rt = RaggedTensor.from_row_splits(EXAMPLE_RAGGED_TENSOR_2D_VALUES,
                                      EXAMPLE_RAGGED_TENSOR_2D_SPLITS)

    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_2D)
    self._TestGetItem(rt, slice_spec, expected)

  # pylint: disable=g-complex-comprehension
  @parameterized.parameters([(start, stop)
                             for start in [-2, -1, None, 0, 1, 2]
                             for stop in [-2, -1, None, 0, 1, 2]])
  def testWithStridedSlices(self, start, stop):
    test_value = [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10], [], [9],
                  [1, 2, 3, 4, 5, 6, 7, 8]]
    rt = ragged_factory_ops.constant(test_value)
    for step in [-3, -2, -1, 1, 2, 3]:
      # Slice outer dimension
      self.assertAllEqual(rt[start:stop:step], test_value[start:stop:step],
                          'slice=%s:%s:%s' % (start, stop, step))
      # Slice inner dimension
      self.assertAllEqual(rt[:, start:stop:step],
                          [row[start:stop:step] for row in test_value],
                          'slice=%s:%s:%s' % (start, stop, step))

  # pylint: disable=invalid-slice-index
  @parameterized.parameters(
      # Tests for out-of-bound errors
      (SLICE_BUILDER[5], (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[-6], (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[0, 2], (IndexError, ValueError,
                             errors.InvalidArgumentError), '.*out of bounds.*'),
      (SLICE_BUILDER[3, 0], (IndexError, ValueError,
                             errors.InvalidArgumentError), '.*out of bounds.*'),

      # Indexing into an inner ragged dimension
      (SLICE_BUILDER[:, 3], ValueError,
       'Cannot index into an inner ragged dimension'),
      (SLICE_BUILDER[:1, 3], ValueError,
       'Cannot index into an inner ragged dimension'),
      (SLICE_BUILDER[..., 3], ValueError,
       'Cannot index into an inner ragged dimension'),

      # Tests for type errors
      (SLICE_BUILDER[0.5], TypeError, re.escape(array_ops._SLICE_TYPE_ERROR)),
      (SLICE_BUILDER[1:3:0.5], TypeError, re.escape(
          array_ops._SLICE_TYPE_ERROR)),
      (SLICE_BUILDER[:, 1:3:0.5], TypeError,
       'slice strides must be integers or None'),
      (SLICE_BUILDER[:, 0.5:1.5], TypeError,
       'slice offsets must be integers or None'),
      (SLICE_BUILDER['foo'], TypeError, re.escape(array_ops._SLICE_TYPE_ERROR)),
      (SLICE_BUILDER[:, 'foo':'foo'], TypeError,
       'slice offsets must be integers or None'),

      # Tests for other errors
      (SLICE_BUILDER[..., 0, 0,
                     0], IndexError, 'Too many indices for RaggedTensor'),
  )
  def testErrorsWithRaggedRank1(self, slice_spec, expected, message):
    """Test that rt.__getitem__(slice_spec) == expected."""
    # Ragged tensor
    rt = RaggedTensor.from_row_splits(EXAMPLE_RAGGED_TENSOR_2D_VALUES,
                                      EXAMPLE_RAGGED_TENSOR_2D_SPLITS)

    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_2D)
    self._TestGetItemException(rt, slice_spec, expected, message)

  @parameterized.parameters(
      # Tests for rt[index, index, ...]
      (SLICE_BUILDER[2, 0], EXAMPLE_RAGGED_TENSOR_4D[2][0]),
      (SLICE_BUILDER[2, 0, 1], EXAMPLE_RAGGED_TENSOR_4D[2][0][1]),
      (SLICE_BUILDER[2, 0, 1, 1], EXAMPLE_RAGGED_TENSOR_4D[2][0][1][1]),
      (SLICE_BUILDER[2, 0, 1:], EXAMPLE_RAGGED_TENSOR_4D[2][0][1:]),
      (SLICE_BUILDER[2, 0, 1:, 1:], [[16], [18]]),
      (SLICE_BUILDER[2, 0, :, 1], [14, 16, 18]),
      (SLICE_BUILDER[2, 0, 1, :], EXAMPLE_RAGGED_TENSOR_4D[2][0][1]),

      # Tests for rt[index, slice, ...]
      (SLICE_BUILDER[0, :], EXAMPLE_RAGGED_TENSOR_4D[0]),
      (SLICE_BUILDER[1, :], EXAMPLE_RAGGED_TENSOR_4D[1]),
      (SLICE_BUILDER[0, :, :, 1], [[2, 4, 6], [8, 10, 12]]),
      (SLICE_BUILDER[1, :, :, 1], []),
      (SLICE_BUILDER[2, :, :, 1], [[14, 16, 18]]),
      (SLICE_BUILDER[3, :, :, 1], [[20]]),

      # Tests for rt[slice, slice, ...]
      (SLICE_BUILDER[:, :], EXAMPLE_RAGGED_TENSOR_4D),
      (SLICE_BUILDER[:, :, :, 1], [[[2, 4, 6], [8, 10, 12]], [], [[14, 16, 18]],
                                   [[20]]]),
      (SLICE_BUILDER[1:, :, :, 1], [[], [[14, 16, 18]], [[20]]]),
      (SLICE_BUILDER[-3:, :, :, 1], [[], [[14, 16, 18]], [[20]]]),

      # Test for ellipsis
      (SLICE_BUILDER[...], EXAMPLE_RAGGED_TENSOR_4D),
      (SLICE_BUILDER[2, ...], EXAMPLE_RAGGED_TENSOR_4D[2]),
      (SLICE_BUILDER[2, 0, ...], EXAMPLE_RAGGED_TENSOR_4D[2][0]),
      (SLICE_BUILDER[..., 0], [[[1, 3, 5], [7, 9, 11]], [], [[13, 15, 17]],
                               [[19]]]),
      (SLICE_BUILDER[2, ..., 0], [[13, 15, 17]]),
      (SLICE_BUILDER[2, 0, ..., 0], [13, 15, 17]),

      # Test for array_ops.newaxis
      (SLICE_BUILDER[array_ops.newaxis, :], [EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, array_ops.newaxis],
       [[row] for row in EXAMPLE_RAGGED_TENSOR_4D]),

      # Empty slice spec.
      ([], EXAMPLE_RAGGED_TENSOR_4D),

      # Slicing inner ragged dimensions.
      (SLICE_BUILDER[:, 1:4], [row[1:4] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, -2:], [row[-2:] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, :-1],
       [[v[:-1] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, 1:2],
       [[v[1:2] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[1:, 1:3, 1:2],
       [[v[1:2] for v in row[1:3]] for row in EXAMPLE_RAGGED_TENSOR_4D[1:]]),

      # Strided slices
      (SLICE_BUILDER[::2], EXAMPLE_RAGGED_TENSOR_4D[::2]),
      (SLICE_BUILDER[::-1], EXAMPLE_RAGGED_TENSOR_4D[::-1]),
      (SLICE_BUILDER[::-2], EXAMPLE_RAGGED_TENSOR_4D[::-2]),
      (SLICE_BUILDER[1::2], EXAMPLE_RAGGED_TENSOR_4D[1::2]),
      (SLICE_BUILDER[:, ::2], [row[::2] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, 1::2], [row[1::2] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, ::2],
       [[v[::2] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, 1::2],
       [[v[1::2] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, ::-1],
       [[v[::-1] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[:, :, ::-2],
       [[v[::-2] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[..., ::-1, :],
       [[v[::-1] for v in row] for row in EXAMPLE_RAGGED_TENSOR_4D]),
      (SLICE_BUILDER[..., ::-1], [[[v[::-1] for v in col] for col in row]
                                  for row in EXAMPLE_RAGGED_TENSOR_4D]),
  )  # pyformat: disable
  def testWithRaggedRank2(self, slice_spec, expected):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_nested_row_splits(
        EXAMPLE_RAGGED_TENSOR_4D_VALUES,
        [EXAMPLE_RAGGED_TENSOR_4D_SPLITS1, EXAMPLE_RAGGED_TENSOR_4D_SPLITS2])
    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_4D)
    self._TestGetItem(rt, slice_spec, expected)

  @parameterized.parameters(
      # Test for errors in unsupported cases
      (SLICE_BUILDER[:, 0], ValueError,
       'Cannot index into an inner ragged dimension.'),
      (SLICE_BUILDER[:, :, 0], ValueError,
       'Cannot index into an inner ragged dimension.'),

      # Test for out-of-bounds errors.
      (SLICE_BUILDER[1, 0], (IndexError, ValueError,
                             errors.InvalidArgumentError), '.*out of bounds.*'),
      (SLICE_BUILDER[0, 0, 3],
       (IndexError, ValueError,
        errors.InvalidArgumentError), '.*out of bounds.*'),
      (SLICE_BUILDER[5], (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[0, 5], (IndexError, ValueError,
                             errors.InvalidArgumentError), '.*out of bounds.*'),
  )
  def testErrorsWithRaggedRank2(self, slice_spec, expected, message):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_nested_row_splits(
        EXAMPLE_RAGGED_TENSOR_4D_VALUES,
        [EXAMPLE_RAGGED_TENSOR_4D_SPLITS1, EXAMPLE_RAGGED_TENSOR_4D_SPLITS2])
    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_4D)
    self._TestGetItemException(rt, slice_spec, expected, message)

  @parameterized.parameters(
      (SLICE_BUILDER[:], []),
      (SLICE_BUILDER[2:], []),
      (SLICE_BUILDER[:-3], []),
  )
  def testWithEmptyTensor(self, slice_spec, expected):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_row_splits([], [0])
    self._TestGetItem(rt, slice_spec, expected)

  @parameterized.parameters(
      (SLICE_BUILDER[0], (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
      (SLICE_BUILDER[-1], (IndexError, ValueError, errors.InvalidArgumentError),
       '.*out of bounds.*'),
  )
  def testErrorsWithEmptyTensor(self, slice_spec, expected, message):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_row_splits([], [0])
    self._TestGetItemException(rt, slice_spec, expected, message)

  @parameterized.parameters(
      (SLICE_BUILDER[-4], EXAMPLE_RAGGED_TENSOR_2D[-4]),
      (SLICE_BUILDER[0], EXAMPLE_RAGGED_TENSOR_2D[0]),
      (SLICE_BUILDER[-3:], EXAMPLE_RAGGED_TENSOR_2D[-3:]),
      (SLICE_BUILDER[:3], EXAMPLE_RAGGED_TENSOR_2D[:3]),
      (SLICE_BUILDER[3:5], EXAMPLE_RAGGED_TENSOR_2D[3:5]),
      (SLICE_BUILDER[0, 1], EXAMPLE_RAGGED_TENSOR_2D[0][1]),
      (SLICE_BUILDER[-3, 0], EXAMPLE_RAGGED_TENSOR_2D[-3][0]),
  )
  def testWithPlaceholderShapes(self, slice_spec, expected):
    """Test that rt.__getitem__(slice_spec) == expected."""
    # Intentionally use an unknown shape for `splits`, to force the code path
    # that deals with having nrows unknown at graph construction time.
    splits = constant_op.constant(
        EXAMPLE_RAGGED_TENSOR_2D_SPLITS, dtype=dtypes.int64)
    splits = array_ops.placeholder_with_default(splits, None)
    rt = RaggedTensor.from_row_splits(EXAMPLE_RAGGED_TENSOR_2D_VALUES, splits)
    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_2D)
    self._TestGetItem(rt, slice_spec, expected)

  @parameterized.parameters(
      (SLICE_BUILDER[..., 2], ValueError,
       'Ellipsis not supported for unknown shape RaggedTensors'),)
  def testErrorsWithPlaceholderShapes(self, slice_spec, expected, message):
    """Test that rt.__getitem__(slice_spec) == expected."""
    if not context.executing_eagerly():
      # Intentionally use an unknown shape for `values`.
      values = array_ops.placeholder_with_default([0], None)
      rt = RaggedTensor.from_row_splits(values, [0, 1])
      self._TestGetItemException(rt, slice_spec, expected, message)

  def testNewAxis(self):
    # rt: [[[['a', 'b'], ['c', 'd']], [], [['e', 'f']]], []]
    splits1 = [0, 3, 3]
    splits2 = [0, 2, 2, 3]
    values = constant_op.constant([['a', 'b'], ['c', 'd'], ['e', 'f']])
    rt = RaggedTensor.from_nested_row_splits(values, [splits1, splits2])
    rt_newaxis0 = rt[array_ops.newaxis]
    rt_newaxis1 = rt[:, array_ops.newaxis]
    rt_newaxis2 = rt[:, :, array_ops.newaxis]
    rt_newaxis3 = rt[:, :, :, array_ops.newaxis]
    rt_newaxis4 = rt[:, :, :, :, array_ops.newaxis]

    self.assertAllEqual(
        rt, [[[[b'a', b'b'], [b'c', b'd']], [], [[b'e', b'f']]], []])
    self.assertAllEqual(
        rt_newaxis0, [[[[[b'a', b'b'], [b'c', b'd']], [], [[b'e', b'f']]], []]])
    self.assertAllEqual(
        rt_newaxis1,
        [[[[[b'a', b'b'], [b'c', b'd']], [], [[b'e', b'f']]]], [[]]])
    self.assertAllEqual(
        rt_newaxis2,
        [[[[[b'a', b'b'], [b'c', b'd']]], [[]], [[[b'e', b'f']]]], []])
    self.assertAllEqual(
        rt_newaxis3,
        [[[[[b'a', b'b']], [[b'c', b'd']]], [], [[[b'e', b'f']]]], []])
    self.assertAllEqual(
        rt_newaxis4,
        [[[[[b'a'], [b'b']], [[b'c'], [b'd']]], [], [[[b'e'], [b'f']]]], []])

    self.assertEqual(rt.ragged_rank, 2)
    self.assertEqual(rt_newaxis0.ragged_rank, 3)
    self.assertEqual(rt_newaxis1.ragged_rank, 3)
    self.assertEqual(rt_newaxis2.ragged_rank, 3)
    self.assertEqual(rt_newaxis3.ragged_rank, 2)
    self.assertEqual(rt_newaxis4.ragged_rank, 2)

    self.assertEqual(rt_newaxis0.shape.as_list(), [1, 2, None, None, 2])
    self.assertEqual(rt_newaxis1.shape.as_list(), [2, 1, None, None, 2])
    self.assertEqual(rt_newaxis2.shape.as_list(), [2, None, 1, None, 2])
    self.assertEqual(rt_newaxis3.shape.as_list(), [2, None, None, 1, 2])
    self.assertEqual(rt_newaxis4.shape.as_list(), [2, None, None, 2, 1])

  @parameterized.parameters(
      # EXAMPLE_RAGGED_TENSOR_3D.shape = [2, 3, None]

      # Indexing into uniform_row_splits dimension:
      (SLICE_BUILDER[:, 1], [r[1] for r in EXAMPLE_RAGGED_TENSOR_3D],
       [2, None]),
      (SLICE_BUILDER[:, 2], [r[2] for r in EXAMPLE_RAGGED_TENSOR_3D],
       [2, None]),
      (SLICE_BUILDER[:, -2], [r[-2] for r in EXAMPLE_RAGGED_TENSOR_3D],
       [2, None]),
      (SLICE_BUILDER[:, -3], [r[-3] for r in EXAMPLE_RAGGED_TENSOR_3D],
       [2, None]),
      (SLICE_BUILDER[1:, 2], [r[2] for r in EXAMPLE_RAGGED_TENSOR_3D[1:]],
       [1, None]),
      (SLICE_BUILDER[:, 1, 1:], [r[1][1:] for r in EXAMPLE_RAGGED_TENSOR_3D],
       [2, None]),
      (SLICE_BUILDER[1:, 1, 1:],
       [r[1][1:] for r in EXAMPLE_RAGGED_TENSOR_3D[1:]],
       [1, None]),

      # Slicing uniform_row_splits dimension:
      (SLICE_BUILDER[:, 2:], [r[2:] for r in EXAMPLE_RAGGED_TENSOR_3D],
       [2, 1, None]),
      (SLICE_BUILDER[:, -2:], [r[-2:] for r in EXAMPLE_RAGGED_TENSOR_3D],
       [2, 2, None]),
      (SLICE_BUILDER[:, :, 1:],
       [[c[1:] for c in r] for r in EXAMPLE_RAGGED_TENSOR_3D],
       [2, 3, None]),
      (SLICE_BUILDER[:, 5:], [r[5:] for r in EXAMPLE_RAGGED_TENSOR_3D],
       [2, 0, None]),

      # Slicing uniform_row_splits dimension with a non-default step size:
      (SLICE_BUILDER[:, ::2], [r[::2] for r in EXAMPLE_RAGGED_TENSOR_3D],
       [2, 2, None]),
      (SLICE_BUILDER[:, ::-1], [r[::-1] for r in EXAMPLE_RAGGED_TENSOR_3D],
       [2, 3, None]),
  )  # pyformat: disable
  def testWithUniformRowLength(self, slice_spec, expected, expected_shape):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_uniform_row_length(
        RaggedTensor.from_row_splits(EXAMPLE_RAGGED_TENSOR_3D_VALUES,
                                     EXAMPLE_RAGGED_TENSOR_3D_SPLITS),
        EXAMPLE_RAGGED_TENSOR_3D_ROWLEN)
    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_3D)
    self.assertIsNot(rt.uniform_row_length, None)
    self._TestGetItem(rt, slice_spec, expected, expected_shape)

    # If the result is 3D, then check that it still has a uniform row length:
    actual = rt.__getitem__(slice_spec)  # pylint: disable=assignment-from-no-return
    if actual.shape.rank == 3:
      self.assertIsNot(actual.uniform_row_length, None)
      self.assertAllEqual(actual.uniform_row_length, expected_shape[1])

  @parameterized.parameters(
      (SLICE_BUILDER[:, 3], errors.InvalidArgumentError, 'out of bounds'),
      (SLICE_BUILDER[:, -4], errors.InvalidArgumentError, 'out of bounds'),
      (SLICE_BUILDER[:, 10], errors.InvalidArgumentError, 'out of bounds'),
      (SLICE_BUILDER[:, -10], errors.InvalidArgumentError, 'out of bounds'),
  )
  def testErrorsWithUniformRowLength(self, slice_spec, expected, message):
    """Test that rt.__getitem__(slice_spec) == expected."""
    rt = RaggedTensor.from_uniform_row_length(
        RaggedTensor.from_row_splits(EXAMPLE_RAGGED_TENSOR_3D_VALUES,
                                     EXAMPLE_RAGGED_TENSOR_3D_SPLITS),
        EXAMPLE_RAGGED_TENSOR_3D_ROWLEN)
    self.assertAllEqual(rt, EXAMPLE_RAGGED_TENSOR_3D)
    self._TestGetItemException(rt, slice_spec, expected, message)


if __name__ == '__main__':
  googletest.main()
