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
"""Tests for image.extract_glimpse()."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test


class ExtractGlimpseTest(test.TestCase):

  def _VerifyValues(self, tensor_in_sizes, glimpse_sizes, offsets,
                    expected_rows, expected_cols):
    """Verifies the output values of the glimpse extraction kernel.

    Args:
      tensor_in_sizes: Input tensor dimensions in [input_rows, input_cols].
      glimpse_sizes: Dimensions of the glimpse in [glimpse_rows, glimpse_cols].
      offsets: Relative location of the center of the glimpse in the input
        image expressed as [row_offset, col_offset].
      expected_rows: A list containing the expected row numbers (None for
         out of bound entries that are expected to be replaced by uniform
         random entries in [0,1) ).
      expected_cols: Same as expected_rows, but for column numbers.
    """

    rows = tensor_in_sizes[0]
    cols = tensor_in_sizes[1]
    # Row Tensor with entries by row.
    # [[ 1 1 1 ... ]
    #  [ 2 2 2 ... ]
    #  [ 3 3 3 ... ]
    #  [ ...
    # ]
    t_rows = array_ops.tile(
        [[1.0 * r] for r in range(1, rows + 1)], [1, cols], name='tile_rows')

    # Shuffle to switch to a convention of (batch_size, height, width, depth).
    t_rows_4d = array_ops.transpose(
        array_ops.expand_dims(array_ops.expand_dims(t_rows, 0), 3),
        [0, 2, 1, 3])

    # Column Tensor with entries by column.
    # [[ 1 2 3 4 ... ]
    #  [ 1 2 3 4 ... ]
    #  [ 1 2 3 4 ... ]
    #  [ ...         ]
    # ]
    t_cols = array_ops.tile(
        [[1.0 * r for r in range(1, cols + 1)]], [rows, 1], name='tile_cols')

    # Shuffle to switch to a convention of (batch_size, height, width, depth).
    t_cols_4d = array_ops.transpose(
        array_ops.expand_dims(array_ops.expand_dims(t_cols, 0), 3),
        [0, 2, 1, 3])

    # extract_glimpses from Row and Column Tensor, respectively.
    # Switch order for glimpse_sizes and offsets to switch from (row, col)
    # convention to tensorflows (height, width) convention.
    t1 = constant_op.constant([glimpse_sizes[1], glimpse_sizes[0]], shape=[2])
    t2 = constant_op.constant([offsets[1], offsets[0]], shape=[1, 2])
    glimpse_rows = (array_ops.transpose(
        image_ops.extract_glimpse(t_rows_4d, t1, t2), [0, 2, 1, 3]))
    glimpse_cols = (array_ops.transpose(
        image_ops.extract_glimpse(t_cols_4d, t1, t2), [0, 2, 1, 3]))

    # Evaluate the TensorFlow Graph.
    with self.cached_session() as sess:
      value_rows, value_cols = self.evaluate([glimpse_rows, glimpse_cols])

    # Check dimensions of returned glimpse.
    self.assertEqual(value_rows.shape[1], glimpse_sizes[0])
    self.assertEqual(value_rows.shape[2], glimpse_sizes[1])
    self.assertEqual(value_cols.shape[1], glimpse_sizes[0])
    self.assertEqual(value_cols.shape[2], glimpse_sizes[1])

    # Check entries.
    min_random_val = 0
    max_random_val = max(rows, cols)
    for i in range(glimpse_sizes[0]):
      for j in range(glimpse_sizes[1]):
        if expected_rows[i] is None or expected_cols[j] is None:
          self.assertGreaterEqual(value_rows[0][i][j][0], min_random_val)
          self.assertLessEqual(value_rows[0][i][j][0], max_random_val)
          self.assertGreaterEqual(value_cols[0][i][j][0], min_random_val)
          self.assertLessEqual(value_cols[0][i][j][0], max_random_val)
        else:
          self.assertEqual(value_rows[0][i][j][0], expected_rows[i])
          self.assertEqual(value_cols[0][i][j][0], expected_cols[j])

  def testCenterGlimpse(self):
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[3, 5],
        offsets=[0.0, 0.0],
        expected_rows=[20, 21, 22],
        expected_cols=[29, 30, 31, 32, 33])

  def testEmptyTensor(self):
    empty_image = np.zeros((0, 4, 3, 0))
    offsets = np.zeros((0, 2))
    with self.cached_session():
      result = image_ops.extract_glimpse(empty_image, [1, 1], offsets)
      self.assertAllEqual(
          np.zeros((0, 1, 1, 0), dtype=np.float32), self.evaluate(result))

  def testLargeCenterGlimpse(self):
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[41, 61],
        offsets=[0.0, 0.0],
        expected_rows=list(range(1, 42)),
        expected_cols=list(range(1, 62)))

  def testTooLargeCenterGlimpse(self):
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[43, 63],
        offsets=[0.0, 0.0],
        expected_rows=[None] + list(range(1, 42)) + [None],
        expected_cols=[None] + list(range(1, 62)) + [None])

  def testGlimpseFullOverlap(self):
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[3, 5],
        offsets=[0.1, 0.3],
        expected_rows=[22, 23, 24],
        expected_cols=[38, 39, 40, 41, 42])

  def testGlimpseFullOverlap2(self):
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[11, 3],
        offsets=[-0.7, -0.7],
        expected_rows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        expected_cols=[8, 9, 10])

  def testGlimpseBeforeLeftMargin(self):
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[11, 5],
        offsets=[-0.7, -0.9],
        expected_rows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        expected_cols=[1, 2, 3, 4, 5])

  def testGlimpseLowerRightCorner(self):
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[7, 5],
        offsets=[1.0, 1.0],
        expected_rows=[38, 39, 40, 41, None, None, None],
        expected_cols=[59, 60, 61, None, None])

  def testGlimpseNoOverlap(self):
    self._VerifyValues(
        tensor_in_sizes=[20, 30],
        glimpse_sizes=[3, 3],
        offsets=[-2.0, 2.0],
        expected_rows=[None, None, None],
        expected_cols=[None, None, None])

  def testGlimpseOnLeftMargin(self):
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[11, 7],
        offsets=[-0.7, -1.0],
        expected_rows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        expected_cols=[None, None, None, 1, 2, 3, 4])

  def testGlimpseUpperMargin(self):
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[7, 5],
        offsets=[-1, 0.9],
        expected_rows=[None, None, None, 1, 2, 3, 4],
        expected_cols=[56, 57, 58, 59, 60])

  def testGlimpseNoiseZeroV1Compatible(self):
    # Note: The old versions of extract_glimpse was incorrect in implementation.
    # This test is for compatibility so that graph save in old versions behave
    # the same. Notice the API uses gen_image_ops.extract_glimpse() on purpose.
    #
    # Image:
    # [  0.   1.   2.   3.   4.]
    # [  5.   6.   7.   8.   9.]
    # [ 10.  11.  12.  13.  14.]
    # [ 15.  16.  17.  18.  19.]
    # [ 20.  21.  22.  23.  24.]
    img = constant_op.constant(
        np.arange(25).reshape((1, 5, 5, 1)), dtype=dtypes.float32)
    with self.test_session():
      # Result 1:
      # [ 0.  0.  0.]
      # [ 0.  0.  0.]
      # [ 0.  0.  0.]
      result1 = gen_image_ops.extract_glimpse(
          img, [3, 3], [[-2, 2]],
          centered=False,
          normalized=False,
          noise='zero',
          uniform_noise=False)
      self.assertAllEqual(
          np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
          self.evaluate(result1)[0, :, :, 0])

      # Result 2:
      # [  0.   0.   0.   0.   0.   0.   0.]
      # [  0.   0.   1.   2.   3.   4.   0.]
      # [  0.   5.   6.   7.   8.   9.   0.]
      # [  0.  10.  11.  12.  13.  14.   0.]
      # [  0.  15.  16.  17.  18.  19.   0.]
      # [  0.  20.  21.  22.  23.  24.   0.]
      # [  0.   0.   0.   0.   0.   0.   0.]
      result2 = gen_image_ops.extract_glimpse(
          img, [7, 7], [[0, 0]],
          normalized=False,
          noise='zero',
          uniform_noise=False)
      self.assertAllEqual(
          np.asarray([[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 3, 4, 0],
                      [0, 5, 6, 7, 8, 9, 0], [0, 10, 11, 12, 13, 14, 0],
                      [0, 15, 16, 17, 18, 19, 0], [0, 20, 21, 22, 23, 24, 0],
                      [0, 0, 0, 0, 0, 0, 0]]),
          self.evaluate(result2)[0, :, :, 0])


  def testGlimpseNoiseZero(self):
    # Image:
    # [  0.   1.   2.   3.   4.]
    # [  5.   6.   7.   8.   9.]
    # [ 10.  11.  12.  13.  14.]
    # [ 15.  16.  17.  18.  19.]
    # [ 20.  21.  22.  23.  24.]
    img = constant_op.constant(
        np.arange(25).reshape((1, 5, 5, 1)), dtype=dtypes.float32)
    with self.test_session():
      # Result 1:
      # [ 0.  0.  0.]
      # [ 0.  0.  0.]
      # [ 0.  0.  0.]
      result1 = image_ops.extract_glimpse_v2(
          img, [3, 3], [[-2, -2]],
          centered=False,
          normalized=False,
          noise='zero')
      self.assertAllEqual(
          np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
          self.evaluate(result1)[0, :, :, 0])

      # Result 2:
      # [ 12.  13.  14.   0.   0.   0.   0.]
      # [ 17.  18.  19.   0.   0.   0.   0.]
      # [ 22.  23.  24.   0.   0.   0.   0.]
      # [  0.   0.   0.   0.   0.   0.   0.]
      # [  0.   0.   0.   0.   0.   0.   0.]
      # [  0.   0.   0.   0.   0.   0.   0.]
      # [  0.   0.   0.   0.   0.   0.   0.]
      result2 = image_ops.extract_glimpse_v2(
          img, [7, 7], [[0, 0]], normalized=False, noise='zero')
      self.assertAllEqual(
          np.asarray([[12, 13, 14, 0, 0, 0, 0], [17, 18, 19, 0, 0, 0, 0],
                      [22, 23, 24, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]),
          self.evaluate(result2)[0, :, :, 0])

  def testGlimpseNonNormalizedNonCentered(self):
    img = constant_op.constant(
        np.arange(25).reshape((1, 5, 5, 1)), dtype=dtypes.float32)
    with self.test_session():
      result1 = image_ops.extract_glimpse_v2(
          img, [3, 3], [[0, 0]], centered=False, normalized=False)
      result2 = image_ops.extract_glimpse_v2(
          img, [3, 3], [[1, 0]], centered=False, normalized=False)
      self.assertAllEqual(
          np.asarray([[0, 1, 2], [5, 6, 7], [10, 11, 12]]),
          self.evaluate(result1)[0, :, :, 0])
      self.assertAllEqual(
          np.asarray([[5, 6, 7], [10, 11, 12], [15, 16, 17]]),
          self.evaluate(result2)[0, :, :, 0])


if __name__ == '__main__':
  test.main()
