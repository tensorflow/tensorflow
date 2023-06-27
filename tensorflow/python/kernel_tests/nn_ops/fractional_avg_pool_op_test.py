# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for fractional average pool operation."""

import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class FractionalAvgTest(test.TestCase):

  # Random number generate with seed.
  _PRNG = np.random.RandomState(341261000)
  _SEED = 341261001

  def _AvgPoolAlongRows(self, input_matrix, row_seq, overlapping):
    """Perform average pool along row of a 2-D matrix based on row_seq.

    Args:
      input_matrix: A 2-D matrix.
      row_seq: Cumulative pooling sequence along row.
      overlapping: Whether or not use overlapping when pooling.

    Returns:
      A 2-D matrix, with
        * num_rows = len(row_seq)-1
        * num_cols = input_matrix.num_cols.
    """
    output_image = np.zeros(input_matrix.shape[1])
    row_max = row_seq[-1]
    for i in range(row_seq.shape[0] - 1):
      row_start = row_seq[i]
      row_end = row_seq[i + 1] + 1 if overlapping else row_seq[i + 1]
      row_end = min(row_end, row_max)
      output_image = np.vstack((output_image, np.mean(
          input_matrix[row_start:row_end, :], axis=0)))  # axis 0 is along row
    # remove the sentinel row
    return output_image[1:, :]

  def _AvgPoolAlongCols(self, input_matrix, col_seq, overlapping):
    """Perform average pool along column of a 2-D matrix based on col_seq.

    Args:
      input_matrix: A 2-D matrix.
      col_seq: Cumulative pooling sequence along column.
      overlapping: Whether or not use overlapping when pooling.

    Returns:
      A 2-D matrix, with
        * num_rows = input_matrix.num_rows
        * num_cols = len(col_seq)-1.
    """
    input_matrix = input_matrix.transpose()
    output_matrix = self._AvgPoolAlongRows(input_matrix, col_seq, overlapping)
    return output_matrix.transpose()

  def _GetExpectedFractionalAvgPoolResult(self, input_tensor, row_seq, col_seq,
                                          overlapping):
    """Get expected fractional average pooling result.

    row_seq and col_seq together defines the fractional pooling region.

    Args:
      input_tensor: Original input tensor, assuming it is a 4-D tensor, with
        dimension as [batch, height/row, width/column, channels/depth].
      row_seq: Cumulative pooling sequence along row.
      col_seq: Cumulative pooling sequence along column.
      overlapping: Use overlapping when doing pooling.

    Returns:
      A 4-D tensor that is the result of average pooling on input_tensor based
        on pooling region defined by row_seq and col_seq, conditioned on whether
        or not overlapping is used.
    """
    input_shape = input_tensor.shape
    output_shape = (input_shape[0], len(row_seq) - 1, len(col_seq) - 1,
                    input_shape[3])
    output_tensor = np.zeros(shape=output_shape, dtype=input_tensor.dtype)
    for batch in range(input_shape[0]):
      for channel in range(input_shape[3]):
        two_dim_slice = input_tensor[batch, :, :, channel]
        tmp = self._AvgPoolAlongRows(two_dim_slice, row_seq, overlapping)
        output_tensor[batch, :, :, channel] = self._AvgPoolAlongCols(
            tmp, col_seq, overlapping)

    return output_tensor

  def _ValidateFractionalAvgPoolResult(self, input_tensor, pooling_ratio,
                                       pseudo_random, overlapping):
    """Validate FractionalAvgPool's result against expected.

    Expected result is computed given input_tensor, and pooling region defined
    by row_seq and col_seq.

    Args:
      input_tensor: A tensor or numpy ndarray.
      pooling_ratio: A list or tuple of length 4, first and last element be 1.
      pseudo_random: Use pseudo random method to generate pooling sequence.
      overlapping: Use overlapping when pooling.

    Returns:
      None
    """
    with self.cached_session() as sess:
      p, r, c = nn_ops.fractional_avg_pool_v2(
          input_tensor,
          pooling_ratio,
          pseudo_random,
          overlapping,
          seed=self._SEED)
      actual, row_seq, col_seq = self.evaluate([p, r, c])
      expected = self._GetExpectedFractionalAvgPoolResult(input_tensor, row_seq,
                                                          col_seq, overlapping)
      self.assertShapeEqual(expected, p)
      self.assertAllClose(expected, actual)

  def _testVisually(self):
    """Manual test by printing out intermediate result of a small random tensor.

    Since _GetExpectedFractionalAvgPoolResult is 'automated', it feels safer to
    have a test case that you can see what's happening.
    This test will generate a small, random, int 2D matrix, and feed it to
    FractionalAvgPool and _GetExpectedFractionalAvgPoolResult.
    """
    num_rows = 6
    num_cols = 6
    tensor_shape = (1, num_rows, num_cols, 1)
    pseudo_random = False
    for overlapping in True, False:
      print("-" * 70)
      print("Testing FractionalAvgPool with overlapping = {}".format(
          overlapping))
      rand_mat = self._PRNG.randint(10, size=tensor_shape)
      pooling_ratio = [1, math.sqrt(2), math.sqrt(2), 1]
      with self.cached_session() as sess:
        p, r, c = nn_ops.fractional_avg_pool_v2(
            rand_mat.astype(np.float32),
            pooling_ratio,
            pseudo_random,
            overlapping,
            seed=self._SEED)
        tensor_output, row_seq, col_seq = self.evaluate([p, r, c])
        expected_result = self._GetExpectedFractionalAvgPoolResult(
            rand_mat.astype(np.float32), row_seq, col_seq, overlapping)
        print("row sequence:")
        print(row_seq)
        print("column sequence:")
        print(col_seq)

        print("Input:")
        # Print input with pooling region marked.
        for i in range(num_rows):
          row_to_print = []
          for j in range(num_cols):
            if j in col_seq:
              row_to_print.append("|")
            row_to_print.append(str(rand_mat[0, i, j, 0]))
          row_to_print.append("|")
          if i in row_seq:
            print("-" * 2 * len(row_to_print))
          print(" ".join(row_to_print))
        print("-" * 2 * len(row_to_print))

        print("Output from FractionalAvgPool:")
        print(tensor_output[0, :, :, 0])
        print("Expected result:")
        print(expected_result[0, :, :, 0])

  def testAllInputOptions(self):
    """Try all possible input options for fractional_avg_pool.
    """
    num_batches = 5
    num_channels = 3
    num_rows = 20
    num_cols = 30
    for pseudo_random in True, False:
      for overlapping in True, False:
        tensor_shape = (num_batches, num_rows, num_cols, num_channels)
        # random tensor with value in [-500.0, 500.0)
        rand_mat = self._PRNG.random_sample(tensor_shape) * 1000 - 500
        self._ValidateFractionalAvgPoolResult(
            rand_mat, [1, math.sqrt(3), math.sqrt(2), 1], pseudo_random,
            overlapping)

  def testIntegerTensorInput(self):
    """Test FractionalAvgPool works fine when input tensor is integer type.
    """
    pseudo_random = True
    overlapping = True
    tensor_shape = (1, 6, 6, 1)
    # pyformat: disable
    mat = np.array([
        [2, 6, 4, 1, 3, 6],
        [8, 9, 1, 6, 6, 8],
        [3, 9, 8, 2, 5, 6],
        [2, 7, 9, 5, 4, 5],
        [8, 5, 0, 5, 7, 4],
        [4, 4, 5, 9, 7, 2]
    ])
    # pyformat: enable
    self._ValidateFractionalAvgPoolResult(mat.reshape(tensor_shape),
                                          [1, math.sqrt(3), math.sqrt(2), 1],
                                          pseudo_random, overlapping)

  def testDifferentTensorShapes(self):
    """Test different shapes of input tensor.

    Mainly test different combinations of num_rows and num_cols.
    """
    pseudo_random = True
    overlapping = True
    for num_batches in [1, 3]:
      for num_channels in [1, 3]:
        for num_rows in [10, 20, 50]:
          for num_cols in [10, 20, 50]:
            tensor_shape = (num_batches, num_rows, num_cols, num_channels)
            # random tensor with value in [-500.0, 500.0)
            rand_mat = self._PRNG.random_sample(tensor_shape) * 1000 - 500
            self._ValidateFractionalAvgPoolResult(
                rand_mat, [1, math.sqrt(3), math.sqrt(2), 1], pseudo_random,
                overlapping)

  def testLargePoolingRatio(self):
    """Test when pooling ratio is not within [1, 2).
    """
    pseudo_random = True
    overlapping = True
    num_batches = 3
    num_channels = 3
    num_rows = 30
    num_cols = 50
    tensor_shape = (num_batches, num_rows, num_cols, num_channels)
    for row_ratio in [math.sqrt(11), math.sqrt(37)]:
      for col_ratio in [math.sqrt(11), math.sqrt(27)]:
        # random tensor with value in [-500.0, 500.0)
        rand_mat = self._PRNG.random_sample(tensor_shape) * 1000 - 500
        self._ValidateFractionalAvgPoolResult(rand_mat,
                                              [1, row_ratio, col_ratio, 1],
                                              pseudo_random, overlapping)

  def testDivisiblePoolingRatio(self):
    """Test when num of rows/cols can evenly divide pooling ratio.

    This is a case regular average pooling can handle. Should be handled by
    fractional pooling as well.
    """
    pseudo_random = True
    overlapping = True
    num_batches = 3
    num_channels = 3
    num_rows = 30
    num_cols = 50
    tensor_shape = (num_batches, num_rows, num_cols, num_channels)
    # random tensor with value in [-500.0, 500.0)
    rand_mat = self._PRNG.random_sample(tensor_shape) * 1000 - 500
    self._ValidateFractionalAvgPoolResult(rand_mat, [1, 2, 2, 1], pseudo_random,
                                          overlapping)

  @test_util.run_deprecated_v1
  def testDifferentInputTensorShape(self):
    """Runs the operation in one session with different input tensor shapes."""
    with self.cached_session() as sess:
      input_holder = array_ops.placeholder(dtypes.float32,
                                           [None, None, None, 3])
      pooling_ratio = [1, 1.5, 1.5, 1]
      pseudo_random = False
      overlapping = False
      p, r, c = nn_ops.fractional_avg_pool_v2(
          input_holder,
          pooling_ratio,
          pseudo_random,
          overlapping,
          seed=self._SEED)
      # First run.
      input_a = np.zeros([3, 32, 32, 3])
      actual, row_seq, col_seq = sess.run([p, r, c], {input_holder: input_a})
      expected = self._GetExpectedFractionalAvgPoolResult(
          input_a, row_seq, col_seq, overlapping)
      self.assertSequenceEqual(expected.shape, actual.shape)
      # Second run.
      input_b = np.zeros([4, 60, 60, 3])
      actual, row_seq, col_seq = sess.run([p, r, c], {input_holder: input_b})
      expected = self._GetExpectedFractionalAvgPoolResult(
          input_b, row_seq, col_seq, overlapping)
      self.assertSequenceEqual(expected.shape, actual.shape)

  def testNegativeSeqValuesForGradOp(self):
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Row sequence tensor values must not be negative.*"):
      y = nn_ops.gen_nn_ops.fractional_avg_pool_grad(
          orig_input_tensor_shape=[2, 2, 2, 2],
          out_backprop=[[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11,
                                                                      12]]]],
          row_pooling_sequence=[-10, 1, 2, 3],
          col_pooling_sequence=[1, 2, 3, 4],
          overlapping=True)

      self.evaluate(y)
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"Column sequence tensor values must not be negative.*"):
        z = nn_ops.gen_nn_ops.fractional_avg_pool_grad(
            orig_input_tensor_shape=[2, 2, 2, 2],
            out_backprop=[[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11,
                                                                        12]]]],
            row_pooling_sequence=[10, 1, 2, 3],
            col_pooling_sequence=[1, 2, -3, 4],
            overlapping=True)

        self.evaluate(z)

  def testPoolingRatioHasMoreDimThanInput(self):
    with self.cached_session() as _:
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"Pooling ratio is higher than input dimension size for dimension 1.*"
      ):
        result = nn_ops.gen_nn_ops.fractional_avg_pool(
            value=constant_op.constant(
                value=[[[[1, 4, 2, 3]]]], dtype=dtypes.int64),
            pooling_ratio=[1.0, 1.44, 1.73, 1.0],
            pseudo_random=False,
            overlapping=False,
            deterministic=False,
            seed=0,
            seed2=0,
            name=None)
        self.evaluate(result)

  def testPoolingRatioIllegalSmallValue(self):
    with self.cached_session() as _:
      # Whether turn on `TF2_BEHAVIOR` generates different error messages
      with self.assertRaisesRegex(
          (errors.InvalidArgumentError, ValueError),
          r"(pooling_ratio cannot be smaller than 1, got: .*)|(is negative)"):
        result = nn_ops.gen_nn_ops.fractional_avg_pool(
            value=np.zeros([3, 30, 30, 3]),
            pooling_ratio=[1, -1, 3, 1],
            pseudo_random=False,
            overlapping=False,
            deterministic=False,
            seed=0,
            seed2=0,
        )
        self.evaluate(result)

  def testPoolingIllegalRatioForBatch(self):
    with self.cached_session() as _:
      with self.assertRaises(errors.UnimplementedError):
        result = nn_ops.gen_nn_ops.fractional_avg_pool(
            np.zeros([3, 30, 50, 3]),
            [2, 3, 1.5, 1],
            True,
            True)
        self.evaluate(result)


class FractionalAvgPoolGradTest(test.TestCase):
  """Tests for FractionalAvgPoolGrad.

  Two types of tests for FractionalAvgPoolGrad.
  1) Test fractional_avg_pool_grad() directly.
    This type of test relies on gen_nn_ops.avg_pool_grad() returns the
  correct result. For example:
    * input_tensor_shape = (1, 10, 10, 1)
    * window_size = (1, 2, 2, 1)
    * stride_size = (1, 2, 2, 1)
    * padding: not really important, since 10/2 is divisible
  avg pooling should generate the same result as fractional avg pooling with:
    * row_sequence = [0, 2, 4, 6, 8, 10]
    * col_sequence = [0, 2, 4, 6, 8, 10]
    * overlapping = False
  This also means their gradients in such case will be the same.

  Similarly, when
    * input_tensor_shape = (1, 7, 7, 1)
    * window_size = (1, 3, 3, 1)
    * stride_size = (1, 2, 2, 1)
    * padding: not important
  avg pooling should generate the same result as fractional avg pooling with:
    * row_sequence = [0, 2, 4, 7]
    * col_sequence = [0, 2, 4, 7]
    * overlapping = True
  2) Test through compute_gradient_error()
  """
  _PRNG = np.random.RandomState(341261004)
  _SEED = 341261005

  def _GenerateRandomInputTensor(self, shape):
    num_elements = 1
    for dim_size in shape:
      num_elements *= dim_size
    x = self._PRNG.rand(num_elements) * 1000
    return x.reshape(shape)

  def testDirectNotUseOverlapping(self):
    for num_batches in [1, 3]:
      for row_window_size in [2, 5]:
        for col_window_size in [2, 4]:
          num_rows = row_window_size * 5
          num_cols = col_window_size * 7
          for num_channels in [1, 2]:
            input_shape = (num_batches, num_rows, num_cols, num_channels)
            with self.cached_session() as _:
              input_tensor = constant_op.constant(
                  self._GenerateRandomInputTensor(input_shape).astype(
                      np.float32))
              window_size = [1, row_window_size, col_window_size, 1]
              stride_size = [1, row_window_size, col_window_size, 1]
              padding = "VALID"
              output_tensor = nn_ops.avg_pool(input_tensor, window_size,
                                              stride_size, padding)
              output_data = self.evaluate(output_tensor)
              num_elements = 1
              for dim_size in output_data.shape:
                num_elements *= dim_size
              output_backprop = (self._PRNG.rand(num_elements) *
                                 1000).reshape(output_data.shape)
              input_backprop_tensor = gen_nn_ops.avg_pool_grad(
                  input_tensor.get_shape(), output_backprop, window_size,
                  stride_size, padding)
              input_backprop = self.evaluate(input_backprop_tensor)
              row_seq = list(range(0, num_rows + 1, row_window_size))
              col_seq = list(range(0, num_cols + 1, col_window_size))
              fap_input_backprop_tensor = gen_nn_ops.fractional_avg_pool_grad(
                  input_tensor.get_shape(),
                  output_backprop,
                  row_seq,
                  col_seq,
                  overlapping=False)
              fap_input_backprop = self.evaluate(fap_input_backprop_tensor)
              self.assertShapeEqual(input_backprop, fap_input_backprop_tensor)
              self.assertAllClose(input_backprop, fap_input_backprop)

  def testDirectUseOverlapping(self):
    for num_batches in [1, 3]:
      for row_window_size in [2, 5]:
        for col_window_size in [2, 4]:
          num_rows = (row_window_size - 1) * 5 + 1
          num_cols = (col_window_size - 1) * 7 + 1
          for num_channels in [1, 2]:
            input_shape = (num_batches, num_rows, num_cols, num_channels)
            with self.cached_session() as _:
              input_tensor = constant_op.constant(
                  self._GenerateRandomInputTensor(input_shape).astype(
                      np.float32))
              window_size = [1, row_window_size, col_window_size, 1]
              stride_size = [1, row_window_size - 1, col_window_size - 1, 1]
              padding = "VALID"
              output_tensor = nn_ops.avg_pool(input_tensor, window_size,
                                              stride_size, padding)
              output_data = self.evaluate(output_tensor)
              num_elements = 1
              for dim_size in output_data.shape:
                num_elements *= dim_size
              output_backprop = (self._PRNG.rand(num_elements) *
                                 1000).reshape(output_data.shape)
              input_backprop_tensor = gen_nn_ops.avg_pool_grad(
                  input_tensor.get_shape(), output_backprop, window_size,
                  stride_size, padding)
              input_backprop = self.evaluate(input_backprop_tensor)
              row_seq = list(range(0, num_rows, row_window_size - 1))
              col_seq = list(range(0, num_cols, col_window_size - 1))
              row_seq[-1] += 1
              col_seq[-1] += 1
              fap_input_backprop_tensor = gen_nn_ops.fractional_avg_pool_grad(
                  input_tensor.get_shape(),
                  output_backprop,
                  row_seq,
                  col_seq,
                  overlapping=True)
              fap_input_backprop = self.evaluate(fap_input_backprop_tensor)
              self.assertShapeEqual(input_backprop, fap_input_backprop_tensor)
              self.assertAllClose(input_backprop, fap_input_backprop)

  @test_util.run_deprecated_v1
  def testAllInputOptionsThroughGradientError(self):
    input_shape = (1, 7, 13, 1)
    input_data = self._GenerateRandomInputTensor(input_shape)
    pooling_ratio = [1, math.sqrt(2), math.sqrt(3), 1]

    for pseudo_random in True, False:
      for overlapping in True, False:
        with self.cached_session() as _:
          input_tensor = constant_op.constant(input_data, shape=input_shape)
          output_tensor, unused_a, unused_b = nn_ops.fractional_avg_pool_v2(
              input_tensor,
              pooling_ratio,
              pseudo_random=pseudo_random,
              overlapping=overlapping,
              seed=self._SEED)
          output_data = self.evaluate(output_tensor)
          output_shape = output_data.shape
          # error_margin and delta setting is similar to avg_pool_grad.
          error_margin = 1e-4
          gradient_error = gradient_checker.compute_gradient_error(
              input_tensor,
              input_shape,
              output_tensor,
              output_shape,
              x_init_value=input_data.reshape(input_shape),
              delta=1e-2)
          self.assertLess(gradient_error, error_margin)

  @test_util.run_deprecated_v1
  def testDifferentTensorShapesThroughGradientError(self):
    pseudo_random = True
    overlapping = True
    pooling_ratio = [1, math.sqrt(3), math.sqrt(2), 1]
    for num_batches in [1, 2]:
      for num_rows in [5, 13]:
        for num_cols in [5, 11]:
          for num_channels in [1, 3]:
            input_shape = (num_batches, num_rows, num_cols, num_channels)
            input_data = self._GenerateRandomInputTensor(input_shape)
            with self.cached_session() as _:
              input_tensor = constant_op.constant(input_data, shape=input_shape)
              output_tensor, unused_a, unused_b = nn_ops.fractional_avg_pool_v2(
                  input_tensor,
                  pooling_ratio,
                  pseudo_random=pseudo_random,
                  overlapping=overlapping,
                  seed=self._SEED)
              output_data = self.evaluate(output_tensor)
              output_shape = output_data.shape
              # error_margin and delta setting is similar to avg_pool_grad.
              error_margin = 1e-4
              gradient_error = gradient_checker.compute_gradient_error(
                  input_tensor,
                  input_shape,
                  output_tensor,
                  output_shape,
                  x_init_value=input_data.reshape(input_shape),
                  delta=1e-2)
              self.assertLess(gradient_error, error_margin)

  @test_util.run_deprecated_v1
  def testLargePoolingRatioThroughGradientError(self):
    input_shape = (1, 17, 23, 1)
    input_data = self._GenerateRandomInputTensor(input_shape)
    pooling_ratio = (1, math.sqrt(13), math.sqrt(7), 1)
    output_shape = [int(a / b) for a, b in zip(input_shape, pooling_ratio)]
    overlapping = True
    pseudo_random = False

    with self.cached_session() as _:
      input_tensor = constant_op.constant(input_data, shape=input_shape)
      output_tensor, unused_a, unused_b = nn_ops.fractional_avg_pool_v2(
          input_tensor,
          pooling_ratio,
          pseudo_random=pseudo_random,
          overlapping=overlapping,
          seed=self._SEED)
      # error_margin and delta setting is similar to avg_pool_grad.
      error_margin = 1e-4
      gradient_error = gradient_checker.compute_gradient_error(
          input_tensor,
          input_shape,
          output_tensor,
          output_shape,
          x_init_value=input_data.reshape(input_shape),
          delta=1e-2)
      self.assertLess(gradient_error, error_margin)

  def testInvalidSeqRaiseErrorForFractionalAvgPoolGrad(self):
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      with self.cached_session() as _:
        overlapping = True
        orig_input_tensor_shape = constant_op.constant(
            -1879048192, shape=[4], dtype=dtypes.int64)
        out_backprop = constant_op.constant([],
                                            shape=[0, 0, 0, 0],
                                            dtype=dtypes.float64)
        row_pooling_sequence = constant_op.constant(
            1, shape=[4], dtype=dtypes.int64)
        col_pooling_sequence = constant_op.constant(
            1, shape=[4], dtype=dtypes.int64)
        t = gen_nn_ops.fractional_avg_pool_grad(
            orig_input_tensor_shape=orig_input_tensor_shape,
            out_backprop=out_backprop,
            row_pooling_sequence=row_pooling_sequence,
            col_pooling_sequence=col_pooling_sequence,
            overlapping=overlapping)
        self.evaluate(t)


if __name__ == "__main__":
  test.main()
