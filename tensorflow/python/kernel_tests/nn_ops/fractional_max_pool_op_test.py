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
"""Tests for fractional max pool operation."""

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


class FractionalMaxPoolTest(test.TestCase):

  # Random number generate with seed.
  _PRNG = np.random.RandomState(341261)
  _SEED = 123456

  def _MaxPoolAlongRows(self, input_matrix, row_seq, overlapping):
    """Perform max pool along row of a 2-D matrix based on row_seq.

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
      output_image = np.vstack((output_image, np.amax(
          input_matrix[row_start:row_end, :], axis=0)))  # axis 0 is along row
    # remove the sentinel row
    return output_image[1:, :]

  def _MaxPoolAlongCols(self, input_matrix, col_seq, overlapping):
    """Perform max pool along column of a 2-D matrix based on col_seq.

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
    output_matrix = self._MaxPoolAlongRows(input_matrix, col_seq, overlapping)
    return output_matrix.transpose()

  def _GetExpectedFractionalMaxPoolResult(self, input_tensor, row_seq, col_seq,
                                          overlapping):
    """Get expected fractional max pool result.

    row_seq and col_seq together defines the fractional pooling region.

    Args:
      input_tensor: Original input tensor, assuming it is a 4-D tensor, with
        dimension as [batch, height/row, width/column, channels/depth].
      row_seq: Cumulative pooling sequence along row.
      col_seq: Cumulative pooling sequence along column.
      overlapping: Use overlapping when doing pooling.

    Returns:
      A 4-D tensor that is the result of max pooling on input_tensor based on
        pooling region defined by row_seq and col_seq, conditioned on whether or
        not overlapping is used.
    """
    input_shape = input_tensor.shape
    output_shape = (input_shape[0], len(row_seq) - 1, len(col_seq) - 1,
                    input_shape[3])
    output_tensor = np.zeros(shape=output_shape, dtype=input_tensor.dtype)
    for batch in range(input_shape[0]):
      for channel in range(input_shape[3]):
        two_dim_slice = input_tensor[batch, :, :, channel]
        tmp = self._MaxPoolAlongRows(two_dim_slice, row_seq, overlapping)
        output_tensor[batch, :, :, channel] = self._MaxPoolAlongCols(
            tmp, col_seq, overlapping)

    return output_tensor

  def _ValidateFractionalMaxPoolResult(self, input_tensor, pooling_ratio,
                                       pseudo_random, overlapping):
    """Validate FractionalMaxPool's result against expected.

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
    with self.cached_session():
      p, r, c = nn_ops.fractional_max_pool_v2(
          input_tensor,
          pooling_ratio,
          pseudo_random,
          overlapping,
          seed=self._SEED)
      actual, row_seq, col_seq = self.evaluate([p, r, c])
      expected = self._GetExpectedFractionalMaxPoolResult(input_tensor, row_seq,
                                                          col_seq, overlapping)
      self.assertShapeEqual(expected, p)
      self.assertAllClose(expected, actual)

  def _testVisually(self):
    """Manual test by printing out intermediate result of a small random tensor.

    Since _GetExpectedFractionalMaxPoolResult is 'automated', it feel safer to
    have a test case that you can see what's happening.
    This test will generate a small, random, int 2D matrix, and feed it to
    FractionalMaxPool and _GetExpectedFractionalMaxPoolResult.
    """
    num_rows = 6
    num_cols = 6
    tensor_shape = (1, num_rows, num_cols, 1)
    pseudo_random = False
    for overlapping in True, False:
      print("-" * 70)
      print("Testing FractionalMaxPool with overlapping = {}".format(
          overlapping))
      rand_mat = self._PRNG.randint(10, size=tensor_shape)
      pooling_ratio = [1, math.sqrt(2), math.sqrt(2), 1]
      with self.cached_session():
        p, r, c = nn_ops.fractional_max_pool_v2(
            rand_mat,
            pooling_ratio,
            pseudo_random,
            overlapping,
            seed=self._SEED)
        tensor_output, row_seq, col_seq = self.evaluate([p, r, c])
        expected_result = self._GetExpectedFractionalMaxPoolResult(rand_mat,
                                                                   row_seq,
                                                                   col_seq,
                                                                   overlapping)
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

        print("Output from FractionalMaxPool:")
        print(tensor_output[0, :, :, 0])
        print("Expected result:")
        print(expected_result[0, :, :, 0])

  def testAllInputOptions(self):
    """Try all possible input options for fractional_max_pool.
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
        self._ValidateFractionalMaxPoolResult(
            rand_mat, [1, math.sqrt(3), math.sqrt(2), 1], pseudo_random,
            overlapping)

  def testIntegerTensorInput(self):
    """Test it works fine when input tensor is integer type.
    """
    num_batches = 5
    num_channels = 3
    num_rows = 20
    num_cols = 30
    pseudo_random = True
    overlapping = True
    tensor_shape = (num_batches, num_rows, num_cols, num_channels)
    rand_mat = self._PRNG.randint(1000, size=tensor_shape)
    self._ValidateFractionalMaxPoolResult(rand_mat,
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
            self._ValidateFractionalMaxPoolResult(
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
        self._ValidateFractionalMaxPoolResult(rand_mat,
                                              [1, row_ratio, col_ratio, 1],
                                              pseudo_random, overlapping)

  def testDivisiblePoolingRatio(self):
    """Test when num of rows/cols can evenly divide pooling ratio.

    This is a case regular max pooling can handle. Should be handled by
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
    self._ValidateFractionalMaxPoolResult(rand_mat, [1, 2, 2, 1], pseudo_random,
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
      p, r, c = nn_ops.fractional_max_pool_v2(
          input_holder,
          pooling_ratio,
          pseudo_random,
          overlapping,
          seed=self._SEED)
      # First run.
      input_a = np.zeros([3, 32, 32, 3])
      actual, row_seq, col_seq = sess.run([p, r, c], {input_holder: input_a})
      expected = self._GetExpectedFractionalMaxPoolResult(
          input_a, row_seq, col_seq, overlapping)
      self.assertSequenceEqual(expected.shape, actual.shape)
      # Second run.
      input_b = np.zeros([4, 45, 45, 3])
      actual, row_seq, col_seq = sess.run([p, r, c], {input_holder: input_b})
      expected = self._GetExpectedFractionalMaxPoolResult(
          input_b, row_seq, col_seq, overlapping)
      self.assertSequenceEqual(expected.shape, actual.shape)

  def testDeterminismExceptionThrowing(self):
    tensor_shape = (5, 20, 20, 3)
    rand_mat = self._PRNG.random_sample(tensor_shape) * 1000 - 500
    with test_util.deterministic_ops():
      with self.assertRaisesRegex(
          ValueError, "requires a non-zero seed to be passed in when "
          "determinism is enabled"):
        nn_ops.fractional_max_pool_v2(rand_mat, [1, 1.5, 1.5, 1])
      nn_ops.fractional_max_pool_v2(rand_mat, [1, 1.5, 1.5, 1], seed=1)

      with self.assertRaisesRegex(ValueError,
                                  'requires "seed" and "seed2" to be non-zero'):
        nn_ops.fractional_max_pool(rand_mat, [1, 1.5, 1.5, 1])
      nn_ops.fractional_max_pool(
          rand_mat, [1, 1.5, 1.5, 1], seed=1, seed2=1, deterministic=True)

  def testPoolingRatioHasMoreDimThanInput(self):
    with self.cached_session() as _:
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"Pooling ratio is higher than input dimension size for dimension 1.*"
      ):
        result = nn_ops.gen_nn_ops.fractional_max_pool(
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

  def testPoolingRatioValueOutOfRange(self):
    with self.cached_session() as _:
      # Whether turn on `TF2_BEHAVIOR` generates different error messages
      with self.assertRaisesRegex(
          (errors.InvalidArgumentError, ValueError),
          r"(pooling_ratio cannot be smaller than 1, got: .*)|(is negative)"):
        result = nn_ops.gen_nn_ops.fractional_max_pool(
            value=np.zeros([3, 30, 30, 3]),
            pooling_ratio=[1, -1, 3, 1],
            pseudo_random=False,
            overlapping=False,
            deterministic=False,
            seed=0,
            seed2=0,
        )
        self.evaluate(result)


class FractionalMaxPoolGradTest(test.TestCase):
  """Tests for FractionalMaxPoolGrad.

  Two types of tests for FractionalMaxPoolGrad.
  1) Test fractional_max_pool_grad() directly.
    This type of test relies on gen_nn_ops.max_pool_grad() returns the correct
  result. For example:
    * input_tensor_shape = (1, 10, 10, 1)
    * window_size = (1, 2, 2, 1)
    * stride_size = (1, 2, 2, 1)
    * padding: not really import, since 10/2 is divisible
  max pooling should generate the same result as fractional max pooling with:
    * row_sequence = [0, 2, 4, 6, 8, 10]
    * col_sequence = [0, 2, 4, 6, 8, 10]
    * overlapping = False
  This also means their gradients in such case will be the same.

    Similarly, when
    * input_tensor_shape = (1, 7, 7, 1)
    * window_size = (1, 3, 3, 1)
    * stride_size = (1, 2, 2, 1)
    * padding: not important
  max pooling should generate the same result as fractional max pooling with:
    * row_sequence = [0, 2, 4, 7]
    * col_sequence = [0, 2, 4, 7]
    * overlapping = True
  2) Test through compute_gradient_error()
  """

  _PRNG = np.random.RandomState(341261)
  _SEED = 123456

  def _GenerateUniqueRandomInputTensor(self, shape):
    """Generate 'unique' random input tensor.

    'Unique' means there's no collision values in the tensor, all elements are
    different. This is done by generating sequence of integers with step of 1
    and then randomly shuffle these integers.

    Args:
      shape: Shape of the tensor desired.

    Returns:
      A numpy ndarray with size = shape and dtype = numpy.float32.
    """
    num_elements = 1
    for size in shape:
      num_elements *= size
    x = np.arange(num_elements, dtype=np.float32)
    self._PRNG.shuffle(x)
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
                  self._GenerateUniqueRandomInputTensor(input_shape))
              window_size = [1, row_window_size, col_window_size, 1]
              stride_size = [1, row_window_size, col_window_size, 1]
              padding = "VALID"
              output_tensor = nn_ops.max_pool(input_tensor, window_size,
                                              stride_size, padding)
              output_data = self.evaluate(output_tensor)
              output_backprop = self._PRNG.randint(100, size=output_data.shape)
              input_backprop_tensor = gen_nn_ops.max_pool_grad(
                  input_tensor, output_tensor, output_backprop, window_size,
                  stride_size, padding)
              input_backprop = self.evaluate(input_backprop_tensor)
              row_seq = list(range(0, num_rows + 1, row_window_size))
              col_seq = list(range(0, num_cols + 1, col_window_size))
              fmp_input_backprop_tensor = gen_nn_ops.fractional_max_pool_grad(
                  input_tensor,
                  output_tensor,
                  output_backprop,
                  row_seq,
                  col_seq,
                  overlapping=False)
              fmp_input_backprop = self.evaluate(fmp_input_backprop_tensor)
              self.assertShapeEqual(input_backprop, fmp_input_backprop_tensor)
              self.assertAllClose(input_backprop, fmp_input_backprop)

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
                  self._GenerateUniqueRandomInputTensor(input_shape))
              window_size = [1, row_window_size, col_window_size, 1]
              stride_size = [1, row_window_size - 1, col_window_size - 1, 1]
              padding = "VALID"
              output_tensor = nn_ops.max_pool(input_tensor, window_size,
                                              stride_size, padding)
              output_data = self.evaluate(output_tensor)
              output_backprop = self._PRNG.randint(100, size=output_data.shape)
              input_backprop_tensor = gen_nn_ops.max_pool_grad(
                  input_tensor, output_tensor, output_backprop, window_size,
                  stride_size, padding)
              input_backprop = self.evaluate(input_backprop_tensor)
              row_seq = list(range(0, num_rows, row_window_size - 1))
              col_seq = list(range(0, num_cols, col_window_size - 1))
              row_seq[-1] += 1
              col_seq[-1] += 1
              fmp_input_backprop_tensor = gen_nn_ops.fractional_max_pool_grad(
                  input_tensor,
                  output_tensor,
                  output_backprop,
                  row_seq,
                  col_seq,
                  overlapping=True)
              fmp_input_backprop = self.evaluate(fmp_input_backprop_tensor)
              self.assertShapeEqual(input_backprop, fmp_input_backprop_tensor)
              self.assertAllClose(input_backprop, fmp_input_backprop)

  @test_util.run_deprecated_v1
  def testAllInputOptionsThroughGradientError(self):
    input_shape = (1, 7, 13, 1)
    input_data = self._GenerateUniqueRandomInputTensor(input_shape)
    # Add some randomness to make input_data not so 'integer'
    input_data += self._PRNG.random_sample(input_shape)
    pooling_ratio = [1, math.sqrt(2), math.sqrt(3), 1]

    for pseudo_random in True, False:
      for overlapping in True, False:
        with self.cached_session() as _:
          input_tensor = constant_op.constant(input_data, shape=input_shape)
          output_tensor, unused_a, unused_b = nn_ops.fractional_max_pool_v2(
              input_tensor,
              pooling_ratio,
              pseudo_random=pseudo_random,
              overlapping=overlapping,
              seed=self._SEED)
          output_data = self.evaluate(output_tensor)
          output_shape = output_data.shape
          # error_margin and delta setting is similar to max_pool_grad.
          error_margin = 1e-3
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
            input_data = self._GenerateUniqueRandomInputTensor(input_shape)
            # Add some randomness to make input_data not so 'integer'
            input_data += self._PRNG.random_sample(input_shape)
            with self.cached_session() as _:
              input_tensor = constant_op.constant(input_data, shape=input_shape)
              output_tensor, unused_a, unused_b = nn_ops.fractional_max_pool_v2(
                  input_tensor,
                  pooling_ratio,
                  pseudo_random=pseudo_random,
                  overlapping=overlapping,
                  seed=self._SEED)
              output_data = self.evaluate(output_tensor)
              output_shape = output_data.shape
              # error_margin and delta setting is similar to max_pool_grad.
              error_margin = 1e-3
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
    input_data = self._GenerateUniqueRandomInputTensor(input_shape)
    # Add some randomness to make input_data not so 'integer'
    input_data += self._PRNG.random_sample(input_shape)
    pooling_ratio = (1, math.sqrt(13), math.sqrt(7), 1)
    output_shape = [int(a / b) for a, b in zip(input_shape, pooling_ratio)]
    overlapping = True
    pseudo_random = False

    with self.cached_session() as _:
      input_tensor = constant_op.constant(input_data, shape=input_shape)
      output_tensor, unused_a, unused_b = nn_ops.fractional_max_pool_v2(
          input_tensor,
          pooling_ratio,
          pseudo_random=pseudo_random,
          overlapping=overlapping,
          seed=self._SEED)
      # error_margin and delta setting is similar to max_pool_grad.
      error_margin = 1e-3
      gradient_error = gradient_checker.compute_gradient_error(
          input_tensor,
          input_shape,
          output_tensor,
          output_shape,
          x_init_value=input_data.reshape(input_shape),
          delta=1e-2)
      self.assertLess(gradient_error, error_margin)

  def testWhenRepeatedMaxValueInPoolingRegion(self):
    """Test when there's repeating value in pooling region.

    There's no formal definition for what the gradient should be when there're
    multiple max value within a pooling cell. Such as
        | 1 5 |
        | 5 3 |
    The expected result depends heavily on implementation, if someone swap the
    order of a nested for loop when walking through the tensor, result would be
    very different.

    The goal of this test is to alert when someone else change the
    implementation. Current implementation scans row-by-row.
    """
    input_data = [5.0, 4.0, 6.0, 7.0,
                  3.0, 5.0, 9.0, 6.0,
                  8.0, 8.0, 9.0, 5.0,
                  7.0, 4.0, 0.0, 0.0]  # pyformat: disable
    input_size = [1, 4, 4, 1]
    output_backprop = [12.0, 15.0,
                       17.0, -5.0,
                       6.0, 21.0]  # pyformat: disable
    row_seq = [0, 1, 3, 4]
    col_seq = [0, 2, 4]
    output_data_not_overlapping = [5.0, 7.0,
                                   8.0, 9.0,
                                   7.0, 0.0]  # pyformat: disable
    output_data_overlapping = [9.0, 9.0,
                               9.0, 9.0,
                               7.0, 0.0]  # pyformat: disable
    output_size = [1, 3, 2, 1]
    expected_input_backprop_not_overlapping = np.reshape(
        [12.0, 0.0, 0.0, 15.0,
         0.0, 0.0, -5.0, 0.0,
         17.0, 0.0, 0.0, 0.0,
         6.0, 0.0, 21.0, 0.0],
        input_size)  # pyformat: disable
    expected_input_backprop_overlapping = np.reshape(
        [0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 39.0, 0.0,
         0.0, 0.0, 0.0, 0.0,
         6.0, 0.0, 21.0, 0.0],
        input_size)  # pyformat: disable
    with self.cached_session() as _:
      # Test when overlapping is False
      input_tensor = constant_op.constant(input_data, shape=input_size)
      output_tensor = constant_op.constant(
          output_data_not_overlapping, shape=output_size)
      grad = constant_op.constant(output_backprop, shape=output_size)
      r = gen_nn_ops.fractional_max_pool_grad(
          input_tensor,
          output_tensor,
          grad,
          row_seq,
          col_seq,
          overlapping=False)
      input_backprop_not_overlapping = self.evaluate(r)
      self.assertShapeEqual(
          np.reshape(expected_input_backprop_not_overlapping, input_size), r)
      self.assertAllClose(expected_input_backprop_not_overlapping,
                          input_backprop_not_overlapping)
      # Test when overlapping is True
      output_tensor = constant_op.constant(
          output_data_overlapping, shape=output_size)
      r = gen_nn_ops.fractional_max_pool_grad(
          input_tensor, output_tensor, grad, row_seq, col_seq, overlapping=True)
      input_backprop_overlapping = self.evaluate(r)
      self.assertShapeEqual(
          np.reshape(expected_input_backprop_overlapping, input_size), r)
      self.assertAllClose(expected_input_backprop_overlapping,
                          input_backprop_overlapping)

  def testInvalidSeqRaiseErrorForFractionalMaxPoolGrad(self):
    with self.assertRaises(errors.InvalidArgumentError):
      with self.cached_session():
        overlapping = True
        orig_input = constant_op.constant(
            .453409232, shape=[1, 7, 13, 1], dtype=dtypes.float32)
        orig_output = constant_op.constant(
            .453409232, shape=[1, 7, 13, 1], dtype=dtypes.float32)
        out_backprop = constant_op.constant(
            .453409232, shape=[1, 7, 13, 1], dtype=dtypes.float32)
        row_pooling_sequence = constant_op.constant(
            0, shape=[5], dtype=dtypes.int64)
        col_pooling_sequence = constant_op.constant(
            0, shape=[5], dtype=dtypes.int64)
        t = gen_nn_ops.FractionalMaxPoolGrad(
            orig_input=orig_input,
            orig_output=orig_output,
            out_backprop=out_backprop,
            row_pooling_sequence=row_pooling_sequence,
            col_pooling_sequence=col_pooling_sequence,
            overlapping=overlapping)
        self.evaluate(t)

  def testOverLargeSeqRaiseErrorForFractionalMaxPoolGrad(self):
    with self.assertRaises(errors.InvalidArgumentError):
      with self.cached_session():
        overlapping = False
        orig_input = [[[[1, 1, 1, 1, 1]]]]
        orig_output = [[[[1, 1, 1]]]]
        out_backprop = [[[[3], [3], [6]]]]
        row_pooling_sequence = [-0x4000000, 1, 1]
        col_pooling_sequence = [-0x4000000, 1, 1]
        t = gen_nn_ops.FractionalMaxPoolGrad(
            orig_input=orig_input,
            orig_output=orig_output,
            out_backprop=out_backprop,
            row_pooling_sequence=row_pooling_sequence,
            col_pooling_sequence=col_pooling_sequence,
            overlapping=overlapping)
        self.evaluate(t)


if __name__ == "__main__":
  test.main()
