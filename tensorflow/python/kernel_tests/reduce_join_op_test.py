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
"""Tests for ReduceJoin op from string_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


def _input_array(num_dims):
  """Creates an ndarray where each element is the binary of its linear index.

  Args:
    num_dims: The number of dimensions to create.

  Returns:
    An ndarray of shape [2] * num_dims.
  """
  formatter = "{:0%db}" % num_dims
  strings = [formatter.format(i) for i in xrange(2**num_dims)]
  return np.array(strings, dtype="S%d" % num_dims).reshape([2] * num_dims)


def _joined_array(num_dims, reduce_dim):
  """Creates an ndarray with the result from reduce_join on input_array.

  Args:
    num_dims: The number of dimensions of the original input array.
    reduce_dim: The dimension to reduce.

  Returns:
    An ndarray of shape [2] * (num_dims - 1).
  """
  formatter = "{:0%db}" % (num_dims - 1)
  result = np.zeros(shape=[2] * (num_dims - 1), dtype="S%d" % (2 * num_dims))
  flat = result.ravel()
  for i in xrange(2**(num_dims - 1)):
    dims = formatter.format(i)
    flat[i] = "".join([(dims[:reduce_dim] + "%d" + dims[reduce_dim:]) % j
                       for j in xrange(2)])
  return result


class UnicodeTestCase(test.TestCase):
  """Test case with Python3-compatible string comparator."""

  def assertAllEqualUnicode(self, truth, actual):
    self.assertAllEqual(
        np.array(truth).astype("U"), np.array(actual).astype("U"))


class ReduceJoinTestHelperTest(UnicodeTestCase):
  """Tests for helper functions."""

  def testInputArray(self):
    num_dims = 3
    truth = ["{:03b}".format(i) for i in xrange(2**num_dims)]
    output_array = _input_array(num_dims).reshape([-1])
    self.assertAllEqualUnicode(truth, output_array)

  def testJoinedArray(self):
    num_dims = 3
    truth_dim_zero = [["000100", "001101"], ["010110", "011111"]]
    truth_dim_one = [["000010", "001011"], ["100110", "101111"]]
    truth_dim_two = [["000001", "010011"], ["100101", "110111"]]
    output_array_dim_zero = _joined_array(num_dims, reduce_dim=0)
    output_array_dim_one = _joined_array(num_dims, reduce_dim=1)
    output_array_dim_two = _joined_array(num_dims, reduce_dim=2)
    self.assertAllEqualUnicode(truth_dim_zero, output_array_dim_zero)
    self.assertAllEqualUnicode(truth_dim_one, output_array_dim_one)
    self.assertAllEqualUnicode(truth_dim_two, output_array_dim_two)


class ReduceJoinTest(UnicodeTestCase):

  def _testReduceJoin(self,
                      input_array,
                      truth,
                      truth_shape,
                      axis,
                      keep_dims=False,
                      separator=""):
    """Compares the output of reduce_join to an expected result.

    Args:
      input_array: The string input to be joined.
      truth: An array or np.array of the expected result.
      truth_shape: An array or np.array of the expected shape.
      axis: The indices to reduce over.
      keep_dims: Whether or not to retain reduced dimensions.
      separator: The separator to use for joining.
    """
    with self.test_session():
      output = string_ops.reduce_join(
          inputs=input_array,
          axis=axis,
          keep_dims=keep_dims,
          separator=separator)
      output_array = output.eval()

    self.assertAllEqualUnicode(truth, output_array)
    self.assertAllEqual(truth_shape, output.get_shape())

  def _testMultipleReduceJoin(self, input_array, axis, separator=" "):
    """Tests reduce_join for one input and multiple axes.

    Does so by comparing the output to that from nested reduce_string_joins.
    The correctness of single-dimension reduce_join is verified by other
    tests below using _testReduceJoin.

    Args:
      input_array: The input to test.
      axis: The indices to reduce.
      separator: The separator to use when joining.
    """
    with self.test_session():
      output = string_ops.reduce_join(
          inputs=input_array, axis=axis, keep_dims=False, separator=separator)
      output_keep_dims = string_ops.reduce_join(
          inputs=input_array, axis=axis, keep_dims=True, separator=separator)

      truth = input_array
      for index in axis:
        truth = string_ops.reduce_join(
            inputs=truth, axis=index, keep_dims=True, separator=separator)
      if not axis:
        truth = constant_op.constant(truth)
      truth_squeezed = array_ops.squeeze(truth, axis=axis)
      output_array = output.eval()
      output_keep_dims_array = output_keep_dims.eval()
      truth_array = truth.eval()
      truth_squeezed_array = truth_squeezed.eval()
    self.assertAllEqualUnicode(truth_array, output_keep_dims_array)
    self.assertAllEqualUnicode(truth_squeezed_array, output_array)
    self.assertAllEqual(truth.get_shape(), output_keep_dims.get_shape())
    self.assertAllEqual(truth_squeezed.get_shape(), output.get_shape())

  def testRankOne(self):
    input_array = ["this", "is", "a", "test"]
    truth = "thisisatest"
    truth_shape = []
    self._testReduceJoin(input_array, truth, truth_shape, axis=0)

  def testRankTwo(self):
    input_array = [["this", "is", "a", "test"],
                   ["please", "do", "not", "panic"]]
    truth_dim_zero = ["thisplease", "isdo", "anot", "testpanic"]
    truth_shape_dim_zero = [4]
    truth_dim_one = ["thisisatest", "pleasedonotpanic"]
    truth_shape_dim_one = [2]
    self._testReduceJoin(
        input_array, truth_dim_zero, truth_shape_dim_zero, axis=0)
    self._testReduceJoin(
        input_array, truth_dim_one, truth_shape_dim_one, axis=1)

    expected_val = "thisisatestpleasedonotpanic"
    expected_shape = []
    self._testReduceJoin(input_array, expected_val, expected_shape, axis=None)

    # Using axis=[] is a no-op.
    expected_val = input_array
    expected_shape = [2, 4]
    self._testReduceJoin(input_array, expected_val, expected_shape, axis=[])

  def testRankFive(self):
    input_array = _input_array(num_dims=5)
    truths = [_joined_array(num_dims=5, reduce_dim=i) for i in xrange(5)]
    truth_shape = [2] * 4
    for i in xrange(5):
      self._testReduceJoin(input_array, truths[i], truth_shape, axis=i)

  def testNegative(self):
    input_array = _input_array(num_dims=5)
    truths = [_joined_array(num_dims=5, reduce_dim=i) for i in xrange(5)]
    truth_shape = [2] * 4
    for i in xrange(5):
      self._testReduceJoin(input_array, truths[i], truth_shape, axis=i - 5)

  def testSingletonDimension(self):
    input_arrays = [
        _input_array(num_dims=5).reshape([2] * i + [1] + [2] * (5 - i))
        for i in xrange(6)
    ]
    truth = _input_array(num_dims=5)
    truth_shape = [2] * 5
    for i in xrange(6):
      self._testReduceJoin(input_arrays[i], truth, truth_shape, axis=i)

  def testSeparator(self):
    input_array = [["this", "is", "a", "test"],
                   ["please", "do", "not", "panic"]]
    truth_dim_zero = ["this  please", "is  do", "a  not", "test  panic"]
    truth_shape_dim_zero = [4]
    truth_dim_one = ["this  is  a  test", "please  do  not  panic"]
    truth_shape_dim_one = [2]

    self._testReduceJoin(
        input_array,
        truth_dim_zero,
        truth_shape_dim_zero,
        axis=0,
        separator="  ")
    self._testReduceJoin(
        input_array,
        truth_dim_one,
        truth_shape_dim_one,
        axis=1,
        separator="  ")

  def testUnknownShape(self):
    input_array = [["a"], ["b"]]
    truth = ["ab"]
    truth_shape = None
    with self.test_session():
      placeholder = array_ops.placeholder(dtypes.string, name="placeholder")
      reduced = string_ops.reduce_join(placeholder, axis=0)
      output_array = reduced.eval(feed_dict={placeholder.name: input_array})
      self.assertAllEqualUnicode(truth, output_array)
      self.assertAllEqual(truth_shape, reduced.get_shape())

  def testUnknownIndices(self):
    input_array = [["this", "is", "a", "test"],
                   ["please", "do", "not", "panic"]]
    truth_dim_zero = ["thisplease", "isdo", "anot", "testpanic"]
    truth_dim_one = ["thisisatest", "pleasedonotpanic"]
    truth_shape = None
    with self.test_session():
      placeholder = array_ops.placeholder(dtypes.int32, name="placeholder")
      reduced = string_ops.reduce_join(input_array, axis=placeholder)
      output_array_dim_zero = reduced.eval(feed_dict={placeholder.name: [0]})
      output_array_dim_one = reduced.eval(feed_dict={placeholder.name: [1]})
      self.assertAllEqualUnicode(truth_dim_zero, output_array_dim_zero)
      self.assertAllEqualUnicode(truth_dim_one, output_array_dim_one)
      self.assertAllEqual(truth_shape, reduced.get_shape())

  def testKeepDims(self):
    input_array = [["this", "is", "a", "test"],
                   ["please", "do", "not", "panic"]]
    truth_dim_zero = [["thisplease", "isdo", "anot", "testpanic"]]
    truth_shape_dim_zero = [1, 4]
    truth_dim_one = [["thisisatest"], ["pleasedonotpanic"]]
    truth_shape_dim_one = [2, 1]

    self._testReduceJoin(
        input_array,
        truth_dim_zero,
        truth_shape_dim_zero,
        axis=0,
        keep_dims=True)
    self._testReduceJoin(
        input_array,
        truth_dim_one,
        truth_shape_dim_one,
        axis=1,
        keep_dims=True)

    expected_val = [["thisisatestpleasedonotpanic"]]
    expected_shape = [1, 1]
    self._testReduceJoin(
        constant_op.constant(input_array), expected_val, expected_shape,
        keep_dims=True, axis=None)

    # Using axis=[] is a no-op.
    expected_val = input_array
    expected_shape = [2, 4]
    self._testReduceJoin(
        input_array, expected_val, expected_shape, keep_dims=True, axis=[])

  def testMultiIndex(self):
    num_dims = 3
    input_array = _input_array(num_dims=num_dims)
    # Also tests [].
    for i in xrange(num_dims + 1):
      for permutation in itertools.permutations(xrange(num_dims), i):
        self._testMultipleReduceJoin(input_array, axis=permutation)

  def testInvalidReductionIndices(self):
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "Invalid reduction dim"):
        string_ops.reduce_join(inputs="", axis=0)
      with self.assertRaisesRegexp(ValueError,
                                   "Invalid reduction dimension -3"):
        string_ops.reduce_join(inputs=[[""]], axis=-3)
      with self.assertRaisesRegexp(ValueError, "Invalid reduction dimension 2"):
        string_ops.reduce_join(inputs=[[""]], axis=2)
      with self.assertRaisesRegexp(ValueError,
                                   "Invalid reduction dimension -3"):
        string_ops.reduce_join(inputs=[[""]], axis=[0, -3])
      with self.assertRaisesRegexp(ValueError, "Invalid reduction dimension 2"):
        string_ops.reduce_join(inputs=[[""]], axis=[0, 2])

  def testZeroDims(self):
    with self.test_session():
      inputs = np.zeros([0, 1], dtype=str)

      # Reduction that drops the dim of size 0.
      output = string_ops.reduce_join(inputs=inputs, axis=0)
      self.assertAllEqualUnicode([""], output.eval())

      # Reduction that keeps the dim of size 0.
      output = string_ops.reduce_join(inputs=inputs, axis=1)
      output_shape = output.eval().shape
      self.assertAllEqual([0], output_shape)

  def testInvalidArgsUnknownShape(self):
    with self.test_session():
      placeholder = array_ops.placeholder(dtypes.string, name="placeholder")
      index_too_high = string_ops.reduce_join(placeholder, axis=1)
      duplicate_index = string_ops.reduce_join(placeholder, axis=[-1, 1])
      with self.assertRaisesOpError("Invalid reduction dimension 1"):
        index_too_high.eval(feed_dict={placeholder.name: [""]})
      with self.assertRaisesOpError("Duplicate reduction dimension 1"):
        duplicate_index.eval(feed_dict={placeholder.name: [[""]]})

  def testInvalidArgsUnknownIndices(self):
    with self.test_session():
      placeholder = array_ops.placeholder(dtypes.int32, name="placeholder")
      reduced = string_ops.reduce_join(["test", "test2"], axis=placeholder)

      with self.assertRaisesOpError("reduction dimension -2"):
        reduced.eval(feed_dict={placeholder.name: -2})
      with self.assertRaisesOpError("reduction dimension 2"):
        reduced.eval(feed_dict={placeholder.name: 2})


if __name__ == "__main__":
  test.main()
