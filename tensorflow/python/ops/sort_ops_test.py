# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the sort wrapper."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.platform import test


class SortTest(test.TestCase):

  def random_array(self, shape, dtype):
    if np.issubdtype(dtype, np.integer):
      imin = np.iinfo(dtype).min
      imax = np.iinfo(dtype).max
      return np.random.randint(imin, imax, shape, dtype)
    else:
      return np.random.random(shape).astype(dtype)

  def _test_sort(self, values, axis, direction):
    expected = np.sort(values, axis=axis)
    if direction == 'DESCENDING':
      expected = np.flip(expected, axis=axis)
    self.assertAllEqual(
        expected,
        sort_ops.sort(
            constant_op.constant(values), axis=axis, direction=direction))

  def testRandom_lowDimensionality(self):
    self._testRandom_lowDimensionality(
        negative_axis=False, dtype=np.float32, direction='ASCENDING')

  def testRandom_lowDimensionality_negative(self):
    self._testRandom_lowDimensionality(
        negative_axis=True, dtype=np.float32, direction='ASCENDING')

  def _testRandom_lowDimensionality(self, negative_axis, dtype, direction):
    np.random.seed(42)
    for _ in range(20):
      rank = np.random.randint(1, 3)
      shape = [np.random.randint(0, 20) for _ in range(rank)]
      arr = self.random_array(shape, dtype)
      sort_axis = np.random.choice(rank)
      if negative_axis:
        sort_axis = -1 - sort_axis
      with self.cached_session():
        self._test_sort(arr, sort_axis, direction)

  def testRandom_highDimensionality(self):
    self._testRandom_highDimensionality(np.float32)

  def _testRandom_highDimensionality(self, dtype):
    np.random.seed(100)
    for _ in range(20):
      rank = np.random.randint(5, 15)
      shape = [np.random.randint(1, 4) for _ in range(rank)]
      arr = self.random_array(shape, dtype)
      sort_axis = np.random.choice(rank)
      with self.cached_session():
        self._test_sort(arr, sort_axis, 'ASCENDING')

  def testIntArray(self):
    dtype = np.int64
    self._testRandom_lowDimensionality(
        negative_axis=False, dtype=dtype, direction='ASCENDING')
    self._testRandom_lowDimensionality(
        negative_axis=False, dtype=dtype, direction='DESCENDING')

    # TODO(b/190410105) re-enable test once proper sort kernel is added.
    if not test_util.is_asan_enabled() and not test_util.is_ubsan_enabled():
      edges = np.linspace(
          np.iinfo(dtype).min, np.iinfo(dtype).max, 10, dtype=dtype)
      self._test_sort(edges, 0, 'ASCENDING')
      self._test_sort(edges, 0, 'DESCENDING')

  def testUIntArray(self):
    dtype = np.uint64
    self._testRandom_lowDimensionality(
        negative_axis=False, dtype=dtype, direction='ASCENDING')
    self._testRandom_lowDimensionality(
        negative_axis=False, dtype=dtype, direction='DESCENDING')
    edges = np.linspace(
        np.iinfo(dtype).min, np.iinfo(dtype).max, 10, dtype=dtype)
    self._test_sort(edges, 0, 'ASCENDING')
    self._test_sort(edges, 0, 'DESCENDING')

  def testScalar(self):
    # Create an empty scalar where the static shape is unknown.
    zeros_length_1 = array_ops.zeros(
        random_ops.random_uniform([1], minval=0, maxval=1, dtype=dtypes.int32),
        dtype=dtypes.int32)
    scalar = array_ops.zeros(zeros_length_1)
    with self.cached_session():
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  'out of bounds'):
        self.evaluate(sort_ops.sort(scalar))

  def testNegativeOutOfBounds_staticShape(self):
    arr = constant_op.constant([3, 4, 5])
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                'slice index .* out of bounds'):
      self.evaluate(sort_ops.sort(arr, axis=-4))

  def testDescending(self):
    arr = np.random.random((10, 5, 5))
    with self.cached_session():
      self.assertAllEqual(
          np.sort(arr, axis=0)[::-1],
          sort_ops.sort(
              constant_op.constant(arr), axis=0, direction='DESCENDING'))

  def testSort_staticallyKnownRank_constantTransposition(self):
    with ops.Graph().as_default():
      # The transposition array should be a constant if the rank of "values" is
      # statically known.
      tensor = random_ops.random_uniform(
          # Rank is statically known to be 5, but the dimension lengths are not
          # known.
          random_ops.random_uniform(
              shape=(5,), minval=0, maxval=10, dtype=dtypes.int32))
      sort_ops.sort(tensor, axis=1)
      transposition = (
          ops.get_default_graph().get_tensor_by_name('sort/transposition:0'))
      self.assertIsNot(tensor_util.constant_value(transposition), None)
      self.assertAllEqual(
          # Swaps "1" and "4" to put "1" at the end.
          tensor_util.constant_value(transposition),
          [0, 4, 2, 3, 1])

  def testArgsort_1d(self):
    arr = np.random.random(42)
    with self.cached_session():
      self.assertAllEqual(
          np.sort(arr), array_ops.gather(arr, sort_ops.argsort(arr)))

  def testArgsortStable(self):
    arr = constant_op.constant([1, 5, 2, 2, 3], dtype=dtypes.int32)
    ascending = [0, 2, 3, 4, 1]
    descending = [1, 4, 2, 3, 0]
    with self.cached_session():
      self.assertAllEqual(
          sort_ops.argsort(arr, direction='ASCENDING', stable=True), ascending)
      self.assertAllEqual(
          sort_ops.argsort(arr, direction='DESCENDING', stable=True),
          descending)

  def testArgsort(self):
    arr = np.random.random((5, 6, 7, 8))
    for axis in range(4):
      with self.cached_session():
        self.assertAllEqual(
            np.argsort(arr, axis=axis), sort_ops.argsort(arr, axis=axis))

  def testArgsortTensorShape(self):
    with ops.Graph().as_default():
      placeholder = array_ops.placeholder(dtypes.float32, shape=[1, None, 5])
      for axis in range(3):
        with self.cached_session():
          self.assertAllEqual(
              placeholder.shape.as_list(),
              sort_ops.argsort(placeholder, axis=axis).shape.as_list())


if __name__ == '__main__':
  test.main()
