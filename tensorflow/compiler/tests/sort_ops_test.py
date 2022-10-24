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
"""Tests for sorting operators."""

import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

ALL_KEY_TYPES = [
    dtypes.bfloat16.as_numpy_dtype, np.float16, np.float32, np.float64,
    np.int32, np.uint32, np.int16, np.uint16, np.int8, np.uint8
]


class XlaSortOpTest(xla_test.XLATestCase, parameterized.TestCase):

  def _assertOpOutputMatchesExpected(self, op, args, expected):
    """Tests that op(*args) == expected."""
    with self.session() as session:
      with self.test_scope():
        placeholders = [
            array_ops.placeholder(dtypes.as_dtype(arg.dtype), arg.shape)
            for arg in args
        ]
        feeds = {placeholders[i]: args[i] for i in range(0, len(args))}
        output = op(*placeholders)
        if isinstance(output, ops.Tensor):
          output = [output]

      results = session.run(output, feeds)
      for result, v in zip(results, expected):
        self.assertAllClose(v, result, rtol=1e-3)

  def _shuffled_arange(self, shape, dtype):
    x = np.arange(np.prod(shape), dtype=dtype)
    np.random.shuffle(x)
    return x.reshape(shape)

  def _supported_key_types(self):
    supported_key_types = set(ALL_KEY_TYPES)
    res = supported_key_types.intersection(self.numeric_types)
    assert res
    return res

  def testSort(self):
    for dtype in self._supported_key_types():
      x = self._shuffled_arange((101,), dtype)
      self._assertOpOutputMatchesExpected(
          xla.sort, [x], expected=[np.arange(101, dtype=dtype)])

  def testKeyValueSort(self):
    for key_type in self._supported_key_types():
      for value_type in self._supported_key_types():
        if key_type == np.uint8 or value_type == np.uint8:
          # I do not understand why the test fails on uint8. We plan to
          # deprecate xla.key_value_sort in favor of xla.variadic_sort anyway.
          continue
        x = self._shuffled_arange((101,), key_type)
        y = (-x).astype(value_type)
        self._assertOpOutputMatchesExpected(
            xla.key_value_sort, [x, y],
            expected=[
                np.arange(101, dtype=key_type),
                -np.arange(101, dtype=value_type)
            ])

  @parameterized.parameters(0, 1, 2)
  def testVariadicSortDimension(self, dimension):
    shape = (2, 3, 4)
    for key_type in self._supported_key_types():
      x = self._shuffled_arange(shape, key_type)
      expected = np.sort(x, axis=dimension)

      @function.Defun(key_type, key_type)
      def compare_lt(x1, x2):
        return x1 < x2

      def wrap_sort(x):
        return xla.variadic_sort([x],
                                 dimension=dimension,
                                 is_stable=False,
                                 comparator=compare_lt)

      self._assertOpOutputMatchesExpected(wrap_sort, [x], expected=[expected])

  def testVariadicSortReverse(self):
    shape = (100,)
    for key_type in self._supported_key_types():
      x = self._shuffled_arange(shape, key_type)
      expected = np.sort(x, axis=0)[::-1]

      @function.Defun(key_type, key_type)
      def compare_gt(x1, x2):
        return x1 > x2

      def wrap_sort(x):
        return xla.variadic_sort([x],
                                 dimension=0,
                                 is_stable=False,
                                 comparator=compare_gt)

      self._assertOpOutputMatchesExpected(wrap_sort, [x], expected=[expected])

  @parameterized.product(dimension=[0, 1, 2], key_type=ALL_KEY_TYPES)
  def testVariadicSortSeveral(self, dimension, key_type):
    if np.__version__ < "1.15":
      raise unittest.SkipTest("np.take_along_axis was added in 1.15")
    if key_type not in self._supported_key_types():
      return
    shape = (2, 3, 4)
    for value_type_1 in self._supported_key_types():
      for value_type_2 in self._supported_key_types():
        inputs = [
            self._shuffled_arange(shape, key_type),
            self._shuffled_arange(shape, value_type_1),
            self._shuffled_arange(shape, value_type_2)
        ]

        # The first array is sorted, and the others are shuffled the same way
        sorted_indices = np.argsort(inputs[0], axis=dimension)
        expected = [
            np.take_along_axis(inp, sorted_indices, axis=dimension)
            for inp in inputs
        ]
        self.assertAllEqual(np.sort(inputs[0], axis=dimension), expected[0])

        @function.Defun(key_type, key_type, value_type_1, value_type_1,
                        value_type_2, value_type_2)
        def compare_lt(x1, x2, y1, y2, z1, z2):
          del y1, y2, z1, z2
          return x1 < x2

        def wrap_sort(*args):
          return xla.variadic_sort(
              args,  # Pass the arguments as a tuple
              comparator=compare_lt,
              dimension=dimension,
              is_stable=False)

        self._assertOpOutputMatchesExpected(
            wrap_sort, inputs, expected=expected)

  @parameterized.parameters(ALL_KEY_TYPES)
  @test_util.disable_mlir_bridge("Not supported yet")
  def testVariadicSortLexicographic(self, key_type_2):
    # Three inputs: the first two are used for lexicographic sort, and the
    # third is just swapped accordingly.
    # The first array will contain only 0 and 1, to test lexicographic order
    if np.__version__ < "1.15":
      raise unittest.SkipTest("np.take_along_axis was added in 1.15")
    shape = (20,)
    if key_type_2 not in self._supported_key_types():
      return
    for key_type_1 in [np.int16, np.uint16, np.int32, np.uint32]:
      for value_type in self._supported_key_types():
        inputs = [
            # Ensure that some keys in the first input are equal
            np.random.uniform(0, 2, shape).astype(key_type_1),
            self._shuffled_arange(shape, key_type_2),
            self._shuffled_arange(shape, value_type)
        ]
        # The first two arrays are sorted lexicographically, and the third
        # is shuffled the same way
        sorted_indices = np.argsort(100 * inputs[0] + inputs[1])
        expected = [
            np.take_along_axis(inp, sorted_indices, axis=0) for inp in inputs
        ]

        @function.Defun(key_type_1, key_type_1, key_type_2, key_type_2,
                        value_type, value_type)
        def compare_lexicographic(x1, x2, y1, y2, z1, z2):
          del z1, z2
          return math_ops.logical_or(
              x1 < x2, math_ops.logical_and(math_ops.equal(x1, x2), y1 < y2))

        def wrap_sort(*args):
          return xla.variadic_sort(
              args,  # Pass the arguments as a tuple
              comparator=compare_lexicographic,
              dimension=0,
              is_stable=False)

        self._assertOpOutputMatchesExpected(
            wrap_sort, inputs, expected=expected)

  @parameterized.product(dimension=[0, 1, 2], key_type=ALL_KEY_TYPES)
  def testVariadicSortSeveralStable(self, dimension, key_type):
    shape = (2, 3, 4)
    if key_type not in self._supported_key_types():
      return
    for value_type_1 in self._supported_key_types():
      for value_type_2 in self._supported_key_types():
        # The first input is all 0s, there should be no changes for
        # stable sort.
        inputs = [
            np.zeros(shape, key_type),
            self._shuffled_arange(shape, value_type_1),
            self._shuffled_arange(shape, value_type_2)
        ]

        @function.Defun(key_type, key_type, value_type_1, value_type_1,
                        value_type_2, value_type_2)
        def compare_lt(x1, x2, y1, y2, z1, z2):
          del y1, y2, z1, z2
          return x1 < x2

        def wrap_sort(*args):
          return xla.variadic_sort(
              args,  # Pass the arguments as a tuple
              comparator=compare_lt,
              dimension=dimension,
              is_stable=True)

        self._assertOpOutputMatchesExpected(wrap_sort, inputs, expected=inputs)

  def testTopK(self):
    supported_types = set([
        dtypes.bfloat16.as_numpy_dtype, np.float16, np.float32, np.float64,
        np.int32, np.uint32, np.int64, np.uint64
    ])
    for dtype in supported_types.intersection(self.numeric_types):
      # Use small input size for bfloat16. Otherwise, we'll get duplicate values
      # after conversion to bfloat16, so the possible resulting index array is
      # no longer unique.
      if dtype in (dtypes.bfloat16.as_numpy_dtype, np.float16):
        array_size = 20
        k_options = [0, 1, 2, 10, 20]
      else:
        array_size = 200 * 1000
        k_options = [0, 1, 2, 10, 20, 100, 1000, 200 * 1000]
      for x in [np.arange(array_size)]:
        np.random.shuffle(x)
        for k in k_options:
          indices = x.argsort()[::-1][:k]

          def topk(v, k=k):
            return nn_ops.top_k(v, k=k, sorted=True)

          self._assertOpOutputMatchesExpected(
              topk, [x.astype(dtype)],
              expected=[x[indices].astype(dtype), indices])

  @parameterized.named_parameters(
      ("HalfPrecision", dtypes.bfloat16.as_numpy_dtype),
      ("HalfFloatPrecision", np.float16),
      ("SinglePrecision", np.float32),
      ("DoublePrecision", np.float64),
      ("Int32", np.int32),
      ("UnsignedInt32", np.uint32),
      ("Int64", np.int64),
      ("UnsignedInt64", np.uint64),
  )
  def testTopK2D(self, dtype):
    if dtype in self.numeric_types:
      # Use small input size for bfloat16. Otherwise, we'll get duplicate values
      # after conversion to bfloat16, so the possible resulting index array is
      # no longer unique.
      if dtype in (dtypes.bfloat16.as_numpy_dtype, np.float16):
        array_size = 10
        k_options = [0, 1, 2, 10]
      else:
        array_size = 200 * 1000
        k_options = [0, 1, 2, 10, 20, 100, 1000, 200 * 1000]
      batch = 16
      for x in [np.arange(batch * array_size)]:
        np.random.shuffle(x)
        x = np.reshape(x, [batch, array_size])
        for k in k_options:
          indices = x.argsort(axis=1)[::, -1:-k - 1:-1]
          expected = np.sort(x, axis=1)[::, -1:-k - 1:-1]

          def topk(v, k=k):
            return nn_ops.top_k(v, k=k, sorted=True)

          self._assertOpOutputMatchesExpected(
              topk, [x.astype(dtype)],
              expected=[expected.astype(dtype), indices])

  def testTopKZeros(self):
    """Tests that positive and negative zeros sort correctly."""
    supported_types = set(
        [dtypes.bfloat16.as_numpy_dtype, np.float16, np.float32, np.float64])
    for dtype in supported_types.intersection(self.numeric_types):
      with self.session() as sess:
        p = array_ops.placeholder(dtype)
        with self.test_scope():
          topk = nn_ops.top_k(p, k=4)
        results = sess.run(
            topk,
            {p: np.array([0., -0., 0., 3., -0., -4., 0., -0.], dtype=dtype)})
        self.assertAllEqual(np.array([3., 0., 0., 0.], dtype=dtype), results[0])
        self.assertEqual(list([3, 0, 2, 6]), list(results[1]))

  def testTopKInfinities(self):
    """Tests that positive and negative infinity sort correctly."""
    supported_types = set(
        [dtypes.bfloat16.as_numpy_dtype, np.float16, np.float32, np.float64])
    for dtype in supported_types.intersection(self.numeric_types):
      with self.session() as sess:
        p = array_ops.placeholder(dtype)
        with self.test_scope():
          topk = nn_ops.top_k(p, k=6)
        results = sess.run(topk, {
            p:
                np.array([1, 2, float("inf"), -float("inf"), -1, -2],
                         dtype=dtype)
        })
        self.assertAllEqual(
            np.array([float("inf"), 2.0, 1.0, -1.0, -2.0, -float("inf")],
                     dtype=dtype), results[0])
        self.assertEqual(list([2, 1, 0, 4, 5, 3]), list(results[1]))

  @parameterized.named_parameters(
      ("Int32", np.int32),
      ("Int64", np.uint64),
  )
  def testInTopK(self, dtype):
    if dtype in self.numeric_types:
      array_size = 200 * 1000
      k_options = [0, 1, 2, 10, 20, 100, 1000, 200 * 1000]
      batch = 16
      for x in [np.arange(batch * array_size)]:
        np.random.shuffle(x)
        x = np.reshape(x, [batch, array_size])
        y = np.random.randint(0, array_size, size=batch)
        for k in k_options:
          indices = x.argsort(axis=1)[::, -1:-k - 1:-1]
          expected = [y[i] in indices[i] for i in range(batch)]

          def in_topk(predictions, targets, k=k):
            return nn_ops.in_top_k(predictions, targets, k)

          self._assertOpOutputMatchesExpected(
              in_topk,
              [x.astype(np.float32), y.astype(dtype)],
              expected=[expected])


if __name__ == "__main__":
  test.main()
