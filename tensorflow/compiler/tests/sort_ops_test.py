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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class XlaSortOpTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self, op, args, expected):
    with self.cached_session() as session:
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

  def testSort(self):
    supported_types = set(
        [dtypes.bfloat16.as_numpy_dtype, np.float32, np.int32, np.uint32])
    for dtype in supported_types.intersection(self.numeric_types):
      x = np.arange(101, dtype=dtype)
      np.random.shuffle(x)
      self._assertOpOutputMatchesExpected(
          xla.sort, [x], expected=[np.arange(101, dtype=dtype)])

  def testKeyValueSort(self):
    supported_key_types = set(
        [dtypes.bfloat16.as_numpy_dtype, np.float32, np.int32, np.uint32])
    supported_value_types = set(
        [dtypes.bfloat16.as_numpy_dtype, np.float32, np.int32, np.uint32,
         dtypes.int64.as_numpy_dtype, dtypes.uint64.as_numpy_dtype])
    for key_type in supported_key_types.intersection(self.numeric_types):
      for value_type in supported_value_types.intersection(self.numeric_types):
        x = np.arange(101, dtype=key_type)
        np.random.shuffle(x)
        y = (-x).astype(value_type)
        self._assertOpOutputMatchesExpected(
            xla.key_value_sort, [x, y],
            expected=[
                np.arange(101, dtype=key_type),
                -np.arange(101, dtype=value_type)
            ])

  def testTopK(self):
    supported_types = set(
        [dtypes.bfloat16.as_numpy_dtype, np.float32, np.int32, np.uint32])
    for dtype in supported_types.intersection(self.numeric_types):
      # Use small input size for bfloat16. Otherwise, we'll get duplicate values
      # after conversion to bfloat16, so the possible resulting index array is
      # no longer unique.
      if dtype == dtypes.bfloat16.as_numpy_dtype:
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

  def testTopK2D(self):
    supported_types = set(
        [dtypes.bfloat16.as_numpy_dtype, np.float32, np.int32, np.uint32])
    for dtype in supported_types.intersection(self.numeric_types):
      # Use small input size for bfloat16. Otherwise, we'll get duplicate values
      # after conversion to bfloat16, so the possible resulting index array is
      # no longer unique.
      if dtype == dtypes.bfloat16.as_numpy_dtype:
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
    # Only bfloat16 is implemented.
    bfloat16 = dtypes.bfloat16.as_numpy_dtype
    if bfloat16 not in self.numeric_types:
      return

    with self.cached_session() as sess:
      p = array_ops.placeholder(dtypes.bfloat16)
      with self.test_scope():
        topk = nn_ops.top_k(p, k=4)
      results = sess.run(
          topk,
          {p: np.array([0., -0., 0., 3., -0., -4., 0., -0.], dtype=bfloat16)})
      self.assertAllEqual(
          np.array([3., 0., 0., 0.], dtype=bfloat16), results[0])
      self.assertEqual(list([3, 0, 2, 6]), list(results[1]))

  def testTopKInfinities(self):
    """Tests that positive and negative infinity sort correctly."""
    # Only bfloat16 is implemented.
    bfloat16 = dtypes.bfloat16.as_numpy_dtype
    if bfloat16 not in self.numeric_types:
      return

    with self.cached_session() as sess:
      p = array_ops.placeholder(dtypes.bfloat16)
      with self.test_scope():
        topk = nn_ops.top_k(p, k=6)
      results = sess.run(topk, {
          p: np.array(
              [1, 2, float("inf"), -float("inf"), -1, -2], dtype=bfloat16)
      })
      self.assertAllEqual(
          np.array(
              [float("inf"), 2.0, 1.0, -1.0, -2.0, -float("inf")],
              dtype=bfloat16), results[0])
      self.assertEqual(list([2, 1, 0, 4, 5, 3]), list(results[1]))

  def testInTopK(self):
    supported_types = set([np.int32, np.int64])
    for dtype in supported_types.intersection(self.numeric_types):
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
