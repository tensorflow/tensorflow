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
    with self.test_session() as session:
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
    # TODO(b/26783907): The Sort HLO is not implemented on CPU or GPU.
    if self.device in ["XLA_CPU", "XLA_GPU"]:
      return

    supported_types = set([dtypes.bfloat16.as_numpy_dtype, np.float32])
    for dtype in supported_types.intersection(self.numeric_types):
      x = np.arange(101, dtype=dtype)
      np.random.shuffle(x)
      self._assertOpOutputMatchesExpected(
          xla.sort, [x], expected=[np.arange(101, dtype=dtype)])

  def testTopK(self):
    # TODO(b/26783907): The Sort HLO is not implemented on CPU or GPU.
    if self.device in ["XLA_CPU", "XLA_GPU"]:
      return

    # Only bfloat16 is implemented.
    bfloat16 = dtypes.bfloat16.as_numpy_dtype
    if bfloat16 in self.numeric_types:
      for x in [np.arange(20)]:
        np.random.shuffle(x)
        for k in [0, 1, 2, 10, 20]:
          indices = x.argsort()[::-1][:k]

          def topk(v, k=k):
            return nn_ops.top_k(v, k=k, sorted=True)

          self._assertOpOutputMatchesExpected(
              topk, [x.astype(bfloat16)],
              expected=[x[indices].astype(bfloat16), indices])

  def testTopKZeros(self):
    """Tests that positive and negative zeros sort correctly."""
    # TODO(b/26783907): The Sort HLO is not implemented on CPU or GPU.
    if self.device in ["XLA_CPU", "XLA_GPU"]:
      return

    # Only bfloat16 is implemented.
    bfloat16 = dtypes.bfloat16.as_numpy_dtype
    if bfloat16 not in self.numeric_types:
      return

    with self.test_session() as sess:
      p = array_ops.placeholder(dtypes.bfloat16)
      with self.test_scope():
        topk = nn_ops.top_k(p, k=4)
      results = sess.run(
          topk,
          {p: np.array([0., -0., 0., 3., -0., -4., 0., -0.], dtype=bfloat16)})
      self.assertAllEqual(
          np.array([3., 0., 0., 0.], dtype=bfloat16), results[0])
      self.assertEqual(list([3, 0, 1, 2]), list(results[1]))

  def testTopKInfinities(self):
    """Tests that positive and negative infinity sort correctly."""
    # TODO(b/26783907): The Sort HLO is not implemented on CPU or GPU.
    if self.device in ["XLA_CPU", "XLA_GPU"]:
      return

    # Only bfloat16 is implemented.
    bfloat16 = dtypes.bfloat16.as_numpy_dtype
    if bfloat16 not in self.numeric_types:
      return

    with self.test_session() as sess:
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


if __name__ == "__main__":
  test.main()
