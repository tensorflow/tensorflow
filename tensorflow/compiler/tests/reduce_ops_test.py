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
"""Tests for reduction operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class ReduceOpsTest(XLATestCase):

  def _testReduction(self,
                     tf_reduce_fn,
                     np_reduce_fn,
                     dtype,
                     test_inputs,
                     rtol=1e-4,
                     atol=1e-4):
    """Tests that the output of 'tf_reduce_fn' matches numpy's output."""

    for test_input in test_inputs:
      with self.test_session() as sess:
        with self.test_scope():
          a = array_ops.placeholder(dtype)
          index = array_ops.placeholder(dtypes.int32)
          out = tf_reduce_fn(a, index)
        result = sess.run(out, {a: test_input, index: [0]})
        self.assertAllClose(
            result, np_reduce_fn(test_input, axis=0), rtol=rtol, atol=atol)

        result = sess.run(out, {a: test_input, index: [1]})
        self.assertAllClose(
            result, np_reduce_fn(test_input, axis=1), rtol=rtol, atol=atol)

        result = sess.run(out, {a: test_input, index: [-1]})
        self.assertAllClose(
            result, np_reduce_fn(test_input, axis=1), rtol=rtol, atol=atol)

        with self.assertRaisesWithPredicateMatch(
            errors_impl.InvalidArgumentError, 'Invalid reduction dim'):
          sess.run(out, {a: test_input, index: [-33]})

        with self.assertRaisesWithPredicateMatch(
            errors_impl.InvalidArgumentError, 'Invalid reduction dim'):
          sess.run(out, {a: test_input, index: [2]})

  REAL_DATA = [
      np.zeros(shape=(2, 0)),
      np.zeros(shape=(0, 30)),
      np.arange(1, 7).reshape(2, 3),
      np.arange(-10, -4).reshape(2, 3),
      np.arange(-4, 2).reshape(2, 3),
  ]
  COMPLEX_DATA = [
      np.zeros(shape=(2, 0)).astype(np.complex64),
      np.zeros(shape=(0, 30)).astype(np.complex64),
      np.arange(1, 13, dtype=np.float32).view(np.complex64).reshape(2, 3),
      np.arange(-14, -2, dtype=np.float32).view(np.complex64).reshape(2, 3),
      np.arange(-4, 8, dtype=np.float32).view(np.complex64).reshape(2, 3),
  ]
  NONEMPTY_REAL_DATA = [x for x in REAL_DATA if np.size(x) > 0]
  NONEMPTY_COMPLEX_DATA = [x for x in COMPLEX_DATA if np.size(x) > 0]
  BOOL_DATA = [
      np.array([], dtype=np.bool).reshape(2, 0),
      np.array([], dtype=np.bool).reshape(0, 3),
      np.array([[False, True, False], [True, True, False]]),
  ]

  def testReduceSumF32(self):
    self._testReduction(math_ops.reduce_sum, np.sum, np.float32, self.REAL_DATA)

  def testReduceSumC64(self):
    self._testReduction(math_ops.reduce_sum, np.sum, np.complex64,
                        self.COMPLEX_DATA)

  def testReduceProdF32(self):
    self._testReduction(math_ops.reduce_prod, np.prod, np.float32,
                        self.REAL_DATA)

  def testReduceProdC64(self):
    self._testReduction(math_ops.reduce_prod, np.prod, np.complex64,
                        self.COMPLEX_DATA)

  def testReduceMin(self):

    def reference_min(dtype, inp, axis):
      """Wrapper around np.amin that returns +infinity for an empty input."""
      if inp.shape[axis] == 0:
        if np.issubdtype(dtype, np.floating):
          return np.full(inp.shape[0:axis] + inp.shape[axis + 1:], float('inf'))
        return np.full(inp.shape[0:axis] + inp.shape[axis + 1:],
                       np.iinfo(dtype).max)
      return np.amin(inp, axis)

    for dtype in set(self.all_types).intersection(
        [np.float32, np.int32, np.int64]):
      self._testReduction(math_ops.reduce_min,
                          functools.partial(reference_min, dtype), dtype,
                          self.REAL_DATA)

  def testReduceMax(self):

    def reference_max(dtype, inp, axis):
      """Wrapper around np.amax that returns -infinity for an empty input."""
      if inp.shape[axis] == 0:
        if np.issubdtype(dtype, np.floating):
          return np.full(inp.shape[0:axis] + inp.shape[axis + 1:],
                         float('-inf'))
        return np.full(inp.shape[0:axis] + inp.shape[axis + 1:],
                       np.iinfo(dtype).min)
      return np.amax(inp, axis)

    for dtype in set(self.all_types).intersection(
        [np.float32, np.int32, np.int64]):
      self._testReduction(math_ops.reduce_max,
                          functools.partial(reference_max, dtype), dtype,
                          self.REAL_DATA)

  def testReduceMeanF32(self):
    # TODO(phawkins): mean on XLA currently returns 0 instead of NaN when
    # reducing across zero inputs.
    self._testReduction(math_ops.reduce_mean, np.mean, np.float32,
                        self.NONEMPTY_REAL_DATA)

  def testReduceMeanC64(self):
    self._testReduction(math_ops.reduce_mean, np.mean, np.complex64,
                        self.NONEMPTY_COMPLEX_DATA)

  def testReduceAll(self):
    self._testReduction(math_ops.reduce_all, np.all, np.bool, self.BOOL_DATA)

  def testReduceAny(self):
    self._testReduction(math_ops.reduce_any, np.any, np.bool, self.BOOL_DATA)


if __name__ == '__main__':
  googletest.main()
