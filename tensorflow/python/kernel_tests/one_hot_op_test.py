# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for tensorflow.ops.one_hot_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class OneHotTest(tf.test.TestCase):

  def _testOneHot(self, truth, use_gpu=False, expected_err_re=None, 
                  raises=None, **inputs):
    with self.test_session(use_gpu=use_gpu):
      if raises is not None:
        with self.assertRaises(raises):
          tf.one_hot(**inputs)
      else:
        ans = tf.one_hot(**inputs)
        if expected_err_re is None:
          tf_ans = ans.eval()
          self.assertAllClose(tf_ans, truth, atol=1e-10)
          self.assertEqual(tf_ans.shape, ans.get_shape())
        else:
          with self.assertRaisesOpError(expected_err_re):
            ans.eval()

  def _testBothOneHot(self, truth, expected_err_re=None, raises=None, **inputs):
    self._testOneHot(truth, True, expected_err_re, raises, **inputs)
    self._testOneHot(truth, False, expected_err_re, raises, **inputs)

  def _testBasic(self, dtype):
    indices = np.asarray([0, 2, -1, 1], dtype=np.int64)
    depth = 3
    on_value = np.asarray(1.0, dtype=dtype)
    off_value = np.asarray(-1.0, dtype=dtype)

    truth = np.asarray(
        [[1.0, -1.0, -1.0],
         [-1.0, -1.0, 1.0],
         [-1.0, -1.0, -1.0],
         [-1.0, 1.0, -1.0]],
        dtype=dtype)

    # axis == -1
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        dtype=dtype,
        truth=truth)

    # axis == 0
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        axis=0,
        dtype=dtype,
        truth=truth.T)  # Output is transpose version in this case

  def _testDefaultBasic(self, dtype):
    indices = np.asarray([0, 2, -1, 1], dtype=dtype)
    depth = 3

    truth = np.asarray(
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            dtype=dtype)

    # axis == -1
    self._testBothOneHot(
            indices=indices,
            depth=depth,
            dtype=dtype,
            truth=truth)

    # axis == 0
    self._testBothOneHot(
            indices=indices,
            depth=depth,
            axis=0,
            dtype=dtype,
            truth=truth.T)  # Output is transpose version in this case

  def testFloatBasic(self):
    self._testBasic(np.float32)
    self._testDefaultBasic(np.float32)

  def testDoubleBasic(self):
    self._testBasic(np.float64)
    self._testDefaultBasic(np.float64)

  def testInt32Basic(self):
    self._testBasic(np.int32)
    self._testDefaultBasic(np.int32)

  def testInt64Basic(self):
    self._testBasic(np.int64)
    self._testDefaultBasic(np.int64)

  def testComplexBasic(self):
    self._testBasic(np.complex64)
    self._testDefaultBasic(np.complex64)

  def _testBatch(self, dtype):
    indices = np.asarray([[0, 2, -1, 1],
                          [1, 0, 1, -1]],
                         dtype=np.int64)
    depth = 3
    on_value = np.asarray(1.0, dtype=dtype)
    off_value = np.asarray(-1.0, dtype=dtype)

    truth = np.asarray(
        [[[1.0, -1.0, -1.0],
          [-1.0, -1.0, 1.0],
          [-1.0, -1.0, -1.0],
          [-1.0, 1.0, -1.0]],
         [[-1.0, 1.0, -1.0],
          [1.0, -1.0, -1.0],
          [-1.0, 1.0, -1.0],
          [-1.0, -1.0, -1.0]]],
        dtype=dtype)

    # axis == -1
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        dtype=dtype,
        truth=truth)

    # axis == 1
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        axis=1,
        dtype=dtype,
        truth=[truth[0].T, truth[1].T])  # Do not transpose the batch

  def _testDefaultValuesBatch(self, dtype):
    indices = np.asarray([[0, 2, -1, 1],
                          [1, 0, 1, -1]],
                         dtype=dtype)
    depth = 3

    truth = np.asarray(
            [[[1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0]],
             [[0.0, 1.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0]]],
            dtype=dtype)

    # axis == -1
    self._testBothOneHot(
            indices=indices,
            depth=depth,
            dtype=dtype,
            truth=truth)

    # axis == 1
    self._testBothOneHot(
            indices=indices,
            depth=depth,
            axis=1,
            dtype=dtype,
            truth=[truth[0].T, truth[1].T])  # Do not transpose the batch

  def _testTypeBatch(self, dtype):
    indices = np.asarray([[0, 2, -1, 1],
                          [1, 0, 1, -1]],
                         dtype=dtype)
    depth = 3

    on_value = np.asarray(1.0, dtype=dtype)
    off_value = np.asarray(-1.0, dtype=dtype)

    truth = np.asarray(
        [[[1.0, -1.0, -1.0],
          [-1.0, -1.0, 1.0],
          [-1.0, -1.0, -1.0],
          [-1.0, 1.0, -1.0]],
         [[-1.0, 1.0, -1.0],
          [1.0, -1.0, -1.0],
          [-1.0, 1.0, -1.0],
          [-1.0, -1.0, -1.0]]],
        dtype=dtype)

    # axis == -1
    self._testBothOneHot(
            indices=indices,
            on_value=on_value,
            off_value=off_value,
            depth=depth,
            truth=truth)

    # axis == 1
    self._testBothOneHot(
            indices=indices,
            on_value=on_value,
            off_value=off_value,
            depth=depth,
            axis=1,
            truth=[truth[0].T, truth[1].T])  # Do not transpose the batch

  def testFloatBatch(self):
    self._testBatch(np.float32)
    self._testDefaultValuesBatch(np.float32)
    self._testTypeBatch(np.float32)

  def testDoubleBatch(self):
    self._testBatch(np.float64)
    self._testDefaultValuesBatch(np.float64)
    self._testTypeBatch(np.float64)

  def testInt32Batch(self):
    self._testBatch(np.int32)
    self._testDefaultValuesBatch(np.int32)
    self._testTypeBatch(np.int32)

  def testInt64Batch(self):
    self._testBatch(np.int64)
    self._testDefaultValuesBatch(np.int64)
    self._testTypeBatch(np.int64)

  def testComplexBatch(self):
    self._testBatch(np.complex64)
    self._testDefaultValuesBatch(np.complex64)
    self._testTypeBatch(np.complex64)

  def testSimpleCases(self):
    indices = [0,1,2]
    depth = 3
    truth = np.asarray(
      [[1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, 1.0]],
       dtype=np.float32)
    self._testBothOneHot(indices=indices, depth=depth, truth=truth)

    indices = [0,1,2]
    depth = 3
    truth = np.asarray(
      [[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]],
       dtype=np.int32)
    self._testBothOneHot(indices=indices, depth=depth, dtype=np.int32, 
                         truth=truth)

    indices = [0,1,2]
    depth = 3
    truth = np.asarray(
      [[1, -1, -1],
       [-1, 1, -1],
       [-1, -1, 1]],
       dtype=np.int32)
    self._testBothOneHot(indices=indices, depth=depth, on_value=1,
                         off_value=-1, truth=truth)

  def testStringDtypeError(self):
    indices = [0,1,2]
    depth = 3
    truth = np.asarray(
      [[1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, 1.0]])
    self._testBothOneHot(indices=indices, depth=depth, on_value=1,
                         off_value=-1, dtype=tf.string, raises=TypeError,
                         truth=truth)

  def testSingleValueGiven(self):
    # Only on_value provided
    indices = [0,1,2]
    depth = 3
    truth = np.asarray(
      [[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]],
       dtype=np.int32)
    self._testBothOneHot(indices=indices, depth=depth, on_value=1, truth=truth)

    # Only off_value provided
    indices = [0,1,2]
    depth = 3
    truth = np.asarray(
      [[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]],
       dtype=np.float32)
    self._testBothOneHot(indices=indices, depth=depth,
                         off_value=0.0, truth=truth)

if __name__ == "__main__":
  tf.test.main()
