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
"""Tests for Repeat."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.repeat.python.ops import repeat_op
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test

repeat = repeat_op.repeat

class RepeatTest(test.TestCase):
  
  def _testRepeat(self, input, repeats, axis, use_gpu=False, expected_err=None):
    if expected_err is None:
      np_repeat = np.repeat(input, repeats, axis)
      tf_repeat_tensor = repeat(input, repeats, axis)
      with self.test_session(use_gpu=use_gpu):
        tf_repeat = tf_repeat_tensor.eval()
      self.assertAllClose(np_repeat, tf_repeat)
      self.assertShapeEqual(np_repeat, tf_repeat_tensor)
    else:
      with self.test_session(use_gpu=use_gpu):
        with self.assertRaisesOpError(expected_err):
          repeat(input, repeats, axis).eval()
    
  def _testScalar(self, dtype):
    input = 5
    repeats = 4
    axis = 0
    self._testRepeat(input, repeats, axis)
    
    input = np.asarray(100 * np.random.randn(200), dtype=dtype)
    repeats = 2
    axis = 0
    self._testRepeat(input, repeats, axis)
    
    input = np.asarray(100 * np.random.randn(3, 2, 4, 5, 6), dtype=dtype)
    repeats = 3
    axis = 1
    self._testRepeat(input, repeats, axis)
    
  def _testVector(self, dtype):
    input = np.asarray(100 * np.random.randn(200), dtype=dtype)
    repeats = np.asarray(10 * np.random.randn(200), dtype=np.int32) % 5
    axis = 0
    self._testRepeat(input, repeats, axis)
    
    input = np.asarray(100 * np.random.randn(3, 2, 4, 5, 6), dtype=dtype)
    repeats = np.asarray(10 * np.random.randn(4), dtype=np.int32) % 5
    axis = 2
    self._testRepeat(input, repeats, axis)
    
  def _testNegativeAxis(self, dtype, use_gpu=False):
    input = np.asarray(100 * np.random.randn(200), dtype=dtype)
    repeats = 2
    axis = -1
    self._testRepeat(input, repeats, axis, use_gpu=use_gpu)
    
    input = np.asarray(100 * np.random.randn(3, 2, 4, 5, 6), dtype=dtype)
    repeats = np.asarray(10 * np.random.randn(5), dtype=np.int32) % 5
    axis = -2
    self._testRepeat(input, repeats, axis, use_gpu=use_gpu)
    
  def testFloat(self):
    self._testScalar(np.float32)
    self._testVector(np.float32)
    self._testNegativeAxis(np.float32)

  def testDouble(self):
    self._testScalar(np.float64)
    self._testVector(np.float64)
    self._testNegativeAxis(np.float64)

  def testInt32(self):
    self._testScalar(np.int32)
    self._testVector(np.int32)
    self._testNegativeAxis(np.int32)

  def testInt64(self):
    self._testScalar(np.int64)
    self._testVector(np.int64)
    self._testNegativeAxis(np.int64)
    
class RepeatGradTest(test.TestCase):
  
  def _testRepeatGrad(self, input, repeats, axis):
    output = repeat(input, repeats, axis)
    in_shape = input.get_shape().as_list()
    out_shape = output.get_shape().as_list()
    with self.test_session():
      err = gradient_checker.compute_gradient_error(
          input, in_shape, output, out_shape)
    self.assertLess(err, 1e-3)
    
  def _testScalar(self, dtype):
    input = constant_op.constant(
        np.asarray(100 * np.random.randn(1), dtype=dtype), shape = [])
    repeats = 5
    axis = 0
    self._testRepeatGrad(input, repeats, axis)
    
    input = constant_op.constant(
        np.asarray(100 * np.random.randn(20), dtype=dtype))
    repeats = 2
    axis = 0
    self._testRepeatGrad(input, repeats, axis)
    
    input = constant_op.constant(
        np.asarray(100 * np.random.randn(3, 2, 4), dtype=dtype))
    repeats = 3
    axis = 1
    self._testRepeatGrad(input, repeats, axis)
    
  def _testVector(self, dtype):
    input = constant_op.constant(
        np.asarray(100 * np.random.randn(20), dtype=dtype))
    repeats = np.asarray(10 * np.random.randn(20), dtype=np.int32) % 5
    axis = 0
    self._testRepeatGrad(input, repeats, axis)
    
    input = constant_op.constant(
        np.asarray(100 * np.random.randn(3, 2, 4), dtype=dtype))
    repeats = np.asarray(10 * np.random.randn(4), dtype=np.int32) % 5
    axis = 2
    self._testRepeatGrad(input, repeats, axis)
    
  def testFloat(self):
    self._testScalar(np.float32)
    self._testVector(np.float32)

  def testDouble(self):
    self._testScalar(np.float64)
    self._testVector(np.float64)
  
if __name__ == "__main__":
  test.main()
