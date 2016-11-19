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
"""Functional tests for basic component wise operations using a GPU device."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import math
import numpy as np
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.gen_array_ops import _broadcast_gradient_args

class GPUBinaryOpsTest(tf.test.TestCase):
  def _compareGPU(self, x, y, np_func, tf_func):
    with self.test_session(use_gpu=True) as sess:
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = sess.run(out)

    with self.test_session(use_gpu=False) as sess:
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = sess.run(out)

    self.assertAllClose(tf_cpu, tf_gpu)
    
  def testFloatBasic(self):
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float32)
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float32)
    self._compareGPU(x, y, np.add, tf.add)
    self._compareGPU(x, y, np.subtract, tf.sub)
    self._compareGPU(x, y, np.multiply, tf.mul)
    self._compareGPU(x, y + 0.1, np.true_divide, tf.truediv)

  def testFloatWithBCast(self):
    x = np.linspace(-5, 20, 15).reshape(3, 5).astype(np.float32)
    y = np.linspace(20, -5, 30).reshape(2, 3, 5).astype(np.float32)
    self._compareGPU(x, y, np.add, tf.add)
    self._compareGPU(x, y, np.subtract, tf.sub)
    self._compareGPU(x, y, np.multiply, tf.mul)
    self._compareGPU(x, y + 0.1, np.true_divide, tf.truediv)

  def testDoubleBasic(self):
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float64)
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float64)
    self._compareGPU(x, y, np.add, tf.add)
    self._compareGPU(x, y, np.subtract, tf.sub)
    self._compareGPU(x, y, np.multiply, tf.mul)
    self._compareGPU(x, y + 0.1, np.true_divide, tf.truediv)

  def testDoubleWithBCast(self):
    x = np.linspace(-5, 20, 15).reshape(3, 5).astype(np.float64)
    y = np.linspace(20, -5, 30).reshape(2, 3, 5).astype(np.float64)
    self._compareGPU(x, y, np.add, tf.add)
    self._compareGPU(x, y, np.subtract, tf.sub)
    self._compareGPU(x, y, np.multiply, tf.mul)
    self._compareGPU(x, y + 0.1, np.true_divide, tf.truediv)


  #def _GetGradientArgs(self, xs, ys):
    #with self.test_session(use_gpu=True) as sess:
     # return sess.run(_broadcast_gradient_args(xs, ys))

  #def testBroadcast(self):
    #r0, r1 = self._GetGradientArgs([2, 3, 5], [1])
    #self.assertAllEqual(r0, [])
    #self.assertAllEqual(r1, [0, 1, 2])
      
if __name__ == "__main__":
  tf.test.main()
