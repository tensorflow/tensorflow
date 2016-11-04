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
"""Functional tests for basic component wise operations using SYCL device."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import math
import numpy as np
from tensorflow.python.ops import gen_math_ops

class SYCLBinaryOpsTest(tf.test.TestCase):
    def _compareSycl(self, x, y, np_func, tf_func):
        np_ans = np_func(x, y)
        with self.test_session(force_gpu=True):
            with tf.device('/job:localhost/replica:0/task:0/device:SYCL:0'):
                inx = tf.convert_to_tensor(x)
                iny = tf.convert_to_tensor(y)
                out = tf_func(inx, iny)
                tf_gpu = out.eval()
                print(tf_gpu)
        self.assertAllClose(np_ans, tf_gpu)
        self.assertShapeEqual(np_ans, out)
    
    def testFloatBasic(self):
      x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float32)
      y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float32)
      self._compareSycl(x, y, np.add, tf.add)
      self._compareSycl(x, y, np.subtract, tf.sub)
      self._compareSycl(x, y, np.multiply, tf.mul)
      self._compareSycl(x, y + 0.1, np.true_divide, tf.truediv)
      
if __name__ == "__main__":
  tf.test.main()
