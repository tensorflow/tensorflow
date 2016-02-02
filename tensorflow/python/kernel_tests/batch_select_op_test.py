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

"""Functional tests for coefficient-wise operations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
# pylint: disable=g-bad-import-order
import tensorflow.python.platform
# pylint: enable=g-bad-import-order
# pylint: enable=unused-import

import numpy as np
import tensorflow as tf


class BatchSelectOpTest(tf.test.TestCase):

  def _compare(self, c, x, y, use_gpu):
    np_ans = np.concatenate(
        [x_i if c_i else y_i for c_i, x_i, y_i in zip(c, x, y)])
    with self.test_session(use_gpu=use_gpu):
      out = tf.batch_select(c, x, y)
      tf_ans = out.eval()
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self, c, x, y):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf.batch_select(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                  s,
                                                  out,
                                                  s,
                                                  x_init_value=x)
    if x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, c, x, y):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf.batch_select(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = tf.test.compute_gradient(iny,
                                                  s,
                                                  out,
                                                  s,
                                                  x_init_value=y)
    if x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def testBasic(self):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [np.float32, np.float64, np.int32, np.int64, np.complex64]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(c, xt, yt, use_gpu=False)
      if t in [np.float32, np.float64]:
        self._compare(c, xt, yt, use_gpu=True)

  def testGradients(self):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [np.float32, np.float64]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compareGradientX(c, xt, yt)
      self._compareGradientY(c, xt, yt)

  def testShapeMismatch(self):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(2, 5, 3) * 100
    for t in [np.float32, np.float64, np.int32, np.int64, np.complex64]:
      xt = x.astype(t)
      yt = y.astype(t)
      with self.assertRaises(ValueError):
        tf.batch_select(c, xt, yt)
