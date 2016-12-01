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
    self._compareGPU(x, y + 0.1, np.floor_divide, tf.floordiv)
    self._compareGPU(x, y, np.power, tf.pow)

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


class MathBuiltinUnaryTest(tf.test.TestCase):
  def _compare(self, x, np_func, tf_func, use_gpu):
    np_out = np_func(x)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(x)
      ofunc = tf_func(inx)
      tf_out = sess.run(ofunc)
    self.assertAllClose(np_out, tf_out)

  def _inv(self, x):
    return 1.0 / x

  def _rsqrt(self, x):
    return self._inv(np.sqrt(x))

  def _testDtype(self, dtype, use_gpu):
    data = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(dtype)
    self._compare(data, np.abs, tf.abs, use_gpu)
    self._compare(data, np.arccos, tf.acos, use_gpu)
    self._compare(data, np.arcsin, tf.asin, use_gpu)
    self._compare(data, np.arctan, tf.atan, use_gpu)
    self._compare(data, np.ceil, tf.ceil, use_gpu)
    self._compare(data, np.cos, tf.cos, use_gpu)
    self._compare(data, np.exp, tf.exp, use_gpu)
    self._compare(data, np.floor, tf.floor, use_gpu)
    self._compare(data, np.log, tf.log, use_gpu)
    self._compare(data, np.log1p, tf.log1p, use_gpu)
    self._compare(data, np.negative, tf.neg, use_gpu)
    self._compare(data, self._rsqrt, tf.rsqrt, use_gpu)
    self._compare(data, np.sin, tf.sin, use_gpu)
    self._compare(data, np.sqrt, tf.sqrt, use_gpu)
    self._compare(data, np.square, tf.square, use_gpu)
    self._compare(data, np.tan, tf.tan, use_gpu)
    self._compare(data, np.tanh, tf.tanh, use_gpu)

  def testTypes(self):
    for dtype in [np.float32]:
      self._testDtype(dtype, use_gpu=True)

  def testFloorDevide(self):
    x = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape([1, 3, 2])
    y = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape([1, 3, 2])

    np_out = np.floor_divide(x, y + 0.1)

    with self.test_session(use_gpu=True) as sess:
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y + 0.1)
      ofunc = inx / iny
      out_func2 = tf.floor(ofunc)
      tf_out = sess.run(out_func2)

    self.assertAllClose(np_out, tf_out)

class BroadcastSimpleTest(tf.test.TestCase):
  def _GetGradientArgs(self, xs, ys):
    with self.test_session(use_gpu=True) as sess:
      return sess.run(_broadcast_gradient_args(xs, ys))

  def testBroadcast(self):
    r0, r1 = self._GetGradientArgs([2, 3, 5], [1])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1, 2])

  _GRAD_TOL = {tf.float32: 1e-3}

  def _compareGradientX(self, x, y, np_func, tf_func,
                      numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      xs = list(x.shape)
      jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                  xs,
                                                  out,
                                                  zs,
                                                  x_init_value=x)
      tol = self._GRAD_TOL[tf.as_dtype(x.dtype)]
      self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGradientY(self, x, y, np_func, tf_func,
                        numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      ys = list(np.shape(y))
      jacob_t, jacob_n = tf.test.compute_gradient(iny,
                                                  ys,
                                                  out,
                                                  zs,
                                                  x_init_value=y)
    tol = self._GRAD_TOL[tf.as_dtype(x.dtype)]
    self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=True):
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = out.eval()
    self.assertAllClose(np_ans, tf_gpu)
    self.assertShapeEqual(np_ans, out)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def testGradient(self):
    x = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape([1, 3, 2])
    y = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape([1, 3, 2])

    self._compareGradientX(x , y, np.true_divide, tf.truediv)
    self._compareGradientY(x, y, np.true_divide, tf.truediv)
    self._compareGpu(x, y, np.true_divide, tf.truediv)
    self._compareGpu(x, y +0.1  , np.floor_divide, tf.floordiv)

if __name__ == "__main__":
  tf.test.main()
