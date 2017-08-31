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

import itertools
import threading

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.gen_array_ops import _broadcast_gradient_args
from tensorflow.python.platform import test


class GPUBinaryOpsTest(test.TestCase):

  def _compareGPU(self, x, y, np_func, tf_func):
    with self.test_session(use_gpu=True) as sess:
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = sess.run(out)

    with self.test_session(use_gpu=False) as sess:
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = sess.run(out)

    self.assertAllClose(tf_cpu, tf_gpu)

  def testFloatBasic(self):
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float32)
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float32)
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareGPU(x, y + 0.1, np.floor_divide, math_ops.floordiv)
    self._compareGPU(x, y, np.power, math_ops.pow)

  def testFloatWithBCast(self):
    x = np.linspace(-5, 20, 15).reshape(3, 5).astype(np.float32)
    y = np.linspace(20, -5, 30).reshape(2, 3, 5).astype(np.float32)
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)

  def testDoubleBasic(self):
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float64)
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float64)
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)

  def testDoubleWithBCast(self):
    x = np.linspace(-5, 20, 15).reshape(3, 5).astype(np.float64)
    y = np.linspace(20, -5, 30).reshape(2, 3, 5).astype(np.float64)
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)


class MathBuiltinUnaryTest(test.TestCase):

  def _compare(self, x, np_func, tf_func, use_gpu):
    np_out = np_func(x)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = ops.convert_to_tensor(x)
      ofunc = tf_func(inx)
      tf_out = sess.run(ofunc)
    self.assertAllClose(np_out, tf_out)

  def _inv(self, x):
    return 1.0 / x

  def _rsqrt(self, x):
    return self._inv(np.sqrt(x))

  def _testDtype(self, dtype, use_gpu):
    data = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(dtype)
    data_gt_1 = data + 2 # for x > 1
    self._compare(data, np.abs, math_ops.abs, use_gpu)
    self._compare(data, np.arccos, math_ops.acos, use_gpu)
    self._compare(data, np.arcsin, math_ops.asin, use_gpu)
    self._compare(data, np.arcsinh, math_ops.asinh, use_gpu)
    self._compare(data_gt_1, np.arccosh, math_ops.acosh, use_gpu)
    self._compare(data, np.arctan, math_ops.atan, use_gpu)
    self._compare(data, np.ceil, math_ops.ceil, use_gpu)
    self._compare(data, np.cos, math_ops.cos, use_gpu)
    self._compare(data, np.cosh, math_ops.cosh, use_gpu)
    self._compare(data, np.exp, math_ops.exp, use_gpu)
    self._compare(data, np.floor, math_ops.floor, use_gpu)
    self._compare(data, np.log, math_ops.log, use_gpu)
    self._compare(data, np.log1p, math_ops.log1p, use_gpu)
    self._compare(data, np.negative, math_ops.negative, use_gpu)
    self._compare(data, self._rsqrt, math_ops.rsqrt, use_gpu)
    self._compare(data, np.sin, math_ops.sin, use_gpu)
    self._compare(data, np.sinh, math_ops.sinh, use_gpu)
    self._compare(data, np.sqrt, math_ops.sqrt, use_gpu)
    self._compare(data, np.square, math_ops.square, use_gpu)
    self._compare(data, np.tan, math_ops.tan, use_gpu)
    self._compare(data, np.tanh, math_ops.tanh, use_gpu)
    self._compare(data, np.arctanh, math_ops.atanh, use_gpu)

  def testTypes(self):
    for dtype in [np.float32]:
      self._testDtype(dtype, use_gpu=True)

  def testFloorDivide(self):
    x = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape(
        [1, 3, 2])
    y = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape(
        [1, 3, 2])

    np_out = np.floor_divide(x, y + 0.1)

    with self.test_session(use_gpu=True) as sess:
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y + 0.1)
      ofunc = inx / iny
      out_func2 = math_ops.floor(ofunc)
      tf_out = sess.run(out_func2)

    self.assertAllClose(np_out, tf_out)


class BroadcastSimpleTest(test.TestCase):

  def _GetGradientArgs(self, xs, ys):
    with self.test_session(use_gpu=True) as sess:
      return sess.run(_broadcast_gradient_args(xs, ys))

  def testBroadcast(self):
    r0, r1 = self._GetGradientArgs([2, 3, 5], [1])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1, 2])

  _GRAD_TOL = {dtypes.float32: 1e-3}

  def _compareGradientX(self,
                        x,
                        y,
                        np_func,
                        tf_func,
                        numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.test_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      xs = list(x.shape)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, xs, out, zs, x_init_value=x)
      tol = self._GRAD_TOL[dtypes.as_dtype(x.dtype)]
      self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGradientY(self,
                        x,
                        y,
                        np_func,
                        tf_func,
                        numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.test_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      ys = list(np.shape(y))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, ys, out, zs, x_init_value=y)
    tol = self._GRAD_TOL[dtypes.as_dtype(x.dtype)]
    self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=True):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = out.eval()
    self.assertAllClose(np_ans, tf_gpu)
    self.assertShapeEqual(np_ans, out)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def testGradient(self):
    x = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape(
        [1, 3, 2])
    y = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape(
        [1, 3, 2])

    self._compareGradientX(x, y, np.true_divide, math_ops.truediv)
    self._compareGradientY(x, y, np.true_divide, math_ops.truediv)
    self._compareGpu(x, y, np.true_divide, math_ops.truediv)
    self._compareGpu(x, y + 0.1, np.floor_divide, math_ops.floordiv)


class GpuMultiSessionMemoryTest(test_util.TensorFlowTestCase):
  """Tests concurrent sessions executing on the same GPU."""

  def _run_session(self, session, results):
    n_iterations = 500
    with session as s:
      data = variables.Variable(1.0)
      with ops.device('/device:GPU:0'):
        random_seed.set_random_seed(1)
        matrix1 = variables.Variable(
            random_ops.truncated_normal([1024, 1]), name='matrix1')
        matrix2 = variables.Variable(
            random_ops.truncated_normal([1, 1024]), name='matrix2')
        x1 = math_ops.multiply(data, matrix1, name='x1')
        x3 = math_ops.matmul(x1, math_ops.matmul(matrix2, matrix1))
        x4 = math_ops.matmul(array_ops.transpose(x3), x3, name='x4')
        s.run(variables.global_variables_initializer())

        for _ in xrange(n_iterations):
          value = s.run(x4)
          results.add(value.flat[0])
          if len(results) != 1:
            break

  def testConcurrentSessions(self):
    n_threads = 4
    threads = []
    results = []
    for _ in xrange(n_threads):
      session = self.test_session(graph=ops.Graph(), use_gpu=True)
      results.append(set())
      args = (session, results[-1])
      threads.append(threading.Thread(target=self._run_session, args=args))

    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    flat_results = set([x for x in itertools.chain(*results)])
    self.assertEqual(1,
                     len(flat_results),
                     'Expected single value, got %r' % flat_results)


if __name__ == '__main__':
  test.main()
