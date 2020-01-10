"""Functional tests for coefficient-wise operations.
"""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker as gc

_ADD = lambda x, y: x + y
_SUB = lambda x, y: x - y
_MUL = lambda x, y: x * y
_DIV = lambda x, y: x / y
_MOD = lambda x, y: x % y
_NEG = lambda x: -x
_ABS = abs

_LT = lambda x, y: x < y
_LE = lambda x, y: x <= y
_GT = lambda x, y: x > y
_GE = lambda x, y: x >= y

_AND = lambda x, y: x & y
_OR = lambda x, y: x | y
_XOR = lambda x, y: x ^ y
_INV = lambda x: ~x


class UnaryOpTest(tf.test.TestCase):

  def _compareCpu(self, x, np_func, tf_func):
    np_ans = np_func(x)
    with self.test_session(use_gpu=False):
      inx = tf.convert_to_tensor(x)
      y = tf_func(inx)
      tf_cpu = y.eval()
      self.assertShapeEqual(np_ans, y)
      self.assertAllClose(np_ans, tf_cpu)
      if x.dtype == np.float32:
        s = list(np.shape(x))
        jacob_t, jacob_n = gc.ComputeGradient(inx, s, y, s, x_init_value=x)
        self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
      elif x.dtype == np.float64:
        s = list(np.shape(x))
        jacob_t, jacob_n = gc.ComputeGradient(inx, s, y, s, x_init_value=x)
        self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGpu(self, x, np_func, tf_func):
    np_ans = np_func(x)
    with self.test_session(use_gpu=True):
      result = tf_func(tf.convert_to_tensor(x))
      tf_gpu = result.eval()
    self.assertShapeEqual(np_ans, result)
    self.assertAllClose(np_ans, tf_gpu)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def _compareBoth(self, x, np_func, tf_func):
    self._compareCpu(x, np_func, tf_func)
    self._compareGpu(x, np_func, tf_func)

  def _inv(self, x):
    return 1.0 / x

  def _rsqrt(self, x):
    return self._inv(np.sqrt(x))

  def _sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  def testFloatBasic(self):
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32)
    y = (x + .5).astype(np.float32)     # no zero
    z = (x + 15.5).astype(np.float32)   # all positive
    self._compareBoth(x, np.abs, tf.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, tf.neg)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, tf.inv)
    self._compareBoth(x, np.square, tf.square)
    self._compareBoth(z, np.sqrt, tf.sqrt)
    self._compareBoth(z, self._rsqrt, tf.rsqrt)
    self._compareBoth(x, np.exp, tf.exp)
    self._compareBoth(z, np.log, tf.log)
    self._compareBoth(x, np.tanh, tf.tanh)
    self._compareBoth(x, self._sigmoid, tf.sigmoid)
    self._compareBoth(y, np.sign, tf.sign)
    self._compareBoth(x, np.sin, tf.sin)
    self._compareBoth(x, np.cos, tf.cos)

  def testFloatTanhEdge(self):
    x = np.arange(40, 40 + 6).reshape(6).astype(np.float32)
    self._compareBoth(x, np.tanh, tf.tanh)
    x = np.arange(-40, -40 + 6).reshape(6).astype(np.float32)
    self._compareBoth(x, np.tanh, tf.tanh)

  def testFloatEmpty(self):
    x = np.empty((2, 0, 5), dtype=np.float32)
    self._compareBoth(x, np.abs, tf.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, tf.neg)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(x, self._inv, tf.inv)
    self._compareBoth(x, np.square, tf.square)
    self._compareBoth(x, np.sqrt, tf.sqrt)
    self._compareBoth(x, self._rsqrt, tf.rsqrt)
    self._compareBoth(x, np.exp, tf.exp)
    self._compareBoth(x, np.log, tf.log)
    self._compareBoth(x, np.tanh, tf.tanh)
    self._compareBoth(x, self._sigmoid, tf.sigmoid)
    self._compareBoth(x, np.sign, tf.sign)
    self._compareBoth(x, np.sin, tf.sin)
    self._compareBoth(x, np.cos, tf.cos)

  def testDoubleBasic(self):
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float64)
    y = (x + .5).astype(np.float64)    # no zero
    z = (x + 15.5).astype(np.float64)  # all positive
    self._compareBoth(x, np.abs, tf.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, tf.neg)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, tf.inv)
    self._compareBoth(x, np.square, tf.square)
    self._compareBoth(z, np.sqrt, tf.sqrt)
    self._compareBoth(z, self._rsqrt, tf.rsqrt)
    self._compareBoth(x, np.exp, tf.exp)
    self._compareBoth(z, np.log, tf.log)
    self._compareBoth(x, np.tanh, tf.tanh)
    self._compareBoth(x, self._sigmoid, tf.sigmoid)
    self._compareBoth(y, np.sign, tf.sign)
    self._compareBoth(x, np.sin, tf.sin)
    self._compareBoth(x, np.cos, tf.cos)

  def testInt32Basic(self):
    x = np.arange(-6, 6, 2).reshape(1, 3, 2).astype(np.int32)
    self._compareCpu(x, np.abs, tf.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareCpu(x, np.negative, tf.neg)
    self._compareCpu(x, np.negative, _NEG)
    self._compareCpu(x, np.square, tf.square)
    self._compareCpu(x, np.sign, tf.sign)

  def testInt64Basic(self):
    x = np.arange(
        -6 << 40, 6 << 40, 2 << 40).reshape(1, 3, 2).astype(np.int64)
    self._compareCpu(x, np.abs, tf.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareCpu(x, np.negative, tf.neg)
    self._compareCpu(x, np.negative, _NEG)
    self._compareCpu(x, np.square, tf.square)
    self._compareCpu(x, np.sign, tf.sign)

  def testComplex64Basic(self):
    x = np.complex(1, 1) * np.arange(-3, 3).reshape(1, 3, 2).astype(
        np.complex64)
    y = x + 0.5  # no zeros
    self._compareCpu(x, np.abs, tf.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareCpu(x, np.negative, tf.neg)
    self._compareCpu(x, np.negative, _NEG)
    self._compareCpu(y, self._inv, tf.inv)
    self._compareCpu(x, np.square, tf.square)
    self._compareCpu(x, np.sqrt, tf.sqrt)
    self._compareCpu(y, self._rsqrt, tf.rsqrt)
    self._compareCpu(x, np.exp, tf.exp)
    self._compareCpu(y, np.log, tf.log)
    self._compareCpu(x, np.tanh, tf.tanh)
    self._compareCpu(x, self._sigmoid, tf.sigmoid)
    self._compareCpu(x, np.sin, tf.sin)
    self._compareCpu(x, np.cos, tf.cos)


class BinaryOpTest(tf.test.TestCase):

  def _compareCpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=False):
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = out.eval()
      # Test that the op takes precedence over numpy operators.
      np_left = tf_func(x, iny).eval()
      np_right = tf_func(inx, y).eval()

    self.assertAllClose(np_ans, tf_cpu)
    self.assertAllClose(np_ans, np_left)
    self.assertAllClose(np_ans, np_right)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self, x, y, np_func, tf_func):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      xs = list(x.shape)
      jacob_t, jacob_n = gc.ComputeGradient(inx, xs, out, zs, x_init_value=x)
      if x.dtype == np.float32:
        self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
      elif x.dtype == np.float64:
        self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, x, y, np_func, tf_func):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      ys = list(np.shape(y))
      jacob_t, jacob_n = gc.ComputeGradient(iny, ys, out, zs, x_init_value=y)
    if x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

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

  def _compareBoth(self, x, y, np_func, tf_func):
    self._compareCpu(x, y, np_func, tf_func)
    if x.dtype == np.float32 or x.dtype == np.float64:
      self._compareGradientX(x, y, np_func, tf_func)
      self._compareGradientY(x, y, np_func, tf_func)
      self._compareGpu(x, y, np_func, tf_func)

  def testFloatBasic(self):
    x = np.linspace(-10, 10, 6).reshape(1, 3, 2).astype(np.float32)
    y = np.linspace(20, -20, 6).reshape(1, 3, 2).astype(np.float32)
    self._compareBoth(x, y, np.add, tf.add)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y + 0.1, np.divide, tf.div)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.divide, _DIV)

  def testFloatDifferentShapes(self):
    x = np.array([1, 2, 3, 4]).reshape(2, 2).astype(np.float32)
    y = np.array([1, 2]).reshape(2, 1).astype(np.float32)
    with self.test_session() as sess:
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      s = tf.reduce_sum(inx * iny)
      gx, gy = sess.run(tf.gradients(s, [inx, iny]))
    # gx is simply the broadcasted y
    self.assertAllEqual(gx, np.array([1, 1, 2, 2])
                        .reshape(2, 2).astype(np.float32))
    # gy is x's column summed up
    self.assertAllEqual(gy, np.array([3, 7]).
                        reshape(2, 1).astype(np.float32))

  def testDoubleBasic(self):
    x = np.linspace(-10, 10, 6).reshape(1, 3, 2).astype(np.float64)
    y = np.linspace(20, -20, 6).reshape(1, 3, 2).astype(np.float64)
    self._compareBoth(x, y, np.add, tf.add)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y + 0.1, np.divide, tf.div)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.divide, _DIV)

  def testInt8Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int8)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int8)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testInt16Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int16)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int16)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testInt32Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int32)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int32)
    self._compareBoth(x, y, np.add, tf.add)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    # NOTE: int32 division is ill-defined.
    self._compareBoth(x, y, np.divide, tf.div)
    self._compareBoth(x, y, np.mod, tf.mod)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    # NOTE: int32 division is ill-defined.
    self._compareBoth(x, y, np.divide, _DIV)
    self._compareBoth(x, y, np.mod, _MOD)

  def testInt64Basic(self):
    x = np.arange(1 << 40, 13 << 40, 2 << 40).reshape(1, 3, 2).astype(np.int64)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int64)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    # NOTE: int64 division is ill-defined.
    self._compareBoth(x, y, np.divide, tf.div)
    self._compareBoth(x, y, np.mod, tf.mod)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    # NOTE: int64 division is ill-defined.
    self._compareBoth(x, y, np.divide, _DIV)
    self._compareBoth(x, y, np.mod, _MOD)

  def testComplex64Basic(self):
    x = np.complex(1, 1) * np.linspace(-10, 10, 6).reshape(1, 3, 2).astype(
        np.complex64)
    y = np.complex(1, 1) * np.linspace(20, -20, 6).reshape(1, 3, 2).astype(
        np.complex64)
    self._compareCpu(x, y, np.add, tf.add)
    self._compareCpu(x, y, np.subtract, tf.sub)
    self._compareCpu(x, y, np.multiply, tf.mul)
    self._compareCpu(x, y + 0.1, np.divide, tf.div)
    self._compareCpu(x, y, np.add, _ADD)
    self._compareCpu(x, y, np.subtract, _SUB)
    self._compareCpu(x, y, np.multiply, _MUL)
    self._compareCpu(x, y + 0.1, np.divide, _DIV)

  def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
    x = (1 + np.linspace(0, 5, np.prod(xs))).astype(dtype).reshape(xs)
    y = (1 + np.linspace(0, 5, np.prod(ys))).astype(dtype).reshape(ys)
    self._compareCpu(x, y, np_func, tf_func)
    if x.dtype == np.float32 or x.dtype == np.float64:
      self._compareGradientX(x, y, np_func, tf_func)
      self._compareGradientY(x, y, np_func, tf_func)
      self._compareGpu(x, y, np_func, tf_func)

  # TODO(josh11b,vrv): Refactor this to use parameterized tests.
  def _testBCastByFunc(self, funcs, xs, ys):
    dtypes = [
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.complex64
    ]
    for dtype in dtypes:
      for (np_func, tf_func) in funcs:
        self._compareBCast(xs, ys, dtype, np_func, tf_func)
        self._compareBCast(ys, xs, dtype, np_func, tf_func)

  def _testBCastA(self, xs, ys):
    funcs = [
        (np.add, tf.add),
        (np.add, _ADD),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastB(self, xs, ys):
    funcs = [
        (np.subtract, tf.sub),
        (np.subtract, _SUB),
        (np.power, tf.pow),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastC(self, xs, ys):
    funcs = [
        (np.multiply, tf.mul),
        (np.multiply, _MUL),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastD(self, xs, ys):
    funcs = [
        (np.divide, tf.div),
        (np.divide, _DIV)
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def testBCast_0A(self):
    self._testBCastA([1, 3, 2], [1])

  def testBCast_0B(self):
    self._testBCastB([1, 3, 2], [1])

  def testBCast_0C(self):
    self._testBCastC([1, 3, 2], [1])

  def testBCast_0D(self):
    self._testBCastD([1, 3, 2], [1])

  def testBCast_1A(self):
    self._testBCastA([1, 3, 2], [2])

  def testBCast_1B(self):
    self._testBCastB([1, 3, 2], [2])

  def testBCast_1C(self):
    self._testBCastC([1, 3, 2], [2])

  def testBCast_1D(self):
    self._testBCastD([1, 3, 2], [2])

  def testBCast_2A(self):
    self._testBCastA([1, 3, 2], [3, 2])

  def testBCast_2B(self):
    self._testBCastB([1, 3, 2], [3, 2])

  def testBCast_2C(self):
    self._testBCastC([1, 3, 2], [3, 2])

  def testBCast_2D(self):
    self._testBCastD([1, 3, 2], [3, 2])

  def testBCast_3A(self):
    self._testBCastA([1, 3, 2], [3, 1])

  def testBCast_3B(self):
    self._testBCastB([1, 3, 2], [3, 1])

  def testBCast_3C(self):
    self._testBCastC([1, 3, 2], [3, 1])

  def testBCast_3D(self):
    self._testBCastD([1, 3, 2], [3, 1])

  def testBCast_4A(self):
    self._testBCastA([1, 3, 2], [1, 3, 2])

  def testBCast_4B(self):
    self._testBCastB([1, 3, 2], [1, 3, 2])

  def testBCast_4C(self):
    self._testBCastC([1, 3, 2], [1, 3, 2])

  def testBCast_4D(self):
    self._testBCastD([1, 3, 2], [1, 3, 2])

  def testBCast_5A(self):
    self._testBCastA([1, 3, 2], [2, 3, 1])

  def testBCast_5B(self):
    self._testBCastB([1, 3, 2], [2, 3, 1])

  def testBCast_5C(self):
    self._testBCastC([1, 3, 2], [2, 3, 1])

  def testBCast_5D(self):
    self._testBCastD([1, 3, 2], [2, 3, 1])

  def testBCast_6A(self):
    self._testBCastA([1, 3, 2], [2, 1, 1])

  def testBCast_6B(self):
    self._testBCastB([1, 3, 2], [2, 1, 1])

  def testBCast_6C(self):
    self._testBCastC([1, 3, 2], [2, 1, 1])

  def testBCast_6D(self):
    self._testBCastD([1, 3, 2], [2, 1, 1])

  def testBCast_7A(self):
    self._testBCastA([1, 3, 2], [1, 3, 1])

  def testBCast_7B(self):
    self._testBCastB([1, 3, 2], [1, 3, 1])

  def testBCast_7C(self):
    self._testBCastC([1, 3, 2], [1, 3, 1])

  def testBCast_7D(self):
    self._testBCastD([1, 3, 2], [1, 3, 1])

  def testBCast_8A(self):
    self._testBCastA([2, 1, 5], [2, 3, 1])

  def testBCast_8B(self):
    self._testBCastB([2, 1, 5], [2, 3, 1])

  def testBCast_8C(self):
    self._testBCastC([2, 1, 5], [2, 3, 1])

  def testBCast_8D(self):
    self._testBCastD([2, 1, 5], [2, 3, 1])

  def testBCast_9A(self):
    self._testBCastA([2, 0, 5], [2, 0, 1])

  def testBCast_9B(self):
    self._testBCastB([2, 0, 5], [2, 0, 1])

  def testBCast_9C(self):
    self._testBCastC([2, 0, 5], [2, 0, 1])

  def testBCast_9D(self):
    self._testBCastD([2, 0, 5], [2, 0, 1])

  def testBCast_10A(self):
    self._testBCastA([2, 3, 0], [2, 3, 1])

  def testBCast_10B(self):
    self._testBCastB([2, 3, 0], [2, 3, 1])

  def testBCast_10C(self):
    self._testBCastC([2, 3, 0], [2, 3, 1])

  def testBCast_10D(self):
    self._testBCastD([2, 3, 0], [2, 3, 1])

  def testBCast_11A(self):
    self._testBCastA([1, 3, 2], [1, 3, 2])

  def testBCast_11B(self):
    self._testBCastB([1, 3, 2], [1, 3, 2])

  def testBCast_11C(self):
    self._testBCastC([1, 3, 2], [1, 3, 2])

  def testBCast_11D(self):
    self._testBCastD([1, 3, 2], [1, 3, 2])

  def testBCast_12A(self):
    self._testBCastA([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_12B(self):
    self._testBCastB([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_12C(self):
    self._testBCastC([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_12D(self):
    self._testBCastD([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_13A(self):
    self._testBCastA([1, 3, 2, 1, 1], [1])

  def testBCast_13B(self):
    self._testBCastB([1, 3, 2, 1, 1], [1])

  def testBCast_13C(self):
    self._testBCastC([1, 3, 2, 1, 1], [1])

  def testBCast_13D(self):
    self._testBCastD([1, 3, 2, 1, 1], [1])

  def testBCast_14A(self):
    self._testBCastA([2, 3, 1, 1, 5], [1])

  def testBCast_14B(self):
    self._testBCastB([2, 3, 1, 1, 5], [1])

  def testBCast_14C(self):
    self._testBCastC([2, 3, 1, 1, 5], [1])

  def testBCast_14D(self):
    self._testBCastD([2, 3, 1, 1, 5], [1])

  def testBCast_15A(self):
    self._testBCastA([10, 3, 1, 2], [3, 1, 2])

  def testBCast_15B(self):
    self._testBCastB([10, 3, 1, 2], [3, 1, 2])

  def testBCast_15C(self):
    self._testBCastC([10, 3, 1, 2], [3, 1, 2])

  def testBCast_15D(self):
    self._testBCastD([10, 3, 1, 2], [3, 1, 2])

  def testMismatchedDimensions(self):
    for func in [tf.add, tf.sub, tf.mul, tf.div,
                 _ADD, _SUB, _MUL, _DIV]:
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Incompatible shapes" in e.message):
        func(tf.convert_to_tensor([10.0, 20.0, 30.0]),
             tf.convert_to_tensor([[40.0, 50.0], [60.0, 70.0]]))


class ComparisonOpTest(tf.test.TestCase):

  def _compare(self, func, x, y, dtype):
    with self.test_session(use_gpu=False):
      out = func(tf.convert_to_tensor(np.array([x]).astype(dtype)),
                 tf.convert_to_tensor(np.array([y]).astype(dtype)))
      ret = out.eval()
    return ret[0]

  def testScalarCompareScalar(self):
    dtypes = [np.float32, np.float64, np.int32, np.int64]
    data = [-1, 0, 1]
    for t in dtypes:
      for x in data:
        for y in data:
          self.assertEqual(self._compare(tf.less, x, y, t),
                           x < y)
          self.assertEqual(self._compare(tf.less_equal, x, y, t),
                           x <= y)
          self.assertEqual(self._compare(tf.greater, x, y, t),
                           x > y)
          self.assertEqual(self._compare(tf.greater_equal, x, y, t),
                           x >= y)
          self.assertEqual(self._compare(tf.equal, x, y, t),
                           x == y)
          self.assertEqual(self._compare(tf.not_equal, x, y, t),
                           x != y)

  def _compareCpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=False):
      out = tf_func(tf.convert_to_tensor(x), tf.convert_to_tensor(y))
      tf_cpu = out.eval()
    self.assertAllEqual(np_ans, tf_cpu)

  def _compareGpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=True):
      out = tf_func(tf.convert_to_tensor(x), tf.convert_to_tensor(y))
      tf_gpu = out.eval()
    self.assertAllEqual(np_ans, tf_gpu)

  def _compareBoth(self, x, y, np_func, tf_func):
    self._compareCpu(x, y, np_func, tf_func)
    if x.dtype == np.float32 or x.dtype == np.float64:
      self._compareGpu(x, y, np_func, tf_func)

  def testTensorCompareTensor(self):
    x = np.linspace(-15, 15, 6).reshape(1, 3, 2)
    y = np.linspace(20, -10, 6).reshape(1, 3, 2)
    for t in [np.float32, np.float64, np.int32, np.int64]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compareBoth(xt, yt, np.less, tf.less)
      self._compareBoth(xt, yt, np.less_equal, tf.less_equal)
      self._compareBoth(xt, yt, np.greater, tf.greater)
      self._compareBoth(xt, yt, np.greater_equal, tf.greater_equal)
      self._compareBoth(xt, yt, np.equal, tf.equal)
      self._compareBoth(xt, yt, np.not_equal, tf.not_equal)
    # TODO(zhifengc): complex64 doesn't work on GPU yet.
    self._compareCpu(x.astype(np.complex64), y.astype(np.complex64),
                     np.equal, tf.equal)
    self._compareCpu(x.astype(np.complex64), y.astype(np.complex64),
                     np.not_equal, tf.not_equal)

  def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
    x = np.linspace(-15, 15, np.prod(xs)).astype(dtype).reshape(xs)
    y = np.linspace(20, -10, np.prod(ys)).astype(dtype).reshape(ys)
    self._compareCpu(x, y, np_func, tf_func)
    self._compareCpu(y, x, np_func, tf_func)
    if x.dtype == np.float32 or x.dtype == np.float64:
      self._compareGpu(x, y, np_func, tf_func)
      self._compareGpu(y, x, np_func, tf_func)

  def _testBCastByFunc(self, np_func, tf_func):
    shapes = [
        ([1, 3, 2], [1]),
        ([1, 3, 2], [2]),
        ([1, 3, 2], [3, 2]),
        ([1, 3, 2], [3, 1]),
        ([1, 3, 2], [1, 3, 2]),
        ([1, 3, 2], [2, 3, 1]),
        ([1, 3, 2], [2, 1, 1]),
        ([1, 3, 2], [1, 3, 1]),
        ([2, 1, 5], [2, 3, 1]),
        ([2, 0, 5], [2, 0, 1]),
        ([2, 3, 0], [2, 3, 1]),
    ]
    dtypes = [
        np.float32,
        np.float64,
        np.int32,
        np.int64,
    ]
    for (xs, ys) in shapes:
      for dtype in dtypes:
        self._compareBCast(xs, ys, dtype, np_func, tf_func)

  def testBCastLess(self):
    self._testBCastByFunc(np.less, tf.less)

  def testBCastLessEqual(self):
    self._testBCastByFunc(np.less_equal, tf.less_equal)

  def testBCastGreater(self):
    self._testBCastByFunc(np.greater, tf.greater)

  def testBCastGreaterEqual(self):
    self._testBCastByFunc(np.greater_equal, tf.greater_equal)

  def testBCastEqual(self):
    self._testBCastByFunc(np.equal, tf.equal)

  def testBCastNotEqual(self):
    self._testBCastByFunc(np.not_equal, tf.not_equal)

  def testShapeMismatch(self):
    dtypes = [np.float32, np.float64, np.int32, np.int64]
    funcs = [tf.less, tf.less_equal, tf.greater,
             tf.greater_equal, tf.equal, tf.not_equal]
    x = np.arange(0, 10).reshape([2, 5])
    y = np.arange(0, 10).reshape([5, 2])
    for t in dtypes:
      for f in funcs:
        with self.assertRaisesWithPredicateMatch(
            ValueError, lambda e: "Incompatible shapes" in e.message):
          f(x.astype(t), y.astype(t))


class LogicalOpTest(tf.test.TestCase):

  def _compareBinary(self, x, y, np_func, tf_func, use_gpu=False):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=use_gpu):
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_val = out.eval()
    self.assertEqual(out.dtype, tf.bool)
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def _not(self, x, use_gpu=False):
    np_ans = np.logical_not(x)
    with self.test_session(use_gpu=use_gpu):
      out = tf.logical_not(tf.convert_to_tensor(x))
      tf_val = out.eval()
    self.assertEqual(out.dtype, tf.bool)
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def testScalar(self):
    data = [np.array([True]), np.array([False])]
    for use_gpu in [True, False]:
      for x in data:
        self._not(x, use_gpu)
      for x in data:
        for y in data:
          self._compareBinary(
              x, y, np.logical_and, tf.logical_and, use_gpu)
          self._compareBinary(
              x, y, np.logical_or, tf.logical_or, use_gpu)
          self._compareBinary(
              x, y, np.logical_xor, tf.logical_xor, use_gpu)

  def testTensor(self):
    x = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    y = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    for use_gpu in [True, False]:
      self._not(x, use_gpu)
      self._compareBinary(x, y, np.logical_and, tf.logical_and, use_gpu)
      self._compareBinary(x, y, np.logical_or, tf.logical_or, use_gpu)
      self._compareBinary(x, y, np.logical_xor, tf.logical_xor, use_gpu)

  def testBCast(self):
    shapes = [
        ([1, 3, 2], [1]),
        ([1, 3, 2], [2]),
        ([1, 3, 2], [3, 2]),
        ([1, 3, 2], [3, 1]),
        ([1, 3, 2], [1, 3, 2]),
        ([1, 3, 2], [2, 3, 1]),
        ([1, 3, 2], [2, 1, 1]),
        ([1, 3, 2], [1, 3, 1]),
        ([2, 1, 5], [2, 3, 1]),
        ([2, 0, 5], [2, 0, 1]),
        ([2, 3, 0], [2, 3, 1]),
    ]
    for (xs, ys) in shapes:
      x = np.random.randint(0, 2, np.prod(xs)).astype(np.bool).reshape(xs)
      y = np.random.randint(0, 2, np.prod(ys)).astype(np.bool).reshape(ys)
      for use_gpu in [True, False]:
        self._compareBinary(x, y, np.logical_and, tf.logical_and, use_gpu)
        self._compareBinary(x, y, np.logical_or, tf.logical_or, use_gpu)
        self._compareBinary(x, y, np.logical_xor, tf.logical_xor, use_gpu)

  def testShapeMismatch(self):
    x = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    y = np.random.randint(0, 2, 6).astype(np.bool).reshape(3, 2, 1)
    for f in [tf.logical_and, tf.logical_or, tf.logical_xor]:
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Incompatible shapes" in e.message):
        f(x, y)


class SelectOpTest(tf.test.TestCase):

  def _compare(self, c, x, y, use_gpu):
    np_ans = np.where(c, x, y)
    with self.test_session(use_gpu=use_gpu):
      out = tf.select(c, x, y)
      tf_ans = out.eval()
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self, c, x, y):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf.select(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = gc.ComputeGradient(inx, s, out, s, x_init_value=x)
    if x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, c, x, y):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf.select(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = gc.ComputeGradient(iny, s, out, s, x_init_value=y)
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
        tf.select(c, xt, yt)


class MinMaxOpTest(tf.test.TestCase):

  def _compare(self, x, y, use_gpu):
    np_min, np_max = np.minimum(x, y), np.maximum(x, y)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      omin, omax = tf.minimum(inx, iny), tf.maximum(inx, iny)
      tf_min, tf_max = sess.run([omin, omax])
    self.assertAllEqual(np_min, tf_min)
    self.assertAllEqual(np_max, tf_max)

  def testBasic(self):
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(1, 3, 2) * 100.
    for t in [np.float32, np.float64, np.int32, np.int64]:
      self._compare(x.astype(t), y.astype(t), use_gpu=False)
      self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testDifferentShapes(self):
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(2) * 100.  # should broadcast
    for t in [np.float32, np.float64, np.int32, np.int64]:
      self._compare(x.astype(t), y.astype(t), use_gpu=False)
      self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testScalar(self):
    x = np.random.rand(1, 3, 2) * 100.
    y = np.asscalar(np.random.rand(1) * 100.)  # should broadcast
    # dropped np.float64, int64 because TF automatically converts to 32 bit
    for t in [np.float32, np.int32]:
      self._compare(x.astype(t), t(y), use_gpu=False)
      self._compare(x.astype(t), t(y), use_gpu=True)

  def _compareGradientX(self, func, x, y):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gc.ComputeGradient(inx, s, out, s, x_init_value=x)
    if x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, func, x, y):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gc.ComputeGradient(iny, s, out, s, x_init_value=y)
    if x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def testGradients(self):
    x = np.random.rand(1, 3, 2) * 100.
    # ensure x != y
    y = x + (np.random.randint(2, size=x.shape) - .5) * 2  # -1 or +1
    self._compareGradientX(tf.maximum, x, y)
    self._compareGradientY(tf.maximum, x, y)
    self._compareGradientX(tf.minimum, x, y)
    self._compareGradientY(tf.minimum, x, y)


class MathOpsOverloadTest(tf.test.TestCase):

  def _computeTensorAndLiteral(self, x, y, dtype, func):
    with self.test_session(use_gpu=False):
      inx = tf.convert_to_tensor(x, dtype=dtype)
      z = func(inx, y)  # Should use __add__, __sub__, etc.
      return z.eval()

  def _computeLiteralAndTensor(self, x, y, dtype, func):
    with self.test_session(use_gpu=False):
      iny = tf.convert_to_tensor(y, dtype=dtype)
      z = func(x, iny)  # Should use __radd__, __rsub__, etc.
      return z.eval()

  def _compareBinary(self, x, y, dtype, np_func, tf_func):
    np_ans = np_func(x, y)
    self.assertAllClose(np_ans, self._computeTensorAndLiteral(
        x, y, dtype, tf_func))
    self.assertAllClose(np_ans, self._computeLiteralAndTensor(
        x, y, dtype, tf_func))

  def _compareUnary(self, x, dtype, np_func, tf_func):
    np_ans = np_func(x)
    with self.test_session(use_gpu=False):
      self.assertAllClose(np_ans, tf_func(tf.convert_to_tensor(x, dtype=dtype)).eval())

  def testOverload(self):
    dtypes = [
        tf.float32,
        tf.float64,
        tf.int32,
        tf.int64,
        tf.complex64,
    ]
    funcs = [
        (np.add, _ADD),
        (np.subtract, _SUB),
        (np.multiply, _MUL),
        (np.divide, _DIV)
    ]
    for dtype in dtypes:
      for np_func, tf_func in funcs:
        self._compareBinary(10, 5, dtype, np_func, tf_func)
    # Mod only works for int32 and int64.
    for dtype in [tf.int32, tf.int64]:
      self._compareBinary(10, 3, dtype, np.mod, _MOD)

  def testOverloadComparisons(self):
    dtypes = [
        tf.float32,
        tf.float64,
        tf.int32,
        tf.int64,
    ]
    funcs = [
        (np.less, _LT),
        (np.less_equal, _LE),
        (np.greater, _GT),
        (np.greater_equal, _GE),
    ]
    for dtype in dtypes:
      for np_func, tf_func in funcs:
        self._compareBinary(10, 5, dtype, np_func, tf_func)
    logical_funcs = [
        (np.logical_and, _AND),
        (np.logical_or, _OR),
        (np.logical_xor, _XOR),
    ]
    for np_func, tf_func in logical_funcs:
      self._compareBinary(True, False, tf.bool, np_func, tf_func)
      self._compareBinary(True, True, tf.bool, np_func, tf_func)
      self._compareBinary(False, False, tf.bool, np_func, tf_func)
      self._compareBinary(False, True, tf.bool, np_func, tf_func)
      self._compareBinary([True, True, False, False],
                          [True, False, True, False],
                          tf.bool, np_func, tf_func)
    self._compareUnary(True, tf.bool, np.logical_not, _INV)
    self._compareUnary(False, tf.bool, np.logical_not, _INV)
    self._compareUnary([True, False], tf.bool, np.logical_not, _INV)


class IsFiniteInfNanTest(tf.test.TestCase):

  def _compare(self, x, use_gpu):
    np_finite, np_inf, np_nan = np.isfinite(x), np.isinf(x), np.isnan(x)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(x)
      ofinite, oinf, onan = tf.is_finite(inx), tf.is_inf(
          inx), tf.is_nan(inx)
      tf_finite, tf_inf, tf_nan = sess.run([ofinite, oinf, onan])
    self.assertAllEqual(np_inf, tf_inf)
    self.assertAllEqual(np_nan, tf_nan)
    self.assertAllEqual(np_finite, tf_finite)
    self.assertShapeEqual(np_inf, oinf)
    self.assertShapeEqual(np_nan, onan)
    self.assertShapeEqual(np_finite, ofinite)

  def _testDtype(self, dtype):
    fi = np.finfo(dtype)
    data = np.array([0, -1, 1, fi.resolution, -fi.resolution, fi.min, fi.max,
                     -np.inf, np.inf, np.nan]).astype(dtype)
    self._compare(data, use_gpu=False)
    self._compare(data, use_gpu=True)

  def testFloat(self):
    self._testDtype(np.float32)

  def testDouble(self):
    self._testDtype(np.float64)


class RoundingTest(tf.test.TestCase):

  def _compare(self, x, use_gpu):
    np_floor, np_ceil = np.floor(x), np.ceil(x)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(x)
      ofloor, oceil = tf.floor(inx), tf.ceil(inx)
      tf_floor, tf_ceil = sess.run([ofloor, oceil])
    self.assertAllEqual(np_floor, tf_floor)
    self.assertAllEqual(np_ceil, tf_ceil)
    self.assertShapeEqual(np_floor, ofloor)
    self.assertShapeEqual(np_ceil, oceil)

  def _testDtype(self, dtype):
    data = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(dtype)
    self._compare(data, use_gpu=True)
    self._compare(data, use_gpu=True)

  def testTypes(self):
    for dtype in [np.float32, np.float64]:
      self._testDtype(dtype)


class ComplexMakeRealImagTest(tf.test.TestCase):

  def _compareMake(self, real, imag, use_gpu):
    np_ans = real + (1j) * imag
    with self.test_session(use_gpu=use_gpu):
      real = tf.convert_to_tensor(real)
      imag = tf.convert_to_tensor(imag)
      tf_ans = tf.complex(real, imag)
      out = tf_ans.eval()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def testMake(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    for use_gpu in [False, True]:
      self._compareMake(real, imag, use_gpu)
      self._compareMake(real, 12.0, use_gpu)
      self._compareMake(23.0, imag, use_gpu)

  def _compareRealImag(self, cplx, use_gpu):
    np_real, np_imag = np.real(cplx), np.imag(cplx)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(cplx)
      tf_real = tf.real(inx)
      tf_imag = tf.imag(inx)
      tf_real_val, tf_imag_val = sess.run([tf_real, tf_imag])
    self.assertAllEqual(np_real, tf_real_val)
    self.assertAllEqual(np_imag, tf_imag_val)
    self.assertShapeEqual(np_real, tf_real)
    self.assertShapeEqual(np_imag, tf_imag)

  def testRealImag(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    cplx = real + (1j) * imag
    self._compareRealImag(cplx, use_gpu=False)
    self._compareRealImag(cplx, use_gpu=True)

  def _compareConj(self, cplx, use_gpu):
    np_ans = np.conj(cplx)
    with self.test_session(use_gpu=use_gpu):
      inx = tf.convert_to_tensor(cplx)
      tf_conj = tf.conj(inx)
      tf_ans = tf_conj.eval()
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, tf_conj)

  def testConj(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    cplx = real + (1j) * imag
    self._compareConj(cplx, use_gpu=False)
    self._compareConj(cplx, use_gpu=True)

  def _compareGradient(self, x):
    # x[:, 0] is real, x[:, 1] is imag.  We combine real and imag into
    # complex numbers. Then, we extract real and imag parts and
    # computes the squared sum. This is obviously the same as sum(real
    # * real) + sum(imag * imag). We just want to make sure the
    # gradient function is checked.
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      real, imag = tf.split(1, 2, inx)
      real, imag = tf.reshape(real, [-1]), tf.reshape(imag, [-1])
      cplx = tf.complex(real, imag)
      cplx = tf.conj(cplx)
      loss = tf.reduce_sum(
          tf.square(tf.real(cplx))) + tf.reduce_sum(
              tf.square(tf.imag(cplx)))
      epsilon = 1e-3
      jacob_t, jacob_n = gc.ComputeGradient(inx, list(x.shape), loss, [1],
                                            x_init_value=x, delta=epsilon)
    self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  def testGradient(self):
    data = np.arange(1, 2, 0.10).reshape([5, 2]).astype(np.float32)
    self._compareGradient(data)

  def _compareMulGradient(self, data):
    # data is a float matrix of shape [n, 4].  data[:, 0], data[:, 1],
    # data[:, 2], data[:, 3] are real parts of x, imaginary parts of
    # x, real parts of y and imaginary parts of y.
    with self.test_session():
      inp = tf.convert_to_tensor(data)
      xr, xi, yr, yi = tf.split(1, 4, inp)

      def vec(x):  # Reshape to a vector
        return tf.reshape(x, [-1])
      xr, xi, yr, yi = vec(xr), vec(xi), vec(yr), vec(yi)

      def cplx(r, i):  # Combine to a complex vector
        return tf.complex(r, i)
      x, y = cplx(xr, xi), cplx(yr, yi)
      # z is x times y in complex plane.
      z = x * y
      # Defines the loss function as the sum of all coefficients of z.
      loss = tf.reduce_sum(tf.real(z) + tf.imag(z))
      epsilon = 0.005
      jacob_t, jacob_n = gc.ComputeGradient(inp, list(data.shape), loss, [1],
                                            x_init_value=data, delta=epsilon)
    self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  def testMulGradient(self):
    data = np.arange(1, 2, 0.125).reshape([2, 4]).astype(np.float32)
    self._compareMulGradient(data)


class AccumulateTest(tf.test.TestCase):

  def testSimple(self):
    with self.test_session():
      random_arrays = [np.random.rand(16, 16, 16, 16).astype(np.float32)
                       for _ in range(20)]
      random_tensors = [tf.convert_to_tensor(x, dtype=tf.float32)
                        for x in random_arrays]
      tf_val = tf.accumulate_n(random_tensors)
      np_val = random_arrays[0]
      for random_array in random_arrays[1:]:
        np_val += random_array
      self.assertAllClose(np_val, tf_val.eval())

  def testZeroArgs(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf_val = tf.accumulate_n([])
        tf_val.eval()

if __name__ == "__main__":
  tf.test.main()
