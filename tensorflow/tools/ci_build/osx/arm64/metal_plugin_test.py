#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import itertools
import math
import gc
import copy
import os
import re

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.eager import backprop
# from tensorflow.python.eager import test as test_eager
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import tensor_util
from six.moves import xrange  # pylint: disable=redefined-builtin
##from tensorflow.python.ops import math_grad
from tensorflow.python.training import gradient_descent
from tensorflow.python.ops import variables
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.eager import function
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_impl
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from absl.testing import parameterized
from tensorflow.python.training import adam
from tensorflow.python.ops import custom_gradient
from tensorflow.python.compat import compat
from tensorflow.python.client import session
from tensorflow.python.platform import test as test_lib
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import image_ops
from tensorflow import raw_ops
from tensorflow import dtypes
from tensorflow.python.ops import resources
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import indexed_slices


from six.moves import xrange  # pylint: disable=redefined-builtin
from PIL import Image

_ADD = lambda x, y: x + y
_SUB = lambda x, y: x - y
_MUL = lambda x, y: x * y
_POW = lambda x, y: x**y
_TRUEDIV = lambda x, y: x / y
_FLOORDIV = lambda x, y: x // y
_MOD = lambda x, y: x % y
_NEG = lambda x: -x
_ABS = abs
_MAX_RANK = 5

def _default_tolerance(dtype):
  """Returns a sensible default tolerance for comparing results of a given type.

  Args:
    dtype: A datatype.
  """
  if dtype == np.float16:
    return 5e-3
  elif dtype in (np.float32, np.complex64):
    return 1e-3
  elif dtype in (np.float64, np.complex128):
    return 1e-5
  else:
    return None  # Fail fast for unexpected types

def _powerset(iterable):
  """Helper for generating all possible reduction_axes arguments.
  Example:
  powerset([0,1,2]): () (0,) (1,) (2,) (0,1) (0,2) (1,2) (0,1,2)
  Args:
    iterable: An iterable of items to generate the powerset of.
  Returns:
    The powerset of all items in iterable.
  """
  s = list(iterable)
  return itertools.chain.from_iterable(
      itertools.combinations(s, r) for r in range(len(s) + 1))

def adam_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      alpha=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-8):
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


def pool_direct_single_axis(
    input,  # pylint: disable=redefined-builtin
    axis,
    window_size,
    pooling_type,
    padding,
    dilation_rate,
    stride):
  effective_window_size = (window_size - 1) * dilation_rate + 1
  input_size = input.shape[axis]
  if padding == "SAME":
    output_size = int(math.ceil(input_size / stride))
    total_padding_amount = max(
        0, (output_size - 1) * stride + effective_window_size - input_size)
    before_padding = total_padding_amount // 2
  elif padding == "VALID":
    output_size = int(
        math.ceil((input_size - effective_window_size + 1) / stride))
    before_padding = 0
  else:
    raise ValueError("Unsupported padding type: %r" % (padding,))

  output_shape = input.shape[:axis] + (output_size,) + input.shape[axis + 1:]
  output = np.zeros(output_shape, input.dtype)
  initial_dim_selector = tuple(np.s_[:] for _ in range(axis))
  if pooling_type == "MAX":
    pooling_func = np.max
  elif pooling_type == "AVG":
    pooling_func = np.mean
  else:
    raise ValueError("Unsupported pooling type: %r" % (pooling_type,))
  for output_pos in range(output_size):
    input_start_pos = output_pos * stride - before_padding
    input_end_pos = min(input_start_pos + effective_window_size, input_size)
    if input_start_pos < 0:
      input_start_pos += dilation_rate
    input_slice = np.s_[input_start_pos:input_end_pos:dilation_rate]

    output[initial_dim_selector + (output_pos,)] = pooling_func(
        input[initial_dim_selector + (input_slice,)], axis=axis)
  return output


def pool_direct(
    input,  # pylint: disable=redefined-builtin
    window_shape,
    pooling_type,
    padding,  # pylint: disable=redefined-builtin
    dilation_rate,
    strides,
    data_format=None):
  if data_format is None or not data_format.startswith("NC"):
    spatial_start_dim = 1
  else:
    spatial_start_dim = 2
  output = input
  for i in range(len(window_shape)):
    output = pool_direct_single_axis(
        input=output,
        axis=i + spatial_start_dim,
        window_size=window_shape[i],
        pooling_type=pooling_type,
        padding=padding,
        dilation_rate=dilation_rate[i],
        stride=strides[i])
  return output

_TEST_TYPES = [dtypes.float32]

class MomentumOptimizerTest(test.TestCase, parameterized.TestCase):

  def _update_nesterov_momentum_numpy(self, var, accum, g, lr, momentum):
    accum = accum * momentum - g * lr
    var += (accum * momentum - g * lr)
    return var, accum

  def testBasic(self):
    for _, dtype in enumerate([dtypes.float32]):
      var0 = variables.Variable([1.0, 2.0], dtype=dtype, name="var0")
      var1 = variables.Variable([3.0, 4.0], dtype=dtype, name="var1")
      grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
      learning_rate = 2.0
      momentum = 0.9
      mom_opt = gradient_descent_v2.SGD(
          learning_rate=learning_rate, momentum=momentum)
      # self.assertFalse(mom_opt._initial_decay)
      mom_update = mom_opt.apply_gradients(
          zip([grads0, grads1], [var0, var1]))

      # Check we have slots
      slot0 = mom_opt.get_slot(var0, "momentum")
      self.assertEqual(slot0.shape, var0.shape)
      slot1 = mom_opt.get_slot(var1, "momentum")
      self.assertEqual(slot1.shape, var1.shape)

      # Step 1: the momentum accumulators where 0. So we should see a normal
      # update: v -= grad * learning_rate
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(mom_update)
      # Check that the momentum accumulators have been updated.
      self.assertAllCloseAccordingToType(
          np.array([-0.2, -0.2]), self.evaluate(slot0))
      self.assertAllCloseAccordingToType(
          np.array([-0.02, -0.02]), self.evaluate(slot1))
      # Check that the parameters have been updated.
      self.assertAllCloseAccordingToType(
          np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]),
          self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]),
          self.evaluate(var1))
      # Step 2: the momentum accumulators contain the previous update.
      self.evaluate(mom_update)
      if context.executing_eagerly():
        mom_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      # Check that the momentum accumulators have been updated.
      self.assertAllCloseAccordingToType(
          np.array([(0.9 * (-0.2) - 2.0 * 0.1), (0.9 * (-0.2) - 2.0 * 0.1)]),
          self.evaluate(slot0))
      self.assertAllCloseAccordingToType(
          np.array([(0.9 * (-0.02) - 2.0 * 0.01),
                    (0.9 * (-0.02) - 2.0 * 0.01)]), self.evaluate(slot1))
      # Check that the parameters have been updated.
      self.assertAllCloseAccordingToType(
          np.array([
              1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
              2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
          ]), self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          np.array([
              2.98 - ((0.9 * 0.01 + 0.01) * 2.0),
              3.98 - ((0.9 * 0.01 + 0.01) * 2.0)
          ]), self.evaluate(var1))

  def testNesterovMomentum(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.float32]:
        var0 = variables.Variable([1.0, 2.0], dtype=dtype, name="var0")
        var1 = variables.Variable([3.0, 4.0], dtype=dtype, name="var1")
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        accum0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        loss = lambda: 5 * var0 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
        mom_op = gradient_descent_v2.SGD(
            learning_rate=2.0, momentum=0.9, nesterov=True)
        opt_op = mom_op.minimize(loss, [var0, var1])
        self.evaluate(variables.global_variables_initializer())
        for _ in range(1, 5):
          self.evaluate(opt_op)
          var0_np, accum0_np = self._update_nesterov_momentum_numpy(
              var0_np, accum0_np, var0_np * 10, 2.0, 0.9)
          var1_np, accum1_np = self._update_nesterov_momentum_numpy(
              var1_np, accum1_np, 3, 2.0, 0.9)
          self.assertAllClose(var0_np, self.evaluate(var0))
          self.assertAllClose(var1_np, self.evaluate(var1))

class ArgMaxTest(test.TestCase):

  def _testArg(self,
               method,
               x,
               axis,
               expected_values,
               use_gpu=False,
               expected_err_re=None):
    with self.session(use_gpu=use_gpu):
      ans = method(x, axis=axis)
      if expected_err_re is None:
        tf_ans = self.evaluate(ans)
        # Defaults to int64 output.
        self.assertEqual(np.int64, tf_ans.dtype)
        self.assertAllEqual(tf_ans, expected_values)
        self.assertShapeEqual(expected_values, ans)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate(ans)

  def _testBothArg(self,
                   method,
                   x,
                   axis,
                   expected_values,
                   expected_err_re=None):
    self._testArg(method, x, axis, expected_values, True, expected_err_re)
    # Compilation time is too large with XLA/CPU autojit.
    if not test_util.is_xla_enabled():
      self._testArg(method, x, axis, expected_values, False, expected_err_re)

  def _testBasic(self, dtype):
    x = np.arange(200, dtype=np.float32).astype(np.bool_).astype(dtype)
    np.random.shuffle(x)

    # Check that argmin and argmax match numpy along the primary axis
    self._testBothArg(math_ops.argmax, x, 0, x.argmax())
    # self._testBothArg(math_ops.argmin, x, 0, x.argmin())

  def _testTieBreaking(self, dtype):
    x = np.zeros(200, dtype=dtype)

    # Check that argmin and argmax match numpy along the primary axis for
    # breaking ties.
    self._testBothArg(math_ops.argmax, x, 0, x.argmax())
    self._testBothArg(math_ops.argmin, x, 0, x.argmin())

  def _testDim(self, dtype):
    shape = (3, 2, 4, 5, 6, 3, 7)
    x = np.arange(
        functools.reduce(lambda x, y: x * y, shape),
        dtype=np.float32).astype(dtype)
    np.random.shuffle(x)
    x = x.reshape(shape)

    # Check that argmin and argmax match numpy along all axes
    for axis in range(-7, 7):
      self._testBothArg(math_ops.argmax, x, axis, x.argmax(axis))
      self._testBothArg(math_ops.argmin, x, axis, x.argmin(axis))

  def testFloat(self):
    self._testBasic(np.float32)
    # self._testTieBreaking(np.float32)
    # self._testDim(np.float32)

  def testFloatInt32Output(self):
    x = np.asarray(100 * np.random.randn(200), dtype=np.float32)
    expected_values = x.argmax()
    with self.session(use_gpu=True):
      ans = math_ops.argmax(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      # The values are equal when comparing int32 to int64 because
      # the values don't have a range that exceeds 32-bit integers.
      self.assertAllEqual(tf_ans, expected_values)
    expected_values = x.argmin()
    with self.session(use_gpu=True):
      ans = math_ops.argmin(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      self.assertAllEqual(tf_ans, expected_values)

class GatherTest(test.TestCase, parameterized.TestCase):

  def _buildParams(self, data, dtype):
    data = data.astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testScalar1D(self):
    with self.cached_session(use_gpu=True):
      data = np.array([0, 1, 2, 3, 7, 5])
      for dtype in _TEST_TYPES:
        for indices in 4, [1, 2, 2, 4, 5]:
          params_np = self._buildParams(data, dtype)
          params = constant_op.constant(params_np)
          indices_tf = constant_op.constant(indices)
          gather_t = array_ops.gather(params, indices_tf)
          gather_val = self.evaluate(gather_t)
          np_val = params_np[indices]
          self.assertAllEqual(np_val, gather_val)
          self.assertEqual(np_val.shape, gather_t.get_shape())

  def testScalar2D(self):
    with self.session(use_gpu=True):
      data = np.array([[0, 1, 2],
                       [3, 4, 5],
                       [6, 7, 8],
                       [9, 10, 11],
                       [12, 13, 14]])
      for dtype in _TEST_TYPES:
        for axis in range(data.ndim):
          params_np = self._buildParams(data, dtype)
          params = constant_op.constant(params_np)
          indices = constant_op.constant(2)
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = self.evaluate(gather_t)
          print ("TF {}".format(gather_val))
          print ("CPU {}".format(np.take(params_np, 2, axis=axis)))
          self.assertAllEqual(np.take(params_np, 2, axis=axis), gather_val)
          expected_shape = data.shape[:axis] + data.shape[axis + 1:]
          self.assertEqual(expected_shape, gather_t.get_shape())

  def testSimpleTwoD32(self):
    with self.session(use_gpu=True):
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [9, 10, 11], [12, 13, 14]])
      for dtype in _TEST_TYPES:
        for axis in range(data.ndim):
          params_np = self._buildParams(data, dtype)
          params = constant_op.constant(params_np)
          # The indices must be in bounds for any axis.
          indices = constant_op.constant([0, 1, 0, 2])
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = self.evaluate(gather_t)
          self.assertAllEqual(np.take(params_np, [0, 1, 0, 2], axis=axis),
                              gather_val)
          expected_shape = data.shape[:axis] + (4,) + data.shape[axis + 1:]
          self.assertEqual(expected_shape, gather_t.get_shape())

class SliceTest(test.TestCase):

  def testEmpty(self):
    inp = np.random.rand(4, 4).astype("f")
    for k in xrange(4):
      with self.cached_session(use_gpu=True):
        a = constant_op.constant(inp, shape=[4, 4], dtype=dtypes.float32)
        slice_t = a[2, k:k]
        slice_val = self.evaluate(slice_t)
      self.assertAllEqual(slice_val, inp[2, k:k])

  def testSimple(self):
    with self.session(use_gpu=True) as sess:
      inp = np.random.rand(4, 4).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.ravel(order="C")],
          shape=[4, 4],
          dtype=dtypes.float32)
      slice_t = array_ops.slice(a, [0, 0], [2, 2])
      slice2_t = a[:2, :2]
      slice_val, slice2_val = self.evaluate([slice_t, slice2_t])
    self.assertAllEqual(slice_val, inp[:2, :2])
    self.assertAllEqual(slice2_val, inp[:2, :2])
    self.assertEqual(slice_val.shape, slice_t.get_shape())
    self.assertEqual(slice2_val.shape, slice2_t.get_shape())

  def testSingleDimension(self):
    for _ in range(10):
      with self.cached_session(use_gpu=True):
        inp = np.random.rand(10).astype("f")
        a = constant_op.constant(inp, shape=[10], dtype=dtypes.float32)

        hi = np.random.randint(0, 9)
        scalar_t = a[hi]
        scalar_val = self.evaluate(scalar_t)
        self.assertAllEqual(scalar_val, inp[hi])

        if hi > 0:
          lo = np.random.randint(0, hi)
        else:
          lo = 0
        slice_t = a[lo:hi]
        slice_val = self.evaluate(slice_t)
        self.assertAllEqual(slice_val, inp[lo:hi])

  def test3Dimension(self):
    with self.cached_session():
      input_shape = [8, 16, 16, 16, 8]
      total_input_size = 1
      for s in input_shape:
        total_input_size *= s
      inputs = [
          i * 1.0 / total_input_size for i in range(1, total_input_size + 1)
      ]
      a = constant_op.constant(inputs, shape=input_shape, dtype=dtypes.float32)

      filter_shape = [1, 1, 1, 8, 8]
      total_filter_size = 1
      for s in filter_shape:
        total_filter_size *= s
      filters = [
          i * 1.0 / total_filter_size for i in range(1, total_filter_size + 1)
      ]
      f = constant_op.constant(
          filters, shape=filter_shape, dtype=dtypes.float32)

      conv_t = nn_ops.conv3d(
          a, filter=f, strides=[1, 1, 1, 1, 1], padding="VALID")
      slice_t = array_ops.slice(conv_t, [0, 1, 1, 1, 0], [1, 1, 1, 1, 8])
      result = self.evaluate(slice_t)
      expected = [
          0.03028321, 0.03132677, 0.03237033, 0.03341389, 0.03445745, 0.035501,
          0.03654456, 0.03758812
      ]
      self.assertAllClose(expected, result.flatten(), rtol=1e-6)

  def testRandom(self):
    # Random dims of rank 6
    input_shape = np.random.randint(0, 20, size=6)
    inp = np.random.rand(*input_shape).astype("f")
    with self.session(use_gpu=True) as sess:
      a = constant_op.constant(
          [float(x) for x in inp.ravel(order="C")],
          shape=input_shape,
          dtype=dtypes.float32)
      indices = [0 if x == 0 else np.random.randint(x) for x in input_shape]
      sizes = [
          np.random.randint(0, input_shape[i] - indices[i] + 1)
          for i in range(6)
      ]
      slice_t = array_ops.slice(a, indices, sizes)
      slice2_t = a[indices[0]:indices[0] + sizes[0], indices[1]:indices[
          1] + sizes[1], indices[2]:indices[2] + sizes[2], indices[3]:indices[3]
                   + sizes[3], indices[4]:indices[4] + sizes[4], indices[5]:
                   indices[5] + sizes[5]]

      slice_val, slice2_val = self.evaluate([slice_t, slice2_t])

    expected_val = inp[indices[0]:indices[0] + sizes[0], indices[1]:indices[
        1] + sizes[1], indices[2]:indices[2] + sizes[2], indices[3]:indices[
            3] + sizes[3], indices[4]:indices[4] + sizes[4], indices[5]:indices[
                5] + sizes[5]]
    self.assertAllEqual(slice_val, expected_val)
    self.assertAllEqual(slice2_val, expected_val)
    self.assertEqual(expected_val.shape, slice_t.get_shape())
    self.assertEqual(expected_val.shape, slice2_t.get_shape())

  def testPartialShapeInference(self):
    z = array_ops.zeros((1, 2, 3))
    self.assertAllEqual(z.get_shape().as_list(), [1, 2, 3])

    m1 = array_ops.slice(z, [0, 0, 0], [-1, -1, -1])
    self.assertAllEqual(m1.get_shape().as_list(), [1, 2, 3])

    m2 = array_ops.slice(z, [0, 0, 0], [constant_op.constant(1) + 0, 2, -1])
    self.assertAllEqual(m2.get_shape().as_list(), [1, 2, 3])

class L2LossTest(test_lib.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testL2Loss(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant(
          [1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="x", dtype=dtype)
      l2loss = nn_ops.l2_loss(x)
      value = self.evaluate(l2loss)
      self.assertAllClose(7.0, value)

  @test_util.run_deprecated_v1
  def testGradient(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    with self.cached_session():
      x = constant_op.constant(x_val, name="x")
      output = nn_ops.l2_loss(x)
      err = gradient_checker.compute_gradient_error(x, x_shape, output, [1])
    print("L2Loss gradient err = %g " % err)
    err_tolerance = 1e-10
    self.assertLess(err, err_tolerance)

class AdamOptimizerTest(test.TestCase):
  def doTestBasic(self, use_resource=False, use_callable_params=False):
    if context.executing_eagerly() and not use_resource:
      self.skipTest(
          "Skipping test with use_resource=False and executing eagerly.")
    for i, dtype in enumerate([dtypes.float32]):
      with self.session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i)
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i)
        else:
          var0 = variables.RefVariable(var0_np)
          var1 = variables.RefVariable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        learning_rate = lambda: 0.001
        beta1 = lambda: 0.9
        beta2 = lambda: 0.999
        epsilon = lambda: 1e-8
        if not use_callable_params:
          learning_rate = learning_rate()
          beta1 = beta1()
          beta2 = beta2()
          epsilon = epsilon()

        opt = adam.AdamOptimizer(learning_rate=learning_rate)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        opt_variables = opt.variables()
        beta1_power, beta2_power = opt._get_beta_accumulators()
        self.assertTrue(beta1_power is not None)
        self.assertTrue(beta2_power is not None)
        self.assertIn(beta1_power, opt_variables)
        self.assertIn(beta2_power, opt_variables)
        # Ensure that non-slot variables are the same type as the requested
        # variables.
        self.assertEqual(
            use_resource,
            resource_variable_ops.is_resource_variable(beta1_power))
        self.assertEqual(
            use_resource,
            resource_variable_ops.is_resource_variable(beta2_power))

        if not context.executing_eagerly():
          with ops.Graph().as_default():
            # Shouldn't return non-slot variables from other graphs.
            self.assertEqual(0, len(opt.variables()))
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          if not context.executing_eagerly():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**(t + 1),
                                             self.evaluate(beta2_power))
          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)
          var0_eval = self.evaluate(var0)
          var1_eval = self.evaluate(var1)
          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0_eval)
          self.assertAllCloseAccordingToType(var1_np, var1_eval)
          if use_resource:
            self.assertEqual("var0_%d/Adam:0" % (i,),
                             opt.get_slot(var=var0, name="m").name)

  def testBasic(self):
     self.doTestBasic(use_resource=True)

  @test_util.run_in_graph_and_eager_modes
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  def testBasicCallableParams(self):
    with context.eager_mode():
      self.doTestBasic(use_resource=True, use_callable_params=True)

  @test_util.run_deprecated_v1
  def testTensorLearningRate(self):
    for dtype in [dtypes.float32]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adam.AdamOptimizer(constant_op.constant(0.001))
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**t,
                                             self.evaluate(beta2_power))
          update.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testSharing(self):
    for dtype in [dtypes.float32]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adam.AdamOptimizer()
        update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 3 steps of intertwined Adam1 and Adam2.
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**t,
                                             self.evaluate(beta2_power))
          if t % 2 == 0:
            update1.run()
          else:
            update2.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testTwoSessions(self):
    optimizer = adam.AdamOptimizer()

    with context.eager_mode():
      var0 = variables.Variable(np.array([1.0, 2.0], dtype=np.float32), name="v0")
      grads0 = constant_op.constant(np.array([0.1, 0.1], dtype=np.float32))
      optimizer.apply_gradients([(grads0, var0)])

    g = ops.Graph()
    with g.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0], dtype=np.float32), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1], dtype=np.float32))
        optimizer.apply_gradients([(grads0, var0)])

    gg = ops.Graph()
    with gg.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))

        # If the optimizer saves any state not keyed by graph the following line
        # fails.
        optimizer.apply_gradients([(grads0, var0)])

  def testSlotsUniqueEager(self):
    with context.eager_mode():
      v1 = resource_variable_ops.ResourceVariable(1.)
      v2 = resource_variable_ops.ResourceVariable(1.)
      opt = adam.AdamOptimizer(1.)
      opt.minimize(lambda: v1 + v2)
      # There should be two non-slot variables, and two unique slot variables
      # for v1 and v2 respectively.
      self.assertEqual(6, len({id(v) for v in opt.variables()}))

class RoundingTest(test.TestCase):

  def _compare_values(self, x, y=None):
    y = np.rint(x) if y is None else np.asarray(y)

    tf_rint = math_ops.rint(x)
    np_rint = self.evaluate(tf_rint)

    self.assertAllEqual(y, np_rint)
    self.assertShapeEqual(y, tf_rint)

  def _compare(self, x):
    np_floor, np_ceil = np.floor(x), np.ceil(x)

    inx = ops.convert_to_tensor(x)
    ofloor, oceil = math_ops.floor(inx), math_ops.ceil(inx)
    tf_floor, tf_ceil = self.evaluate([ofloor, oceil])

    self.assertAllEqual(np_floor, tf_floor)
    self.assertAllEqual(np_ceil, tf_ceil)
    self.assertShapeEqual(np_floor, ofloor)
    self.assertShapeEqual(np_ceil, oceil)

  def _testDtype(self, dtype):
    data = (np.arange(-3, 3) / 4.).reshape(1, 3, 2).astype(dtype)
    self._compare(data)
    # TODO: rint op is not supported for float16
    if dtype is np.float16:
      return
    self._compare_values(data)
    x = [0.5, 0.5000001]
    y = [0.0, 1.0]
    self._compare_values(x, y=y)

    # numpy example
    x = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]
    y = [-2., -2., -0., 0., 2., 2., 2.]
    self._compare_values(x, y=y)

  def testTypes(self):
    self.skipTest("b/131162241")
    for dtype in [np.float16, np.float32, np.float64]:
      self._testDtype(dtype)


class ReverseSequenceTest(test.TestCase):
  def _validateReverseSequence(self,
                           x,
                           batch_axis,
                           seq_axis,
                           seq_lengths,
                           truth,
                           use_gpu=False):
    with self.cached_session(use_gpu=use_gpu):
      ans = array_ops.reverse_sequence(
          x, batch_axis=batch_axis, seq_axis=seq_axis, seq_lengths=seq_lengths)
      tf_ans = self.evaluate(ans)
      self.assertAllClose(tf_ans, truth, atol=1e-10)
      self.assertShapeEqual(truth, ans)

  def _testBasic(self, dtype, len_dtype=np.int64):
    x = np.asarray(
        [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]],
         [[17, 18, 19, 20], [21, 22, 23, 24]]],
        dtype=dtype)
    x = x.reshape(3, 2, 4, 1, 1)
    x = x.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    # reverse dim 2 up to (0:3, none, 0:4) along dim=0
    seq_lengths = np.asarray([3, 0, 4], dtype=len_dtype)

    truth_orig = np.asarray(
        [
            [[3, 2, 1, 4], [7, 6, 5, 8]],  # reverse 0:3
            [[9, 10, 11, 12], [13, 14, 15, 16]],  # reverse none
            [[20, 19, 18, 17], [24, 23, 22, 21]]
        ],  # reverse 0:4 (all)
        dtype=dtype)
    truth_orig = truth_orig.reshape(3, 2, 4, 1, 1)
    truth = truth_orig.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    seq_axis = 0  # permute seq_axis and batch_axis (originally 2 and 0, resp.)
    batch_axis = 2
    self._validateReverseSequence(x, batch_axis, seq_axis, seq_lengths, truth, use_gpu=True)

  def testFloat(self):
    self._testBasic(np.float32, len_dtype=np.int32)
    self._testBasic(np.float32, len_dtype=np.int64)

class TopKTest(test.TestCase):
  def _validateTopK(self,
                inputs,
                k,
                expected_values,
                expected_indices):
    np_expected_values = np.array(expected_values)
    np_expected_indices = np.array(expected_indices)

    with self.cached_session(use_gpu=True) as sess:
        values_op, indices_op = nn_ops.top_k(inputs, k)
        values, indices = self.evaluate([values_op, indices_op])

        self.assertShapeEqual(np_expected_values, values_op)
        self.assertShapeEqual(np_expected_indices, indices_op)

        self.assertAllClose(np_expected_values, values_op)


  def testTop1(self):
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.3, 0.2]]
    self._validateTopK(inputs, 1, [[0.4], [0.3]], [[3], [1]])

  def testTop2(self):
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.4, 0.2]]
    self._validateTopK(inputs, 2, [[0.4, 0.3], [0.4, 0.3]], [[3, 1], [2, 1]])

  def testTop3(self):
    k = 5
    inputs = np.random.permutation(np.linspace(0, 100, 6140, dtype=np.float32))
    indices = np.argsort(-inputs)[:k]
    values = -np.sort(-inputs)[:k]
    self._validateTopK(inputs, k, values, indices)

  def testTensorK(self):
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.4, 0.2]]
    k = constant_op.constant(2)
    self._validateTopK(inputs, k, [[0.4, 0.3], [0.4, 0.3]], [[3, 1], [2, 1]])


class InTopKTest(test.TestCase):
  def _validateInTopK(self,
                    predictions,
                    target,
                    k,
                    expected):
    np_ans = np.array(expected, np.bool)
    with self.cached_session(use_gpu=True) as sess:
        output = nn_ops.in_top_k(predictions, target, k)
        nn_ans = self.evaluate(output)
        self.assertAllEqual(np_ans, nn_ans)
        self.assertShapeEqual(np_ans, output)

  def testInTop1(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [3, 2]
    self._validateInTopK(predictions, target, 1, [True, False])

  def testInTop2(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [2, 2]
    self._validateInTopK(predictions, target, 2, [False, True])

  def testInTop2Tie(self):
    # Class 2 and 3 tie for 2nd, so both are considered in top 2.
    predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
    target = [2, 3]
    self._validateInTopK(predictions, target, 2, [True, True])

  def testInTop2_int64Target(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = np.asarray([0, 2]).astype(np.int64)
    self._validateInTopK(predictions, target, 2, [False, True])

  def testTensorK(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [0, 2]
    k = constant_op.constant(3)
    self._validateInTopK(predictions, target, k, [False, True])

class SplitTest(test.TestCase):

  def testSpecialCase2(self):
    #Test that fails if the Special case 2 is enabled in split_v_op.cc
    split_dim = 0
    shape = (86, 2, 2, 4, 4)
    size_splits = [4, 2, 4, 7, 5, 7, 4, 6, 2, 3, 7, 6, 5, 2, 5, 2, 4, 6, 5]
    x = np.random.rand(*shape).astype(np.float32)
    tf_ans = self.evaluate(array_ops.split(x, size_splits, split_dim))

  def testRandomVariableSlices(self):
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    split_dim = np.random.randint(-5, 5)
    num_split = np.random.randint(2, 25)
    size_splits = np.random.randint(2, 8, num_split, dtype=np.int32)
    shape[split_dim] = np.sum(size_splits)
    x = np.random.rand(*shape).astype(np.float32)
    with self.cached_session(use_gpu=True):
      result = self.evaluate(array_ops.split(x, size_splits, split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + size_splits[i])
      offset += size_splits[i]
      self.assertAllEqual(result[i], x[tuple(slices)])

  def testRegularSlices(self):
    shape = np.random.randint(1, 5, size=5)
    split_dim = np.random.randint(-5, 5)
    num_split = np.random.randint(2,10)
    shape[split_dim] = shape[split_dim]*num_split
    x = np.random.rand(*shape).astype(np.float32)
    with self.cached_session(use_gpu=True):
      result = self.evaluate(array_ops.split(x, num_split, split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    length = shape[split_dim] // num_split
    for i in range(num_split):
        slices[split_dim] = slice(offset, offset + length)
        offset += length
        self.assertAllEqual(result[i], x[tuple(slices)])



class ResizeBilinearTest(test.TestCase):
  def _testResize(self,
                  x,
                  y,
                  use_gpu=False):
    with self.cached_session(use_gpu=use_gpu):
        ans = image_ops.resize_bilinear(x, y, half_pixel_centers=True)
        tf_ans = self.evaluate(ans)
        ref_ans = self._refResize(x, y)
        self.assertAllEqual(tf_ans.shape, ref_ans.shape)
        self.assertAllClose(tf_ans, ref_ans)

  def _refResize(self, x, y):
    """
    PIL has to treat each channel separately. Additionally it expects the new shape
    to be given (width, height), where as tensorflow expects (height, width)
    """
    resized_array = []
    for array in x:
        img_channels = []
        for channel_ind in range(array.shape[-1]):
            channel = array[:,:,channel_ind]
            pil_img = Image.fromarray(channel)
            resized_img = np.asarray(pil_img.resize(size=(y[1], y[0]), resample=Image.BILINEAR))
            img_channels.append(resized_img)
        img = np.stack(img_channels, axis=-1)
        resized_array.append(img)

    resized_array = np.array(resized_array)
    return resized_array

  def testFloatBasic(self):
    x = np.random.rand(3, 24, 24, 3)
    x = x.astype(np.float32)
    y = np.asarray([48, 48], dtype=np.int32)
    self._testResize(x, y, use_gpu=True)

  def testFloatUneven(self):
    x = np.random.rand(3, 24, 48, 3)
    x = x.astype(np.float32)
    y = np.asarray([96, 64])
    self._testResize(x, y, use_gpu=True)

  def testFloatLarge(self):
    x = np.random.rand(3, 256, 256, 3)
    x = x.astype(np.float32)
    y = np.asarray([1024, 1024])
    self._testResize(x, y, use_gpu=True)


class OneHotTest(test.TestCase):

  def _testOneHot(self,
                  truth,
                  use_gpu=False,
                  expected_err_re=None,
                  raises=None,
                  **inputs):
    with self.cached_session(use_gpu=use_gpu):
      if raises is not None:
        with self.assertRaises(raises):
          array_ops.one_hot(**inputs)
      else:
        ans = array_ops.one_hot(**inputs)

        if expected_err_re is None:
          tf_ans = self.evaluate(ans)
          self.assertEqual(tf_ans.shape, ans.get_shape())
          self.assertAllEqual(tf_ans, truth)
        else:
          with self.assertRaisesOpError(expected_err_re):
            self.evaluate(ans)

  def _testBothOneHot(self, truth, expected_err_re=None, raises=None, **inputs):
    self._testOneHot(truth, True, expected_err_re, raises, **inputs)
    self._testOneHot(truth, False, expected_err_re, raises, **inputs)

  def _testBasic(self, dtype):
    indices = np.asarray([0, 2, -1, 1], dtype=np.int32)
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
    indices = np.asarray([0, 2, -1, 1], dtype=np.int32)
    depth = 3

    truth = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=dtype)

    # axis == -1
    self._testBothOneHot(indices=indices, depth=depth, truth=truth)

    # axis == 0
    self._testBothOneHot(
        indices=indices, depth=depth, axis=0,
        truth=truth.T)  # Output is transpose version in this case

  def testFloatBasic(self):
    self._testBasic(np.float32)
    self._testDefaultBasic(np.float32)

def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NHWC", False), ("NHWC", True)]
  return test_configs


class Conv2DTest(test.TestCase):

  def _DtypesToTest(self, use_gpu):
    # double datatype is currently not supported for convolution ops
    # on the ROCm platform
    optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
    if use_gpu and not test_util.GpuSupportsHalfMatMulAndConv():
      return [dtypes.float32] + optional_float64
    else:
      # It is important that float32 comes before float16 here,
      # as we will be using its gradients as reference for fp16 gradients.
      return [dtypes.float32, dtypes.float16] + optional_float64

  def _CreateNumpyTensor(self, shape):
    total_size = 1
    for s in shape:
      total_size *= s
    return np.arange(1, total_size + 1, dtype=np.float32).reshape(shape)

  def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, dilations,
                            strides, padding, data_format, dtype, use_gpu):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      dilations: Dilated rate: [col_dilation, row_dilation]
      strides: Stride: [col_stride, row_stride]
      padding: Padding type.
      data_format: Format of the data tensors.
      dtype: Data type for inputs and outputs.
      use_gpu: True if the operations should be run on GPU
    Returns:
      Symbolic tensor value that can be used to execute the computation
    """
    x1 = self._CreateNumpyTensor(tensor_in_sizes)
    x2 = self._CreateNumpyTensor(filter_in_sizes)

    with test_util.device(use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)
      strides = [1] + strides + [1]
      dilations = [1] + dilations + [1]
      if isinstance(padding, (list, tuple)):
        padding = [(0, 0)] + padding + [(0, 0)]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)
        dilations = test_util.NHWCToNCHW(dilations)
        if isinstance(padding, (list, tuple)):
          padding = test_util.NHWCToNCHW(padding)
      conv = nn_ops.conv2d(
          t1,
          t2,
          dilations=dilations,
          strides=strides,
          padding=padding,
          data_format=data_format)
      self.assertEqual(conv.dtype, dtype)
      if data_format == "NCHW":
        conv = test_util.NCHWToNHWC(conv)

      return conv

  def _CompareFwdValues(self, tensor_in_sizes, filter_in_sizes, conv_strides,
                        padding):
    """Verifies that CPU and GPU produce the same values.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      conv_strides: [row_stride, col_stride] for the convolution;
      padding: Padding type.
    """
    x1 = np.random.rand(*tensor_in_sizes).astype(np.float32)
    x2 = np.random.rand(*filter_in_sizes).astype(np.float32)

    def _SetupVal(data_format, use_gpu):
      with test_util.device(use_gpu):
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t1 = test_util.NHWCToNCHW(t1)
          strides = test_util.NHWCToNCHW(strides)
        conv = nn_ops.conv2d(
            t1, t2, strides=strides, padding=padding, data_format=data_format)
        if data_format == "NCHW":
          conv = test_util.NCHWToNHWC(conv)
        return conv

    tensors = []
    for (data_format, use_gpu) in GetTestConfigs():
      tensors.append(_SetupVal(data_format, use_gpu))
    values = self.evaluate(tensors)
    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-3, atol=1e-3)

  def _ComputeReferenceDilatedConv(self, tensor_in_sizes, filter_in_sizes,
                                   stride, dilation, padding, data_format,
                                   use_gpu):
    x1 = self._CreateNumpyTensor(tensor_in_sizes)
    x2 = self._CreateNumpyTensor(filter_in_sizes)
    with test_util.device(use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      if isinstance(stride, collections_abc.Iterable):
        strides = list(stride)
      else:
        strides = [stride, stride]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        full_strides = [1, 1] + strides
        full_dilation = [1, 1] + dilation
      else:
        full_strides = [1] + strides + [1]
        full_dilation = [1] + dilation + [1]
      expected = nn_ops.convolution(
          t1,
          t2,
          padding=padding,
          strides=strides,
          dilation_rate=dilation,
          data_format=data_format)
      computed = nn_ops.conv2d(
          t1,
          t2,
          strides=full_strides,
          dilations=full_dilation,
          padding=padding,
          data_format=data_format)
      if data_format == "NCHW":
        expected = test_util.NCHWToNHWC(expected)
        computed = test_util.NCHWToNHWC(computed)
    return expected, computed

  def _VerifyDilatedConvValues(self, tensor_in_sizes, filter_in_sizes, strides,
                               padding, dilations, rtol=1e-4):
    expected_results = []
    computed_results = []
    for data_format, use_gpu in GetTestConfigs():
      expected, computed = self._ComputeReferenceDilatedConv(
          tensor_in_sizes, filter_in_sizes, strides, dilations, padding,
          data_format, use_gpu)
      expected_results.append(expected)
      computed_results.append(computed)
      tolerance = 1e-2 if use_gpu else 1e-5
      expected_values = self.evaluate(expected_results)
      computed_values = self.evaluate(computed_results)
      for e_value, c_value in zip(expected_values, computed_values):
        tf_logging.debug("expected = %s", e_value)
        tf_logging.debug("actual = %s", c_value)
        self.assertAllClose(
            e_value.flatten(), c_value.flatten(), atol=tolerance, rtol=rtol)

  def _VerifyValues(self,
                    tensor_in_sizes,
                    filter_in_sizes,
                    strides,
                    padding,
                    expected,
                    dilations=(1, 1),
                    gpu_only=False,
                    test_grappler_layout_optimizer=False,
                    tol=1e-5,
                    fp16_tol=1e-3):
    if gpu_only and not test.is_gpu_available(cuda_only=True):
      return
    tensors = []
    dilations = list(dilations)
    for (data_format, use_gpu) in GetTestConfigs():
      if gpu_only and not use_gpu:
        continue
      dtypes_to_test = self._DtypesToTest(use_gpu)
      if not test_grappler_layout_optimizer and data_format == "NHWC":
        dtypes_to_test.append(dtypes.int32)
      for dtype in dtypes_to_test:
        result = self._SetupValuesForDevice(
            tensor_in_sizes,
            filter_in_sizes,
            dilations,
            strides,
            padding,
            data_format,
            dtype,
            use_gpu=use_gpu)
        if test_grappler_layout_optimizer and data_format == "NHWC" and use_gpu:
          # Grappler's layout optimizer will not optimize a fetch node, so
          # this identity allows Grappler to optimize the Conv2D node.
          result = array_ops.identity(result)
        tensors.append(result)
      values = self.evaluate(tensors)
      for i in range(len(tensors)):
        conv = tensors[i]
        value = values[i]
        tf_logging.debug("expected = %s", expected)
        tf_logging.debug("actual = %s", value)
        tol_to_use = fp16_tol if value.dtype == np.float16 else tol
        if np.issubdtype(value.dtype, np.integer):
          self.assertAllEqual(np.rint(expected), np.ravel(value))
        else:
          self.assertAllClose(expected, np.ravel(value), atol=tol_to_use,
                              rtol=tol_to_use)
        self.assertShapeEqual(value, conv)
        self.assertEqual(value.dtype, conv.dtype.as_numpy_dtype)

  def _VerifyExplicitPaddings(self,
                              tensor_in_sizes,
                              filter_in_sizes,
                              strides,
                              padding,
                              dilations=(1, 1),
                              test_grappler_layout_optimizer=False,
                              tol=1e-5,
                              fp16_tol=1e-3):
    """Verifies Conv2D with explicit padding generates correct values.

    It does this by comparing with Conv2D without explicit padding. This
    function assumes Conv2D without explicit padding works correctly.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [kernel_rows, kernel_cols,
        input_depth, output_depth].
      strides: [row_stride, col_stride] for the convolution;
      padding: Explicit padding amounts.
      dilations: Dilation values
      test_grappler_layout_optimizer: If True, allow the Grappler layout
        optimizer to run, which turns NHWC Conv2Ds on the GPU to NCHW Conv2Ds.
      tol: The absolute and relative tolerance for non-fp16 dtypes.
      fp16_tol: The absolute and relative tolerance for fp16.
    """
    input_tensor = self._CreateNumpyTensor(tensor_in_sizes)
    filter_tensor = self._CreateNumpyTensor(filter_in_sizes)
    input_tensor = array_ops.pad(input_tensor, [(0, 0)] + padding + [(0, 0)])
    dilations = list(dilations)
    conv2d_result = nn_ops.conv2d(
        input_tensor,
        filter_tensor, [1] + list(strides) + [1],
        "VALID",
        dilations=[1] + dilations + [1])
    expected = list(self.evaluate(array_ops.reshape(conv2d_result, [-1])))
    self._VerifyValues(
        tensor_in_sizes,
        filter_in_sizes,
        strides,
        padding,
        expected,
        dilations,
        test_grappler_layout_optimizer=test_grappler_layout_optimizer,
        tol=tol,
        fp16_tol=fp16_tol)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D1x1Filter(self):
    expected_output = [
        30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0,
        204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2Filter2x1Dilation(self):
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[1, 4, 4, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID")

  @test_util.run_in_graph_and_eager_modes
  def testConv2DEmpty(self):
    expected_output = []
    self._VerifyValues(
        tensor_in_sizes=[0, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DEmptyDilation(self):
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[0, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID")

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2Filter(self):
    # The outputs are computed using third_party/py/IPython/notebook.
    expected_output = [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterDilation(self):
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        dilations=[1, 2],
        padding="VALID")

  @test_util.run_in_graph_and_eager_modes
  def testConv2D1x2Filter(self):
    # The outputs are computed using third_party/py/IPython/notebook.
    expected_output = [
        231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0,
        936.0, 1029.0
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 2, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D1x2FilterDilation(self):
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 2, 3, 3],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID")

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterStride2(self):
    expected_output = [2271.0, 2367.0, 2463.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[2, 2],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterStride2Same(self):
    expected_output = [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[2, 2],
        padding="SAME",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterStride1x2(self):
    expected_output = [58.0, 78.0, 98.0, 118.0, 138.0, 158.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 6, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[1, 2],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DKernelSmallerThanStrideValid(self):
    expected_output = [65, 95, 275, 305]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[3, 3],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DKernelSmallerThanStrideSame(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1],
        strides=[2, 2],
        padding="SAME",
        expected=[1, 3, 7, 9])

    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 1],
        filter_in_sizes=[1, 1, 1, 1],
        strides=[2, 2],
        padding="SAME",
        expected=[1, 3, 9, 11])

    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[3, 3],
        padding="SAME",
        expected=[44, 28, 41, 16])

  @test_util.run_in_graph_and_eager_modes
  def testConv2DKernelSizeMatchesInputSize(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 2, 1],
        filter_in_sizes=[2, 2, 1, 2],
        strides=[1, 1],
        padding="VALID",
        expected=[50, 60])

  @test_util.run_in_graph_and_eager_modes
  def testConv2DKernelSizeMatchesInputSizeDilation(self):
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[1, 3, 3, 1],
        filter_in_sizes=[2, 2, 1, 2],
        strides=[1, 1],
        dilations=[2, 2],
        padding="VALID")

  @test_util.run_in_graph_and_eager_modes()
  def testConv2D0x0Padding(self):
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        padding=[[0, 0], [0, 0]])

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[3, 4, 3, 2],
        filter_in_sizes=[1, 1, 2, 1],
        strides=[2, 2],
        padding=[[0, 0], [0, 0]])

  @test_util.run_in_graph_and_eager_modes()
  def testConv2D1x1Padding(self):
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[2, 2, 2, 2],
        strides=[1, 1],
        padding=[[1, 1], [1, 1]])

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 2, 1],
        filter_in_sizes=[1, 1, 1, 2],
        strides=[1, 1],
        padding=[[1, 1], [1, 1]])

  @test_util.run_in_graph_and_eager_modes()
  def testConv2D2x2Padding(self):
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 1, 2],
        filter_in_sizes=[2, 1, 2, 1],
        strides=[1, 1],
        padding=[[2, 2], [2, 2]])

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 1, 2],
        filter_in_sizes=[1, 1, 2, 1],
        strides=[2, 1],
        padding=[[2, 2], [2, 2]])


  @test_util.run_in_graph_and_eager_modes()
  def testConv2DOnlyTopRightPadding(self):
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 2],
        strides=[1, 1],
        padding=[[1, 0], [0, 2]],
        tol=5e-5)

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 4, 2],
        filter_in_sizes=[2, 2, 2, 2],
        strides=[1, 3],
        padding=[[1, 0], [0, 2]])

  @test_util.run_in_graph_and_eager_modes()
  def testConv2DLotsPadding(self):
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 1, 1, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        padding=[[3, 4], [4, 2]])

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 1, 1],
        filter_in_sizes=[2, 2, 1, 3],
        strides=[2, 1],
        padding=[[3, 4], [4, 2]])

  @test_util.run_in_graph_and_eager_modes()
  def testConv2DExplicitPaddingWithDilations(self):
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 3, 2, 1],
        filter_in_sizes=[1, 2, 1, 2],
        strides=[1, 1],
        padding=[[1, 0], [0, 1]],
        dilations=[2, 1])

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[3, 2, 2, 1],
        strides=[1, 1],
        padding=[[2, 1], [1, 2]],
        dilations=[2, 3])

  def testConv2DExplicitPaddingWithLayoutOptimizer(self):
    # Test with Grappler's layout optimizer, to ensure the layout optimizer
    # handles explicit padding correctly.
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 3, 2, 1],
        filter_in_sizes=[1, 2, 1, 2],
        strides=[1, 1],
        padding=[[1, 0], [0, 1]],
        dilations=[2, 1],
        test_grappler_layout_optimizer=True)

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[3, 2, 2, 1],
        strides=[1, 1],
        padding=[[2, 1], [1, 2]],
        dilations=[2, 3],
        test_grappler_layout_optimizer=True)

  # Testing for backprops
  def _RunAndVerifyBackpropInput(self,
                                 input_sizes,
                                 filter_sizes,
                                 output_sizes,
                                 strides,
                                 padding,
                                 expected,
                                 data_format,
                                 use_gpu,
                                 err,
                                 dilations=(1, 1)):
    if use_gpu and not test.is_gpu_available(cuda_only=True):
      return
    x1 = self._CreateNumpyTensor(filter_sizes)
    x2 = self._CreateNumpyTensor(output_sizes)
    dilations = list(dilations)
    with test_util.device(use_gpu):
      if data_format == "NCHW":
        input_sizes = test_util.NHWCToNCHW(input_sizes)
      t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
      t1 = constant_op.constant(x1, shape=filter_sizes)
      t2 = constant_op.constant(x2, shape=output_sizes)
      strides = [1] + strides + [1]
      dilations = [1] + dilations + [1]
      if isinstance(padding, (list, tuple)):
        padding = [(0, 0)] + padding + [(0, 0)]
      if data_format == "NCHW":
        t2 = test_util.NHWCToNCHW(t2)
        strides = test_util.NHWCToNCHW(strides)
        dilations = test_util.NHWCToNCHW(dilations)
        if isinstance(padding, (list, tuple)):
          padding = test_util.NHWCToNCHW((padding))
      conv = nn_ops.conv2d_backprop_input(
          t0,
          t1,
          t2,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)
      if data_format == "NCHW":
        conv = test_util.NCHWToNHWC(conv)
      # "values" consists of two tensors for two backprops
      value = self.evaluate(conv)
      self.assertShapeEqual(value, conv)
    tf_logging.debug("expected = %s", expected)
    tf_logging.debug("actual = %s", value)
    self.assertAllCloseAccordingToType(expected, value.flatten(), atol=1e-5)

  def _CompareBackpropInput(self, input_sizes, filter_sizes, output_sizes,
                            conv_strides, padding):
    x1 = np.random.rand(*filter_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(data_format, use_gpu):
      with test_util.device(use_gpu):
        if data_format == "NCHW":
          new_input_sizes = test_util.NHWCToNCHW(input_sizes)
        else:
          new_input_sizes = input_sizes
        t0 = constant_op.constant(new_input_sizes, shape=[len(new_input_sizes)])
        t1 = constant_op.constant(x1, shape=filter_sizes)
        t2 = constant_op.constant(x2, shape=output_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t2 = test_util.NHWCToNCHW(t2)
          strides = test_util.NHWCToNCHW(strides)
        conv = nn_ops.conv2d_backprop_input(
            t0,
            t1,
            t2,
            strides=strides,
            padding=padding,
            data_format=data_format)
        if data_format == "NCHW":
          conv = test_util.NCHWToNHWC(conv)
        ret = self.evaluate(conv)
        self.assertShapeEqual(ret, conv)
        return ret

    values = []
    for (data_format, use_gpu) in GetTestConfigs():
      values.append(_GetVal(data_format, use_gpu))

    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DEmptyBackpropInput(self):
    expected_output = []
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropInput(
          input_sizes=[0, 2, 3, 1],
          filter_sizes=[2, 2, 1, 1],
          output_sizes=[0, 1, 2, 1],
          strides=[1, 1],
          padding="VALID",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu,
          err=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DStrideTwoFilterOneSameBackpropInput(self):
    expected_output = [
        1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ]
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropInput(
          input_sizes=[1, 4, 4, 1],
          filter_sizes=[1, 1, 1, 1],
          output_sizes=[1, 2, 2, 1],
          strides=[2, 2],
          padding="SAME",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu,
          err=1e-5)



class PoolingTest(test.TestCase):

  def _test(self, input_shape, dtype, **kwargs):
    # Use negative numbers to make sure there isn't any zero padding getting
    # used.
    x = -np.arange(
        np.prod(input_shape), dtype=dtype).reshape(input_shape) - 1
    y1 = pool_direct(input=x, **kwargs)
    y2 = nn_ops.pool(input=x, **kwargs)
    self.assertAllClose(y1, self.evaluate(y2), rtol=1e-2, atol=1e-2)

  def _test_gradient(self, input_shape, dtype, **kwargs):
    x_val = -np.arange(
        np.prod(input_shape), dtype=dtype).reshape(input_shape) - 1
    x = constant_op.constant(x_val, name="x", dtype=dtype)
    output = nn_ops.pool(input=x, **kwargs)
    y_shape = output.get_shape().as_list()
    err = gradient_checker.compute_gradient_error(
        [x], [input_shape], output, y_shape, x_init_value=[x_val])
    err_tolerance = 1e-2
    if (dtype==dtypes.float16):
        err_tolerance = 1.1
        #TODO: this is too high.
        #investigate precision issues.
    self.assertLess(err, err_tolerance)

  def testPoolSimple(self):
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["MAX", "AVG"]:
            for dtype in [np.float32, np.float16]:
              self._test(
                  input_shape=[1, 1, 10, 1],
                  window_shape=[1, 3],
                  padding=padding,
                  pooling_type=pooling_type,
                  dilation_rate=[1, 1],
                  strides=[1, 2],
                  dtype=dtype)

  def testPool1D(self):
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for dtype in [np.float32, np.float16]:
            for pooling_type in ["MAX", "AVG"]:
              for input_shape in [[2, 9, 2], [2, 10, 2]]:
                for window_shape in [[1], [2], [3]]:
                  if padding != "SAME":
                    for dilation_rate in [[1], [2], [3]]:
                      self._test(
                          input_shape=input_shape,
                          window_shape=window_shape,
                          padding=padding,
                          pooling_type=pooling_type,
                          dilation_rate=dilation_rate,
                          strides=[1],
                          dtype=dtype)
                  for strides in [[1], [2], [3]]:
                    if np.any(np.array(strides) > window_shape):
                      continue
                    self._test(
                        input_shape=input_shape,
                        window_shape=window_shape,
                        padding=padding,
                        pooling_type=pooling_type,
                        dilation_rate=[1],
                        strides=strides,
                        dtype=dtype)

  def testPool2D(self):
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for dtype in [np.float32, np.float16]:
            for pooling_type in ["MAX", "AVG"]:
              for input_shape in [[2, 9, 10, 2], [2, 10, 9, 2]]:
                for window_shape in [[1, 1], [2, 1], [2, 3]]:
                  if padding != "SAME":
                    for dilation_rate in [[1, 1], [2, 1], [1, 2], [2, 3]]:
                      self._test(
                          input_shape=input_shape,
                          window_shape=window_shape,
                          padding=padding,
                          pooling_type=pooling_type,
                          dilation_rate=dilation_rate,
                          strides=[1, 1],
                          dtype=dtype)
                  for strides in [[1, 1], [2, 1], [1, 2], [2, 3]]:
                    if np.any(np.array(strides) > window_shape):
                      continue
                    self._test(
                        input_shape=input_shape,
                        window_shape=window_shape,
                        padding=padding,
                        pooling_type=pooling_type,
                        dilation_rate=[1, 1],
                        strides=strides,
                        dtype=dtype)

  @test_util.run_deprecated_v1
  def testGradient2D(self):
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for dtype in [np.float32, np.float16]:
            for pooling_type in ["AVG", "MAX"]:
              for input_shape in [[2, 4, 5, 2], [1, 5, 4, 1]]:
                for window_shape in [[1, 1], [2, 1], [2, 2]]:
                  if padding != "SAME":
                    for dilation_rate in [[1, 1], [2, 1], [2, 2]]:
                      self._test_gradient(
                          input_shape=input_shape,
                          window_shape=window_shape,
                          padding=padding,
                          pooling_type=pooling_type,
                          dilation_rate=dilation_rate,
                          strides=[1, 1],
                          dtype=dtype)
                  for strides in [[1, 1], [2, 1], [1, 2], [2, 2]]:
                    if np.any(np.array(strides) > window_shape):
                      continue
                    self._test_gradient(
                        input_shape=input_shape,
                        window_shape=window_shape,
                        padding=padding,
                        pooling_type=pooling_type,
                        dilation_rate=[1, 1],
                        strides=strides,
                        dtype=dtype)

class FractionalMaxPoolGradTest(test.TestCase):
  _PRNG = np.random.RandomState(341261)
  _SEED = 123456

  def _GenerateUniqueRandomInputTensor(self, shape):
    num_elements = 1
    for size in shape:
      num_elements *= size
    x = np.arange(num_elements, dtype=np.float32)
    self._PRNG.shuffle(x)
    return x.reshape(shape)

  def testDirectNotUseOverlapping(self):
    for num_batches in [1]:
      for row_window_size in [2, 5]:
        for col_window_size in [2, 4]:
          num_rows = row_window_size
          num_cols = col_window_size
          for num_channels in [1]:
            input_shape = (num_batches, num_rows, num_cols, num_channels)
            with self.cached_session() as _:
              input_tensor = constant_op.constant(
                  self._GenerateUniqueRandomInputTensor(input_shape))
              window_size = [1, row_window_size, col_window_size, 1]
              stride_size = [1, row_window_size, col_window_size, 1]
              padding = "VALID"
              output_tensor = nn_ops.max_pool(input_tensor, window_size,
                                              stride_size, padding)
              output_data = self.evaluate(output_tensor)
              output_backprop = self._PRNG.randint(100, size=output_data.shape)
              input_backprop_tensor = gen_nn_ops.max_pool_grad(
                  input_tensor, output_tensor, output_backprop, window_size,
                  stride_size, padding)
              input_backprop = self.evaluate(input_backprop_tensor)



class RandomOpTestCommon(test.TestCase):

  # Checks that executing the same rng_func multiple times rarely produces the
  # same result.
  def _testSingleSessionNotConstant(self,
                                    rng_func,
                                    num,
                                    dtype,
                                    min_or_mean,
                                    max_or_stddev,
                                    use_gpu,
                                    op_seed=None,
                                    graph_seed=None):
    with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
      if graph_seed is not None:
        random_seed.set_random_seed(graph_seed)
      x = rng_func([num], min_or_mean, max_or_stddev, dtype=dtype, seed=op_seed)

      y = self.evaluate(x)
      z = self.evaluate(x)
      w = self.evaluate(x)

      # We use exact equality here. If the random-number generator is producing
      # the same output, all three outputs will be bitwise identical.
      self.assertTrue((not np.array_equal(y, z)) or
                      (not np.array_equal(z, w)) or (not np.array_equal(y, w)))

@test_util.for_all_test_methods(test_util.disable_xla,
                                "This never passed on XLA")
class RandomUniformTest(RandomOpTestCommon):

  def _Sampler(self, num, minv, maxv, dtype, use_gpu, seed=None):

    def func():
      with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
        rng = random_ops.random_uniform(
            [num], minval=minv, maxval=maxv, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in xrange(10):
          ret[i, :] = self.evaluate(rng)
      return ret

    return func

  def testRange(self):
    for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
               dtypes.int64):
      sampler = self._Sampler(1000, minv=-2, maxv=8, dtype=dt, use_gpu=True)
      x = sampler()
      self.assertTrue(-2 <= np.min(x))
      self.assertTrue(np.max(x) < 8)

  # Asserts that different trials (1000 samples per trial) is unlikely
  # to see the same sequence of values. Will catch buggy
  # implementations which uses the same random number seed.
  def testDistinct(self):
    for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
               dtypes.int64):
      maxv = 1.0 if dt.is_floating else 1 << 30
      sampler = self._Sampler(1000, minv=0, maxv=maxv, dtype=dt, use_gpu=True)
      x = sampler()
      y = sampler()
      count = (x == y).sum()
      count_limit = 50 if dt == dtypes.float16 else 10
      if count >= count_limit:
        print("x = ", x)
        print("y = ", y)
        print("count = ", count)
      self.assertTrue(count < count_limit)

  @test_util.run_deprecated_v1
  def testUniformIntsWithInvalidShape(self):
    for dtype in dtypes.int32, dtypes.int64:
      with self.assertRaisesRegexp(
          ValueError, "minval must be a scalar; got a tensor of shape"):
        random_ops.random_uniform(
            [1000], minval=[1, 2], maxval=3, dtype=dtype)
      with self.assertRaisesRegexp(
          ValueError, "maxval must be a scalar; got a tensor of shape"):
        random_ops.random_uniform(
            [1000], minval=1, maxval=[2, 3], dtype=dtype)

  # Check that uniform ints actually follow a uniform distribution.
  @test_util.run_deprecated_v1
  def testUniformInts(self):
    minv = -2
    maxv = 15
    n = 100000
    p = 1 / (maxv - minv)
    # The counts should follow an (n, p) binomial distribution.
    mean = p * n
    std = np.sqrt(n * p * (1 - p))
    for dt in dtypes.int32, dtypes.int64:
      # Use a fixed seed here to make the test deterministic.
      # Without the fixed seed, the 5 * std bound will (very rarely) fail.
      sampler = self._Sampler(
          n // 10, minv=minv, maxv=maxv, dtype=dt, use_gpu=True, seed=17)
      x = sampler().ravel()
      self.assertEqual(x.shape, (n,))
      counts, _ = np.histogram(x, bins=maxv - minv)
      self.assertEqual(counts.shape, (maxv - minv,))
      self.assertEqual(counts.sum(), n)
      error = np.abs(counts - mean)
      self.assertLess(error.max(), 5 * std)

  # Check that minval = maxval is fine iff we're producing no numbers
  def testUniformIntsDegenerate(self):
    for dt in dtypes.int32, dtypes.int64:
      def sample(n):
        return self._Sampler(n, minv=0, maxv=0, dtype=dt, use_gpu=True)()
      self.assertEqual(sample(0).shape, (10, 0))
      with self.assertRaisesOpError('Need minval < maxval, got 0 >= 0'):
        sample(1)

  @test_util.run_deprecated_v1
  def testSeed(self):
    for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
               dtypes.int64):
      for seed in [345, 2**100, -2**100]:
        sx = self._Sampler(1000, 0, 17, dtype=dt, use_gpu=True, seed=seed)
        sy = self._Sampler(1000, 0, 17, dtype=dt, use_gpu=True, seed=seed)
        self.assertAllEqual(sx(), sy())

  @test_util.run_deprecated_v1
  def testNoCSE(self):
    shape = [2, 3, 4]
    for dtype in dtypes.float16, dtypes.float32, dtypes.int32:
      with self.session(use_gpu=True):
        rnd1 = random_ops.random_uniform(shape, 0, 17, dtype=dtype)
        rnd2 = random_ops.random_uniform(shape, 0, 17, dtype=dtype)
        diff = (rnd2 - rnd1).eval()
        self.assertTrue(np.linalg.norm(diff) > 0.1)

  @test_util.run_deprecated_v1
  def testSingleSessionNotConstant(self):
    for use_gpu in [False, True]:
      for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
                 dtypes.int64):
        self._testSingleSessionNotConstant(
            random_ops.random_uniform, 100, dt, 0, 17, use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testSingleSessionOpSeedNotConstant(self):
    for use_gpu in [False, True]:
      for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
                 dtypes.int64):
        self._testSingleSessionNotConstant(
            random_ops.random_uniform,
            100,
            dt,
            10,
            20,
            use_gpu=use_gpu,
            op_seed=1345)

  @test_util.run_deprecated_v1
  def testSingleSessionGraphSeedNotConstant(self):
    for use_gpu in [False, True]:
      for dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
                 dtypes.int64):
        self._testSingleSessionNotConstant(
            random_ops.random_uniform,
            100,
            dt,
            20,
            200,
            use_gpu=use_gpu,
            graph_seed=965)


class BroadcastToTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testBroadcastToBasic(self):
    for dtype in [np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64]:
      with self.session(use_gpu=True):
        x = np.array([1, 2, 3], dtype=dtype)
        v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToString(self):
    with self.session(use_gpu=True):
      x = np.array([b"1", b"2", b"3"])
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToBool(self):
    with self.session(use_gpu=True):
      x = np.array([True, False, True], dtype=np.bool)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShape(self):
    for input_dim in range(1, 6):
      for output_dim in range(input_dim, 6):
        with self.cached_session(use_gpu=True):
          input_shape = [2] * input_dim
          output_shape = [2] * output_dim
          x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
          v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
          v_np = np.broadcast_to(x, output_shape)
          self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeInnerDim(self):
    input_shape = [2, 1, 3]
    output_shape = [2, 5, 3]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeLargerDim(self):
    input_shape = [2, 1, 3, 2, 2, 2]
    output_shape = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 15, 3, 2, 2, 2]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeLargerDim2(self):
    input_shape = [2, 1, 3, 2, 2, 2, 1, 1, 1]
    output_shape = [1, 1, 1, 2, 5, 3, 2, 2, 2, 3, 3, 3]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToScalar(self):
    with self.session(use_gpu=True):
      x = np.array(1, dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastScalarToNonScalar(self):
    with self.session(use_gpu=True):
      x = np.array(1.0, dtype=np.float)
      v_tf = array_ops.broadcast_to(constant_op.constant(1.0), [2, 3, 4,
                                                                1, 1, 1])
      v_np = np.broadcast_to(x, [2, 3, 4, 1, 1, 1])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeTypeAndInference(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.cached_session(use_gpu=True):
        x = np.array([1, 2, 3])
        v_tf = array_ops.broadcast_to(
            constant_op.constant(x),
            constant_op.constant([3, 3], dtype=dtype))
        shape = v_tf.get_shape().as_list()
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf.eval(), v_np)
        # check shape inference when shape input is constant
        self.assertAllEqual(shape, v_np.shape)

  def testBroadcastToBadOutputShape(self):
    with context.eager_mode():
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "Unable to broadcast tensor of shape"):
        self.evaluate(
            array_ops.broadcast_to(
                constant_op.constant([0, 1]), constant_op.constant([2, 1])))

  @test_util.run_deprecated_v1
  def testGradientForScalar(self):
    x = constant_op.constant(1, dtype=dtypes.float32)
    v = array_ops.broadcast_to(x, [2, 4, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(x, x.get_shape(), out,
                                                    out.get_shape())
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithSameRank(self):
    x = constant_op.constant(np.reshape(np.arange(6), (2, 1, 3)),
                             dtype=dtypes.float32)
    v = array_ops.broadcast_to(x, [2, 5, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(x, x.get_shape(),
                                                    out, out.get_shape())
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithIncreasingRank(self):
    x = constant_op.constant([[1], [2]],
                             dtype=dtypes.float32)
    v = array_ops.broadcast_to(x, [5, 2, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(x, x.get_shape(),
                                                    out, out.get_shape())
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithBroadcastAllDimensions(self):
    x = constant_op.constant([1], dtype=dtypes.float32)
    v = array_ops.broadcast_to(x, [5, 2, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(x, x.get_shape(),
                                                    out, out.get_shape())
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithLargeDim(self):
    input_shape = [2, 1, 3, 2, 2, 2, 1, 1, 1]
    output_shape = [1, 1, 1, 2, 5, 3, 2, 2, 2, 3, 3, 3]
    x = constant_op.constant(np.array(np.random.randn(*input_shape),
                                      dtype=np.float32))
    v = array_ops.broadcast_to(x, output_shape)
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(x, x.get_shape(),
                                                    out, out.get_shape())
    self.assertLess(err, 1e-4)


class GPUBinaryOpsTest(test.TestCase):

  def _compareGPU(self, x, y, np_func, tf_func):
    with self.cached_session(use_gpu=True) as sess:
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = self.evaluate(out)

    with self.cached_session(use_gpu=False) as sess:
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = self.evaluate(out)

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

  def testHalfBasic(self):
    x = np.linspace(-5, 20, 15, dtype=np.float16).reshape(1, 3, 5).astype(np.float16)
    y = np.linspace(20, -5, 15, dtype=np.float16).reshape(1, 3, 5).astype(np.float16)
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareGPU(x, y + 0.1, np.floor_divide, math_ops.floordiv)
    self._compareGPU(x, y, np.power, math_ops.pow)


class LogicalOpTest(test.TestCase):

  def _compareBinary(self, x, y, np_func, tf_func, use_gpu=False):
    np_ans = np_func(x, y)
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_val = self.evaluate(out)
    self.assertEqual(out.dtype, dtypes_lib.bool)
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def _not(self, x, use_gpu=False):
    np_ans = np.logical_not(x)
    with test_util.device(use_gpu=use_gpu):
      out = math_ops.logical_not(ops.convert_to_tensor(x))
      tf_val = self.evaluate(out)
    self.assertEqual(out.dtype, dtypes_lib.bool)
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def testScalar(self):
    data = [np.array([True]), np.array([False])]
    for use_gpu in [True, False]:
      for x in data:
        self._not(x, use_gpu)
      for x in data:
        for y in data:
          self._compareBinary(x, y, np.logical_and, math_ops.logical_and, use_gpu)
          self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
          self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor,
                              use_gpu)

  def testTensor(self):
    x = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    y = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    for use_gpu in [True, False]:
      self._not(x, use_gpu)
      self._compareBinary(x, y, np.logical_and, math_ops.logical_and, use_gpu)
      self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
      self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor, use_gpu)

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
        self._compareBinary(x, y, np.logical_and, math_ops.logical_and, use_gpu)
        self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
        self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor, use_gpu)

class XentTest(test.TestCase):

  def _npXent(self, features, labels, dim=-1):
    if dim == -1:
      dim = len(features.shape) - 1
    print ("dim ", dim)
    one_only_on_dim = list(features.shape)
    one_only_on_dim[dim] = 1
    e = np.exp(
        features - np.reshape(np.amax(features, axis=dim), one_only_on_dim))
    probs = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
    bp = (probs - labels)
    tmp = labels* np.log(probs + 1.0e-20)
    print ("before reduction ", tmp)
    l = -np.sum(tmp, axis=dim)
    return l, bp

  # TODO(b/123860949): The values are constant folded for XLA, so placeholders
  # are needed.
  def _testXent(self,
                np_features,
                np_labels,
                use_gpu=True,
                with_placeholders=False):
    np_loss, np_backprop = self._npXent(np_features, np_labels)
    with self.cached_session(use_gpu=use_gpu) as sess:
      if with_placeholders:
        features_placeholder = array_ops.placeholder(np_features.dtype)
        labels_placeholder = array_ops.placeholder(np_labels.dtype)
        loss, backprop = gen_nn_ops.softmax_cross_entropy_with_logits(
            labels=labels_placeholder, features=features_placeholder)
        tf_loss, tf_backprop = sess.run([loss, backprop],
                                        feed_dict={
                                            labels_placeholder: np_labels,
                                            features_placeholder: np_features
                                        })
      else:
        loss, backprop = gen_nn_ops.softmax_cross_entropy_with_logits(
            np_features, np_labels)
        tf_loss, tf_backprop = self.evaluate([loss, backprop])
    self.assertAllCloseAccordingToType(np_backprop, tf_backprop)

  def _testXentWrapper(self, np_features, np_labels, dim=-1, use_gpu=False):
    np_loss, _ = self._npXent(np_features, np_labels, dim=dim)
    with self.cached_session(use_gpu=use_gpu) as sess:
      loss = gen_nn_ops.softmax_cross_entropy_with_logits(
          labels=np_labels, logits=np_features, dim=dim)
      tf_loss = self.evaluate(loss)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)

  def _testAll(self, features, labels, with_placeholders=False):
    self._testXent(
        features, labels, use_gpu=True, with_placeholders=with_placeholders)


  def testFloat(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32),
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float32))

  def testHalf(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16),
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float16))


class AddNTest(test.TestCase):
  # AddN special-cases adding the first M inputs to make (N - M) divisible by 8,
  # after which it adds the remaining (N - M) tensors 8 at a time in a loop.
  # Test N in [1, 10] so we check each special-case from 1 to 9 and one
  # iteration of the loop.
  _MAX_N = 10

  def _supported_types(self):
    if test.is_gpu_available():
      return [
          dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64,
          dtypes.complex128, dtypes.int64, dtypes.bfloat16
      ]
    return [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
            dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64,
            dtypes.complex128, dtypes.bfloat16]

  def _buildData(self, shape, dtype):
    data = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testAddN(self):
    np.random.seed(12345)
    with self.session(use_gpu=True) as sess:
      for dtype in self._supported_types():
        for count in range(1, self._MAX_N + 1):
          data = [self._buildData((2, 2), dtype) for _ in range(count)]
          actual = self.evaluate(math_ops.add_n(data))
          expected = np.sum(np.vstack(
              [np.expand_dims(d, 0) for d in data]), axis=0)
          tol = 5e-3 if dtype == dtypes.float16 else 5e-7
          if dtype == dtypes.bfloat16:
            tol = 2e-2
          self.assertAllClose(expected, actual, rtol=tol, atol=tol)

  def testBigAddN(self):
    np.random.seed(12345)
    with self.session(use_gpu=True) as sess:
      for dtype in self._supported_types():
        for count in range(10, 31):
          data = [self._buildData((2, 2), dtype) for _ in range(count)]
          actual = self.evaluate(math_ops.add_n(data))
          expected = np.sum(np.vstack(
              [np.expand_dims(d, 0) for d in data]), axis=0)
          tol = 5e-2 if dtype in [dtypes.float16, dtypes.bfloat16] else 5e-6
          self.assertAllClose(expected, actual, rtol=tol, atol=tol)

class ResourceVariableOpsTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

  def tearDown(self):
    gc.collect()
    # This will only contain uncollectable garbage, i.e. reference cycles
    # involving objects with __del__ defined.
    self.assertEmpty(gc.garbage)
    super(ResourceVariableOpsTest, self).tearDown()

  @test_util.run_gpu_only
  def testGPUInt64(self):
    with context.eager_mode(), context.device("gpu:0"):
      v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.int64)
      self.assertAllEqual(1, v.numpy())

  def testEagerNameNotIdentity(self):
    with context.eager_mode():
      v0 = resource_variable_ops.ResourceVariable(1.0, name="a")
      v1 = resource_variable_ops.ResourceVariable(2.0, name="a")
      self.assertAllEqual(v0.numpy(), 1.0)
      self.assertAllEqual(v1.numpy(), 2.0)

  def testEagerNameNotNeeded(self):
    with context.eager_mode():
      v0 = resource_variable_ops.ResourceVariable(1.0)
      self.assertAllEqual(v0.numpy(), 1.0)

  def testReadVariableDtypeMismatchEager(self):
    with context.eager_mode():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      resource_variable_ops.assign_variable_op(handle, 1)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "Trying to read variable with wrong dtype. "
          "Expected float got int32"):
        _ = resource_variable_ops.read_variable_op(handle, dtype=dtypes.float32)

  def testEagerInitializedValue(self):
    with context.eager_mode():
      variable = resource_variable_ops.ResourceVariable(1.0, name="eager-init")
      self.assertAllEqual(variable.numpy(), 1.0)
      self.assertAllEqual(variable.initialized_value().numpy(), 1.0)

  def testInitializeVariableUsingInitializedValue(self):
    var1 = resource_variable_ops.ResourceVariable(1.0, name="var1")
    var2 = resource_variable_ops.ResourceVariable(var1.initialized_value(),
                                                  name="var2")
    self.assertAllEqual(var2.initialized_value(), 1.0)

  def testEagerBool(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(False, name="bool_test")
      self.assertAllEqual(bool(v), False)

  def testEagerDeepCopy(self):
    with context.eager_mode():
      init_value = np.ones((4, 4, 4))
      variable = resource_variable_ops.ResourceVariable(init_value,
                                                        name="init")

      copied_variable = copy.deepcopy(variable)
      self.assertEqual(variable.name, copied_variable.name)
      self.assertEqual(variable.shape, copied_variable.shape)
      self.assertEqual(variable.device, copied_variable.device)

      # The copied variable should have the same value as the original.
      self.assertAllEqual(variable.numpy(), copied_variable.numpy())

      # Updates to the copy should not be reflected in the original.
      copied_variable.assign(4 * np.ones((4, 4, 4)))
      self.assertNotAllEqual(variable.numpy(), copied_variable.numpy())

  def testVariableShape(self):
    v = resource_variable_ops.ResourceVariable([1., 1.])
    self.assertAllEqual(
        tensor_util.constant_value(
            resource_variable_ops.variable_shape(v.handle)),
        [2])

  def testAssignVariableDtypeMismatchEager(self):
    with context.eager_mode():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      resource_variable_ops.assign_variable_op(
          handle, constant_op.constant([1]))
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError, "Trying to assign variable with wrong "
          "dtype. Expected int32 got float"):
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([1.], dtype=dtypes.float32))

  @test_util.run_in_graph_and_eager_modes
  def testCreateRead(self):
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    self.evaluate(resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    value = self.evaluate(
        resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32))
    self.assertAllEqual(1, value)

  @test_util.run_in_graph_and_eager_modes
  def testManyAssigns(self):
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    create = resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32))
    with ops.control_dependencies([create]):
      first_read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.int32)
    with ops.control_dependencies([first_read]):
      write = resource_variable_ops.assign_variable_op(
          handle, constant_op.constant(2, dtype=dtypes.int32))
    with ops.control_dependencies([write]):
      second_read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.int32)
    f, s = self.evaluate([first_read, second_read])
    self.assertEqual(f, 1)
    self.assertEqual(s, 2)

  def testAssignAdd(self):
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    self.evaluate(resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    self.evaluate(resource_variable_ops.assign_add_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    read = self.evaluate(
        resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32))
    self.assertEqual(read, 2)
  @test_util.run_in_graph_and_eager_modes
  def testAssignAddMethod(self):
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign_add(1.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign_add(1.0, read_value=True)
    self.assertEqual(3.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign_add(1.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(4.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes
  def testAssignSubMethod(self):
    v = resource_variable_ops.ResourceVariable(3.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign_sub(1.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign_sub(1.0, read_value=True)
    self.assertEqual(1.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign_sub(1.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(0.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes
  def testScatterAdd(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_add(
            handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterSub(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_sub(
            handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[-1]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMul(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_mul(
            handle, [0], constant_op.constant([[5]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[5]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterDiv(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_div(
            handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[2]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMin(self):
    with ops.device("cpu:0"):
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      self.evaluate(
          resource_variable_ops.assign_variable_op(handle,
                                                   constant_op.constant(
                                                       [[6]],
                                                       dtype=dtypes.int32)))
      self.evaluate(
          resource_variable_ops.resource_scatter_min(handle, [0],
                                                     constant_op.constant(
                                                         [[3]],
                                                         dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMax(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_max(
            handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[6]])


  @test_util.run_in_graph_and_eager_modes
  def testScatterSubScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_sub(
            handle, [0], constant_op.constant(2, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[-1]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMulScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_mul(
            handle, [0], constant_op.constant(5, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[5]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterDivScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_div(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[2]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMinScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_min(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMaxScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_max(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[6]])

class GradientDescentOptimizerTest(test.TestCase):
  dtypes_ = [dtypes.float16, dtypes.float32]

  def testBasic(self):
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        optimizer = gradient_descent.GradientDescentOptimizer(3.0)
        sgd_op = optimizer.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))
        self.assertEqual(0, len(optimizer.variables()))

  def testBasicResourceVariable(self):
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))

  def testBasicCallableParams(self):
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        lr = lambda: 3.0
        sgd_op = gradient_descent.GradientDescentOptimizer(lr).apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))

  def testMinimizeResourceVariable(self):
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(var0, x) + var1
        loss = pred * pred
        sgd_op = gradient_descent.GradientDescentOptimizer(1.0).minimize(loss)
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
        np_grad = 2 * np_pred
        self.assertAllCloseAccordingToType(
            [[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - np_grad], self.evaluate(var1))

  def testMinimizeSparseResourceVariable(self):
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        pred += var1
        loss = pred * pred
        sgd_op = gradient_descent.GradientDescentOptimizer(1.0).minimize(loss)
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
        np_grad = 2 * np_pred
        self.assertAllCloseAccordingToType(
            [[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - np_grad], self.evaluate(var1))

  def testTensorLearningRate(self):
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        lrate = constant_op.constant(3.0)
        sgd_op = gradient_descent.GradientDescentOptimizer(
            lrate).apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))

  def testGradWrtRef(self):
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        opt = gradient_descent.GradientDescentOptimizer(3.0)
        values = [1.0, 3.0]
        vars_ = [variables.Variable([v], dtype=dtype) for v in values]
        grads_and_vars = opt.compute_gradients(vars_[0] + vars_[1], vars_)
        self.evaluate(variables.global_variables_initializer())
        for grad, _ in grads_and_vars:
          self.assertAllCloseAccordingToType([1.0], self.evaluate(grad))

  def testWithGlobalStep(self):
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        global_step = variables.Variable(0, trainable=False)
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step)
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params and global_step
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))
        self.assertAllCloseAccordingToType(1, self.evaluate(global_step))

  def testSparseBasic(self):
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
        var1 = variables.Variable([[3.0], [4.0]], dtype=dtype)
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant(
                [0.1], shape=[1, 1], dtype=dtype),
            constant_op.constant([0]),
            constant_op.constant([2, 1]))
        grads1 = indexed_slices.IndexedSlices(
            constant_op.constant(
                [0.01], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([2, 1]))
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0], [2.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([[3.0], [4.0]], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([[1.0 - 3.0 * 0.1], [2.0]],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([[3.0], [4.0 - 3.0 * 0.01]],
                                           self.evaluate(var1))


class BiasAddTestBase(test.TestCase):

  def _npBias(self, inputs, bias):
    assert len(bias.shape) == 1
    assert inputs.shape[-1] == bias.shape[0]
    return inputs + bias.reshape(([1] *
                                  (len(inputs.shape) - 1)) + [bias.shape[0]])

  def testNpBias(self):
    self.assertAllClose(
        np.array([[11, 22, 33], [41, 52, 63]]),
        self._npBias(
            np.array([[10, 20, 30], [40, 50, 60]]), np.array([1, 2, 3])))

  def _testBias(self, np_inputs, np_bias, use_gpu=False):
    np_val = self._npBias(np_inputs, np_bias)
    tf_val = nn_ops.bias_add(np_inputs, np_bias)
    self.assertAllCloseAccordingToType(np_val, tf_val)

  def _AtLeast3d(self, np_value):
    # fill the input value to at least 3-dimension
    if np_value.ndim < 3:
      return np.reshape(np_value, (1,) * (3 - np_value.ndim) + np_value.shape)
    return np_value

  def _NHWCToNCHW(self, np_value):
    # fill the input value to at least 3-dimension
    np_value = self._AtLeast3d(np_value)
    # move the last dimension to second
    np_dim = list(range(np_value.ndim))
    np_dim_new = list(np_dim[0:1]) + list(np_dim[-1:]) + list(np_dim[1:-1])
    return np.transpose(np_value, np_dim_new)

  def _NCHWToNHWC(self, np_value):
    assert len(np_value.shape) >= 3
    np_dim = list(range(np_value.ndim))
    # move the second dimension to the last
    np_dim_new = list(np_dim[0:1]) + list(np_dim[2:]) + list(np_dim[1:2])
    return np.transpose(np_value, np_dim_new)

  def _testBiasNCHW(self, np_inputs, np_bias, use_gpu):
    np_val = self._npBias(np_inputs, np_bias)
    np_inputs = self._NHWCToNCHW(np_inputs)
    tf_val = nn_ops.bias_add(np_inputs, np_bias, data_format="NCHW")
    tf_val = self._NCHWToNHWC(tf_val)
    self.assertAllCloseAccordingToType(self._AtLeast3d(np_val), tf_val)

  def _testAll(self, np_inputs, np_bias):
    if np_inputs.dtype in [np.float32, np.float16]:
      self._testBias(np_inputs, np_bias, use_gpu=True)
      self._testBiasNCHW(np_inputs, np_bias, use_gpu=True)

  def testFloatTypes(self):
    for t in [np.float32, np.float16]:
      self._testAll(
          np.random.rand(4, 3, 3).astype(t),
          np.random.rand(3).astype(t))
      self._testAll(
          np.random.rand(7, 5, 13).astype(t),
          np.random.rand(13).astype(t))
      self._testAll(
          np.random.rand(9, 9).astype(t),
          np.random.rand(9).astype(t))


  def _testGradient(self, np_input, bias, dtype, data_format, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      if data_format == "NCHW":
        np_input = self._NHWCToNCHW(np_input)
      input_tensor = constant_op.constant(
          np_input, shape=np_input.shape, dtype=dtype)
      bias_tensor = constant_op.constant(bias, shape=bias.shape, dtype=dtype)

      if dtype == dtypes.float16:
          delta = 4.0/1024
      else:
          delta = 1.0/1024

      output_tensor = nn_ops.bias_add(
          input_tensor, bias_tensor, data_format=data_format)
      tensor_jacob_t, tensor_jacob_n = gradient_checker.compute_gradient(
          input_tensor, np_input.shape, output_tensor, np_input.shape, delta=delta)
      bias_jacob_t, bias_jacob_n = gradient_checker.compute_gradient(
          bias_tensor, bias.shape, output_tensor, np_input.shape, delta=delta)

      # Test gradient of BiasAddGrad
      bias_add_grad = gradients_impl.gradients(
          nn_ops.l2_loss(output_tensor), bias_tensor)[0]
      grad_jacob_t, grad_jacob_n = gradient_checker.compute_gradient(
          output_tensor, np_input.shape, bias_add_grad, bias.shape, delta=delta)

      threshold = 5e-3
      if dtype == dtypes.float64:
        threshold = 1e-10
      if dtype == dtypes.float16:
          threshold = 2e-2
          # threshold for fp16 < threshold for fp32 since precision is lower.

      self.assertAllClose(tensor_jacob_t, tensor_jacob_n, threshold, threshold)
      self.assertAllClose(bias_jacob_t, bias_jacob_n, threshold, threshold)
      self.assertAllClose(grad_jacob_t, grad_jacob_n, threshold, threshold)

  @test_util.run_deprecated_v1
  def testGradientTensor2D(self):
    for (data_format, use_gpu) in [("NHWC", True)]:
      for dtype in [dtypes.float32, dtypes.float16]:
        np_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            dtype=dtype.as_numpy_dtype).reshape(3, 2)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  @test_util.run_deprecated_v1
  def testGradientTensor3D(self):
    for (data_format, use_gpu) in [("NHWC", True)]:
      for dtype in (dtypes.float32, dtypes.float64, dtypes.float16):
        print (data_format)
        np_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            10.0, 11.0, 12.0],
                            dtype=dtype.as_numpy_dtype).reshape(2, 3, 2)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  @test_util.run_deprecated_v1
  def testEmpty(self):
    np.random.seed(7)
    for shape in (0, 0), (2, 0), (0, 2), (4, 3, 0), (4, 0, 3), (0, 4, 3):
      self._testAll(np.random.randn(*shape), np.random.randn(shape[-1]))

  @test_util.run_deprecated_v1
  def testEmptyGradient(self):
    for (data_format, use_gpu) in ("NHWC", False), ("NHWC", True):
      for shape in (0, 0), (2, 0), (0, 2):
        self._testGradient(
            np.random.randn(*shape), np.random.randn(shape[-1]), dtypes.float64,
            data_format, use_gpu)

    for (data_format, use_gpu) in [("NHWC", False), ("NHWC", True),
                                   ("NCHW", False), ("NCHW", True)]:
      for shape in (4, 3, 0), (4, 0, 3), (0, 4, 3):
        self._testGradient(
            np.random.randn(*shape), np.random.randn(shape[-1]), dtypes.float64,
            data_format, use_gpu)

class LeakyReluTest(test.TestCase):

  def _npLeakyRelu(self, np_features, alpha=0.1):
    return np.maximum(np_features, alpha * np_features)

  def testNpLeakyRelu(self):
    self.assertAllClose(
        np.array([[-0.09, 0.7, -0.05, 0.3, -0.01],
                  [0.1, -0.03, 0.5, -0.07, 0.9]]),
        self._npLeakyRelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]]),
            alpha=0.1))

  def _testLeakyRelu(self, np_features, alpha):
    np_leaky_relu = self._npLeakyRelu(np_features, alpha)
    tf_leaky_relu = nn_ops.leaky_relu(np_features, alpha)
    self.assertAllClose(np_leaky_relu, tf_leaky_relu)
    self.assertShapeEqual(np_leaky_relu, tf_leaky_relu)

  def testNumbersCPU(self):
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testLeakyRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            alpha=0.2)

  def testNumbersGPU(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float32, np.float64]:
      self._testLeakyRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          alpha=0.1)
  def testGradGradFloat16(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float16
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.leaky_relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float16,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-4)

  def testGradientFloat16(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float16,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.leaky_relu, [x]))
      print(err)
    self.assertLess(err, 6e-2) # check if this is too high.

  def testGradientFloat32(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.leaky_relu, [x]))
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.leaky_relu, [x]))
    self.assertLess(err, 1e-10)

  def testGradGradFloat32(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.leaky_relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-4)

class ReluTest(test.TestCase):

  def _npRelu(self, np_features):
    return np.maximum(np_features, np.zeros(np_features.shape))

  def testNpRelu(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
        self._npRelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]])))

  def _testRelu(self, np_features):
    np_relu = self._npRelu(np_features)
    tf_relu = nn_ops.relu(np_features)
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, tf_relu)

  def testNumbersCPU(self):
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float32]:
      self._testRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNoElement(self):
    self._testRelu(np.array([[], []], dtype=np.float32))

  def testGradGradFloat32(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
          dy=  tape.gradient(y, x)
          return dy

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-4)

  def testGradGradFloat16(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float16
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
          dy=  tape.gradient(y, x)
          return dy

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float16,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-4)


class Relu6Test(test.TestCase):

  def _npRelu6(self, np_features):
    sixes = np.copy(np_features)
    sixes.fill(6.0)
    return np.minimum(
        np.maximum(np_features, np.zeros(np_features.shape)), sixes)

  def testNpRelu6(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 6.0], [0.1, 0.0, 6.0, 0.0, 0.9]]),
        self._npRelu6(
            np.array([[-0.9, 0.7, -0.5, 0.3, 6.0], [0.1, -0.3, 6.5, -0.7,
                                                    0.9]])))

  def _testRelu6(self, np_features):
    np_relu6 = self._npRelu6(np_features)
    tf_relu6 = nn_ops.relu6(np_features)
    self.assertAllClose(np_relu6, tf_relu6)
    self.assertShapeEqual(np_relu6, tf_relu6)

  def testNumbersCPU(self):
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testRelu6(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float, np.double]:
      print(t)
      self._testRelu6(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  # The gradient test for ReLU6 is a bit tricky as the derivative is
  # not well defined at around zero and six and we want to avoid that
  # in terms of input values.
  def testGradientFloat32(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x]))
    self.assertLess(err, 1e-4)

  def testGradientFloat16(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float16,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x]))
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x]))
    self.assertLess(err, 1e-10)




class SoftmaxTest(test.TestCase):

  def _npSoftmax(self, features, dim=-1, log=False):
    if dim == -1:
      dim = len(features.shape) - 1
    one_only_on_dim = list(features.shape)
    one_only_on_dim[dim] = 1
    is_fp16 = features.dtype == np.float16
    if is_fp16:
      # Do the compute in fp32 and cast the input back to fp32.
      features = features.astype(np.float32)
    e = np.exp(features - np.reshape(
        np.amax(
            features, axis=dim), one_only_on_dim))
    softmax = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
    if log:
      res = np.log(softmax)
    else:
      res = softmax
    if is_fp16:
      res = res.astype(np.float16)
    return res

  def _testSoftmax(self, np_features, dim=-1, log=False, use_gpu=False):
    # A previous version of the code checked the op name rather than the op type
    # to distinguish between log and non-log.  Use an arbitrary name to catch
    # this bug in future.
    name = "arbitrary"
    np_softmax = self._npSoftmax(np_features, dim=dim, log=log)
    with self.cached_session(use_gpu=use_gpu):
      if log:
        tf_softmax = nn_ops.log_softmax(np_features, axis=dim, name=name)
      else:
        tf_softmax = nn_ops.softmax(np_features, axis=dim, name=name)
      out = self.evaluate(tf_softmax)
    self.assertAllCloseAccordingToType(np_softmax, out)
    self.assertShapeEqual(np_softmax, tf_softmax)
    if not log:
      # Bonus check: the softmaxes should add to one in dimension dim.
      sum_along_dim = np.sum(out, axis=dim)
      self.assertAllCloseAccordingToType(
          np.ones(sum_along_dim.shape), sum_along_dim)

  def _testAll(self, features):
    self._testSoftmax(features, use_gpu=True)
    self._testSoftmax(features, log=True, use_gpu=True)
    self._testOverflow(use_gpu=True)

  def testNpSoftmax(self):
    features = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    # Batch 0: All exps are 1.  The expected result is
    # Softmaxes = [0.25, 0.25, 0.25, 0.25]
    # LogSoftmaxes = [-1.386294, -1.386294, -1.386294, -1.386294]
    #
    # Batch 1:
    # exps = [1., 2.718, 7.389, 20.085]
    # sum = 31.192
    # Softmaxes = exps / sum = [0.0320586, 0.08714432, 0.23688282, 0.64391426]
    # LogSoftmaxes = [-3.44019 , -2.44019 , -1.44019 , -0.44019]
    np_sm = self._npSoftmax(np.array(features))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, 0.25],
                  [0.0320586, 0.08714432, 0.23688282, 0.64391426]]),
        np_sm,
        rtol=1.e-5,
        atol=1.e-5)
    np_lsm = self._npSoftmax(np.array(features), log=True)
    self.assertAllClose(
        np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                  [-3.4401897, -2.4401897, -1.4401897, -0.4401897]]),
        np_lsm,
        rtol=1.e-5,
        atol=1.e-5)

  def _testOverflow(self, use_gpu=False):
    if use_gpu:
      type = np.float32  # pylint: disable=redefined-builtin
    else:
      type = np.float64  # pylint: disable=redefined-builtin
    max = np.finfo(type).max  # pylint: disable=redefined-builtin
    features = np.array([[1., 1., 1., 1.], [max, 1., 2., 3.]]).astype(type)
    with self.cached_session(use_gpu=use_gpu):
      tf_log_softmax = nn_ops.log_softmax(features)
      out = self.evaluate(tf_log_softmax)
    self.assertAllClose(
        np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                  [0, -max, -max, -max]]),
        out,
        rtol=1.e-5,
        atol=1.e-5)

  def testFloat(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32))
  def testHalf(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16))

class BaseReductionTest(test.TestCase):

  def _tf_reduce(self, x, reduction_axes, keepdims):
    raise NotImplementedError()

  def _np_reduce(self, x, reduction_axes, keepdims):
    raise NotImplementedError()

  def _makeIncremental(self, shape, dtype):
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype.as_numpy_dtype)
    if dtype.is_complex:
      data -= 2j * data
    return data

  def _makeRandom(self, shape, dtype):
    data = np.random.rand(*shape).astype(dtype.as_numpy_dtype)
    if dtype.is_complex:
      data -= 2j * data
    return data

  def _compareGradient(self, x, reduction_axes, rtol=1e-8, atol=1e-8):
    if reduction_axes is not None and np.shape(reduction_axes) == (1,):
      # Test scalar reduction_axes argument
      self._compareGradient(x, reduction_axes[0], rtol=rtol, atol=atol)
    with self.cached_session(use_gpu=True):
      t = ops.convert_to_tensor(x)
      su = self._tf_reduce(t, reduction_axes, False)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, x.shape, su, su.get_shape().as_list(), x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=rtol, atol=atol)

  def _compareGradientAxes(self, x, rtol=1e-8, atol=1e-8):
    self._compareGradient(x, None, rtol=rtol, atol=atol)
    self._compareGradient(x, [], rtol=rtol, atol=atol)
    self._compareGradient(x, 0, rtol=rtol, atol=atol)
    self._compareGradient(x, [1], rtol=rtol, atol=atol)
    self._compareGradient(x, [2], rtol=rtol, atol=atol)
    self._compareGradient(x, [1, 2], rtol=rtol, atol=atol)
    self._compareGradient(x, [0, 1, 2, 3], rtol=rtol, atol=atol)


class ConcatOpTest(test.TestCase):

  def _testRandom(self, dtype):
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    # Random number of tensors, but always > 1.
    num_tensors = np.random.randint(2, 10)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    params = {}
    if dtype == dtypes.bfloat16:
      dtype_feed = dtypes.float32
    else:
      dtype_feed = dtype
    with self.cached_session(use_gpu=True):
      p = []
      for i in np.arange(num_tensors):
        input_shape = shape
        input_shape[concat_dim] = np.random.randint(1, 5)
        placeholder = array_ops.placeholder(dtype_feed, shape=input_shape)
        p.append(placeholder)

        t = dtype_feed.as_numpy_dtype
        params[placeholder] = np.random.rand(*input_shape).astype(t)

      if dtype != dtype_feed:
        concat_inputs = [math_ops.cast(p_i, dtype) for p_i in p]
      else:
        concat_inputs = p
      c = array_ops.concat(concat_inputs, concat_dim)
      if dtype != dtype_feed:
        c = math_ops.cast(c, dtype_feed)
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    cur_offset = 0

    for i in np.arange(num_tensors):
      # The index into the result is the ':' along all dimensions
      # except the concat_dim. slice(0, size) is used for ':', and
      # a list of slices is used to index into result.
      ind = [slice(0, params[p[i]].shape[j]) for j in np.arange(5)]
      ind[concat_dim] = slice(cur_offset,
                              cur_offset + params[p[i]].shape[concat_dim])
      cur_offset += params[p[i]].shape[concat_dim]
      if dtype == dtype_feed:
        self.assertAllEqual(result[tuple(ind)], params[p[i]])
      else:
        self.assertAllClose(result[tuple(ind)], params[p[i]], 0.01)


  @test_util.run_deprecated_v1
  def testRandom(self):
    self._testRandom(dtypes_lib.bfloat16.as_numpy_dtype)
    self._testRandom(dtypes.float16)
    self._testRandom(dtypes.float32)
    self._testRandom(dtypes.int32)
    self._testRandom(dtypes.int64)

  def _RunAndVerifyGradientsRandom(self, dtype=dtypes.float32.as_numpy_dtype):
    # Random dims of rank 5
    input_shape = np.random.randint(1, 5, size=5)
    # Random number of tensors
    num_tensors = np.random.randint(12, 20)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    concat_dim_sizes = np.random.randint(1, 5, size=num_tensors)
    with test_util.use_gpu():
      inp = []
      inp_tensors = []
      for x in concat_dim_sizes:
        shape = input_shape
        shape[concat_dim] = x
        t = np.random.rand(*shape).astype(dtype)
        inp.append(t)
        inp_tensors.append(
            constant_op.constant(t.flatten(), shape=shape,
                                 dtype=dtype))
      c = array_ops.concat(inp_tensors, concat_dim)
      output_shape = input_shape
      output_shape[concat_dim] = concat_dim_sizes.sum()
      grad_inp = np.random.rand(*output_shape).astype(dtype)
      grad_tensor = constant_op.constant(grad_inp.flatten(), shape=output_shape)
      grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = array_ops.concat(grad, concat_dim)
      result = self.evaluate(concated_grad)

    self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testGradientsRandom(self):
    for _ in range(5):
      self._RunAndVerifyGradientsRandom()
      self._RunAndVerifyGradientsRandom(dtypes_lib.bfloat16.as_numpy_dtype)


class TileTest(test.TestCase, parameterized.TestCase):

  def testSimple(self):
    # multiples could be int32 or int64
    for dtype in [dtypes.int32, dtypes.int64]:
      for in_type in [np.float32, dtypes.bfloat16.as_numpy_dtype]:
        with self.cached_session(use_gpu=True):
          inp = np.random.rand(4, 1).astype(in_type)
          a = constant_op.constant(inp)
          tiled = array_ops.tile(a, constant_op.constant([1, 4], dtype=dtype))
          result = self.evaluate(tiled)
        self.assertEqual(result.shape, (4, 4))
        self.assertEqual([4, 4], tiled.get_shape())
        self.assertTrue((result == np.tile(inp, (1, 4))).all())


class PadOpTest(test.TestCase):

  def _npPad(self, inp, paddings, mode, constant_values=0):
    mode = mode.lower()
    if mode == "constant":
      return np.pad(inp, paddings, mode=mode, constant_values=constant_values)
    else:
      return np.pad(inp, paddings, mode=mode)

  def _testPad(self, np_inputs, paddings, mode, constant_values):
    np_val = self._npPad(np_inputs, paddings, mode=mode,
                         constant_values=constant_values)
    with self.cached_session(use_gpu=True):
      tf_val = array_ops.pad(np_inputs, paddings, mode=mode,
                             constant_values=constant_values)
      out = self.evaluate(tf_val)
    self.assertAllEqual(np_val, out)
    self.assertShapeEqual(np_val, tf_val)

  def _testPadGradient(self, x, a, mode, constant_values):
    with self.cached_session(use_gpu=True):
      inx = ops.convert_to_tensor(x)
      xs = list(x.shape)
      ina = ops.convert_to_tensor(a)
      y = array_ops.pad(inx, ina, mode=mode, constant_values=constant_values)
      # Expected y's shape to be:
      ys = list(np.array(x.shape) + np.sum(np.array(a), axis=1))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, xs, y, ys, x_init_value=x)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _testPaddingAll(self, np_inputs, paddings, constant_values):
    for mode in ("CONSTANT", "REFLECT", "SYMMETRIC", "reflect", "symmetric",
                 "constant"):
      # Zero-sized input is not allowed for REFLECT mode, but we still want
      # zero-sized input test cases for the other modes.
      if np_inputs.size or mode.upper() != "REFLECT":
        self._testPad(np_inputs, paddings, mode=mode,
                      constant_values=constant_values)
        if np_inputs.dtype == np.float32:
          self._testPadGradient(np_inputs, paddings, mode=mode,
                             constant_values=constant_values)

  @test_util.run_deprecated_v1
  def testPadding(self):
    for t in [np.float32]:
      self._testPaddingAll(np.random.rand(2, 5).astype(t), [[1, 0], [2, 0]], 0.0)
      self._testPaddingAll(np.random.rand(2, 3, 4).astype(t),
                    [[0, 0], [0, 0], [0, 0]], -1234.0)
      self._testPaddingAll(np.random.rand(0, 3, 4).astype(t),
                    [[0, 0], [2, 1], [2, 3]], 0.0)

class RandomOpsCorrectnessTest(test.TestCase):
    shapes = [[1,5], [2,6,5], [5,3,6,2], [100, 100]]
    seeds = [2, 16, 1582, 12]
    minvals = [-10.0, 0.5, 10.0, 1000.0]
    maxvals = [-5.0, 1.0, 20.0, 2000.0]
    means = [-5.0, 1.0, 100.0, 1000.0]
    stddevs = [0.1, 1.0, 10.0 ,100.0]

    def _testRandomDefault(self, rnfunc, shape, seed, dtype):
        with test_util.device(use_gpu=False):
            ref = rnfunc(shape, seed=seed, dtype=dtype)
        with test_util.device(use_gpu=True):
            result = rnfunc(shape, seed=seed, dtype=dtype)
        if (dtype == dtypes.float16):
            self.assertAllClose(result,ref, atol=1e-3)
        else:
            self.assertAllClose(result,ref, atol=1e-5)

    def _testRandomMinvalMaxval(self, rnfunc, shape, seed, minvalue, maxvalue, dtype):
        with test_util.device(use_gpu=False):
            ref = rnfunc(shape, seed=seed, minval=minvalue, maxval=maxvalue, dtype=dtype)
        with test_util.device(use_gpu=True):
            result = rnfunc(shape, seed=seed, minval=minvalue, maxval=maxvalue, dtype=dtype)
        if (dtype == dtypes.float16):
            self.assertAllClose(result,ref, atol=1e-3)
        else:
            self.assertAllClose(result,ref, atol=1e-5)

    def _testRandomMeanStd(self, rnfunc, shape, seed, mean, stddev, dtype):
        with test_util.device(use_gpu=False):
            ref = rnfunc(shape, seed=seed, mean=mean, stddev=stddev, dtype=dtype)
        with test_util.device(use_gpu=True):
            result = rnfunc(shape, seed=seed, mean=mean, stddev=stddev, dtype=dtype)
        if (dtype == dtypes.float16):
            self.assertAllClose(result,ref, atol=5e-1)
            #TODO: this tolerance is too high.
            # For the random normal case & truncated normal, this particular test has 1/10000 mismatched
            # element in the (100, 100) shape.
        else:
            self.assertAllClose(result,ref, atol=1e-5)

    def testRandomUniformCorrectness_1(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            for i in range(len(self.shapes)):
                self._testRandomDefault(random_ops.random_uniform, self.shapes[i], self.seeds[i], dtype)

    def testRandomUniformCorrectness_2(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            for i in range(len(self.shapes)):
                self._testRandomMinvalMaxval(random_ops.random_uniform, self.shapes[i], self.seeds[i], self.minvals[i], self.maxvals[i], dtype)

    def testRandomNormalCorrectness_1(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            for i in range(len(self.shapes)):
                self._testRandomDefault(random_ops.random_normal, self.shapes[i], self.seeds[i], dtype)

    def testRandomNormalCorrectness_2(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            for i in range(len(self.shapes)):
                self._testRandomMeanStd(random_ops.random_normal, self.shapes[i], self.seeds[i], self.means[i], self.stddevs[i], dtype)

    def testRandomTruncatedCorrectness_1(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            for i in range(len(self.shapes)):
                self._testRandomDefault(random_ops.truncated_normal, self.shapes[i], self.seeds[i], dtype)

    def testRandomTruncatedCorrectness_2(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            for i in range(len(self.shapes)):
                self._testRandomMeanStd(random_ops.truncated_normal, self.shapes[i], self.seeds[i], self.means[i], self.stddevs[i], dtype)

class StatelessRandomOpsCorrectnessTest(test.TestCase):
    shapes = [[1,5], [2,6,5], [5,3,6,2], [100, 100]]
    seeds = [[2, 1], [16, 12], [1582, 10230], [12,23101]]
    dtypes = [dtypes.float32, dtypes.float32, dtypes.half, dtypes.half]

    def _testStatelessRandomDefault(self, rnfunc, shape, seed, dtype):
        with test_util.device(use_gpu=False):
            ref = rnfunc(shape=shape, seed=seed, dtype=dtype)
        with test_util.device(use_gpu=True):
            result = rnfunc(shape=shape, seed=seed, dtype=dtype)
        if(dtype==dtypes.float32):
            self.assertAllClose(result,ref, atol=1e-5)
        elif(dtype==dtypes.float16):
            self.assertAllClose(result,ref, atol=1e-3)

    def testRandomUniformCorrectness_1(self):
        for i in range(len(self.shapes)):
            self._testStatelessRandomDefault(raw_ops.StatelessRandomUniform, self.shapes[i], self.seeds[i], self.dtypes[i])

    def testRandomNormalCorrectness_1(self):
        for i in range(len(self.shapes)):
            self._testStatelessRandomDefault(raw_ops.StatelessRandomNormal, self.shapes[i], self.seeds[i], self.dtypes[i])

    def testTruncatedNormalCorrectness_1(self):
        for i in range(len(self.shapes)):
            self._testStatelessRandomDefault(raw_ops.StatelessTruncatedNormal, self.shapes[i], self.seeds[i], self.dtypes[i])

class StatelessRandomOpsCorrectnessTestV2(test.TestCase):
    shapes = [[1,5], [2,6,5], [5,3,6,2], [100, 100]]
    seeds = [[2, 1], [16, 12], [1582, 10230], [12,23101]]
    key = [[2], [16], [1582], [12]]
    counters = [[23,11], [11,23], [2000312, 0], [0,0]]
    itypes = [dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64]
    dtypes = [dtypes.float32, dtypes.float32, dtypes.half, dtypes.half]
    
    def _testStatelessRandomDefault(self, rnfunc, shape, seed, dtype):
        with test_util.device(use_gpu=False):
            ref = rnfunc(shape=shape, seed=seed, dtype=dtype)
        with test_util.device(use_gpu=True):
            result = rnfunc(shape=shape, seed=seed, dtype=dtype)
        if(dtype==dtypes.float32):
            self.assertAllClose(result,ref, atol=1e-5)
        elif(dtype==dtypes.float16):
            self.assertAllClose(result,ref, atol=1e-3)
    
    def _testStatelessRandomDefaultV2(self, rnfunc, shape, key, counter, dtype, alg=1):
        with test_util.device(use_gpu=False):
            ref = rnfunc(shape=shape, key=[key[0]], alg=alg, counter=counter)
        with test_util.device(use_gpu=True):
            result = rnfunc(shape=shape, key=[key[0]], alg=alg, counter=counter)
        self.assertAllClose(result,ref, atol=1e-5)
    
    def _testStatelessRandomUniformFullIntV2(self, rnfunc, shape, key, counter, dtype):
        with test_util.device(use_gpu=False):
            ref = rnfunc(shape=shape, alg=1, key=key, counter=counter, dtype=dtype)
        with test_util.device(use_gpu=True):
            result = rnfunc(shape=shape, alg=1, key=key, counter=counter, dtype=dtype)
        self.assertEqual(result.shape,ref.shape)
    
    def testRandomUniformCorrectness_1(self):
        for i in range(len(self.shapes)):
            self._testStatelessRandomDefault(raw_ops.StatelessRandomUniform, self.shapes[i], self.seeds[i], self.dtypes[i])
    
    def testRandomUniformV2Correctness_1(self):
        for i in range(len(self.shapes)):
            self._testStatelessRandomDefaultV2(raw_ops.StatelessRandomUniformV2, self.shapes[i], self.seeds[i], self.counters[i], self.dtypes[i])
    
    def testRandomNormalCorrectness_1(self):
        for i in range(len(self.shapes)):
            self._testStatelessRandomDefault(raw_ops.StatelessRandomNormal, self.shapes[i], self.seeds[i], self.dtypes[i])
    
    def testRandomNormalV2Correctness_1(self):
        for i in range(len(self.shapes)):
            self._testStatelessRandomDefaultV2(raw_ops.StatelessRandomNormalV2, self.shapes[i], self.seeds[i], self.counters[i], self.dtypes[i])
    
    def testTruncatedNormalCorrectness_1(self):
        for i in range(len(self.shapes)):
            self._testStatelessRandomDefault(raw_ops.StatelessTruncatedNormal, self.shapes[i], self.seeds[i], self.dtypes[i])
    
    def testTruncatedNormalV2Correctness_1(self):
        for i in range(len(self.shapes)):
            self._testStatelessRandomDefaultV2(raw_ops.StatelessTruncatedNormalV2, self.shapes[i], self.seeds[i], self.counters[i], self.dtypes[i])
    
    def testRandomUniformFullIntV2Functional_1(self):
        for i in range(len(self.shapes)):
            self._testStatelessRandomUniformFullIntV2(raw_ops.StatelessRandomUniformFullIntV2, self.shapes[i], self.key[i], self.counters[i], self.itypes[i])

class BatchNormTest(test.TestCase):

  def _batch_norm(self, x, mean, var, offset, scale, epsilon):
    # We compute the batch norm manually in this function because
    # nn_impl.batch_normalization does not support float16 yet.
    # TODO(reedwm): Add float16 support to nn_impl.batch_normalization.
    inv = math_ops.rsqrt(var + epsilon) * scale
    y = math_ops.cast(x, scale.dtype) * inv + (offset - mean * inv)
    return math_ops.cast(y, x.dtype)

  def _running_mean(self, old_mean, new_val, factor):
    if factor == 1.0:
      return new_val
    else:
      return (1.0 - factor) * old_mean + factor * new_val

  def _training_ref(self, x, scale, offset, old_mean, old_var,
                    exponential_avg_factor, epsilon, data_format):
    if data_format not in ['NHWC', 'NCHW']:
      raise ValueError('data_format must be NCHW or NHWC, '
                       'got %s.' % data_format)
    if data_format == 'NCHW':
      x = array_ops.transpose(x, [0, 2, 3, 1])
    batch_mean, batch_var = nn_impl.moments(
        math_ops.cast(x, scale.dtype), [0, 1, 2], keep_dims=False)

    y = self._batch_norm(x, batch_mean, batch_var, offset, scale, epsilon)
    if data_format == 'NCHW':
      y = array_ops.transpose(y, [0, 3, 1, 2])

    # This is for Bessel's correction. tf.nn.moments uses n, instead of n-1, as
    # the denominator in the formula to calculate variance, while
    # tf.compat.v1.nn.fused_batch_norm has Bessel's correction built in.
    sample_size = math_ops.cast(
        array_ops.size(x) / array_ops.size(scale), scale.dtype)
    batch_var_corrected = batch_var * sample_size / (
        math_ops.maximum(sample_size - 1.0, 1.0))

    mean = self._running_mean(old_mean, batch_mean, exponential_avg_factor)
    var = self._running_mean(old_var, batch_var_corrected,
                             exponential_avg_factor)
    return self.evaluate(y), self.evaluate(mean), self.evaluate(var)

  def _test_training(self,
                     x_shape,
                     x_dtype,
                     scale_shape,
                     scale_dtype,
                     use_gpu=True,
                     exponential_avg_factor=1.0,
                     data_format='NHWC'):
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(x_dtype)
    scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    if exponential_avg_factor == 1.0:
      old_mean_val = None
      old_var_val = None
    else:
      old_mean_val = np.random.random_sample(scale_shape).astype(scale_dtype)
      old_var_val = np.random.random_sample(scale_shape).astype(scale_dtype)

    with self.cached_session(use_gpu=use_gpu) as sess:
      x = constant_op.constant(x_val, name='x')
      scale = constant_op.constant(scale_val, name='scale')
      offset = constant_op.constant(offset_val, name='offset')
      epsilon = 0.001
      y, mean, var = nn_impl.fused_batch_norm(
          x,
          scale,
          offset,
          mean=old_mean_val,
          variance=old_var_val,
          epsilon=epsilon,
          exponential_avg_factor=exponential_avg_factor,
          data_format=data_format,
          is_training=True)
      y_val, mean_val, var_val = self.evaluate([y, mean, var])
      y_ref, mean_ref, var_ref = self._training_ref(x, scale, offset,
                                                    old_mean_val, old_var_val,
                                                    exponential_avg_factor,
                                                    epsilon, data_format)
    y_atol = 2e-3 if x_dtype == np.float16 else 1e-3
    self.assertAllClose(y_ref, y_val, atol=y_atol)
    self.assertAllClose(mean_ref, mean_val, atol=1e-3)
    self.assertAllClose(var_ref, var_val, atol=1e-3)

  def _inference_ref(self, x, scale, offset, mean, var, epsilon, data_format):
    if data_format not in ['NHWC', 'NCHW']:
      raise ValueError('data_format must be NCHW or NHWC, '
                       'got %s.' % data_format)
    if data_format == 'NCHW':
      x = array_ops.transpose(x, [0, 2, 3, 1])
    y = self._batch_norm(x, mean, var, offset, scale, epsilon)
    if data_format == 'NCHW':
      y = array_ops.transpose(y, [0, 3, 1, 2])
    return self.evaluate(y)

  def _test_inference(self,
                      x_shape,
                      x_dtype,
                      scale_shape,
                      scale_dtype,
                      use_gpu=True,
                      exponential_avg_factor=1.0,
                      data_format='NHWC'):
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(x_dtype)
    scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    mean_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    var_val = np.random.random_sample(scale_shape).astype(scale_dtype)

    with self.cached_session(use_gpu=use_gpu) as sess:
      x = constant_op.constant(x_val, name='x')
      scale = constant_op.constant(scale_val, name='scale')
      offset = constant_op.constant(offset_val, name='offset')
      mean = constant_op.constant(mean_val, name='mean')
      var = constant_op.constant(var_val, name='variance')
      epsilon = 0.001
      y, _, _ = nn_impl.fused_batch_norm(
          x,
          scale,
          offset,
          mean=mean,
          variance=var,
          epsilon=epsilon,
          exponential_avg_factor=exponential_avg_factor,
          data_format=data_format,
          is_training=False)
      y_val = self.evaluate(y)
      y_ref = self._inference_ref(x, scale, offset, mean, var, epsilon,
                                  data_format)
    # An atol value of 1e-3 is too small for float16's, because some adjacent
    # float16 values that y_val can take are greater than 1e-3 apart, e.g.
    # 2.16602 and 2.16797.
    atol = 2e-3 if x_dtype == np.float16 else 1e-3
    self.assertAllClose(y_ref, y_val, atol=atol)

  def _runtests(self, x_shape, is_training, gradient_test=False):
    use_gpu_vals = [False]
    if test.is_gpu_available(cuda_only=True):
      use_gpu_vals += [True]
    factors = [
        1.0,
    ]
    if compat.forward_compatible(2020, 3, 6):
      factors += [
          0.6,
      ]
    for dtype in [np.float16, np.float32]:
      for use_gpu in use_gpu_vals:
        for data_format in ['NHWC', 'NCHW']:
          if data_format == 'NHWC':
            scale_shape = x_shape[-1:]
          else:
            scale_shape = x_shape[1:2]
          for exponential_avg_factor in factors:
            if gradient_test:
              self._test_gradient(
                  x_shape,
                  dtype,
                  scale_shape,
                  np.float32,
                  use_gpu=use_gpu,
                  data_format=data_format,
                  is_training=is_training,
                  exponential_avg_factor=exponential_avg_factor)
            else:
              if is_training:
                self._test_training(
                    x_shape,
                    dtype,
                    scale_shape,
                    np.float32,
                    use_gpu=use_gpu,
                    data_format=data_format,
                    exponential_avg_factor=exponential_avg_factor)
              else:
                self._test_inference(
                    x_shape,
                    dtype,
                    scale_shape,
                    np.float32,
                    use_gpu=use_gpu,
                    data_format=data_format,
                    exponential_avg_factor=exponential_avg_factor)

  def testInferenceShape1(self):
    x_shape = [1, 1, 6, 1]
    self._runtests(x_shape, False)

  def testInferenceShape2(self):
    x_shape = [1, 1, 6, 2]
    self._runtests(x_shape, False)

  def testInferenceShape3(self):
    x_shape = [1, 2, 1, 6]
    self._runtests(x_shape, False)

  def testInferenceShape4(self):
    x_shape = [27, 131, 127, 6]
    self._runtests(x_shape, False)

  def testInferenceShape5(self):
    x_shape = [0, 131, 127, 6]
    self._runtests(x_shape, False)

  def testTrainingShape1(self):
    x_shape = [1, 1, 6, 1]
    self._runtests(x_shape, True)

  def testTrainingShape2(self):
    x_shape = [1, 1, 6, 2]
    self._runtests(x_shape, True)

  def testTrainingShape3(self):
    x_shape = [1, 2, 1, 6]
    self._runtests(x_shape, True)

  def testTrainingShape4(self):
    x_shape = [27, 131, 127, 6]
    self._runtests(x_shape, True)

  def _test_gradient(self,
                     x_shape,
                     x_dtype,
                     scale_shape,
                     scale_dtype,
                     use_gpu=True,
                     data_format='NHWC',
                     is_training=True):
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(x_dtype)
    scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)

    with self.cached_session(use_gpu=use_gpu):
      x = constant_op.constant(x_val, name='x')
      scale = constant_op.constant(scale_val, name='scale')
      offset = constant_op.constant(offset_val, name='offset')
      if is_training:
        pop_mean = None
        pop_var = None
      else:
        pop_mean = np.random.random_sample(scale_shape).astype(scale_dtype)
        pop_var = np.random.random_sample(scale_shape).astype(scale_dtype)
      y, _, _ = nn_impl.fused_batch_norm(
          x,
          scale,
          offset,
          mean=pop_mean,
          variance=pop_var,
          data_format=data_format,
          is_training=is_training)
      if x_dtype != np.float16:
        err_x = gradient_checker.compute_gradient_error(x, x_shape, y, x_shape)
        err_scale = gradient_checker.compute_gradient_error(
            scale, scale_shape, y, x_shape)
        err_offset = gradient_checker.compute_gradient_error(
            offset, scale_shape, y, x_shape)
      else:
        x32 = constant_op.constant(x_val, name='x32', dtype=dtypes.float32)
        y32, _, _ = nn_impl.fused_batch_norm(
            x32,
            scale,
            offset,
            mean=pop_mean,
            variance=pop_var,
            data_format=data_format,
            is_training=is_training)
        err_x = self._compute_gradient_error_float16(x, x32, x_shape, y, y32,
                                                     x_shape)
        err_scale = self._compute_gradient_error_float16(
            scale, scale, scale_shape, y, y32, x_shape)
        err_offset = self._compute_gradient_error_float16(
            offset, offset, scale_shape, y, y32, x_shape)

    x_err_tolerance = 2e-3 if x_dtype == np.float16 else 1e-3
    scale_err_tolerance = 1e-3
    self.assertLess(err_x, x_err_tolerance)
    self.assertLess(err_scale, scale_err_tolerance)
    self.assertLess(err_offset, scale_err_tolerance)

  @test_util.run_deprecated_v1
  def testBatchNormGradShape1(self):
    for is_training in [True, False]:
      x_shape = [1, 1, 6, 1]
      for dtype in [np.float32]:
        if test.is_gpu_available(cuda_only=True):
          self._test_gradient(
              x_shape,
              dtype, [1],
              np.float32,
              use_gpu=True,
              data_format='NHWC',
              is_training=is_training)
          self._test_gradient(
              x_shape,
              dtype, [1],
              np.float32,
              use_gpu=True,
              data_format='NCHW',
              is_training=is_training)
        self._test_gradient(
            x_shape,
            dtype, [1],
            np.float32,
            use_gpu=False,
            data_format='NHWC',
            is_training=is_training)
        self._test_gradient(
            x_shape,
            dtype, [1],
            np.float32,
            use_gpu=False,
            data_format='NCHW',
            is_training=is_training)

  @test_util.run_deprecated_v1
  def testBatchNormGradShape2(self):
    for is_training in [True, False]:
      x_shape = [1, 1, 6, 2]
      for dtype in [np.float32]:
        if test.is_gpu_available(cuda_only=True):
          self._test_gradient(
              x_shape,
              dtype, [2],
              np.float32,
              use_gpu=True,
              data_format='NHWC',
              is_training=is_training)
        self._test_gradient(
            x_shape,
            dtype, [2],
            np.float32,
            use_gpu=False,
            data_format='NHWC',
            is_training=is_training)

  @test_util.run_deprecated_v1
  def testBatchNormGradShape3(self):
    for is_training in [True, False]:
      x_shape = [1, 2, 1, 6]
      for dtype in [np.float32]:
        if test.is_gpu_available(cuda_only=True):
          self._test_gradient(
              x_shape,
              dtype, [2],
              np.float32,
              use_gpu=True,
              data_format='NCHW',
              is_training=is_training)
        self._test_gradient(
            x_shape,
            dtype, [2],
            np.float32,
            use_gpu=False,
            data_format='NCHW',
            is_training=is_training)

class SumReductionTest(BaseReductionTest):

  def _tf_reduce(self, x, reduction_axes, keepdims):
    return math_ops.reduce_sum(x, reduction_axes, keepdims)

  def _np_reduce(self, x, reduction_axes, keepdims):
    if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                      np.ndarray):
      reduction_axes = tuple(reduction_axes)
    return np.sum(x, axis=reduction_axes, keepdims=keepdims)

  def testAxesType(self):
    for dtype in [dtypes.int64, dtypes.int32]:
      with self.cached_session(use_gpu=True) as sess:
        v = math_ops.reduce_sum([0, 0], constant_op.constant(0, dtype=dtype))
        tf_v = self.evaluate(v)
      self.assertAllEqual(tf_v, 0)

  def testFloat32(self):
    for _ in range(5):
      size_x = int(2**np.random.uniform(0, 15))
      size_y = int(2**np.random.uniform(0, 15))
      if size_x * size_y > 1e7:
        size_y = int(1e7 / size_x)
      if size_x % 2 :
          size_x = size_x + 1
      if (size_y % 2) :
          size_y = size_y + 1
      arr = np.ones([size_x, size_y], dtype=np.float32)
      col_sum = np.sum(arr, axis=0)
      row_sum = np.sum(arr, axis=1)
      full_sum = np.sum(arr, axis=-1, keepdims=True)
      with self.cached_session(use_gpu=True) as sess:
        tf_row_sum = self._tf_reduce(arr, 1, False)
        tf_col_sum = self._tf_reduce(arr, 0, False)
        tf_full_sum = self._tf_reduce(arr, -1,keepdims=True)
        tf_out_col = self.evaluate(tf_col_sum)
        tf_out_row = self.evaluate(tf_row_sum)
        tf_out_full = self.evaluate(tf_full_sum)
      self.assertAllClose(col_sum, tf_out_col)
      self.assertAllClose(row_sum, tf_out_row)
      self.assertAllClose(full_sum, tf_out_full)

    for size_x in [4, 16, 32]:
      for size_y in [4, 16, 32]:
        for size_z in [4, 16, 32]:
          arr = np.ones([size_x, size_y, size_z], dtype=np.float32)
          sum_y = np.sum(arr, axis=1)
          sum_xz = np.sum(arr, axis=(0, 2))

          with self.cached_session(use_gpu=True) as sess:
            tf_sum_xz = self._tf_reduce(arr, [0, 2], False)
            tf_sum_y = self._tf_reduce(arr, 1, False)
            tf_out_sum_xz, tf_out_sum_y = self.evaluate([tf_sum_xz, tf_sum_y])
          self.assertAllClose(sum_y, tf_out_sum_y)
          self.assertAllClose(sum_xz, tf_out_sum_xz)

class MinMaxOpTest(test.TestCase):

  def _compare(self, x, y, use_gpu):
    np_min, np_max = np.minimum(x, y), np.maximum(x, y)
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      omin, omax = math_ops.minimum(inx, iny), math_ops.maximum(inx, iny)
      tf_min, tf_max = self.evaluate([omin, omax])
    self.assertAllEqual(np_min, tf_min)
    self.assertAllEqual(np_max, tf_max)

  def testBasic(self):
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(1, 3, 2) * 100.
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      self._compare(x.astype(t), y.astype(t), use_gpu=False)
      self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testDifferentShapes(self):
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(2) * 100.  # should broadcast
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      self._compare(x.astype(t), y.astype(t), use_gpu=False)
      self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testScalar(self):
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(1).item() * 100.  # should broadcast
    # dropped np.float64, int64 because TF automatically converts to 32 bit
    for t in [np.float32, np.int32]:
      self._compare(x.astype(t), t(y), use_gpu=False)
      self._compare(x.astype(t), t(y), use_gpu=True)

  def _compareGradientX(self, func, x, y):
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, s, out, s, x_init_value=x)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, func, x, y):
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, s, out, s, x_init_value=y)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  @test_util.run_deprecated_v1
  def testGradients(self):
    x = np.random.rand(1, 3, 2) * 100.
    # ensure x != y
    y = x + (np.random.randint(2, size=x.shape) - .5) * 2  # -1 or +1
    self._compareGradientX(math_ops.maximum, x, y)
    self._compareGradientY(math_ops.maximum, x, y)
    self._compareGradientX(math_ops.minimum, x, y)
    self._compareGradientY(math_ops.minimum, x, y)

class MPSFillTest(test.TestCase):

  def _compare(self, dims, val, np_ans, use_gpu):
    ctx = context.context()
    device = "GPU:0" if (use_gpu and ctx.num_gpus()) else "CPU:0"
    with ops.device(device):
      tf_ans = array_ops.fill(dims, val, name="fill")
      out = tf_ans.numpy()
    self.assertAllClose(np_ans, out)

  def _compareAll(self, dims, val, np_ans):
    self._compare(dims, val, np_ans, False)
    self._compare(dims, val, np_ans, True)

  def testFillFloat(self):
    np_ans = np.array([[3.1415] * 3] * 2).astype(np.float32)
    self._compareAll([2, 3], np_ans[0][0], np_ans)



@test_util.run_all_in_graph_and_eager_modes
class SquaredDifferenceTest(test_util.TensorFlowTestCase):

  def testSquaredDifference(self):
    for dtype in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
      y = np.array([-3, -2, -1], dtype=dtype)
      z = (x - y) * (x - y)
      with test_util.device(use_gpu=True):
        z_tf = self.evaluate(math_ops.squared_difference(x, y))
        self.assertAllClose(z, z_tf)

  def testComplexSquaredDifference(self):
    for dtype in [np.complex64, np.complex128]:
      x = np.array([[1 + 3j, 2 + 2j, 3 + 1j], [4 - 1j, 5 - 2j, 6 - 3j]],
                   dtype=dtype)
      y = np.array([-3 + 1j, -2 + 2j, -1 + 3j], dtype=dtype)
      z = np.conj(x - y) * (x - y)
      with test_util.device(use_gpu=False):
        z_tf = self.evaluate(math_ops.squared_difference(x, y))
        self.assertAllClose(z, z_tf)

class MPSOnesLikeTest(test.TestCase):

  def testOnesLike(self):
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64
    ]:
      numpy_dtype = dtype.as_numpy_dtype
      # Creates a tensor of non-zero values with shape 2 x 3.
      d = constant_op.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
      # Constructs a tensor of zeros of the same dimensions and type as "d".
      z_var = array_ops.ones_like(d)
      # Test that the type is correct
      self.assertEqual(z_var.dtype, dtype)
      z_value = z_var.numpy()

      # Test that the value is correct
      self.assertTrue(np.array_equal(z_value, np.array([[1] * 3] * 2)))
      self.assertEqual([2, 3], z_var.get_shape())

class SelectOpTest(test.TestCase):

  def _compare(self, fn, c, x, y, use_gpu):
    np_ans = np.where(c, x, y)
    with test_util.device(use_gpu=use_gpu):
      out = fn(c, x, y)
      tf_ans = self.evaluate(out)
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self,
                        fn,
                        c,
                        x,
                        y,
                        numeric_gradient_type=None,
                        x_init_value=None):
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = fn(c, inx, iny)
      s = list(np.shape(c))
      if x_init_value is None:
        x_init_value = x
      if x.shape != y.shape:
        x_init_value = np.broadcast_to(y, x.shape)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, s, out, s, x_init_value=x_init_value)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = fn(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, s, outf, s, x_init_value=xf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, fn, c, x, y, numeric_gradient_type=None):
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = fn(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, s, out, s, x_init_value=x, delta=1.0)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = fn(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inyf, s, outf, s, x_init_value=yf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _testScalar(self, fn):
    c = True
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testScalar(self):
    self._testScalar(array_ops.where)
    self._testScalar(array_ops.where_v2)

  def _testScalarBroadcast(self, fn, c, x, y):
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testScalarBroadcast(self):
    c = True
    # where_v2 only
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(3, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)

  def _testBasic(self, fn):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [ np.float32 ]:
      xt = x.astype(t)
      yt = y.astype(t)
      # self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float32]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testBasic(self):
    self._testBasic(array_ops.where)
    self._testBasic(array_ops.where_v2)

  def _testBasicBroadcast(self, fn, c, x, y):
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testBasicBroadcast(self):
    c0 = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    c1 = np.random.randint(0, 2, 2).astype(np.bool).reshape(1, 1, 2)
    c2 = np.random.randint(0, 2, 3).astype(np.bool).reshape(1, 3, 1)
    c3 = np.random.randint(0, 2, 1).astype(np.bool).reshape(1, 1, 1)
    for c in [c0, c1, c2, c3]:
      # where_v2 only
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 3, 1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 2) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 2) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(3, 2) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)

  def _testGradients(self, fn):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [np.float16, np.float32, np.float64]:
      xt = x.astype(t)
      yt = y.astype(t)
      if t == np.float16:
        # Compare fp16 theoretical gradients to fp32 numerical gradients,
        # since fp16 numerical gradients are too imprecise unless great
        # care is taken with choosing the inputs and the delta. This is
        # a weaker check (in particular, it does not test the op itself,
        # only its gradient), but it's much better than nothing.
        self._compareGradientX(fn, c, xt, yt, np.float)
        self._compareGradientY(fn, c, xt, yt, np.float)
      else:
        self._compareGradientX(fn, c, xt, yt)
        self._compareGradientY(fn, c, xt, yt)

  @test_util.run_deprecated_v1
  def testGradients(self):
    self._testGradients(array_ops.where)
    self._testGradients(array_ops.where_v2)

  @test_util.run_deprecated_v1
  def testGradientsBroadcast(self):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    for t in [np.float32, np.float64]:
      # where_v2 only
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 3, 1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 2) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 2) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(3, 2) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))

  def _testShapeMismatch(self, fn):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(2, 5, 3) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      with self.assertRaises(ValueError):
        fn(c, xt, yt)

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    self._testShapeMismatch(array_ops.where)
    self._testShapeMismatch(array_ops.where_v2)

  def _testEmptyTensor(self, fn):
    c = np.random.randint(0, 3, 0).astype(np.bool).reshape(1, 3, 0)
    x = np.random.rand(1, 3, 0) * 100
    y = np.random.rand(1, 3, 0) * 100
    z_expected = np.zeros((1, 3, 0), dtype=np.float32)
    with self.cached_session():
      xt = x.astype(np.float32)
      yt = y.astype(np.float32)
      z = fn(c, xt, yt).eval()
      self.assertAllEqual(z_expected, z)

  @test_util.run_deprecated_v1
  def testEmptyTensor(self):
    self._testEmptyTensor(array_ops.where)
    self._testEmptyTensor(array_ops.where_v2)

  def _testNan(self, fn):
    with self.cached_session():
      for c in False, True:
        for a in 7.0, np.nan:
          for b in 5.0, np.nan:
            x = fn(c, a, b).eval()
            y = a if c else b
            self.assertEqual(np.isnan(x), np.isnan(y))

  @test_util.run_deprecated_v1
  def testNan(self):
    """Verify that nans don't propagate where they shouldn't."""
    self._testNan(array_ops.where)
    self._testNan(array_ops.where_v2)

class ZerosLikeTest(test.TestCase):

  def _compareZeros(self, dtype, use_gpu):
    # Creates a tensor of non-zero values with shape 2 x 3.
    # NOTE(kearnes): The default numpy dtype associated with tf.string is
    # np.object (and can't be changed without breaking a lot things), which
    # causes a TypeError in constant_op.constant below. Here we catch the
    # special case of tf.string and set the numpy dtype appropriately.
    if dtype == dtypes_lib.string:
      numpy_dtype = np.string_
    else:
      numpy_dtype = dtype.as_numpy_dtype
    d = constant_op.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
    # Constructs a tensor of zeros of the same dimensions and type as "d".
    z_var = array_ops.zeros_like(d)
    # Test that the type is correct
    self.assertEqual(z_var.dtype, dtype)
    # Test that the shape is correct
    self.assertEqual([2, 3], z_var.get_shape())

    # Test that the value is correct
    z_value = z_var.numpy()
    self.assertFalse(np.any(z_value))
    self.assertEqual((2, 3), z_value.shape)

  def testZerosLikeCPU(self):
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64,
    ]:
      self._compareZeros(dtype, use_gpu=False)

  def testZerosLikeGPU(self):
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.bool, dtypes_lib.int64,
    ]:
      self._compareZeros(dtype, use_gpu=True)

  def testZerosLikeDtype(self):
    # Make sure zeros_like works even for dtypes that cannot be cast between
    shape = (3, 5)
    dtypes = np.float32, np.complex64
    for in_type in dtypes:
      x = np.arange(15).astype(in_type).reshape(*shape)
      for out_type in dtypes:
        y = array_ops.zeros_like(x, dtype=out_type).numpy()
        self.assertEqual(y.dtype, out_type)
        self.assertEqual(y.shape, shape)
        self.assertAllEqual(y, np.zeros(shape, dtype=out_type))

class MpsTest(test.TestCase):
  def _compareCpu(self, x, y, np_func, tf_func, also_compare_variables=False):
    np_ans = np_func(x, y)
    with test_util.force_cpu():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = self.evaluate(out)
      # Test that the op takes precedence over numpy operators.
      np_left = self.evaluate(tf_func(x, iny))
      np_right = self.evaluate(tf_func(inx, y))

      if also_compare_variables:
        var_x = variables.Variable(x)
        var_y = variables.Variable(y)
        self.evaluate(variables.global_variables_initializer())
        print(type(x), type(y), type(var_x), type(var_y))
        print(type(tf_func(x, var_y)), type(tf_func(var_x, y)))
        np_var_left = self.evaluate(tf_func(x, var_y))
        np_var_right = self.evaluate(tf_func(var_x, y))

    if np_ans.dtype != np.object:
      self.assertAllClose(np_ans, tf_cpu)
      self.assertAllClose(np_ans, np_left)
      self.assertAllClose(np_ans, np_right)
      if also_compare_variables:
        self.assertAllClose(np_ans, np_var_left)
        self.assertAllClose(np_ans, np_var_right)
    self.assertShapeEqual(np_ans, out)

  def _inv(self, x):
    return 1.0 / x

  def _rsqrt(self, x):
    return self._inv(np.sqrt(x))

  def _sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  def _log_sigmoid(self, x):
    return np.log(self._sigmoid(x))

  def _replace_domain_error_with_inf(self, fn):

    def func(x):
      try:
        return fn(x)
      except ValueError as e:
        if "domain error" in str(e):
          return np.inf * np.ones_like(x)
        else:
          raise e

    return func

  def _compareTanhGrad(self, x, y):

    default = gen_math_ops.tanh_grad(x, y)

    with test_util.device(use_gpu=False):
      cpu = gen_math_ops.tanh_grad(x, y)

    self.assertAllClose(cpu, default)


  def testTanhGrad(self):
    x = np.random.uniform(-2.0, 2.0, size = [4, 4]).astype(np.float32)
    y = np.random.uniform(-2.0, 2.0, size = [4, 4]).astype(np.float32)
    self._compareTanhGrad(x, y)

  _GRAD_TOL = {
      dtypes_lib.float16: 1e-3,
      dtypes_lib.float32: 1e-3,
      dtypes_lib.complex64: 1e-2,
      dtypes_lib.float64: 1e-5,
      dtypes_lib.complex128: 1e-4
  }
  def _compareGradientX(self,
                        x,
                        y,
                        np_func,
                        tf_func,
                        numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      xs = list(x.shape)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, xs, out, zs, x_init_value=x)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = tf_func(inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, xs, outf, zs, x_init_value=xf, delta=1e-3)
        jacob_n = jacob_n.astype(x.dtype)
      tol = self._GRAD_TOL[dtypes_lib.as_dtype(x.dtype)]
      self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGradientY(self,
                        x,
                        y,
                        np_func,
                        tf_func,
                        numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      ys = list(np.shape(y))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, ys, out, zs, x_init_value=y)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = tf_func(inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inyf, ys, outf, zs, x_init_value=yf)
        jacob_n = jacob_n.astype(x.dtype)
    tol = self._GRAD_TOL[dtypes_lib.as_dtype(x.dtype)]
    self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)


  def compareUnaryGradient_CPU_GPU(self, inx, func, testName):

    with test_util.force_cpu():
      with backprop.GradientTape() as t:
        t.watch(inx)
        y = func(inx)
      cpu_gradient =  t.gradient(y, inx)
      print(testName, " (CPU) = ", cpu_gradient)

    with test_util.force_gpu():
      with backprop.GradientTape() as t:
        t.watch(inx)
        y = func(inx)
      gpu_gradient =  t.gradient(y, inx)
      print(testName, " (GPU) = ", gpu_gradient)

    tol = self._GRAD_TOL[dtypes_lib.as_dtype(inx.dtype)]
    self.assertAllClose(cpu_gradient, gpu_gradient, rtol=tol, atol=tol)

  def _sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  def _compareGpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with test_util.use_gpu():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = self.evaluate(out)
    self.assertAllClose(np_ans, tf_gpu)
    self.assertShapeEqual(np_ans, out)

  def _compareBoth(self, x, y, np_func, tf_func, also_compare_variables=False):
    self._compareCpu(x, y, np_func, tf_func, also_compare_variables)
    self._compareGpu(x, y, np_func, tf_func)

  def _compare(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with test_util.use_gpu():
      out = tf_func(ops.convert_to_tensor(x), ops.convert_to_tensor(y))
      tf_ans = self.evaluate(out)
    self.assertAllEqual(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testGradGrad(self):
    np.random.seed(7)
    shape = (5,)
    dtype_tols = [(np.float32, 5e-4), (np.float64, 1e-6), (np.complex64, 5e-4),
                  (np.complex128, 1e-6)]
    op_range = [
        (gen_math_ops.tanh_grad, [-2, 2]),
    ]

    def rand(dtype, real_range):
      x = np.random.uniform( real_range[0], real_range[1], size=shape[0]).astype(dtype)
      return x

    for op, real_range in op_range:
      with self.cached_session():
        for dtype, tol in dtype_tols:
          x = constant_op.constant(rand(dtype, real_range))
          y = constant_op.constant(rand(dtype, real_range))
          z = op(x, y)
          grads = gradient_checker.compute_gradient(
              [x, y], [shape, shape],
              z,
              shape,
              x_init_value=[rand(dtype, real_range),
                            rand(dtype, real_range)])
          if isinstance(grads, tuple):
            grads = [grads]
          for analytical, numerical in grads:
            self.assertAllClose(analytical, numerical, rtol=tol, atol=tol)

  def testFloatCompareTensor(self):
    x = np.linspace(-15, 15, 6).reshape(1, 3, 2)
    y = np.linspace(20, -10, 6).reshape(1, 3, 2)
    for t in [np.float32, np.float16]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(xt, yt, np.less, math_ops.less)
      self._compare(xt, yt, np.less_equal, math_ops.less_equal)
      self._compare(xt, yt, np.greater, math_ops.greater)
      self._compare(xt, yt, np.greater_equal, math_ops.greater_equal)
      self._compare(xt, yt, np.equal, math_ops.equal)
      self._compare(xt, yt, np.not_equal, math_ops.not_equal)

  def testFloatBasic(self):
    x = np.linspace(-5, 20, 30).reshape(1, 2, 3, 5).astype(np.float32)
    y = np.linspace(20, -5, 30).reshape(1, 2, 3, 5).astype(np.float32)
    self._compareBoth(x, y, np.add, math_ops.add, True)
    self._compareBoth(x, y, np.subtract, math_ops.subtract, True)
    self._compareBoth(x, y, np.multiply, math_ops.multiply, True)
    self._compareBoth(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y + 0.1, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testHalfBasic(self):
    x = np.linspace(-5, 20, 30).reshape(1, 2, 3, 5).astype(np.float16)
    y = np.linspace(20, -5, 30).reshape(1, 2, 3, 5).astype(np.float16)
    self._compareBoth(x, y, np.add, math_ops.add, True)
    self._compareBoth(x, y, np.subtract, math_ops.subtract, True)
    self._compareBoth(x, y, np.multiply, math_ops.multiply, True)
    self._compareBoth(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y + 0.1, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testIntBasic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int32)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int32)
    self._compareBoth(x, y, np.add, math_ops.add)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.mod, math_ops.mod)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y, np.floor_divide, _FLOORDIV)
    self._compareBoth(x, y, np.mod, _MOD)
    # _compareBoth tests on GPU only for floating point types, so test
    # _MOD for int32 on GPU by calling _compareGpu
    self._compareGpu(x, y, np.mod, _MOD)
  def testZeroElementBinaryOp(self):
    x = array_ops.ones([0, 3])
    y = 4.0
    self._compareBoth(x, y, np.add, math_ops.add, True)
    self._compareBoth(x, y, np.subtract, math_ops.subtract, True)
    self._compareBoth(x, y, np.multiply, math_ops.multiply, True)
    self._compareBoth(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testAssignMethod(self):
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign(2.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign(3.0, read_value=True)
    self.assertEqual(3.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign(4.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(4.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes
  def testAssignIncompatibleShape(self):
    v = resource_variable_ops.ResourceVariable([0, 1, 2, 3])
    self.evaluate(v.initializer)
    pattern = re.compile("shapes must be equal", re.IGNORECASE)
    with self.assertRaisesRegexp(Exception, pattern):
      self.evaluate(v.assign_add(1))

  def _compareUnaryCpu(self, x, np_func, tf_func, grad_rtol=None, grad_atol=None):
    if grad_rtol is None:
      grad_rtol = _default_tolerance(x.dtype)
    if grad_atol is None:
       grad_atol = _default_tolerance(x.dtype)
    np_ans = np_func(x)
    with self.cached_session(use_gpu=False):
      inx = ops.convert_to_tensor(x)
      if x.dtype in (np.float32, np.float64,
                     dtypes_lib.bfloat16.as_numpy_dtype):
        y = 1.1 * tf_func(inx)
        np_ans *= 1.1
      else:
        y = tf_func(inx)
      tf_cpu = self.evaluate(y)
      self.assertShapeEqual(np_ans, y)
      if x.dtype == np.float16:
        self.assertAllClose(np_ans, tf_cpu, rtol=1e-3, atol=1e-3)
      elif x.dtype == dtypes_lib.bfloat16.as_numpy_dtype:
        self.assertAllClose(np_ans, tf_cpu, rtol=1e-2, atol=1e-2)
      else:
        self.assertAllClose(np_ans, tf_cpu)

      if x.dtype in (np.complex64, np.complex128) and tf_func == math_ops.sign:
        return  # Return early

      if x.dtype == np.float16:
        s = list(np.shape(x))
        jacob_t, _ = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x)
        xf = x.astype(np.float)
        inxf = ops.convert_to_tensor(xf)
        yf = tf_func(inxf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, s, yf, s, x_init_value=xf, delta=1e-2)
        jacob_n = jacob_n.astype(np.float16)
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)
      elif x.dtype in (np.float32, np.complex64):
        s = list(np.shape(x))
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x, delta=1e-3)
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)
      elif x.dtype in (np.float64, np.complex128):
        s = list(np.shape(x))
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x, delta=1e-5)
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)
  def _compareUnaryGpu(self, x, np_func, tf_func):
    np_ans = np_func(x)
    with test_util.use_gpu():
      result = tf_func(ops.convert_to_tensor(x))
      tf_gpu = self.evaluate(result)
    if x.dtype == np.float16:
      self.assertAllClose(np_ans, tf_gpu, rtol=1e-3, atol=1e-3)
    else:
      self.assertAllClose(np_ans, tf_gpu)
  def _compareUnaryBoth(self, x, np_func, tf_func):
    self._compareUnaryGpu(x, np_func, tf_func)

  def compareConv2d(self, input, filter, padding, format = 'NHWC', dilations = None):
    stride = 2

    strides = [stride, stride]

    with test_util.force_gpu():
      gpu = nn_ops.conv2d(input = input, filter = filter, strides = strides, padding = padding,
                          data_format = format, dilations = dilations)

    with test_util.force_cpu():
      if format == 'NCHW':
        input = array_ops.transpose(input, [0, 2, 3, 1])
        if not isinstance(padding, str):
          padding = [ padding[0], padding[2], padding[3], padding[1] ]
      cpu = nn_ops.conv2d(input = input, filter = filter, strides = strides, padding = padding,
                          data_format = 'NHWC', dilations = dilations)
      if format == 'NCHW':
        cpu = array_ops.transpose(cpu, [0, 3, 1, 2])

      if math_ops.reduce_any(math_ops.not_equal(cpu, gpu)):
        print('Error: padding: {0} format: {1} dilations: {2}'.format(padding, format, dilations))
        print("CPU: ", cpu)
        print("GPU: ", gpu)
      else:
        print('Passed: padding: {0} format: {1} dilations: {2}'.format(padding, format, dilations))
        print("CPU: ", cpu)
        print("GPU: ", gpu)

    self.assertAllEqual(cpu, gpu)

  def testConvolution(self):
    input = constant_op.constant([[
      [ [ 1], [ 2.0 ], [3.0], [4.0] ],
      [ [ 6], [ 7 ], [8], [9] ],
      [ [ 10], [ 11 ], [12], [13] ],
      [ [ 14], [ 15 ], [16], [17] ]

      ]])

    input2 = constant_op.constant([[
      [ [ 1], [ 2.0 ], [3.0], [4.0], [5.0] ],
      [ [ 6], [ 7 ], [8], [9], [15.0] ],
      [ [ 10], [ 11 ], [12], [13], [25.0] ],
      [ [ 14], [ 15 ], [16], [17], [35.0] ]

      ]])


    input4 = constant_op.constant([[
      [ [ 1], [ 2.0 ], [3.0], [4.0], [5.0], [ 1], [ 2.0 ] ],
      [ [ 6], [ 7 ], [8], [9], [15.0], [ 1], [ 2.0 ] ],
      [ [ 10], [ 11 ], [12], [13], [25.0], [ 1], [ 2.0 ] ],
      [ [ 14], [ 15 ], [16], [17], [35.0], [ 1], [ 2.0 ] ],
      [ [ 6], [ 7 ], [8], [9], [15.0], [ 1], [ 2.0 ] ],
      [ [ 10], [ 11 ], [12], [13], [25.0], [ 1], [ 2.0 ] ]

      ]])


    print("input: ", input)


    ## (2,2,1,1)
    filter2x2 = constant_op.constant(
    [
    [[ [1.0] ], [ [1] ]],
    [[ [1.0] ], [ [1] ]],
    ],
    )

    ## (3,2,1,1)
    filter3x2 = constant_op.constant(
    [
    [[ [1.0] ], [ [1] ]],
    [[ [1.0] ], [ [1] ]],
    [[ [1.0] ], [ [1] ]]
    ],
    )

    ## (4,2,1,1)
    filter4x2 = constant_op.constant(
    [
    [[ [1.0] ], [ [1] ]],
    [[ [1.0] ], [ [1] ]],
    [[ [1.0] ], [ [1] ]],
    [[ [1.0] ], [ [1] ]]
    ],
    )

    ## (5,2,1,1)
    filter5x2 = constant_op.constant(
    [
    [[ [1.0] ], [ [1] ]],
    [[ [1.0] ], [ [1] ]],
    [[ [1.0] ], [ [1] ]],
    [[ [1.0] ], [ [1] ]],
    [[ [1.0] ], [ [1] ]]
    ],
    )


    print("filter2x2: ", filter2x2)

    self.compareConv2d(input, filter2x2, "VALID")
    self.compareConv2d(input, filter3x2, "VALID")
    self.compareConv2d(input, filter4x2, "VALID")
    self.compareConv2d(input, filter5x2, "VALID")

    self.compareConv2d(input, filter2x2, "SAME")
    self.compareConv2d(input, filter3x2, "SAME")
    self.compareConv2d(input, filter4x2, "SAME")
    self.compareConv2d(input, filter5x2, "SAME")

    self.compareConv2d(input2, filter2x2, "VALID")
    self.compareConv2d(input2, filter2x2, "SAME")

    pad_top = 2
    pad_bottom = 3
    pad_left = 1
    pad_right = 5
    self.compareConv2d(input2, filter2x2, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    self.compareConv2d(input2, filter2x2, "VALID", dilations = [2, 2])
    self.compareConv2d(input2, filter2x2, "SAME", dilations = [2, 2])

    self.compareConv2d(input4, filter2x2, "VALID", dilations = [2, 3])
    self.compareConv2d(input4, filter2x2, "SAME", dilations = [3, 2])

    self.compareConv2d(input4, filter3x2, "VALID", dilations = [2, 3])
    self.compareConv2d(input4, filter3x2, "SAME", dilations = [3, 2])

    self.compareConv2d(input4, filter5x2, "VALID", dilations = [2, 3])
    self.compareConv2d(input4, filter5x2, "SAME", dilations = [3, 2])
    self.compareConv2d(input2, filter2x2, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], dilations = [2, 2])

    input3 = constant_op.constant([[
     [[ 1, 2.0, 3.0, 4.0, 5.0],
      [ 6, 7, 8, 9, 15 ],
      [ 10, 11, 12, 13, 25.0 ],
      [ 14, 15, 16, 17, 35.0]]

    ]])

    self.compareConv2d(input3, filter2x2, "VALID", "NCHW")
    self.compareConv2d(input3, filter2x2, "SAME", "NCHW")
    self.compareConv2d(input3, filter2x2, [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]], "NCHW")


  def compareTranspose(self, input, perm):
    with test_util.force_gpu():
      gpu = array_ops.transpose(input, perm)

    with test_util.force_cpu():
      cpu = array_ops.transpose(input, perm)

      if math_ops.reduce_any(math_ops.not_equal(cpu, gpu)):
          print('Error')
          print("CPU: ", cpu)
          print("GPU: ", gpu)
      else:
        print('Passed')

    self.assertAllEqual(cpu, gpu)


  def testTranspose(self):

    for dtype in [dtypes.float32, dtypes.bfloat16]:
      input = tf.convert_to_tensor(np.arange(0.0, 5 * 2 * 13), dtype=dtype);
      input = array_ops.reshape(input, [5, 2, 13])

      self.compareTranspose(input, [1, 2, 0])
      self.compareTranspose(input, [0, 2, 1])
      self.compareTranspose(input, [2, 0, 1])
      self.compareTranspose(input, [2, 1, 0])

      input = tf.convert_to_tensor(np.arange(0.0, 2 * 4 * 3 * 5), dtype=dtype);
      input = array_ops.reshape(input, [2, 4, 3, 5])

      self.compareTranspose(input, [1, 0, 2, 3])
      self.compareTranspose(input, [0, 3, 1, 2])
      self.compareTranspose(input, [3, 2, 1, 0])

  def testUnaryHalfBasic(self):
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float16)
    w = x - x.min() + 1.02  # all greater than 1
    y = (x + .5).astype(np.float16)  # no zero
    z = (x + 15.5).astype(np.float16)  # all positive
    k = np.arange(-0.90, 0.90, 0.25).astype(np.float16)  # between -1 and 1

    self._compareUnaryBoth(x, np.abs, math_ops.abs)
    self._compareUnaryBoth(x, np.abs, _ABS)
    self._compareUnaryBoth(x, np.negative, math_ops.negative)
    self._compareUnaryBoth(x, np.negative, _NEG)
    self._compareUnaryBoth(y, self._inv, math_ops.reciprocal)
    self._compareUnaryBoth(z, np.log, math_ops.log)
    self._compareUnaryBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareUnaryBoth(z, np.sqrt, math_ops.sqrt)
    self._compareUnaryBoth(z, self._rsqrt, math_ops.rsqrt)
    self._compareUnaryBoth(x, np.exp, math_ops.exp)
    self._compareUnaryBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareUnaryBoth(x, np.square, math_ops.square)
    self._compareUnaryBoth(y, np.sign, math_ops.sign)
    self._compareUnaryBoth(x, np.tanh, math_ops.tanh)

  def testUnaryFloatBasic(self):
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32)
    w = x - x.min() + 1.02  # all greater than 1
    y = (x + .5).astype(np.float32)  # no zero
    z = (x + 15.5).astype(np.float32)  # all positive
    k = np.arange(-0.90, 0.90, 0.25).astype(np.float32)  # between -1 and 1

    self._compareUnaryBoth(x, np.abs, math_ops.abs)
    self._compareUnaryBoth(x, np.abs, _ABS)
    self._compareUnaryBoth(x, np.negative, math_ops.negative)
    self._compareUnaryBoth(x, np.negative, _NEG)
    self._compareUnaryBoth(y, self._inv, math_ops.reciprocal)
    self._compareUnaryBoth(z, np.log, math_ops.log)
    self._compareUnaryBoth(x, np.square, math_ops.square)
    self._compareUnaryBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareUnaryBoth(z, np.sqrt, math_ops.sqrt)
    self._compareUnaryBoth(z, self._rsqrt, math_ops.rsqrt)
    self._compareUnaryBoth(x, np.exp, math_ops.exp)
    self._compareUnaryBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareUnaryBoth(z, np.log1p, math_ops.log1p)
    self._compareUnaryBoth(x, np.square, math_ops.square)
    self._compareUnaryBoth(y, np.sign, math_ops.sign)
    self._compareUnaryBoth(x, np.tanh, math_ops.tanh)

    x = np.array([ 0.5, 0.7 ], np.float32)
    inx = ops.convert_to_tensor(x);

    print("\nsigmoidGrad:\n")

    self.compareUnaryGradient_CPU_GPU(inx, gen_math_ops.sigmoid, "sigmoidGrad")

    gradient = gen_math_ops.sigmoid_grad(gen_math_ops.sigmoid(inx), constant_op.constant(1.0))
    print("gen_math_ops.sigmoid_grad(y) = ", gradient)

  def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
    x = (1 + np.linspace(0, 5, np.prod(xs))).astype(dtype).reshape(xs)
    y = (1 + np.linspace(0, 5, np.prod(ys))).astype(dtype).reshape(ys)
    self._compareCpu(x, y, np_func, tf_func)
    if x.dtype in (np.float16, np.float32, np.float64):
      self._compareGpu(x, y, np_func, tf_func)

  def _testBCastByFunc(self, funcs, xs, ys):
    dtypes = [
        np.float32,
    ]
    for dtype in dtypes:
      for (np_func, tf_func) in funcs:
        self._compareBCast(xs, ys, dtype, np_func, tf_func)
        self._compareBCast(ys, xs, dtype, np_func, tf_func)

  def _testBCastA(self, xs, ys):
    funcs = [
        (np.add, math_ops.add),
        (np.add, _ADD),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastB(self, xs, ys):
    funcs = [
        (np.subtract, math_ops.subtract),
        (np.subtract, _SUB),
        (np.power, math_ops.pow),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastC(self, xs, ys):
    funcs = [
        (np.multiply, math_ops.multiply),
        (np.multiply, _MUL),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastD(self, xs, ys):
    funcs = [
        (np.true_divide, math_ops.truediv),
        (np.true_divide, _TRUEDIV),
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
    self._testBCastA([2, 3, 2], [2])

  def testBCast_1B(self):
    self._testBCastB([1, 3, 2], [2])

  def testBCast_1C(self):
    self._testBCastC([1, 3, 2], [2])

  def testBCast_1D(self):
    self._testBCastD([1, 3, 2], [2])

  def testBCast_2A(self):
    self._testBCastA([2, 3, 2], [3, 2])

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

def run_benchmark(func, num_iters, execution_mode=None):
  ctx = context.context()
  with context.execution_mode(execution_mode):
    # call func to warm up
    func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    start = time.time()
    for _ in xrange(num_iters):
      func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    end = time.time()

    return end - start

import numpy as np
if __name__ == "__main__":
  test.main()
