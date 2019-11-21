# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gc
import weakref

from absl.testing import parameterized
import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import forwardprop
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import tape as tape_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import normalization_v2
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import test
from tensorflow.python.util import nest


_X11_35_DERIVATIVES = [
    1.1 ** 3.5,
    3.5 * 1.1 ** 2.5,
    3.5 * 2.5 * 1.1 ** 1.5,
    3.5 * 2.5 * 1.5 * 1.1 ** 0.5]


# TODO(allenl): Move this somewhere useful once forward gradients are stable.
def _jvp(f, primals, tangents):
  """Compute the jacobian of `f` at `primals` multiplied by `tangents`."""
  with forwardprop.ForwardAccumulator(primals, tangents) as acc:
    primals_out = f(*primals)
  return primals_out, acc.jvp(
      primals_out, unconnected_gradients=UnconnectedGradients.ZERO)


def _jacfwd(f, primals):
  """Compute the jacobian of `f` at `primals` using forward-mode autodiff."""
  jac_flat = []
  flat_primals = nest.flatten(primals)
  tangent_mask = [array_ops.zeros_like(primal) for primal in flat_primals]
  for primal_index, primal in enumerate(flat_primals):
    primal_vector = array_ops.reshape(primal, [-1])
    primal_vector_length = array_ops.size(primal_vector)
    jac_columns = []
    for element_index in math_ops.range(primal_vector_length):
      mask = array_ops.one_hot(element_index, primal_vector_length)
      tangent_mask[primal_index] = array_ops.reshape(mask,
                                                     array_ops.shape(primal))
      jac_columns.append(
          nest.map_structure(
              functools.partial(array_ops.reshape, shape=[-1]),
              _jvp(f, primals,
                   nest.pack_sequence_as(primals, tangent_mask))[1]))
    jac_flat.append(array_ops.stack(jac_columns, axis=1))
    tangent_mask[primal_index] = array_ops.zeros_like(primal)
  return nest.pack_sequence_as(primals, jac_flat)


def _grad(f, argnums=0):
  """Return a function which computes the gradient of `f`."""

  def _f(*params):
    with backprop.GradientTape() as tape:
      tape.watch(params)
      primals_out = f(*params)
    return tape.gradient(
        primals_out,
        params[argnums],
        unconnected_gradients=UnconnectedGradients.ZERO)

  return _f


def _gradfwd(f, argnums=0, f_out_dtypes=dtypes.float32):
  """Return a function which computes the gradient of `f` in forward mode."""

  def _f(*params):
    def _single_jvp(param_mask):
      with forwardprop.ForwardAccumulator(primals=[params[argnums]],
                                          tangents=param_mask) as acc:
        primals_out = f(*params)
      return acc.jvp(primals_out)
    # Building up a function to run with pfor takes a bit too long since we're
    # only running it a handful of times.
    return _vectorize_parameters(_single_jvp, [params[argnums]],
                                 use_pfor=False, dtype=f_out_dtypes)
  return _f


def _hvp(f, primals, tangents):
  """Compute a forward-over-back Hessian-vector product."""
  with forwardprop.ForwardAccumulator(primals, tangents) as acc:
    with backprop.GradientTape() as tape:
      tape.watch(primals)
      f_out = f(*primals)
      f_out.shape.assert_is_compatible_with([])
    return acc.jvp(tape.gradient(f_out, primals))


def _vectorize_parameters(f, params, use_pfor, dtype):
  """Loop over `params`, providing a one-hot mask to `f` for each."""
  parameter_sizes = [array_ops.size(param) for param in params]
  total_size = math_ops.add_n(parameter_sizes)

  def _wrapper(index):
    full_onehot = array_ops.one_hot(index, total_size)
    split_onehot = array_ops.split(full_onehot, parameter_sizes)
    tangents = [array_ops.reshape(v, array_ops.shape(param))
                for param, v in zip(params, split_onehot)]
    return f(tangents)

  if use_pfor:
    return control_flow_ops.vectorized_map(_wrapper, math_ops.range(total_size))
  else:
    return map_fn.map_fn(_wrapper, math_ops.range(total_size), dtype)


def _forward_over_back_hessian(f, params, use_pfor, dtype=None):
  """Computes the full Hessian matrix for the scalar-valued f(*params).

  Args:
    f: A function taking `params` and returning a scalar.
    params: A possibly nested structure of tensors.
    use_pfor: If true, uses `tf.vectorized_map` calls instead of looping.
    dtype: Required if `use_pfor=False`. A possibly nested structure of dtypes
      (e.g. `tf.float32`) matching the structure of `f`'s returns.

  Returns:
    A possibly nested structure of matrix slices corresponding to `params`. Each
    slice has shape [P, p_s] where `p_s` is the number of parameters (`tf.size`)
    in the corresponding element of `params` and `P` is the total number of
    parameters (`sum_s(p_s)`). The full matrix can be obtained by concatenating
    along the second axis.
  """
  return _vectorize_parameters(
      functools.partial(_hvp, f, params),
      params, use_pfor=use_pfor, dtype=dtype)


def _test_gradients(testcase,
                    f,
                    primals,
                    order,
                    delta=1e-3,
                    rtol=1e-2,
                    atol=1e-6):
  """Tests forward/backward jacobians of `f`'s [0, `order`)-order gradients."""
  if order < 1:
    raise ValueError(
        "`order` should be a positive integer, got '{}'.".format(order))
  if order > 1:
    _test_gradients(
        testcase=testcase,
        f=_grad(f),
        primals=primals,
        order=order - 1,
        delta=delta,
        rtol=rtol,
        atol=atol)
  sym_jac_back, num_jac = gradient_checker_v2.compute_gradient(
      f, primals, delta=delta)
  testcase.assertAllClose(num_jac, sym_jac_back, rtol=rtol, atol=atol)
  sym_jac_fwd = _jacfwd(f, primals)
  testcase.assertAllClose(num_jac, sym_jac_fwd, rtol=rtol, atol=atol)
  # And the symbolic computations should be much closer.
  testcase.assertAllClose(sym_jac_back, sym_jac_fwd)


class ForwardpropTest(test.TestCase, parameterized.TestCase):

  def testJVPFunction(self):
    add_outputs = (constant_op.constant(4.),)
    vp, = forwardprop._jvp_dispatch(
        op_name="Add",
        attr_tuple=(),
        inputs=(constant_op.constant(1.), constant_op.constant(3.)),
        outputs=add_outputs,
        tangents=(
            constant_op.constant(1.),
            constant_op.constant(5.),
        ))
    self.assertAllClose(1. + 5., self.evaluate(vp))

    mul_outputs = (constant_op.constant([20.]),)
    vp, = forwardprop._jvp_dispatch(
        op_name="Mul",
        attr_tuple=(),
        inputs=(constant_op.constant([4.]), constant_op.constant([5.])),
        outputs=mul_outputs,
        tangents=(
            constant_op.constant([2.]),
            constant_op.constant([3.]),
        ))
    self.assertAllClose([2. * 5. + 3. * 4.], self.evaluate(vp))

  def testJVPFunctionUsedByAccumulatorForOps(self):
    previous_fn = forwardprop._jvp_dispatch
    try:
      x = constant_op.constant(1.)
      with forwardprop.ForwardAccumulator(x, 2.) as acc:
        y = x + x
        pywrap_tensorflow.TFE_Py_RegisterJVPFunction(
            lambda *args, **kwargs: [constant_op.constant(-15.)])
        z = x + x
      self.assertAllClose(4., acc.jvp(y))
      self.assertAllClose(-15., acc.jvp(z))
    finally:
      pywrap_tensorflow.TFE_Py_RegisterJVPFunction(previous_fn)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testFunctionCacheLimited(self):
    # Every time this test is executed, it will create a slightly larger Tensor
    # and push it through Add's gradient. Since we check for new pyobjects after
    # the warmup, retracing each time without cleaning up old traces fails the
    # test. It works because of experimental_relax_shapes.
    for _ in range(forwardprop._TRACE_COUNT_LIMIT):
      execution_count = getattr(self, "_execution_count", 0)
      self._execution_count = execution_count + 1
      x = array_ops.zeros([execution_count])
      with forwardprop.ForwardAccumulator(
          x, array_ops.ones_like(x)) as acc:
        y = x + x
      self.assertAllClose(2. * array_ops.ones_like(x), acc.jvp(y))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testMultipleWatchesAdd(self):
    x = constant_op.constant(-2.)
    with self.assertRaisesRegexp(ValueError, "multiple times"):
      with forwardprop.ForwardAccumulator(
          [x, x], [1., 2.]):
        pass
    with forwardprop.ForwardAccumulator(
        [x], [3.]) as acc:
      self.assertAllClose(3., acc.jvp(x))
      acc._watch(x, constant_op.constant(10.))
      self.assertAllClose(13., acc.jvp(x))
      acc._watch(x, constant_op.constant(11.))
      self.assertAllClose(24., acc.jvp(x))
      y = constant_op.constant(3.) * x
    self.assertAllClose(24., acc.jvp(x))
    self.assertAllClose(24. * 3., acc.jvp(y))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testReenter(self):
    x = constant_op.constant(-2.)
    with forwardprop.ForwardAccumulator(x, 1.5) as acc:
      self.assertAllClose(1.5, acc.jvp(x))
      y = 4. * x
      self.assertAllClose(6., acc.jvp(y))
      with self.assertRaisesRegexp(ValueError, "already recording"):
        with acc:
          pass
    z = 4. * x
    self.assertIsNone(acc.jvp(z))
    with acc:
      yy = y * y
    self.assertAllClose(6. * -8. * 2., acc.jvp(yy))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testDeadTensorsJVPCleared(self):
    x = array_ops.ones([100])
    x_weak = weakref.ref(x)
    grad_tensor = constant_op.constant(array_ops.zeros([100]))
    grad_tensor_weak = weakref.ref(grad_tensor)
    with forwardprop.ForwardAccumulator(x, grad_tensor) as acc:
      derived_tensor = constant_op.constant(2.) * x
      del grad_tensor
      self.assertAllClose(array_ops.zeros([100]), acc.jvp(x))
      del x
      self.assertIsNone(x_weak())
      self.assertIsNone(grad_tensor_weak())
      derived_tensor_weak = weakref.ref(derived_tensor)
      derived_tensor_grad = acc.jvp(derived_tensor)
      derived_tensor_grad_weak = weakref.ref(derived_tensor_grad)
      del derived_tensor
      del derived_tensor_grad
      self.assertIsNone(derived_tensor_weak())
      self.assertIsNone(derived_tensor_grad_weak())

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testJVPManual(self):
    primal, tangent = _jvp(math_ops.sin, (constant_op.constant(0.1),),
                           (constant_op.constant(0.2),))
    self.assertAllClose(math_ops.sin(0.1), primal)
    self.assertAllClose(math_ops.cos(0.1) * 0.2, tangent)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testNumericHigherOrder(self):

    def f(x):
      pointwise = math_ops.sin(x) * math_ops.tan(x)
      return math_ops.reduce_prod(
          pointwise + math_ops.reduce_sum(pointwise), axis=1)

    _test_gradients(
        self, f, [constant_op.constant([[2.0, 3.0], [1.0, 4.0]])], order=3)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testCustomGradient(self):

    @custom_gradient.custom_gradient
    def f(x):

      def grad(dy):
        return dy * math_ops.cos(x)

      return np.sin(x.numpy()), grad

    _test_gradients(self, f, [constant_op.constant([1., 2.])], order=3)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testCustomGradientRecomputeGrad(self):

    @custom_gradient.recompute_grad
    def f(x):
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    _test_gradients(self, f, [constant_op.constant([1.])], order=3)

  def testExceptionInCustomGradientNotSwallowed(self):

    @custom_gradient.custom_gradient
    def f(unused_x):
      def grad(unused_dy):
        raise ValueError("test_error_string")
      return 1., grad

    c = constant_op.constant(1.)
    d = constant_op.constant(2.)
    with forwardprop.ForwardAccumulator(c, d):
      with self.assertRaisesRegexp(ValueError, "test_error_string"):
        f(c)

  @parameterized.named_parameters(
      [("EluM5", -0.5, nn_ops.elu),
       ("EluP5", [0.5], nn_ops.elu),
       ("SwishP5", 0.5, nn_impl.swish),
       ("SwishM5", [-0.5], nn_impl.swish)])
  def testElementwiseNNOps(self, value, op_fn):
    _test_gradients(self, op_fn, [constant_op.constant(value)], order=3)

  @parameterized.named_parameters(
      [("Dense", [[0.1]], functools.partial(core.Dense, 5)),
       ("Conv2D",
        np.reshape(np.arange(start=-1., stop=1., step=2. / (1 * 2 * 4 * 4)),
                   [1, 2, 4, 4]),
        functools.partial(convolutional.Conv2D, 2, 2), 1e-3)])
  def testKerasLayers(self, value, op_fn, atol=1e-6):
    layer = op_fn()
    input_value = constant_op.constant(value, dtype=dtypes.float32)
    layer.build(input_value.shape)
    # Make sure the test is deterministic by avoiding random variable
    # initialization.
    for v in layer.trainable_variables:
      v.assign(array_ops.reshape(
          math_ops.range(
              -1., 1., 2. / array_ops.size(v, out_type=dtypes.float32),
              dtype=dtypes.float32),
          v.shape))
    _test_gradients(
        self, layer, [input_value], atol=atol,
        # These are linear, so second-order is pretty boring.
        order=2)

  @parameterized.named_parameters(
      [("NonFused", [[0.1], [0.2], [-0.3]],
        functools.partial(normalization_v2.BatchNormalization, fused=False)),
       ("Fused", [[[[0.1, 2.]]], [[[0.2, -3.]]], [[[-0.3, 4.]]]],
        functools.partial(normalization_v2.BatchNormalization, fused=True))])
  def testBatchNorm(self, value, op_fn):
    for training in [True, False]:
      layer = op_fn()
      input_value = constant_op.constant(value, dtype=dtypes.float32)
      layer.build(input_value.shape)
      _test_gradients(
          self, functools.partial(layer, training=training), [input_value],
          order=2, atol=1e-3)

  @parameterized.named_parameters(
      [("NonFused", [[0.1], [0.2], [-0.3]],
        functools.partial(normalization_v2.BatchNormalization, fused=False)),
       ("Fused", [[[[0.1, 2.]]], [[[0.2, -3.]]], [[[-0.3, 4.]]]],
        functools.partial(normalization_v2.BatchNormalization, fused=True))])
  def testBatchNormLayerParamGrads(self, value, op_fn):
    for training in [True, False]:
      layer = op_fn()
      with backprop.GradientTape() as tape:
        input_value = constant_op.constant(value, dtype=dtypes.float32)
        tape.watch(input_value)
        output = layer(input_value, training=training)
      jac_back = tape.jacobian(
          output, [input_value] + layer.trainable_variables)
      jac_forward = _jacfwd(
          lambda *args: layer(args[0], training=training),  # pylint:disable=cell-var-from-loop
          [input_value] + layer.trainable_variables)
      for backward, forward in zip(jac_back, jac_forward):
        forward = array_ops.reshape(forward, array_ops.shape(backward))
        self.assertAllClose(backward, forward)

  @parameterized.named_parameters(
      [("NCHW", "NCHW", [4, 3, 2, 2], 3, False),
       ("NHWC", "NHWC", [4, 2, 2, 3], 3, False),
       ("NCHWForward", "NCHW", [2, 2, 1, 1], 2, True),
       ("NHWCForward", "NHWC", [2, 1, 1, 2], 2, True),])
  def testFusedBatchNormGradsTraining(self, data_format, x_shape, channels,
                                      test_back_over_forward):
    increment = 3. / math_ops.reduce_prod(
        constant_op.constant(x_shape, dtype=dtypes.float32))
    x = array_ops.reshape(math_ops.range(-2., 1., increment), x_shape)
    scale = constant_op.constant([1., 1.1, 0.9])[:channels]
    offset = constant_op.constant([-0.5, -0.6, -0.7])[:channels]
    epsilon = 0.001

    def _bn_fused(x_arg, scale_arg, offset_arg):
      return nn_impl.fused_batch_norm(x_arg, scale_arg, offset_arg,
                                      epsilon=epsilon, is_training=True,
                                      data_format=data_format)[0]
    _test_gradients(self, _bn_fused, [x, scale, offset], order=2, atol=1e-3)
    if test_back_over_forward:
      # Note that this uses a loop over parameters, and so is quite slow. Thus
      # it's skipped for the larger test cases.
      gradfwd_x = _gradfwd(_bn_fused, 0)
      _test_gradients(self, gradfwd_x, [x, scale, offset], order=1,
                      atol=1e-3)
      gradfwd_scale = _gradfwd(_bn_fused, 1)
      _test_gradients(self, gradfwd_scale, [x, scale, offset], order=1,
                      atol=1e-3)
      gradfwd_offset = _gradfwd(_bn_fused, 2)
      _test_gradients(self, gradfwd_offset, [x, scale, offset], order=1,
                      atol=1e-3)

  def testFusedBatchNormGradsInference(self):

    if test.is_built_with_rocm():
      # This test was addeded recently and has been failing on the ROCm
      # platform, since it was added.
      # TODO(rocm): do root cause analysis of test failure and fix it.
      self.skipTest("Test fails on ROCm platform, needs further analysis")

    x_shape = [4, 10, 10, 2]
    increment = 3. / math_ops.reduce_prod(
        constant_op.constant(x_shape, dtype=dtypes.float32))
    x = array_ops.reshape(math_ops.range(-2., 1., increment), x_shape)
    scale = constant_op.constant([1., 1.1])
    offset = constant_op.constant([-0.5, -0.6])
    mean = constant_op.constant([-1.3, 1.4])
    variance = constant_op.constant([0.7, 0.9])
    epsilon = 0.001

    def _bn_fused(x_arg, scale_arg, offset_arg):
      return nn_impl.fused_batch_norm(x_arg, scale_arg, offset_arg,
                                      mean, variance,
                                      epsilon=epsilon, is_training=False)[0]
    _test_gradients(self, _bn_fused, [x, scale, offset],
                    order=2, atol=1e-2)

  @parameterized.named_parameters(
      [("Function", def_function.function),
       ("NoFunction", lambda f: f)])
  def testVariablesHVP(self, decorator):

    if test.is_built_with_rocm():
      # TODO(rocm)
      # This test was recently added and has never passed on the
      # ROCm platform. Remove this skip once the test is passing again
      self.skipTest("NoFunction decorator test fails on the ROCm platform")

    class _Model(module.Module):

      def __init__(self):
        self._first_dense = core.Dense(18)
        self._conv = convolutional.Conv2D(2, 2)
        self._norm = normalization_v2.BatchNormalization()
        self._second_dense = core.Dense(1)

      def __call__(self, x):
        x = self._first_dense(x)
        x = nn_ops.relu(x)
        x = self._norm(x)
        x = nn_ops.relu(self._conv(array_ops.reshape(x, [-1, 2, 3, 3])))
        return self._second_dense(x)

    model = _Model()
    def _loss():
      input_value = constant_op.constant([[-0.5, 1.], [0.5, -1.]])
      target = constant_op.constant([[-1.], [2.]])
      return math_ops.reduce_sum((model(input_value) - target) ** 2.)

    @decorator
    def _compute_hvps():
      with backprop.GradientTape() as tape:
        loss = _loss()
      vector = tape.gradient(loss, model.trainable_variables)
      variable_input_fn = lambda unused_variables: _loss()
      forward_over_back_hvp, = _hvp(
          variable_input_fn, [model.trainable_variables], [vector])
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        loss = _loss()
        first_grads = tape.gradient(loss, model.trainable_variables)
      back_over_back_hvp = tape.gradient(
          first_grads, model.trainable_variables, output_gradients=vector)
      return forward_over_back_hvp, back_over_back_hvp
    self.assertAllClose(*_compute_hvps(), rtol=1e-5, atol=1e-5)

  def testPushPopAccumulatorState(self):
    # Note that this example is somewhat contrived. push_forwardprop_state is
    # probably only useful in practice for building functions that compute jvps
    # alongside their usual outputs.
    c = constant_op.constant(1.)
    d = constant_op.constant(2.)
    with forwardprop.ForwardAccumulator(c, d) as acc:

      @custom_gradient.custom_gradient
      def f(x):
        y = math_ops.sin(x.numpy())

        def grad(dy):
          with forwardprop_util.push_forwardprop_state():
            x_copy = constant_op.constant(x.numpy())
            acc._watch(x_copy, dy)
            y_copy = math_ops.sin(x_copy)
          return dy * acc.jvp(y_copy)

        return y, grad

      output = f(c)
      self.assertAllClose(d * math_ops.cos(c), acc.jvp(output))

  @parameterized.named_parameters(
      [("Order{}".format(order), order, expected)
       for order, expected in enumerate(_X11_35_DERIVATIVES)])
  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testHigherOrderPureForward(self, order, expected):

    def _forwardgrad(f):
      def _compute_forwardgrad(primal):
        tangent = constant_op.constant(1.)
        with forwardprop.ForwardAccumulator(primal, tangent) as acc:
          primal_out = f(primal)
        return acc.jvp(primal_out)
      return _compute_forwardgrad

    def _forward(x):
      return x ** 3.5

    f = _forward
    primal = constant_op.constant(1.1)
    for _ in range(order):
      f = _forwardgrad(f)
    self.assertAllClose(expected, f(primal))

  @parameterized.named_parameters(
      [("Function", def_function.function),
       ("NoFunction", lambda f: f)])
  def testGradPureForward(self, decorator):

    @decorator
    def f(x):
      return x ** 3.5

    primal = constant_op.constant(1.1)
    with forwardprop.ForwardAccumulator(
        primal, constant_op.constant(1.)) as outer_acc:
      with forwardprop.ForwardAccumulator(
          primal, constant_op.constant(1.)) as acc:
        primal_out = f(primal)
    inner_jvp = acc.jvp(primal_out)
    outer_jvp = outer_acc.jvp(inner_jvp)
    self.assertAllClose(1.1 ** 3.5, primal_out)
    self.assertAllClose(3.5 * 1.1 ** 2.5, inner_jvp)
    self.assertAllClose(3.5 * 2.5 * 1.1 ** 1.5, outer_jvp)
    self.assertIsNone(acc.jvp(outer_acc.jvp(primal_out)))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testJVPPacking(self):
    two = constant_op.constant(2.)
    primal_in = constant_op.constant(1.)
    inner_jvp = constant_op.constant(3.)
    with forwardprop.ForwardAccumulator(
        [primal_in, inner_jvp],
        [constant_op.constant(2.), constant_op.constant(4.)]) as outer_acc:
      with forwardprop.ForwardAccumulator(
          primal_in, inner_jvp) as inner_acc:
        packed_input_indices, packed_input_tangents = (
            forwardprop_util.pack_tangents([primal_in]))
        self.assertAllClose([3., 2., 4.], packed_input_tangents)
        expected_indices = (
            # inner_acc watches primal_in
            ((0, 1),),
            # outer_acc watches primal_in and inner_jvp
            ((0, 2),
             (1, 3)))
        self.assertAllEqual(expected_indices, packed_input_indices)
        primal_out = primal_in * two
        self.assertAllClose(6., inner_acc.jvp(primal_out))
        self.assertAllClose(4., outer_acc.jvp(primal_out))
        self.assertAllClose(8., outer_acc.jvp(inner_acc.jvp(primal_out)))
        packed_output_indices, packed_output_tangents = (
            forwardprop_util.pack_tangents([primal_out]))
        self.assertAllClose([6., 4., 8.], packed_output_tangents)
        self.assertAllEqual(expected_indices, packed_output_indices)

  def testFunctionGradInFunctionPureForward(self):

    @def_function.function
    def take_gradients():
      @def_function.function
      def f(x):
        return x ** 3.5

      primal = constant_op.constant(1.1)
      with forwardprop.ForwardAccumulator(
          primal, constant_op.constant(1.)) as outer_acc:
        with forwardprop.ForwardAccumulator(
            primal, constant_op.constant(1.)) as acc:
          primal_out = f(primal)
      inner_jvp = acc.jvp(primal_out)
      outer_jvp = outer_acc.jvp(inner_jvp)
      self.assertIsNone(acc.jvp(outer_acc.jvp(primal_out)))
      return primal_out, inner_jvp, outer_jvp

    primal_out, inner_jvp, outer_jvp = take_gradients()
    self.assertAllClose(1.1 ** 3.5, primal_out)
    self.assertAllClose(3.5 * 1.1 ** 2.5, inner_jvp)
    self.assertAllClose(3.5 * 2.5 * 1.1 ** 1.5, outer_jvp)

  def testFunctionGrad(self):

    @def_function.function
    def f(x):
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    _test_gradients(
        self,
        f,
        [constant_op.constant([1., 2.])],
        order=3)

  def testReusingJVP(self):
    m1 = random_ops.random_uniform((256, 2096))
    m2 = array_ops.identity(m1)
    tangent1 = random_ops.random_uniform((256, 2096))
    tangent2 = random_ops.random_uniform((256, 2096))
    matmul = def_function.function(math_ops.matmul)

    with forwardprop.ForwardAccumulator(
        primals=[m1, m2], tangents=[tangent1, tangent2]) as acc:
      result1 = matmul(m1, m1, transpose_b=True)
      result2 = matmul(m2, m2, transpose_b=True)

    def _expected(mat, tangent):
      return (math_ops.matmul(tangent, mat, transpose_b=True)
              + math_ops.matmul(mat, tangent, transpose_b=True))

    self.assertAllClose(result1, result2)
    self.assertAllClose(_expected(m1, tangent1), acc.jvp(result1))
    self.assertAllClose(_expected(m2, tangent2), acc.jvp(result2))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testHVPMemory(self):

    def fun(x):
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    primals = constant_op.constant([1., 2., 3.])
    tangents = constant_op.constant([3., 4., 5.])
    _hvp(fun, (primals,), (tangents,))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testHVPCorrectness(self):

    def fun(x):
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    primals = constant_op.constant([1., 2., 3.])
    tangents = constant_op.constant([3., 4., 5.])
    forwardback_hvp_eager, = _hvp(fun, (primals,), (tangents,))
    forwardback_hvp_function, = def_function.function(_hvp)(fun, (primals,),
                                                            (tangents,))

    with backprop.GradientTape(persistent=True) as g:
      g.watch(primals)
      with backprop.GradientTape() as gg:
        gg.watch(primals)
        out = fun(primals)
      grad = array_ops.unstack(gg.gradient(out, primals))
    hessian = []
    for i in range(3):
      hessian.append(g.gradient(grad[i], primals))
    hessian = array_ops.stack(hessian, axis=0)
    backback_hvp = math_ops.tensordot(hessian, tangents, axes=1)

    self.assertAllClose(backback_hvp, forwardback_hvp_eager)
    self.assertAllClose(backback_hvp, forwardback_hvp_function)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testShouldRecordAndStopRecord(self):
    c = constant_op.constant(1.)
    c_tangent = constant_op.constant(2.)
    with forwardprop.ForwardAccumulator(c, c_tangent) as acc:
      with backprop.GradientTape() as tape:
        self.assertFalse(tape_lib.should_record_backprop([c]))
        self.assertEqual(
            1, pywrap_tensorflow.TFE_Py_TapeSetPossibleGradientTypes([c]))
        tape.watch(c)
        self.assertEqual(
            2, pywrap_tensorflow.TFE_Py_TapeSetPossibleGradientTypes([c]))
        self.assertTrue(tape_lib.should_record_backprop([c]))
        with tape_lib.stop_recording():
          self.assertEqual(
              0, pywrap_tensorflow.TFE_Py_TapeSetPossibleGradientTypes([c]))
          self.assertFalse(tape_lib.should_record_backprop([c]))
          d = c * 2.
        self.assertEqual(
            2, pywrap_tensorflow.TFE_Py_TapeSetPossibleGradientTypes([c]))
        self.assertTrue(tape_lib.should_record_backprop([c]))
        self.assertFalse(tape_lib.should_record_backprop([d]))
        self.assertIsNone(acc.jvp(d))
      self.assertIsNone(tape.gradient(d, c))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testRecordingSelectively(self):
    c = constant_op.constant(1.)
    c_tangent = constant_op.constant(2.)
    with forwardprop.ForwardAccumulator(c, c_tangent) as acc:
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(c)
        with tape_lib.stop_recording():
          two = constant_op.constant(2.)
          d = c * two
          three = constant_op.constant(3.)
          e = c * three
        self.assertIsNone(acc.jvp(d))
        self.assertIsNone(acc.jvp(e))
        self.assertIsNone(tape.gradient(d, c))
        self.assertIsNone(tape.gradient(e, c))
        tape_lib.record_operation_forwardprop_only(
            "CustomForwardMul", [d], [c, two],
            lambda dd: (two * dd, c * dd), None)
        tape_lib.record_operation_backprop_only(
            "CustomBackwardMul", [e], [c, three],
            lambda de: (three * de, c * de))
        self.assertAllClose(4., acc.jvp(d))
        self.assertIsNone(acc.jvp(e))
        self.assertIsNone(tape.gradient(d, c))
        self.assertAllClose(3., tape.gradient(e, c))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testOpWithNoTrainableOutputs(self):
    v = variables.Variable(1.)
    with forwardprop.ForwardAccumulator(v, 11.):
      v.assign_sub(0.5)
      self.assertAllClose(0.5, self.evaluate(v))

  # TODO(b/141025187): Add a no_new_pyobjects decorator.
  def testVariableReadInFunction(self):
    v = variables.Variable(1.)
    with forwardprop.ForwardAccumulator(v, 11.) as acc:
      @def_function.function
      def f():
        return v.read_value(), 2. * v.read_value()
      result = f()
      self.assertAllClose((1.0, 2.), result)
      self.assertAllClose((11., 22.), acc.jvp(result))

  @parameterized.named_parameters(
      [("ForwardPropFirst", True),
       ("TapeFirst", False)])
  def testForwardOverBackwardMemoryEfficiency(self, forward_prop_first):
    # Watching depends depends on nesting, not creation order
    c = constant_op.constant(1.)
    if forward_prop_first:
      forward_accumulator = forwardprop.ForwardAccumulator(c, .1)
      gradient_tape = backprop.GradientTape()
    else:
      gradient_tape = backprop.GradientTape()
      forward_accumulator = forwardprop.ForwardAccumulator(c, .1)
    try:
      gc.disable()
      with gradient_tape as tape:
        # Adding and removing the tape multiple times in different nesting
        # patterns does not affect watch ordering.
        pass
      with forward_accumulator as acc:
        with gradient_tape as tape:
          tape.watch(c)
          d = math_ops.cos(c)
          self.assertFalse(tape_lib.should_record_backprop((acc.jvp(d),)))
          e = math_ops.cos(acc.jvp(d))
          math_ops.cos(e)
          weak_e = weakref.ref(e)
          del e
          self.assertIsNone(weak_e())
        self.assertIsNone(tape.gradient(acc.jvp(d), c))
    finally:
      gc.enable()

  @parameterized.named_parameters(
      [("ForwardPropFirst", True),
       ("TapeFirst", False)])
  def testBackwardOverForward(self, forward_prop_first):
    c = constant_op.constant(1.)
    # Watching depends depends on nesting, not creation order
    if forward_prop_first:
      forward_accumulator = forwardprop.ForwardAccumulator(c, .1)
      gradient_tape = backprop.GradientTape()
    else:
      gradient_tape = backprop.GradientTape()
      forward_accumulator = forwardprop.ForwardAccumulator(c, .1)
    with gradient_tape as tape:
      with forward_accumulator as acc:
        tape.watch(c)
        d = math_ops.cos(c)
        self.assertTrue(tape_lib.should_record_backprop((acc.jvp(d),)))
      self.assertAllClose(-.1 * math_ops.cos(1.),
                          tape.gradient(acc.jvp(d), c))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testRecordingWithJVPIndices(self):
    c = constant_op.constant(1.)
    with forwardprop.ForwardAccumulator(c, 10.) as acc:
      packed_input_tangents = forwardprop_util.pack_tangents([c]).tangents
      self.assertAllClose([10.], packed_input_tangents)
      d = constant_op.constant(2.)
      d_tangent = constant_op.constant(3.)
      tape_lib.record_operation_forwardprop_only(
          "FunctionWithInlineJVPs",
          [d] + [d_tangent],
          [c] + packed_input_tangents,
          None, (((0, 1),),))
      self.assertAllClose(3., acc.jvp(d))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testSpecialForwardFunctionUsed(self):
    c = constant_op.constant(1.)
    d = constant_op.constant(2.)
    e = constant_op.constant(3.)
    with forwardprop.ForwardAccumulator(c, 10.) as acc:
      tape_lib.record_operation(
          "ForwardIsSpecial",
          [d], [c],
          None, lambda jvp: [-2. * jvp])
      self.assertAllClose(-20., acc.jvp(d))
      tape_lib.record_operation(
          "ForwardIsSpecial2",
          [], [],
          None, lambda: [])
      tape_lib.record_operation(
          "ForwardIsSpecial3",
          [e], [d],
          None, lambda x: [x])
      self.assertAllClose(-20., acc.jvp(e))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testVariableWatched(self):
    v = variables.Variable([1., 2., 3.])
    with forwardprop.ForwardAccumulator(
        v, constant_op.constant([.1, -.2, .3])) as acc:
      self.assertAllClose([.1, -.2, .3], acc.jvp(v))
      x = v * 2.
      self.assertAllClose([.2, -.4, .6], acc.jvp(x))
      x2 = v + .1
      self.assertAllClose([.1, -.2, .3], acc.jvp(x2))

  def testUnconnectedGradients(self):
    x = constant_op.constant(-1.)
    with forwardprop.ForwardAccumulator(x, 0.1) as acc:
      self.assertAllClose(0.1, acc.jvp(x, unconnected_gradients="zero"))
      self.assertAllClose(0.1, acc.jvp(x, unconnected_gradients="none"))
      y = constant_op.constant(-2.)
      self.assertAllClose(0.0, acc.jvp(y, unconnected_gradients="zero"))
      self.assertIsNone(acc.jvp(y, unconnected_gradients="none"))

  # TODO(kkb): One weakref instance is created with warmup_iters=2, investigate.
  @test_util.assert_no_new_pyobjects_executing_eagerly(warmup_iters=3)
  def testVariableWatchedFunction(self):

    class _Model(module.Module):

      def __init__(self):
        self._v = None

      @def_function.function
      def compute_jvps(self):
        if self._v is None:
          self._v = variables.Variable([1., 2., 3.])
        with forwardprop.ForwardAccumulator(
            self._v, constant_op.constant([.1, -.2, .3])) as acc:
          x = self._v * 2.
          x2 = self._v + .1
        return acc.jvp((self._v, x, x2))

    model = _Model()
    v_jvp, x_jvp, x2_jvp = model.compute_jvps()
    self.assertAllClose([.1, -.2, .3], v_jvp)
    self.assertAllClose([.2, -.4, .6], x_jvp)
    self.assertAllClose([.1, -.2, .3], x2_jvp)

  # NOTE: assert_no_new_pyobjects_executing_eagerly fails flakily on this
  # test... could be something wrong with the test decorator, or some sort of
  # nondeterminstic caching.
  def testMirroredVariableWatched(self):

    def _replicated(input_tangent):
      with forwardprop.ForwardAccumulator(v, input_tangent) as acc:
        self.assertAllClose([.1, -.2, .3], acc.jvp(v))
        x = v * 2.
        self.assertAllClose([.2, -.4, .6], acc.jvp(x))
        x2 = v + .1
        self.assertAllClose([.1, -.2, .3], acc.jvp(x2))

    strategy = mirrored_strategy.MirroredStrategy()
    with strategy.scope():
      v = variables.Variable([1., 2., 3.])
      strategy.experimental_run_v2(
          _replicated, args=(constant_op.constant([.1, -.2, .3]),))

  # TODO(b/141025187): Add a no_new_pyobjects decorator.
  def testArgumentUnused(self):
    v = constant_op.constant(1.)
    with forwardprop.ForwardAccumulator(v, 11.) as acc:

      @def_function.function
      def _f(x):
        del x
        return constant_op.constant(1.)

      result = _f(v)
      self.assertAllClose(1.0, result)
      self.assertIsNone(acc.jvp(result))


@def_function.function
def _has_loop(iters, y):
  ret = 0.
  for i in math_ops.range(iters):
    ret += y * math_ops.cast(i, dtypes.float32)
  return ret


@def_function.function
def _has_cond(k, y):
  if k > 1:
    ret = 3. * y
  else:
    ret = 0.
  return ret


@def_function.function
def _fprop_while(iters, y):
  with forwardprop.ForwardAccumulator(y, 1.) as acc:
    ret = 0.
    for i in math_ops.range(iters):
      ret += y * math_ops.cast(i, dtypes.float32)
  return acc.jvp(ret)


@def_function.function
def _fprop_cond(k, y):
  with forwardprop.ForwardAccumulator(y, 1.) as acc:
    if k > 1:
      ret = 3. * y
    else:
      ret = 0.
  return acc.jvp(ret)


class ControlFlowTests(test.TestCase):

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testOfFunctionWhile(self):
    y = constant_op.constant(1.)
    with forwardprop.ForwardAccumulator(y, 1.) as acc:
      self.assertAllClose(
          10., acc.jvp(_has_loop(constant_op.constant(5), y)))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testOfFunctionCond(self):
    y = constant_op.constant(1.)
    with forwardprop.ForwardAccumulator(y, 1.) as acc:
      self.assertAllClose(
          3., acc.jvp(_has_cond(constant_op.constant(5), y)))
      self.assertAllClose(
          0., acc.jvp(_has_cond(constant_op.constant(0), y)))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testInFunctionWhile(self):
    self.assertAllClose(
        10., _fprop_while(constant_op.constant(5), constant_op.constant(1.)))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testInFunctionCond(self):
    self.assertAllClose(
        3., _fprop_cond(constant_op.constant(5), constant_op.constant(1.)))
    self.assertAllClose(
        0., _fprop_cond(constant_op.constant(0), constant_op.constant(1.)))


class HessianTests(test.TestCase, parameterized.TestCase):

  def testHessian1D(self):
    # Note: stolen from ops/gradients_test.py
    m = 4
    rng = np.random.RandomState([1, 2, 3])
    mat_value = rng.randn(m, m).astype("float32")
    x_value = rng.randn(m).astype("float32")
    hess_value = mat_value + mat_value.T
    mat = variables.Variable(mat_value)

    def _f(x):
      return math_ops.reduce_sum(x[:, None] * mat * x[None, :])

    hessian_eager, = _forward_over_back_hessian(
        _f, [constant_op.constant(x_value)],
        use_pfor=False, dtype=[dtypes.float32])
    self.assertAllClose(hess_value, hessian_eager)
    hessian_function, = def_function.function(_forward_over_back_hessian)(
        _f, [constant_op.constant(x_value)],
        use_pfor=False, dtype=[dtypes.float32])
    self.assertAllClose(hess_value, hessian_function)
    hessian_pfor, = def_function.function(_forward_over_back_hessian)(
        _f, [constant_op.constant(x_value)],
        use_pfor=True, dtype=[dtypes.float32])
    self.assertAllClose(hess_value, hessian_pfor)

  @parameterized.named_parameters(
      [("PFor", True),
       ("MapFn", False)])
  def testHessianOfVariables(self, use_pfor):
    model = core.Dense(1)
    model.build([2])

    def _loss(*unused_args):
      input_value = constant_op.constant([[-0.5, 1.], [0.5, -1.]])
      target = constant_op.constant([[-1.], [2.]])
      return math_ops.reduce_sum((model(input_value) - target) ** 2.)

    kernel_hess, bias_hess = _forward_over_back_hessian(
        _loss, [model.kernel, model.bias], use_pfor=use_pfor,
        dtype=[dtypes.float32, dtypes.float32])
    # 3 total parameters, the whole hessian is the 3x3 concatenation
    self.assertEqual([3, 2, 1], kernel_hess.shape)
    self.assertEqual([3, 1], bias_hess.shape)
    full_hessian = array_ops.concat(
        [array_ops.reshape(kernel_hess, [3, 2]), bias_hess], axis=1)
    # The full Hessian should be symmetric.
    self.assertAllClose(full_hessian, array_ops.transpose(full_hessian))


if __name__ == "__main__":
  # TODO(allenl): Also test with 1.x-style graph mode.
  ops.enable_eager_execution()
  test.main()
