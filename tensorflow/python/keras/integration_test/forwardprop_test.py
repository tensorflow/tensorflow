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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf


def _jvp(f, primals, tangents):
  """Compute the jacobian of `f` at `primals` multiplied by `tangents`."""
  with tf.autodiff.ForwardAccumulator(primals, tangents) as acc:
    primals_out = f(*primals)
  return primals_out, acc.jvp(
      primals_out, unconnected_gradients=tf.UnconnectedGradients.ZERO)


def _jacfwd(f, primals):
  """Compute the jacobian of `f` at `primals` using forward-mode autodiff."""
  jac_flat = []
  flat_primals = tf.nest.flatten(primals)
  tangent_mask = [tf.zeros_like(primal) for primal in flat_primals]
  for primal_index, primal in enumerate(flat_primals):
    primal_vector = tf.reshape(primal, [-1])
    primal_vector_length = tf.size(primal_vector)
    jac_columns = []
    for element_index in tf.range(primal_vector_length):
      mask = tf.one_hot(element_index, primal_vector_length)
      tangent_mask[primal_index] = tf.reshape(mask, tf.shape(primal))
      jac_columns.append(
          tf.nest.map_structure(
              functools.partial(tf.reshape, shape=[-1]),
              _jvp(f, primals, tf.nest.pack_sequence_as(primals,
                                                        tangent_mask))[1]))
    jac_flat.append(tf.stack(jac_columns, axis=1))
    tangent_mask[primal_index] = tf.zeros_like(primal)
  return tf.nest.pack_sequence_as(primals, jac_flat)


def _grad(f, argnums=0):
  """Return a function which computes the gradient of `f`."""

  def _f(*params):
    with tf.GradientTape() as tape:
      tape.watch(params)
      primals_out = f(*params)
    return tape.gradient(
        primals_out,
        params[argnums],
        unconnected_gradients=tf.UnconnectedGradients.ZERO)

  return _f


def _hvp(f, primals, tangents):
  """Compute a forward-over-back Hessian-vector product."""
  with tf.autodiff.ForwardAccumulator(primals, tangents) as acc:
    with tf.GradientTape() as tape:
      tape.watch(primals)
      f_out = f(*primals)
      f_out.shape.assert_is_compatible_with([])
    return acc.jvp(tape.gradient(f_out, primals))


def _vectorize_parameters(f, params, use_pfor, dtype):
  """Loop over `params`, providing a one-hot mask to `f` for each."""
  parameter_sizes = [tf.size(param) for param in params]
  total_size = tf.math.add_n(parameter_sizes)

  def _wrapper(index):
    full_onehot = tf.one_hot(index, total_size)
    split_onehot = tf.split(full_onehot, parameter_sizes)
    tangents = [
        tf.reshape(v, tf.shape(param))
        for param, v in zip(params, split_onehot)
    ]
    return f(tangents)

  if use_pfor:
    return tf.vectorized_map(_wrapper, tf.range(total_size))
  else:
    return tf.map_fn(_wrapper, tf.range(total_size), dtype)


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
  sym_jac_back, num_jac = tf.test.compute_gradient(f, primals, delta=delta)
  testcase.assertAllClose(num_jac, sym_jac_back, rtol=rtol, atol=atol)
  sym_jac_fwd = _jacfwd(f, primals)
  testcase.assertAllClose(num_jac, sym_jac_fwd, rtol=rtol, atol=atol)
  # And the symbolic computations should be much closer.
  testcase.assertAllClose(sym_jac_back, sym_jac_fwd)


class ForwardpropTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ("Dense", [[0.1]], functools.partial(tf.keras.layers.Dense, 5)),
      ("Conv2D",
       np.reshape(
           np.arange(start=-1., stop=1., step=2. / (1 * 2 * 4 * 4)),
           [1, 2, 4, 4]), functools.partial(tf.keras.layers.Conv2D, 2, 2), 1e-3)
  ])
  def testKerasLayers(self, value, op_fn, atol=1e-6):
    layer = op_fn()
    input_value = tf.constant(value, dtype=tf.float32)
    layer.build(input_value.shape)
    # Make sure the test is deterministic by avoiding random variable
    # initialization.
    for v in layer.trainable_variables:
      v.assign(
          tf.reshape(
              tf.range(
                  -1.,
                  1.,
                  2. / tf.size(v, out_type=tf.float32),
                  dtype=tf.float32), v.shape))
    _test_gradients(
        self, layer, [input_value], atol=atol,
        # These are linear, so second-order is pretty boring.
        order=2)

  @parameterized.named_parameters([
      ("NonFused", [[0.1], [0.2], [-0.3]],
       functools.partial(tf.keras.layers.BatchNormalization, fused=False)),
      ("Fused", [[[[0.1, 2.]]], [[[0.2, -3.]]], [[[-0.3, 4.]]]],
       functools.partial(tf.keras.layers.BatchNormalization, fused=True))
  ])
  def testBatchNorm(self, value, op_fn):
    for training in [True, False]:
      layer = op_fn()
      input_value = tf.constant(value, dtype=tf.float32)
      layer.build(input_value.shape)
      _test_gradients(
          self, functools.partial(layer, training=training), [input_value],
          order=2, atol=1e-3)

  @parameterized.named_parameters([
      ("NonFused", [[0.1], [0.2], [-0.3]],
       functools.partial(tf.keras.layers.BatchNormalization, fused=False)),
      ("Fused", [[[[0.1, 2.]]], [[[0.2, -3.]]], [[[-0.3, 4.]]]],
       functools.partial(tf.keras.layers.BatchNormalization, fused=True))
  ])
  def testBatchNormLayerParamGrads(self, value, op_fn):
    for training in [True, False]:
      layer = op_fn()
      with tf.GradientTape() as tape:
        input_value = tf.constant(value, dtype=tf.float32)
        tape.watch(input_value)
        output = layer(input_value, training=training)
      jac_back = tape.jacobian(
          output, [input_value] + layer.trainable_variables)
      jac_forward = _jacfwd(
          lambda *args: layer(args[0], training=training),  # pylint:disable=cell-var-from-loop
          [input_value] + layer.trainable_variables)
      for backward, forward in zip(jac_back, jac_forward):
        forward = tf.reshape(forward, tf.shape(backward))
        self.assertAllClose(backward, forward)

  @parameterized.named_parameters([("Function", tf.function),
                                   ("NoFunction", lambda f: f)])
  def testVariablesHVP(self, decorator):

    if tf.test.is_built_with_rocm():
      # TODO(rocm)
      # This test was recently added and has never passed on the
      # ROCm platform. Remove this skip once the test is passing again
      self.skipTest("NoFunction decorator test fails on the ROCm platform")

    class _Model(tf.Module):

      def __init__(self):
        self._first_dense = tf.keras.layers.Dense(18)
        self._conv = tf.keras.layers.Conv2D(2, 2)
        self._norm = tf.keras.layers.BatchNormalization()
        self._second_dense = tf.keras.layers.Dense(1)

      def __call__(self, x):
        x = self._first_dense(x)
        x = tf.nn.relu(x)
        x = self._norm(x)
        x = tf.nn.relu(self._conv(tf.reshape(x, [-1, 2, 3, 3])))
        return self._second_dense(x)

    model = _Model()
    def _loss():
      input_value = tf.constant([[-0.5, 1.], [0.5, -1.]])
      target = tf.constant([[-1.], [2.]])
      return tf.math.reduce_sum((model(input_value) - target)**2.)

    @decorator
    def _compute_hvps():
      with tf.GradientTape() as tape:
        loss = _loss()
      vector = tape.gradient(loss, model.trainable_variables)
      variable_input_fn = lambda unused_variables: _loss()
      forward_over_back_hvp, = _hvp(
          variable_input_fn, [model.trainable_variables], [vector])
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        loss = _loss()
        first_grads = tape.gradient(loss, model.trainable_variables)
      back_over_back_hvp = tape.gradient(
          first_grads, model.trainable_variables, output_gradients=vector)
      return forward_over_back_hvp, back_over_back_hvp
    self.assertAllClose(*_compute_hvps(), rtol=1e-5, atol=1e-5)


class HessianTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      [("PFor", True),
       ("MapFn", False)])
  def testHessianOfVariables(self, use_pfor):
    model = tf.keras.layers.Dense(1)
    model.build([None, 2])

    def _loss(*unused_args):
      input_value = tf.constant([[-0.5, 1.], [0.5, -1.]])
      target = tf.constant([[-1.], [2.]])
      return tf.math.reduce_sum((model(input_value) - target)**2.)

    kernel_hess, bias_hess = _forward_over_back_hessian(
        _loss, [model.kernel, model.bias],
        use_pfor=use_pfor,
        dtype=[tf.float32, tf.float32])
    # 3 total parameters, the whole hessian is the 3x3 concatenation
    self.assertEqual([3, 2, 1], kernel_hess.shape)
    self.assertEqual([3, 1], bias_hess.shape)
    full_hessian = tf.concat([tf.reshape(kernel_hess, [3, 2]), bias_hess],
                             axis=1)
    # The full Hessian should be symmetric.
    self.assertAllClose(full_hessian, tf.transpose(full_hessian))


if __name__ == "__main__":
  tf.test.main()
