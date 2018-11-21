# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Functional test for OptimizerV2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class OptimizerTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testBasic(self):
    for _, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
        if not context.executing_eagerly():
          loss = loss()
        sgd = gradient_descent.SGD(3.0)

        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd through optimizer
        opt_op = sgd.minimize(loss, var_list=[var0, var1])
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(opt_op)
        # Validate updated params
        self.assertAllClose([-14., -13.], self.evaluate(var0))
        self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testAdaptiveLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)

      def loss():
        return 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop

      sgd = gradient_descent.SGD(1.0)

      self.evaluate(variables.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd through optimizer
      opt_op = sgd.minimize(loss, [var0, var1])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)
      # Validate updated params
      # var0 = [1., 2.] - 1.0 * [5, 5]
      self.assertAllClose([-4., -3.], self.evaluate(var0))
      # var1 = [3., 4.] - 1.0 * [3, 3]
      self.assertAllClose([0., 1.], self.evaluate(var1))

      sgd.learning_rate = 0.5
      if context.executing_eagerly():
        sgd.minimize(loss, [var0, var1])
      else:
        self.evaluate(opt_op)
      # Validate updated params
      # var0 = [-4., -3.] - 0.5 * [5, 5]
      self.assertAllClose([-6.5, -5.5], self.evaluate(var0))
      # var1 = [0., 1.] - 0.5 * [3, 3]
      self.assertAllClose([-1.5, -0.5], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testAggregationMethod(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
        if not context.executing_eagerly():
          loss = loss()
        sgd = gradient_descent.SGD(3.0)

        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd through optimizer
        opt_op = sgd.minimize(
            loss,
            var_list=[var0, var1],
            aggregation_method=gradients_impl.AggregationMethod
            .EXPERIMENTAL_ACCUMULATE_N)
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(opt_op)
        # Validate updated params
        self.assertAllClose([-14., -13.], self.evaluate(var0))
        self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testPrecomputedGradient(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
        if not context.executing_eagerly():
          loss = loss()
        grad_loss = constant_op.constant([42, -42], dtype=dtype)
        sgd = gradient_descent.SGD(3.0)

        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd through optimizer
        opt_op = sgd.minimize(loss, var_list=[var0, var1], grad_loss=grad_loss)
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(opt_op)
        # Validate updated params
        self.assertAllClose([1.0 - 3 * 5 * 42.0, 2.0 - 3 * 5 * (-42.0)],
                            self.evaluate(var0))
        self.assertAllClose([3.0 - 3 * 3 * 42.0, 4.0 - 3 * 3 * (-42.0)],
                            self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testNoGradients(self):
    for _, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0  # pylint: disable=cell-var-from-loop
        if not context.executing_eagerly():
          loss = loss()
        sgd_op = gradient_descent.SGD(3.0)
        with self.assertRaisesRegexp(ValueError, 'No gradients'):
          # var1 has no gradient
          sgd_op.minimize(loss, var_list=[var1])

  @test_util.run_in_graph_and_eager_modes
  def testNoGradientsForAnyVariables_Minimize(self):
    for _, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        loss = lambda: constant_op.constant(5.0)
        if not context.executing_eagerly():
          loss = loss()

        sgd_op = gradient_descent.SGD(3.0)
        with self.assertRaisesRegexp(ValueError,
                                     'No gradients provided for any variable'):
          sgd_op.minimize(loss, var_list=[var0, var1])

  @test_util.run_in_graph_and_eager_modes
  def testNoGradientsForAnyVariables_ApplyGradients(self):
    for _, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        sgd_op = gradient_descent.SGD(3.0)
        with self.assertRaisesRegexp(ValueError,
                                     'No gradients provided for any variable'):
          sgd_op.apply_gradients([(None, var0), (None, var1)])

  @test_util.run_in_graph_and_eager_modes
  def testGradientsAsVariables(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
        if not context.executing_eagerly():
          loss = loss()

        sgd = gradient_descent.SGD(3.0)
        grads_and_vars = sgd.compute_gradients(loss, [var0, var1])
        # Convert gradients to tf.Variables
        converted_grads = [
            resource_variable_ops.ResourceVariable(
                array_ops.zeros([2], dtype), name='c_%d_%d' % (i, j))
            for j, gv in enumerate(grads_and_vars)
        ]
        convert_ops = [
            state_ops.assign(converted_grads[j], gv[0])
            for j, gv in enumerate(grads_and_vars)
        ]

        # Run convert_ops to achieve the gradients converting
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(convert_ops)
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 1 step of sgd through optimizer
        converted_grads_and_vars = list(zip(converted_grads, [var0, var1]))
        opt_op = sgd.apply_gradients(converted_grads_and_vars)
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(convert_ops)
        self.evaluate(opt_op)

        # Validate updated params
        self.assertAllClose([-14., -13.], self.evaluate(var0))
        self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testComputeGradientsWithTensors(self):
    with self.cached_session():
      x = ops.convert_to_tensor(1.0)

      def f():
        return x * x

      sgd = gradient_descent.SGD(3.0)
      grads_and_vars = sgd.compute_gradients(f, [x])
      self.assertEqual(1, len(grads_and_vars))
      grad, x_as_var = grads_and_vars[0]
      self.assertIs(x, x_as_var)
      self.assertEqual(2.0, self.evaluate(grad))

      with self.assertRaises(NotImplementedError):
        sgd.apply_gradients(grads_and_vars)

  @test_util.run_in_graph_and_eager_modes
  def testConstraint(self):
    constraint_01 = lambda x: clip_ops.clip_by_value(x, -0.1, 0.)
    constraint_0 = lambda x: clip_ops.clip_by_value(x, 0., 1.)
    with self.cached_session():
      var0 = variables.Variable([1.0, 2.0],
                                constraint=constraint_01)
      var1 = variables.Variable([3.0, 4.0],
                                constraint=constraint_0)
      loss = lambda: 5 * var0 + 3 * var1
      if not context.executing_eagerly():  # pylint: disable=cell-var-from-loop
        loss = loss()
      sgd = gradient_descent.SGD(3.0)

      self.evaluate(variables.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd through optimizer
      opt_op = sgd.minimize(loss, var_list=[var0, var1])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)
      # Validate updated params
      self.assertAllClose([-0.1, -0.1], self.evaluate(var0))
      self.assertAllClose([0., 0.], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testIterationWithoutMinimize(self):
    with self.cached_session():
      sgd = gradient_descent.SGD(3.0)
      self.evaluate(sgd.iterations.initializer)
      self.assertEqual(0, self.evaluate(sgd.iterations))

  @test_util.run_in_graph_and_eager_modes
  def testSerializationWithinDefun(self):
    with self.cached_session():
      sgd = gradient_descent.SGD(3.0)
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0],
                                                    dtype=dtypes.float32)
      loss = lambda: 3 * var0
      sgd.minimize(loss, [var0])

      def serialize():
        config = sgd.get_config()
        gradient_descent.SGD.from_config(config)

      compiled_serialize = function.defun(serialize)
      with self.assertRaisesRegexp(RuntimeError, 'inside Tensorflow graph'):
        compiled_serialize()

  @test_util.run_in_graph_and_eager_modes
  def testConfig(self):
    with self.cached_session():
      opt = gradient_descent.SGD(learning_rate=1.0)
      config = opt.get_config()
      opt2 = gradient_descent.SGD.from_config(config)
      # assert both are equal float values.
      self.assertEqual(
          opt._get_hyper('learning_rate'), opt2._get_hyper('learning_rate'))
      var0 = variables.Variable([[1.0], [2.0]], dtype=dtypes.float32)
      loss = lambda: 3 * var0
      # learning rate variable created when calling minimize.
      opt.minimize(loss, [var0])
      self.evaluate(variables.global_variables_initializer())
      config = opt.get_config()
      opt3 = gradient_descent.SGD.from_config(config)
      self.assertEqual(
          self.evaluate(opt._get_hyper('learning_rate')),
          opt3._get_hyper('learning_rate'))

  @test_util.run_in_graph_and_eager_modes
  def testWeights(self):
    with self.cached_session():
      opt1 = adam.Adam(learning_rate=1.0)
      var1 = resource_variable_ops.ResourceVariable([1.0, 2.0],
                                                    dtype=dtypes.float32)
      loss1 = lambda: 3 * var1
      opt_op_1 = opt1.minimize(loss1, [var1])
      self.evaluate(variables.global_variables_initializer())
      config = opt1.get_config()
      opt2 = adam.Adam.from_config(config)
      var2 = resource_variable_ops.ResourceVariable([1.0, 2.0],
                                                    dtype=dtypes.float32)
      loss2 = lambda: 3 * var2
      opt_op_2 = opt2.minimize(loss2, [var2])
      weights = opt1.get_weights()

      # Assert set_weights and both variables get updated to same value.
      self.evaluate(variables.global_variables_initializer())
      opt2.set_weights(weights)
      self.evaluate([opt_op_1, opt_op_2])
      self.assertAllClose(self.evaluate(var1), self.evaluate(var2))
      self.assertEqual(1, self.evaluate(opt1.iterations))
      self.assertEqual(1, self.evaluate(opt2.iterations))

      var3 = resource_variable_ops.ResourceVariable([1.0, 2.0, 3.0],
                                                    dtype=dtypes.float32)
      var4 = resource_variable_ops.ResourceVariable([4.0, 5.0, 6.0],
                                                    dtype=dtypes.float32)
      loss3 = lambda: 3 * var3 + 5 * var4
      opt_op_3 = opt1.minimize(loss3, [var3, var4])

      # Assert set_weights with ValueError since weight list does not match.
      self.evaluate(variables.global_variables_initializer())
      weights = opt1.get_weights()
      with self.assertRaisesRegexp(ValueError, 'but the optimizer was'):
        opt2.set_weights(weights)

      # Assert set_weights and variables get updated to same value.
      var5 = resource_variable_ops.ResourceVariable([1.0, 2.0, 3.0],
                                                    dtype=dtypes.float32)
      var6 = resource_variable_ops.ResourceVariable([4.0, 5.0, 6.0],
                                                    dtype=dtypes.float32)
      loss4 = lambda: 3 * var5 + 5 * var6
      opt_op_4 = opt2.minimize(loss4, [var5, var6])
      self.evaluate(variables.global_variables_initializer())
      opt2.set_weights(weights)
      self.evaluate([opt_op_3, opt_op_4])
      self.assertAllClose(
          self.evaluate([var3, var4]), self.evaluate([var5, var6]))

  @test_util.run_in_graph_and_eager_modes
  def testGettingHyperParameters(self):
    opt = adam.Adam(learning_rate=1.0)
    var = resource_variable_ops.ResourceVariable([1.0, 2.0],
                                                 dtype=dtypes.float32)
    loss = lambda: 3 * var
    opt_op = opt.minimize(loss, [var])
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(opt_op)

    lr = self.evaluate(opt.lr)
    self.assertEqual(1.0, lr)

    opt.lr = 2.0
    lr = self.evaluate(opt.lr)
    self.assertEqual(2.0, lr)

    self.evaluate(opt.lr.assign(3.0))
    lr = self.evaluate(opt.lr)
    self.assertEqual(3.0, lr)

    with self.assertRaises(AttributeError):
      opt.not_an_attr += 3

  @test_util.run_in_graph_and_eager_modes
  def testOptimizerWithKerasModel(self):
    a = input_layer.Input(shape=(3,), name='input_a')
    b = input_layer.Input(shape=(3,), name='input_b')

    dense = core.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = core.Dropout(0.5, name='dropout')(c)

    model = training.Model([a, b], [d, e])

    optimizer = gradient_descent.SGD(learning_rate=0.001)
    loss = 'mse'
    model.compile(optimizer, loss, metrics=['mae'])

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))

    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=1,
              batch_size=5)

  @test_util.run_in_graph_and_eager_modes
  def testOptimizerWithCallbacks(self):
    input_np = np.random.random((10, 3))
    output_np = np.random.random((10, 4))
    a = input_layer.Input(shape=(3,), name='input_a')
    model = sequential.Sequential()
    model.add(core.Dense(4, name='dense'))
    model.add(core.Dropout(0.5, name='dropout'))
    model(a)
    optimizer = gradient_descent.SGD(learning_rate=0.1)
    model.compile(optimizer, loss='mse', metrics=['mae'])
    # This does not reduce the LR after the first epoch (due to low delta).
    cbks = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, min_delta=0, patience=1, cooldown=5)
    ]
    model.fit(
        input_np,
        output_np,
        batch_size=10,
        validation_data=(input_np, output_np),
        callbacks=cbks,
        epochs=5,
        verbose=0)
    self.assertAllClose(
        float(backend.get_value(model.optimizer.lr)), 0.1, atol=1e-4)

    # This should reduce the LR after the first epoch (due to high delta).
    cbks = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            min_delta=10,
            patience=1,
            cooldown=5)
    ]
    model.fit(
        input_np,
        output_np,
        batch_size=10,
        validation_data=(input_np, output_np),
        callbacks=cbks,
        epochs=5,
        verbose=2)
    self.assertAllClose(
        float(backend.get_value(model.optimizer.lr)), 0.01, atol=1e-4)


# Note: These tests are kept in a separate class to avoid bugs in some
# distributions of Python that break AutoGraph which is used by tf.function.
class OptimizerWithFunctionTest(test.TestCase):

  def testBasic(self):
    with context.eager_mode():
      var = resource_variable_ops.ResourceVariable([1.0, 2.0],
                                                   dtype=dtypes.float32)
      loss = lambda: 3 * var
      opt = adam.Adam(learning_rate=1.0)

      @def_function.function
      def fn():
        opt.minimize(loss, [var])
        return var

      self.assertAllClose([0., 1.], fn(), atol=1e-4)
      self.assertAllClose([-1, 0.], fn(), atol=1e-4)


if __name__ == '__main__':
  test.main()
