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

import collections

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adadelta
from tensorflow.python.keras.optimizer_v2 import adagrad
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import adamax
from tensorflow.python.keras.optimizer_v2 import ftrl
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import nadam
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import momentum
from tensorflow.python.training import training_util
from tensorflow.python.training.tracking import util as trackable_utils


_DATA_TYPES = [dtypes.half, dtypes.float32, dtypes.float64]
# TODO(b/141710709): complex support in NVCC and ROCM.
if (not test_util.IsBuiltWithNvcc() and not test.is_built_with_rocm()):
  _DATA_TYPES += [dtypes.complex64, dtypes.complex128]


class OptimizerTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testBasic(self):
    for dtype in _DATA_TYPES:
      with testing_utils.use_gpu():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
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

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testAdaptiveLearningRate(self):
    for dtype in _DATA_TYPES:
      with self.test_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)

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

        sgd.learning_rate = learning_rate_schedule.InverseTimeDecay(
            0.5, decay_steps=1.0, decay_rate=0.5)
        if context.executing_eagerly():
          sgd.minimize(loss, [var0, var1])
        else:
          self.evaluate(opt_op)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testPrecomputedGradient(self):
    for dtype in _DATA_TYPES:
      with testing_utils.use_gpu():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
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

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testNoGradients(self):
    for dtype in _DATA_TYPES:
      with testing_utils.use_gpu():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0  # pylint: disable=cell-var-from-loop
        sgd_op = gradient_descent.SGD(3.0)
        with self.assertRaisesRegex(ValueError, 'No gradients'):
          # var1 has no gradient
          sgd_op.minimize(loss, var_list=[var1])

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testNoGradientsForAnyVariables_Minimize(self):
    for dtype in _DATA_TYPES:
      with testing_utils.use_gpu():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        loss = lambda: constant_op.constant(5.0)

        sgd_op = gradient_descent.SGD(3.0)
        with self.assertRaisesRegex(ValueError,
                                    'No gradients provided for any variable'):
          sgd_op.minimize(loss, var_list=[var0, var1])

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testNoGradientsForAnyVariables_ApplyGradients(self):
    for dtype in _DATA_TYPES:
      with testing_utils.use_gpu():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        sgd_op = gradient_descent.SGD(3.0)
        with self.assertRaisesRegex(ValueError,
                                    'No gradients provided for any variable'):
          sgd_op.apply_gradients([(None, var0), (None, var1)])

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testGradientsAsVariables(self):
    for i, dtype in enumerate(_DATA_TYPES):
      with testing_utils.use_gpu():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        loss = lambda: 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop

        sgd = gradient_descent.SGD(3.0)
        grads_and_vars = sgd._compute_gradients(loss, [var0, var1])
        # Convert gradients to tf.Variables
        converted_grads = [
            variables.Variable(
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

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testComputeGradientsWithTensors(self):
    with testing_utils.use_gpu():
      x = ops.convert_to_tensor_v2_with_dispatch(1.0)

      def f():
        return x * x

      sgd = gradient_descent.SGD(3.0)
      grads_and_vars = sgd._compute_gradients(f, [x])
      self.assertLen(grads_and_vars, 1)
      grad, x_as_var = grads_and_vars[0]
      self.assertIs(x, x_as_var)
      self.assertEqual(2.0, self.evaluate(grad))

      with self.assertRaises(NotImplementedError):
        sgd.apply_gradients(grads_and_vars)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testConstraint(self):
    constraint_01 = lambda x: clip_ops.clip_by_value(x, -0.1, 0.)
    constraint_0 = lambda x: clip_ops.clip_by_value(x, 0., 1.)
    with testing_utils.use_gpu():
      var0 = variables.Variable([1.0, 2.0],
                                constraint=constraint_01)
      var1 = variables.Variable([3.0, 4.0],
                                constraint=constraint_0)
      loss = lambda: 5 * var0 + 3 * var1
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

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testIterationWithoutMinimize(self):
    with testing_utils.use_gpu():
      sgd = gradient_descent.SGD(3.0)
      self.evaluate(sgd.iterations.initializer)
      self.assertEqual(0, self.evaluate(sgd.iterations))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testConfig(self):
    with testing_utils.use_gpu():
      opt = gradient_descent.SGD(learning_rate=1.0)
      config = opt.get_config()
      opt2 = gradient_descent.SGD.from_config(config)
      lr = opt._get_hyper('learning_rate')
      lr2 = opt2._get_hyper('learning_rate')
      self.evaluate(variables.global_variables_initializer())
      # assert both are equal float values.
      self.assertEqual(self.evaluate(lr), self.evaluate(lr2))
      var0 = variables.Variable([[1.0], [2.0]], dtype=dtypes.float32)
      loss = lambda: 3 * var0
      # learning rate variable created when calling minimize.
      opt.minimize(loss, [var0])
      opt3 = gradient_descent.SGD.from_config(config)
      lr3 = opt3._get_hyper('learning_rate')
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(self.evaluate(lr), self.evaluate(lr3))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testConfigWithLearningRateDecay(self):
    with testing_utils.use_gpu():
      var0 = variables.Variable([[1.0], [2.0]], dtype=dtypes.float32)
      for decay_schedule in [
          learning_rate_schedule.InverseTimeDecay(
              0.5, decay_steps=1.0, decay_rate=0.1),
          learning_rate_schedule.PiecewiseConstantDecay(
              [5], [1., .5])
      ]:
        step = 10
        opt = gradient_descent.SGD(decay_schedule)
        config = opt.get_config()
        opt2 = gradient_descent.SGD.from_config(config)
        # assert both are equal float values.
        self.assertAllEqual(
            decay_schedule(step),
            opt._get_hyper('learning_rate')(step))
        self.assertAllEqual(
            decay_schedule(step),
            opt2._get_hyper('learning_rate')(step))
        loss = lambda: 3 * var0
        # learning rate variable is created when calling minimize.
        opt.minimize(loss, [var0])
        self.evaluate(variables.global_variables_initializer())
        config = opt.get_config()
        opt3 = gradient_descent.SGD.from_config(config)
        self.assertAllEqual(
            self.evaluate(opt._get_hyper('learning_rate')(step)),
            opt3._get_hyper('learning_rate')(step))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testGradClipValue(self):
    with testing_utils.use_gpu():
      var = variables.Variable([1.0, 2.0])
      loss = lambda: 3 * var
      opt = gradient_descent.SGD(learning_rate=1.0, clipvalue=1.0)
      opt_op = opt.minimize(loss, [var])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)
      self.assertAllClose([0., 1.], self.evaluate(var))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testGradClipNorm(self):
    with testing_utils.use_gpu():
      var = variables.Variable([1.0])
      loss = lambda: 3 * var
      opt = gradient_descent.SGD(learning_rate=1.0, clipnorm=1.0)
      opt_op = opt.minimize(loss, [var])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)
      self.assertAllClose([0.], self.evaluate(var))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testGradGlobalClipNorm(self):
    with testing_utils.use_gpu():
      # l2 norm is 5.0
      var1 = variables.Variable([1.0])
      var2 = variables.Variable([2.0])
      loss = lambda: 3 * var1 + 4 * var2
      opt = gradient_descent.SGD(learning_rate=1.0, global_clipnorm=2.0)
      opt_op = opt.minimize(loss, [var1, var2])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)
      # grad1 = 3.0 * 2.0 / 5.0 = 1.2
      self.assertAllClose([-.2], self.evaluate(var1))
      # grad2 = 4.0 * 2.0 / 5.0 = 1.6
      self.assertAllClose([.4], self.evaluate(var2))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInvalidClipNorm(self):
    with self.assertRaisesRegex(ValueError, '>= 0'):
      gradient_descent.SGD(learning_rate=1.0, clipnorm=-1.0)

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'],
          clip_type=['clipnorm', 'global_clipnorm', 'clipvalue']))
  def testConfigWithCliping(self, clip_type):
    opt = gradient_descent.SGD(learning_rate=1.0, **{clip_type: 2.0})
    config = opt.get_config()
    opt = gradient_descent.SGD.from_config(config)
    self.assertEqual(getattr(opt, clip_type), 2.0)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInvalidKwargs(self):
    with self.assertRaisesRegex(TypeError, 'Unexpected keyword argument'):
      gradient_descent.SGD(learning_rate=1.0, invalidkwargs=1.0)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testWeights(self):
    with testing_utils.use_gpu():
      opt1 = adam.Adam(learning_rate=1.0)
      var1 = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
      loss1 = lambda: 3 * var1
      opt_op_1 = opt1.minimize(loss1, [var1])
      self.evaluate(variables.global_variables_initializer())
      config = opt1.get_config()
      opt2 = adam.Adam.from_config(config)
      var2 = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
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

      var3 = variables.Variable([1.0, 2.0, 3.0], dtype=dtypes.float32)
      var4 = variables.Variable([4.0, 5.0, 6.0], dtype=dtypes.float32)
      loss3 = lambda: 3 * var3 + 5 * var4
      opt_op_3 = opt1.minimize(loss3, [var3, var4])

      # Assert set_weights with ValueError since weight list does not match.
      self.evaluate(variables.global_variables_initializer())
      weights = opt1.get_weights()
      with self.assertRaisesRegex(ValueError, 'but the optimizer was'):
        opt2.set_weights(weights)

      # Assert set_weights and variables get updated to same value.
      var5 = variables.Variable([1.0, 2.0, 3.0], dtype=dtypes.float32)
      var6 = variables.Variable([4.0, 5.0, 6.0], dtype=dtypes.float32)
      loss4 = lambda: 3 * var5 + 5 * var6
      opt_op_4 = opt2.minimize(loss4, [var5, var6])
      self.evaluate(variables.global_variables_initializer())
      opt2.set_weights(weights)
      self.evaluate([opt_op_3, opt_op_4])
      self.assertAllClose(
          self.evaluate([var3, var4]), self.evaluate([var5, var6]))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testGettingHyperParameters(self):
    with self.test_session():
      opt = adam.Adam(learning_rate=1.0)
      var = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
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

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testGettingHyperParametersWithLrInConstructor(self):
    with self.test_session():
      opt = gradient_descent.SGD(lr=3.0)
      var = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
      loss = lambda: 3 * var
      opt_op = opt.minimize(loss, [var])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)

      self.assertIsInstance(opt.lr, variables.Variable)
      self.assertIsInstance(opt.learning_rate, variables.Variable)

      lr = self.evaluate(opt.lr)
      self.assertEqual(3.0, lr)

      opt.lr = 2.0
      lr = self.evaluate(opt.lr)
      self.assertEqual(2.0, lr)

      self.evaluate(opt.lr.assign(4.0))
      lr = self.evaluate(opt.lr)
      self.assertEqual(4.0, lr)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testDir(self):
    opt = gradient_descent.SGD(learning_rate=1.0, momentum=0.1)
    dir_result = set(dir(opt))
    self.assertIn('learning_rate', dir_result)  # Hyperparameter
    self.assertIn('lr', dir_result)  # Hyperparameter
    self.assertIn('momentum', dir_result)  # Hyperparameter
    self.assertIn('nesterov', dir_result)  # Attribute
    self.assertIn('minimize', dir_result)  # Attribute

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
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

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testOptimizerWithCallbacks(self):
    np.random.seed(1331)
    input_np = np.random.random((10, 3))
    output_np = np.random.random((10, 4))
    a = input_layer.Input(shape=(3,), name='input_a')
    model = sequential.Sequential()
    model.add(core.Dense(4, kernel_initializer='zeros', name='dense'))
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
        epochs=2,
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
        epochs=2,
        verbose=2)
    self.assertAllClose(
        float(backend.get_value(model.optimizer.lr)), 0.01, atol=1e-4)

  def testOptimizerSetIterations(self):
    global_step = training_util.get_or_create_global_step()
    opt = adam.Adam(learning_rate=1.0)
    opt.iterations = global_step
    var = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
    self.evaluate(variables.global_variables_initializer())
    init_step_value = self.evaluate(global_step)
    loss = lambda: 3 * var
    opt_op = opt.minimize(loss, [var])
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(opt_op)
    new_step_value = self.evaluate(global_step)
    self.assertEqual(new_step_value, init_step_value + 1)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testOptimizerWithCallableVarList(self):
    train_samples = 20
    input_dim = 1
    num_classes = 2
    (x, y), _ = testing_utils.get_test_data(
        train_samples=train_samples,
        test_samples=10,
        input_shape=(input_dim,),
        num_classes=num_classes)
    y = np_utils.to_categorical(y)

    num_hidden = 1
    model = testing_utils.get_small_sequential_mlp(
        num_hidden=num_hidden, num_classes=num_classes)
    opt = adam.Adam()

    loss = lambda: losses.mean_squared_error(model(x), y)
    var_list = lambda: model.trainable_weights

    with self.assertRaisesRegex(
        ValueError, 'Weights for model .* have not yet been created'):
      var_list()
    train_op = opt.minimize(loss, var_list)
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(
          [[0.]], self.evaluate(opt.get_slot(var_list()[0], 'm')))
      self.evaluate(train_op)
    self.assertNotEqual(
        [[0.]], self.evaluate(opt.get_slot(var_list()[0], 'm')))
    self.assertLen(var_list(), 4)

  def testVarKey(self):
    with ops.get_default_graph().as_default():
      a = variables.Variable([1., 2.], name='var')
      b = variables.Variable([1.], name='var')
      self.assertTrue(a._in_graph_mode)
      self.assertTrue(b._in_graph_mode)
      var_key = optimizer_v2._var_key(a)
      self.assertEqual('var', var_key)
      var_key = optimizer_v2._var_key(b)
      self.assertEqual('var_1', var_key)

  def testVarName(self):
    with ops.get_default_graph().as_default():
      var = variables.Variable([1., 2.], name='var')
      loss = var + 1.
      opt = adam.Adam()
      opt.get_updates(loss, [var])
      opt_vars = opt.variables()
      self.assertLen(opt_vars, 3)
      self.assertEqual('Adam/iter:0', opt_vars[0].name)
      self.assertEqual('Adam/var/m:0', opt_vars[1].name)
      var_2 = variables.Variable([1., 2.], name='var_2')
      loss = var_2 + 1.
      with backend.name_scope('outter'):
        opt.get_updates(loss, [var_2])
      opt_vars = opt.variables()
      self.assertLen(opt_vars, 5)
      self.assertEqual('outter/Adam/var_2/m:0', opt_vars[3].name)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testEmptyVarList(self):
    opt = gradient_descent.SGD(1.)
    opt.minimize(lambda: constant_op.constant(1.), [])
    opt.apply_gradients([])

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testAggregationTrue(self):
    # Test that experimental_aggregate_gradients=True works without distributed
    # strategy.
    var = variables.Variable([1., 2.])
    opt = gradient_descent.SGD(3.0)

    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose([1., 2.], self.evaluate(var))
    opt_op = opt.apply_gradients([([0.1, 0.1], var)],
                                 experimental_aggregate_gradients=True)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(opt_op)
    self.assertAllClose([0.7, 1.7], self.evaluate(var))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testAggregationFalse(self):
    # Test that experimental_aggregate_gradients=False works without distributed
    # strategy.
    var = variables.Variable([1., 2.])
    opt = gradient_descent.SGD(3.0)

    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose([1., 2.], self.evaluate(var))
    opt_op = opt.apply_gradients([([0.1, 0.1], var)],
                                 experimental_aggregate_gradients=False)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(opt_op)
    self.assertAllClose([0.7, 1.7], self.evaluate(var))

  @combinations.generate(combinations.combine(mode=['eager']))
  def testRestoringIterationsWithoutAnOptimizer(self):
    opt = gradient_descent.SGD(3.0)
    opt.iterations.assign(5)
    checkpoint = trackable_utils.Checkpoint(optimizer=opt)
    path = checkpoint.save(self.get_temp_dir())

    # Following verifies that the `iterations` can be restored with the absence
    # of an `Optimizer` object (using a `Checkpoint` as a placeholder).
    iterations_var = variables.Variable(0, dtype=dtypes.int64)
    optimizer_checkpoint = trackable_utils.Checkpoint(iter=iterations_var)
    checkpoint_to_restore = trackable_utils.Checkpoint(
        optimizer=optimizer_checkpoint)
    checkpoint_to_restore.restore(path)

    self.assertEqual(5, self.evaluate(iterations_var))

  @combinations.generate(combinations.combine(mode=['eager']))
  def testSlotWithNonstandardShapeRestoresBasedOnCheckpoint(self):
    # First create an optimizer and a slot variable with a non-standard shape.
    x = variables.Variable([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float32)
    slot_shape = [2, 1]
    optimizer_1 = optimizer_v2.OptimizerV2(name='test')
    optimizer_1.add_slot(x, 'test_slot', 'ones', shape=slot_shape)

    # Then save the variable and optimizer to a checkpoint.
    checkpoint_1 = trackable_utils.Checkpoint(var=x, optimizer=optimizer_1)
    checkpoint_path = checkpoint_1.save(self.get_temp_dir())

    # Create a new optimizer and call restore on it (and x)
    optimizer_2 = optimizer_v2.OptimizerV2(name='test')
    checkpoint_2 = trackable_utils.Checkpoint(var=x, optimizer=optimizer_2)
    checkpoint_2.restore(checkpoint_path)

    self.assertEqual(slot_shape,
                     optimizer_2.get_slot(x, 'test_slot').shape.as_list())

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_gradient_aggregator(self):
    def gradient_aggregator(grads_and_vars):
      # Simulate an all-reduce where the other replica has zeros for gradients,
      # by dividing each gradient by 2.
      grads = [g for g, _ in grads_and_vars]
      vars = [v for _, v in grads_and_vars]  # pylint: disable=redefined-builtin
      all_reduced_grads = [g / 2 for g in grads]
      return list(zip(all_reduced_grads, vars))

    var = variables.Variable(2.0)
    sgd = gradient_descent.SGD(1.0, gradient_aggregator=gradient_aggregator)
    loss = lambda: 2 * var
    opt_op = sgd.minimize(loss, var_list=[var])
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(opt_op)
    self.assertEqual(self.evaluate(var), 1.0)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_override_aggregate_gradients(self):
    class MyOptimizer(gradient_descent.SGD):

      def _aggregate_gradients(self, grads_and_vars):
        # Simulate an all-reduce where the other replica has zeros for
        # gradients, by dividing each gradient by 2.
        grads = [g for g, _ in grads_and_vars]
        vars = [v for _, v in grads_and_vars]  # pylint: disable=redefined-builtin
        all_reduced_grads = [g / 2 for g in grads]
        return list(zip(all_reduced_grads, vars))

    var = variables.Variable(2.0)
    sgd = MyOptimizer(1.0)
    loss = lambda: 2 * var
    opt_op = sgd.minimize(loss, var_list=[var])
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(opt_op)
    self.assertEqual(self.evaluate(var), 1.0)


@keras_parameterized.run_all_keras_modes
class OptimizersCompatibilityTest(keras_parameterized.TestCase):

  def _testOptimizersCompatibility(self, opt_v1, opt_v2, test_weights=True):
    if context.executing_eagerly():
      self.skipTest(
          'v1 optimizer does not run in eager mode')
    np.random.seed(1331)
    with testing_utils.use_gpu():
      train_samples = 20
      input_dim = 3
      num_classes = 2
      (x, y), _ = testing_utils.get_test_data(
          train_samples=train_samples,
          test_samples=10,
          input_shape=(input_dim,),
          num_classes=num_classes)
      y = np_utils.to_categorical(y)

      num_hidden = 5
      model_v1 = testing_utils.get_small_sequential_mlp(
          num_hidden=num_hidden, num_classes=num_classes, input_dim=input_dim)
      model_v1.compile(
          opt_v1,
          loss='categorical_crossentropy',
          metrics=[],
          run_eagerly=testing_utils.should_run_eagerly())
      model_v1.fit(x, y, batch_size=5, epochs=1)

      model_v2 = testing_utils.get_small_sequential_mlp(
          num_hidden=num_hidden, num_classes=num_classes, input_dim=input_dim)
      model_v2.set_weights(model_v1.get_weights())
      model_v2.compile(
          opt_v2,
          loss='categorical_crossentropy',
          metrics=[],
          run_eagerly=testing_utils.should_run_eagerly())
      if not ops.executing_eagerly_outside_functions():
        model_v2._make_train_function()
      if test_weights:
        opt_v2.set_weights(opt_v1.get_weights())

      hist_1 = model_v1.fit(x, y, batch_size=5, epochs=1, shuffle=False)
      hist_2 = model_v2.fit(x, y, batch_size=5, epochs=1, shuffle=False)
      self.assertAllClose(model_v1.get_weights(), model_v2.get_weights(),
                          rtol=1e-5, atol=1e-5)
      self.assertAllClose(hist_1.history['loss'], hist_2.history['loss'],
                          rtol=1e-5, atol=1e-5)

  def testAdadeltaCompatibility(self):
    opt_v1 = optimizer_v1.Adadelta(lr=0.01)
    opt_v2 = adadelta.Adadelta(learning_rate=0.01)
    self._testOptimizersCompatibility(opt_v1, opt_v2)

  def testAdagradCompatibility(self):
    opt_v1 = optimizer_v1.Adagrad(lr=0.01)
    opt_v2 = adagrad.Adagrad(learning_rate=0.01)
    self._testOptimizersCompatibility(opt_v1, opt_v2)

  def testAdamCompatibility(self):
    opt_v1 = optimizer_v1.Adam()
    opt_v2 = adam.Adam()
    self._testOptimizersCompatibility(opt_v1, opt_v2)

  def testAdamaxCompatibility(self):
    opt_v1 = optimizer_v1.Adamax(lr=0.01)
    opt_v2 = adamax.Adamax(learning_rate=0.01)
    self._testOptimizersCompatibility(opt_v1, opt_v2)

  def testNadamCompatibility(self):
    opt_v1 = optimizer_v1.Nadam(lr=0.001)
    opt_v2 = nadam.Nadam(learning_rate=0.001)
    self._testOptimizersCompatibility(opt_v1, opt_v2)

  def testMomentumCompatibility(self):
    opt_v1 = optimizer_v1.SGD(lr=0.01, momentum=0.9)
    opt_v2 = gradient_descent.SGD(learning_rate=0.01, momentum=0.9)
    self._testOptimizersCompatibility(opt_v1, opt_v2)

  def testRMSpropCompatibility(self):
    opt_v1 = optimizer_v1.RMSprop()
    opt_v2 = rmsprop.RMSprop()
    self._testOptimizersCompatibility(opt_v1, opt_v2)

  def testSGDCompatibility(self):
    opt_v1 = optimizer_v1.SGD(lr=0.01)
    opt_v2 = gradient_descent.SGD(learning_rate=0.01)
    self._testOptimizersCompatibility(opt_v1, opt_v2, False)

  def testNumericEquivalenceForNesterovMomentum(self):
    if context.executing_eagerly():
      self.skipTest(
          'v1 optimizer does not run in eager mode')
    np.random.seed(1331)
    with testing_utils.use_gpu():
      train_samples = 20
      input_dim = 3
      num_classes = 2
      (x, y), _ = testing_utils.get_test_data(
          train_samples=train_samples,
          test_samples=10,
          input_shape=(input_dim,),
          num_classes=num_classes)
      y = np_utils.to_categorical(y)

      num_hidden = 5
      model_k_v1 = testing_utils.get_small_sequential_mlp(
          num_hidden=num_hidden, num_classes=num_classes, input_dim=input_dim)
      model_k_v2 = testing_utils.get_small_sequential_mlp(
          num_hidden=num_hidden, num_classes=num_classes, input_dim=input_dim)
      model_k_v2.set_weights(model_k_v1.get_weights())
      model_tf = testing_utils.get_small_sequential_mlp(
          num_hidden=num_hidden, num_classes=num_classes, input_dim=input_dim)
      model_tf.set_weights(model_k_v2.get_weights())

      opt_k_v1 = optimizer_v1.SGD(momentum=0.9, nesterov=True)
      opt_k_v2 = gradient_descent.SGD(momentum=0.9, nesterov=True)
      opt_tf = momentum.MomentumOptimizer(
          learning_rate=0.01, momentum=0.9, use_nesterov=True)

      model_k_v1.compile(
          opt_k_v1,
          loss='categorical_crossentropy',
          metrics=[],
          run_eagerly=testing_utils.should_run_eagerly())
      model_k_v2.compile(
          opt_k_v2,
          loss='categorical_crossentropy',
          metrics=[],
          run_eagerly=testing_utils.should_run_eagerly())
      model_tf.compile(
          opt_tf,
          loss='categorical_crossentropy',
          metrics=[],
          run_eagerly=testing_utils.should_run_eagerly())

      hist_k_v1 = model_k_v1.fit(x, y, batch_size=5, epochs=10, shuffle=False)
      hist_k_v2 = model_k_v2.fit(x, y, batch_size=5, epochs=10, shuffle=False)
      hist_tf = model_tf.fit(x, y, batch_size=5, epochs=10, shuffle=False)

      self.assertAllClose(model_k_v1.get_weights(), model_tf.get_weights())
      self.assertAllClose(model_k_v1.get_weights(), model_k_v2.get_weights())
      self.assertAllClose(opt_k_v1.get_weights(), opt_k_v2.get_weights())
      self.assertAllClose(hist_k_v1.history['loss'], hist_tf.history['loss'])
      self.assertAllClose(hist_k_v1.history['loss'], hist_k_v2.history['loss'])

  def testNumericEquivalenceForAmsgrad(self):
    if context.executing_eagerly():
      self.skipTest(
          'v1 optimizer does not run in eager mode')
    np.random.seed(1331)
    with testing_utils.use_gpu():
      train_samples = 20
      input_dim = 3
      num_classes = 2
      (x, y), _ = testing_utils.get_test_data(
          train_samples=train_samples,
          test_samples=10,
          input_shape=(input_dim,),
          num_classes=num_classes)
      y = np_utils.to_categorical(y)

      num_hidden = 5
      model_k_v1 = testing_utils.get_small_sequential_mlp(
          num_hidden=num_hidden, num_classes=num_classes, input_dim=input_dim)
      model_k_v2 = testing_utils.get_small_sequential_mlp(
          num_hidden=num_hidden, num_classes=num_classes, input_dim=input_dim)
      model_k_v2.set_weights(model_k_v1.get_weights())

      opt_k_v1 = optimizer_v1.Adam(amsgrad=True)
      opt_k_v2 = adam.Adam(amsgrad=True)

      model_k_v1.compile(
          opt_k_v1,
          loss='categorical_crossentropy',
          metrics=[],
          run_eagerly=testing_utils.should_run_eagerly())
      model_k_v2.compile(
          opt_k_v2,
          loss='categorical_crossentropy',
          metrics=[],
          run_eagerly=testing_utils.should_run_eagerly())

      hist_k_v1 = model_k_v1.fit(x, y, batch_size=5, epochs=10, shuffle=False)
      hist_k_v2 = model_k_v2.fit(x, y, batch_size=5, epochs=10, shuffle=False)

      self.assertAllClose(model_k_v1.get_weights(), model_k_v2.get_weights())
      self.assertAllClose(opt_k_v1.get_weights(), opt_k_v2.get_weights())
      self.assertAllClose(hist_k_v1.history['loss'], hist_k_v2.history['loss'])


# Note: These tests are kept in a separate class to avoid bugs in some
# distributions of Python that break AutoGraph which is used by tf.function.
@combinations.generate(combinations.combine(mode=['eager']))
class OptimizerWithFunctionTest(test.TestCase, parameterized.TestCase):

  def testBasic(self):
    var = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
    loss = lambda: 3 * var
    opt = adam.Adam(learning_rate=1.0)

    @def_function.function
    def fn():
      opt.minimize(loss, [var])
      return var

    self.assertAllClose([0., 1.], fn(), atol=1e-4)
    self.assertAllClose([-1, 0.], fn(), atol=1e-4)

  def testBasicWithConstantDecay(self):
    var = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
    loss = lambda: 3 * var
    opt = adam.Adam(learning_rate=1.0)

    @def_function.function
    def fn():
      opt.minimize(loss, [var])
      return var

    self.assertAllClose([0., 1.], fn(), atol=1e-4)
    self.assertAllClose([-1, 0.], fn(), atol=1e-4)

  def testVarKeyWithVarCreatedInEager(self):
    a = variables.Variable([1., 2.], name='var')
    b = variables.Variable([1.], name='var')

    @test_util.also_run_as_tf_function
    def var_key_test():
      self.assertFalse(a._in_graph_mode)
      self.assertFalse(b._in_graph_mode)
      var_key_a = optimizer_v2._var_key(a)
      self.assertStartsWith(var_key_a, 'var_')
      var_key_b = optimizer_v2._var_key(b)
      self.assertStartsWith(var_key_b, 'var_')
      self.assertNotEqual(var_key_a, var_key_b)

    var_key_test()

  def testLearningRateDecayUsedInTwoFunctions(self):
    a = variables.Variable([1., 2.], name='var')
    b = variables.Variable([1.], name='var')

    learning_rate_decay = learning_rate_schedule.InverseTimeDecay(
        0.5, decay_steps=1.0, decay_rate=0.5)
    opt = adam.Adam(learning_rate=learning_rate_decay)
    loss_a = lambda: 3 * a
    loss_b = lambda: 2 * b

    @def_function.function
    def fn_a():
      opt.minimize(loss_a, [a])
      return a

    @def_function.function
    def fn_b():
      opt.minimize(loss_b, [b])
      return b

    fn_a()
    fn_b()


_NUM_LEARNERS = 50
APPLY_SCOPE = 'debug_apply'
ALLOWLIST = [
    # optimizer_v2._deduplicate_indexed_slices contains an indexed slice:
    #   array_ops.shape(unique_indices)[0]
    # which winds up expanding to [0:1:1] thereby creating three constants
    # to represent the indices.
    ('embeddings/strided_slice/stack', 'Const'),
]


def get_inputs(op):
  op_inputs = list(op.inputs) + op.control_inputs
  names = [i.name for i in op_inputs]
  op_inputs = [getattr(i, 'op', i) for i in op_inputs]
  return op_inputs, names


def strip_name(node):
  if 'Placeholder' in node.op:
    return
  node.name = ''


def topological_sort(graph):
  graph_ops = graph.get_operations()

  sources = []
  result = []

  inputs = {}
  outputs = collections.defaultdict(set)
  for op in graph_ops:
    op_inputs = get_inputs(op)[0]
    if not op_inputs:
      sources.append(op)

    inputs[op] = set(op_inputs)
    for i in op_inputs:
      outputs[i].add(op)

  while sources:
    op = sources.pop()
    for op_output in outputs[op]:
      inputs[op_output].remove(op)
      if not inputs[op_output]:
        sources.append(op_output)

    result.append(op)

  # Check correctness.
  if len(result) != len(graph_ops):
    raise ValueError('Sort result has {} ops, source graph has {}.'
                     .format(len(result), len(graph_ops)))

  sort_check_seen = set()
  for op in result:
    sort_check_seen.add(op)
    for i in get_inputs(op)[0]:
      assert i in sort_check_seen

  return result


def identify_redundant_ops(graph):
  """Implements basic common subexpression elimination.

  This is not intended to replicate the graph semantics of TensorFlow Graphs
  (for instance it does not handle stateful op ordering), nor is it intended to
  replace the common subexpression elimination Grappler pass. Rather, it
  provides a high level sanity check that clearly redundant ops are not being
  created.

  Args:
    graph: The graph to be analyzed.

  Returns:
    A count of the duplicate ops and a description of the structure of each.
  """
  sorted_ops = topological_sort(graph)
  duplicates = collections.defaultdict(list)
  unified_node_defs = {}
  name_map = {}

  for op in sorted_ops:
    input_names = []
    for op_input, name in zip(*get_inputs(op)):
      input_def = op_input.node_def

      # Operations can have multiple outputs. We track which is used to prevent
      # overzealous elimination.
      input_def.name = name

      input_def.input[:] = [name_map.get(i, i) for i in input_def.input]
      strip_name(input_def)

      # NodeDef.SerializeToString() does not provide identical serialized
      # representations for identical NodeDefs, so we instead use string
      # representation as a dict key.
      key = repr(input_def)

      if key in unified_node_defs:
        input_names.append(unified_node_defs[key])

      else:
        unified_node_defs[key] = op_input.name
        input_names.append(name)

    node_def = op.node_def
    node_def.input[:] = input_names
    strip_name(node_def)

    key = repr(node_def)
    duplicates[key].append(op)
    name_map[op.name] = duplicates[key][0].name

  num_duplicates = 0
  duplicate_types = []
  for standard_def, op_defs in duplicates.items():
    # We are only interested in testing the apply method of the optimizer
    op_defs = [i for i in op_defs if APPLY_SCOPE in i.name]

    # We only check for per-apply redundant ops.
    if len(op_defs) < _NUM_LEARNERS:
      continue

    # Certain ops are simply not worth eliminating, and are instead simply
    # ignored.
    name, op_type = op_defs[0].name, op_defs[0].type
    if any(allowlisted_scope in name and op_type == allowlisted_type
           for allowlisted_scope, allowlisted_type in ALLOWLIST):
      continue

    num_duplicates += len(op_defs)
    traceback = []
    for level in op_defs[0].traceback:
      traceback.append('  {} {}:{}'.format(level[0], level[2], level[1]))

    duplicate_types.append(
        '# Example name: {}\n# Op creation stack:\n{}\n{}'.format(
            op_defs[0].name,
            '\n'.join(traceback),
            standard_def))

  return num_duplicates, duplicate_types


def make_model():
  r"""Constructs a simple ensemble of weak learners model.

  ---------    ---------             ---------    ---------
  | Input |    | Input |     ...     | Input |    | Input |
  ---------    ---------             ---------    ---------
      |            |                     |            |
      V            V                     V            V
  ---------    ---------             ---------    ---------
  | Embed |    | Embed |     ...     | Embed |    | Embed |
  ---------    ---------             ---------    ---------
      |            |                     |            |
      V            V                     V            V
  ---------    ---------             ---------    ---------
  | Dense |    | Dense |     ...     | Dense |    | Dense |
  ---------    ---------             ---------    ---------
      \            |                     |            /
       \           |                     |           /
        ---------------------------------------------
                              |
                          ---------
                          | Dense |
                          ---------

  This topology is chosen because it exercises both dense and sparse update
  paths.

  Returns:
    A model for testing optimizer coefficient reuse.
  """
  inputs = []
  intermediates = []
  for _ in range(_NUM_LEARNERS):
    inp = keras.layers.Input(shape=(1,), dtype=dtypes.int32)
    layer = keras.layers.Embedding(1, 4)(inp)
    layer = keras.layers.Dense(1)(layer)

    inputs.append(inp)
    intermediates.append(layer)

  layer = keras.layers.Concatenate(axis=-1)(intermediates)
  layer = keras.layers.Dense(1)(layer)

  return keras.models.Model(inputs, layer)


COEFFICIENT_PARAMS = (
    ('Adadelta', adadelta.Adadelta, None),
    ('Adagrad', adagrad.Adagrad, None),
    ('Adam', adam.Adam, None),
    ('Adam_amdgrad', adam.Adam, dict(amsgrad=True)),
    ('Adamax', adamax.Adamax, None),
    ('Ftrl', ftrl.Ftrl, None),
    ('Ftrl_l2_shrinkage', ftrl.Ftrl,
     dict(l2_shrinkage_regularization_strength=0.1)),
    ('SGD', gradient_descent.SGD, None),
    ('SGD_momentum', gradient_descent.SGD, dict(momentum=0.5)),
    ('Nadam', nadam.Nadam, None),
    ('RMSprop', rmsprop.RMSprop, None),
    ('RMSprop_centered', rmsprop.RMSprop, dict(centered=True)),
    ('RMSprop_momentum', rmsprop.RMSprop, dict(momentum=0.5)),
    ('RMSprop_momentum_centered', rmsprop.RMSprop,
     dict(momentum=0.5, centered=True)),
)


class OptimizerCoefficientTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(*COEFFICIENT_PARAMS)
  def test_duplicate_ops(self, optimizer_class, init_kwargs=None):
    init_kwargs = init_kwargs or {}
    optimizer = optimizer_class(**init_kwargs)

    graph = ops.Graph()
    with graph.as_default():
      model = make_model()
      trainable_variables = model.trainable_variables
      grads = optimizer.get_gradients(model.outputs[0], trainable_variables)

      with backend.name_scope(APPLY_SCOPE):
        optimizer.apply_gradients(zip(grads, trainable_variables))

    num_duplicates, duplicate_types = identify_redundant_ops(graph)
    if num_duplicates:
      # Avoid spamming logs.
      if len(duplicate_types) > 3:
        duplicate_types = duplicate_types[:3] + ['...']

      num_total = len(graph.get_operations())
      raise ValueError('{} of {} ({:.1f}%) ops were duplicates:\n\n{}'.format(
          num_duplicates, num_total, num_duplicates / num_total * 100,
          '\n'.join(duplicate_types)))

  @parameterized.named_parameters(*COEFFICIENT_PARAMS)
  def test_subclass_compat(self, optimizer_class, init_kwargs=None):
    """Ensure that subclassed optimizers without apply_state still work."""

    class SubclassedOptimizer(optimizer_class):

      def _resource_apply_dense(self, grad, var):  # pylint: disable=useless-super-delegation
        return super(SubclassedOptimizer, self)._resource_apply_dense(grad, var)

      def _resource_apply_sparse(self, grad, var, indices):  # pylint: disable=useless-super-delegation
        return super(SubclassedOptimizer, self)._resource_apply_sparse(
            grad, var, indices)

    init_kwargs = init_kwargs or {}
    optimizer = SubclassedOptimizer(**init_kwargs)

    graph = ops.Graph()
    with graph.as_default():
      model = make_model()
      trainable_variables = model.trainable_variables
      grads = optimizer.get_gradients(model.outputs[0], trainable_variables)

      with backend.name_scope(APPLY_SCOPE):
        optimizer.apply_gradients(zip(grads, trainable_variables))


if __name__ == '__main__':
  test.main()
