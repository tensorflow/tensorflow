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
"""Tests mixed precision works correctly with Keras layers and models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.keras.mixed_precision.experimental import test_util as mp_test_util
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class AssertTypeLayer(base_layer.Layer):
  """A layer which asserts it's inputs are a certain type."""

  def __init__(self, assert_type=None, **kwargs):
    self._assert_type = assert_type
    super(AssertTypeLayer, self).__init__(**kwargs)

  def assert_input_types(self, inputs):
    """Asserts `inputs` are of the correct type. Should be called in call()."""
    if self._assert_type:
      inputs_flattened = nest.flatten(inputs)
      for inp in inputs_flattened:
        assert inp.dtype.base_dtype == self._assert_type, (
            'Input tensor has type %s which does not match assert type %s' %
            (inp.dtype.name, self._assert_type.name))


class AddLayer(AssertTypeLayer):
  """A layer which adds it's input to a scalar variable."""

  def __init__(self, regularizer=None, use_operator=False, **kwargs):
    """Initializes the AddLayer.

    Args:
      regularizer: The regularizer on the scalar variable.
      use_operator: If True, add using the + operator. If False, add using
        tf.add.
      **kwargs: Passed to AssertTypeLayer constructor.
    """
    self._regularizer = regularizer
    self._use_operator = use_operator
    super(AddLayer, self).__init__(**kwargs)

  def build(self, _):
    self.v = self.add_weight('v', (), initializer='ones',
                             regularizer=self._regularizer)
    self.built = True

  def call(self, inputs):
    self.assert_input_types(inputs)
    assert inputs.dtype == self.v.dtype
    return self._add(inputs, self.v)

  def _add(self, x, y):
    if self._use_operator:
      return x + y
    else:
      return math_ops.add(x, y)


class AddLayerWithoutAutoCast(AddLayer):
  """Same as AddLayer, but does not use AutoCastVariables."""

  def build(self, _):
    dtype = self.dtype
    if dtype in ('float16', 'bfloat16'):
      dtype = 'float32'
    self.v = self.add_weight('v', (), initializer='ones', dtype=dtype,
                             experimental_autocast=False,
                             regularizer=self._regularizer)
    self.built = True

  def call(self, inputs):
    self.assert_input_types(inputs)
    assert self.v.dtype in (dtypes.float32, dtypes.float64)
    return self._add(inputs, math_ops.cast(self.v, inputs.dtype))


class IdentityRegularizer(regularizers.Regularizer):

  def __call__(self, x):
    assert x.dtype == dtypes.float32
    return array_ops.identity(x)


def create_one_device_strategy():
  return one_device_strategy.OneDeviceStrategy('cpu:0')


def create_mirrored_strategy():
  if context.num_gpus() >= 1:
    return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
  else:
    return mirrored_strategy.MirroredStrategy(['cpu:0'])


TESTCASES = ({
    'testcase_name': 'base',
    'strategy_fn': create_one_device_strategy
}, {
    'testcase_name': 'distribute',
    'strategy_fn': create_mirrored_strategy
})


class KerasLayerTest(test.TestCase, parameterized.TestCase):
  """Test mixed precision with Keras layers."""

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_variables_in_float32(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():
      with policy.policy_scope('infer_float32_vars'):
        layer = AddLayer(assert_type=dtypes.float16)
        y = layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float32)
        self.assertEqual(y.dtype, dtypes.float16)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(y), 2.)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_layer_with_non_autocast_variable(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():
      with policy.policy_scope('infer_float32_vars'):
        layer = AddLayerWithoutAutoCast(assert_type=dtypes.float16)
        y = layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float32)
        self.assertEqual(y.dtype, dtypes.float16)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(y), 2.)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_layer_regularizer_runs_in_float32(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():
      with policy.policy_scope('infer_float32_vars'):
        # Test on AddLayer
        layer = AddLayer(assert_type=dtypes.float16,
                         regularizer=IdentityRegularizer())
        layer(x)
        (regularizer_loss,) = layer.losses
        self.assertEqual(regularizer_loss.dtype, dtypes.float32)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(regularizer_loss), 1.)

        # Test on AddLayerWithoutAutoCast
        layer = AddLayerWithoutAutoCast(assert_type=dtypes.float16,
                                        regularizer=IdentityRegularizer())
        layer(x)
        (regularizer_loss,) = layer.losses
        self.assertEqual(regularizer_loss.dtype, dtypes.float32)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(regularizer_loss), 1.)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_passing_policy_to_layer(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():
      # Passing a Policy to 'dtype' sets the policy for that layer.
      layer = AddLayer(assert_type=dtypes.float16,
                       dtype=policy.Policy('infer_float32_vars'))
      # layer.dtype refers to the variable dtype
      self.assertEqual(layer.dtype, dtypes.float32)
      layer(x)
      self.assertEqual(layer.v.dtype, dtypes.float32)
      with policy.policy_scope('infer_float32_vars'):
        # Passing a Policy to dtype overrides the global Policy
        layer = AddLayer(assert_type=dtypes.float16,
                         dtype=policy.Policy('infer'))
        # layer dtype is not yet known
        self.assertEqual(layer.dtype, None)
        layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float16)
        self.assertEqual(layer.dtype, dtypes.float16)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_gradient(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope() as strategy:
      with policy.policy_scope('infer_float32_vars'):
        layer = AddLayer(assert_type=dtypes.float16)
        def run_fn():
          with backprop.GradientTape() as tape:
            y = layer(x)
            # Divide by num_replicas_in_sync, as the effective total loss is the
            # sum of each of the replica's losses.
            y /= strategy.num_replicas_in_sync

          # Learning rate is small enough that if applied to a float16 variable,
          # the variable will not change. So this tests the learning rate is not
          # applied to a float16 value, but instead the float32 variable.
          opt = gradient_descent.SGD(2 ** -14)
          grad = tape.gradient(y, layer.v)
          return opt.apply_gradients([(grad, layer.v)])

        op = strategy.experimental_run(run_fn)
        if not context.executing_eagerly():
          self.evaluate(variables.global_variables_initializer())
          self.evaluate(op)
        # The gradient with respective to the variable is 1. Since the
        # variable is initialized with 1 and the learning rate is 2**-14, the
        # new variable value should be: init_val - gradient * learning_rate,
        # which is  1 - 1 * 2**-14
        self.assertEqual(self.evaluate(layer.v), 1 - 2 ** -14)


class KerasModelTest(test.TestCase, parameterized.TestCase):
  """Test mixed precision with Keras models."""

  @parameterized.named_parameters({
      'testcase_name': 'base',
      'strategy_fn': create_one_device_strategy,
  }, {
      'testcase_name': 'distribute',
      'strategy_fn': create_mirrored_strategy,
  }, {
      'testcase_name': 'operator',
      'strategy_fn': create_mirrored_strategy,
      'use_operator': True
  }, {
      'testcase_name': 'regularizer',
      'strategy_fn': create_mirrored_strategy,
      'use_regularizer': True
  })
  @test_util.run_in_graph_and_eager_modes
  def test_model(self, strategy_fn, use_operator=False, use_regularizer=False):
    regularizer = IdentityRegularizer() if use_regularizer else None
    with strategy_fn().scope():
      with policy.policy_scope('infer_float32_vars'):
        x = layers.Input(shape=(), batch_size=2, dtype=dtypes.float16)
        layer = AddLayer(assert_type=dtypes.float16, use_operator=use_operator,
                         regularizer=regularizer)
        y = layer(x)
        y = math_ops.cast(y, dtypes.float32)
        model = models.Model(inputs=x, outputs=y)

        def loss_fn(y_true, y_pred):
          del y_true
          return math_ops.reduce_mean(y_pred)

        # Learning rate is small enough that if applied to a float16 variable,
        # the variable will not change. So this tests the learning rate not
        # applied to a float16 value, but instead the float32 variable.
        opt = gradient_descent.SGD(2 ** -14)
        model.compile(opt, loss=loss_fn)

      self.assertEqual(backend.eval(layer.v), 1)
      x = np.ones((2, 1))
      y = np.ones((2, 1))
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(2)
      model.fit(dataset)
      # Variable starts at 1, and should have gradient of 2 ** -14 subtracted
      # from it.
      expected = 1 - 2 ** -14
      if use_regularizer:
        # Regularizer adds another 2 ** -14 to the gradient.
        expected -= 2 ** -14
      self.assertEqual(backend.eval(layer.v), expected)

  @parameterized.named_parameters({
      'testcase_name': 'base',
      'strategy_fn': create_one_device_strategy,
  }, {
      'testcase_name': 'distribute',
      'strategy_fn': create_mirrored_strategy,
  }, {
      'testcase_name': 'loss_scaling',
      'strategy_fn': create_mirrored_strategy,
      'use_loss_scaling': True
  })
  @test_util.run_in_graph_and_eager_modes
  def test_advanced_model(self, strategy_fn, use_loss_scaling=False):

    # The advanced model tests mixed-precision-related features that would occur
    # in a resnet50 model. It tests a model that has:
    #  * Multiple layers, some which use auto-cast variables and some which do
    #    not
    #  * Regularization on some variables and not others.
    #  * Loss scaling (if use_loss_scaling is True)

    strategy = strategy_fn()
    if use_loss_scaling:
      loss_scale = 8.
    learning_rate = 2 ** -14

    with strategy.scope():
      with policy.policy_scope(policy.Policy('infer_float32_vars')):
        x = layers.Input(shape=(), batch_size=2, dtype=dtypes.float16)
        layer1 = AddLayer(assert_type=dtypes.float16,
                          regularizer=IdentityRegularizer(), use_operator=True)
        layer2 = AddLayerWithoutAutoCast(assert_type=dtypes.float16,
                                         use_operator=True)
        layer3 = AddLayer(assert_type=dtypes.float16, use_operator=False)
        layer4 = AddLayerWithoutAutoCast(assert_type=dtypes.float16,
                                         regularizer=IdentityRegularizer(),
                                         use_operator=False)
        y = layer1(x)
        y = layer2(y)
        y = layer3(y)
        y = layer4(y)
        if use_loss_scaling:
          # The gradient of 'y' at this point is 1. With loss scaling, the
          # gradient is 'loss_scale'. The DistributionStrategy additionally
          # scales the gradient by 1/num_replicas in_sync. We divide by the
          # batch size of 2 since the loss is averaged across batch elements.
          expected_gradient = loss_scale / strategy.num_replicas_in_sync / 2
          identity_with_grad_check_fn = (
              mp_test_util.create_identity_with_grad_check_fn(
                  expected_dtype=dtypes.float16,
                  expected_gradient=[expected_gradient] * 2))
          y = core.Lambda(identity_with_grad_check_fn)(y)
        y = math_ops.cast(y, dtypes.float32)
        model = models.Model(inputs=x, outputs=y)

        def loss_fn(y_true, y_pred):
          self.assertEqual(y_true.dtype, dtypes.float32)
          self.assertEqual(y_pred.dtype, dtypes.float32)
          return math_ops.reduce_mean(y_pred)

        opt = gradient_descent.SGD(learning_rate)
        if use_loss_scaling:
          opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
        model.compile(opt, loss=loss_fn)

      x = np.ones((2, 1))
      y = np.ones((2, 1))
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(2)
      model.fit(dataset)
      for layer in (layer1, layer2, layer3, layer4):
        if layer.losses:
          # Layer has weight regularizer
          self.assertEqual(backend.eval(layer.v), 1 - 2 * learning_rate)
        else:
          # Layer does not have weight regularizer
          self.assertEqual(backend.eval(layer.v), 1 - learning_rate)


if __name__ == '__main__':
  test.main()
