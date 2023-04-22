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
"""Tests keras.layers.Layer works properly with mixed precision."""

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.mixed_precision import get_layer_policy
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.mixed_precision import test_util as mp_test_util
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import util as trackable_utils


class MultiplyLayerWithFunction(mp_test_util.MultiplyLayer):
  """Same as MultiplyLayer, but _multiply is decorated with a tf.function."""

  @def_function.function
  def _multiply(self, x, y):
    return super(MultiplyLayerWithFunction, self)._multiply(x, y)


# If called outside any strategy.scope() calls, this will return the default
# strategy.
default_strategy_fn = distribution_strategy_context.get_strategy


def create_mirrored_strategy():
  """Create a MirroredStrategy, using a GPU if it is available."""
  if tf_config.list_logical_devices('GPU'):
    return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
  else:
    return mirrored_strategy.MirroredStrategy(['cpu:0'])


def create_central_storage_strategy():
  """Create a CentralStorageStrategy, using a GPU if it is available."""
  compute_devices = ['cpu:0', 'gpu:0'] if (
      tf_config.list_logical_devices('GPU')) else ['cpu:0']
  return central_storage_strategy.CentralStorageStrategy(
      compute_devices, parameter_device='cpu:0')


TESTCASES = ({
    'testcase_name': 'base',
    'strategy_fn': default_strategy_fn
}, {
    'testcase_name': 'distribute',
    'strategy_fn': create_mirrored_strategy
})


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class LayerTest(keras_parameterized.TestCase):
  """Test mixed precision with Keras layers."""

  @parameterized.named_parameters(*TESTCASES)
  def test_mixed_policies_(self, strategy_fn):
    strategy = strategy_fn()
    for dtype in 'float16', 'bfloat16':
      x = constant_op.constant([1.])
      policy_name = 'mixed_' + dtype
      with strategy.scope(), policy.policy_scope(policy_name):
        layer = mp_test_util.MultiplyLayer(assert_type=dtype)
        self.assertEqual(layer.dtype, dtypes.float32)
        self.assertEqual(get_layer_policy.get_layer_policy(layer).name,
                         policy_name)
        y = layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float32)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(layer.dtype_policy.name, policy_name)
        self.assertIsInstance(layer.dtype_policy, policy.Policy)
        self.assertEqual(layer.compute_dtype, dtype)
        self.assertEqual(layer.dtype, dtypes.float32)
        self.assertEqual(layer.variable_dtype, dtypes.float32)
        self.assertEqual(get_layer_policy.get_layer_policy(layer).name,
                         policy_name)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(y), 1.)

  def test_layer_with_int_variable(self):
    class LayerWithIntVar(base_layer.Layer):

      def build(self, _):
        self.v = self.add_weight('v', dtype='int32', trainable=False)

      def call(self, inputs):
        # Only float variables should be autocasted. This will fail if self.v is
        # autocasted to float32
        return math_ops.cast(inputs, 'int32') + self.v

    x = constant_op.constant([1.])
    layer = LayerWithIntVar(dtype='mixed_float16')
    self.assertEqual(layer(x).dtype, 'int32')

  @parameterized.named_parameters(*TESTCASES)
  def test_layer_with_non_autocast_variable(self, strategy_fn):
    x = constant_op.constant([1.])
    with strategy_fn().scope():
      with policy.policy_scope('mixed_float16'):
        layer = mp_test_util.MultiplyLayerWithoutAutoCast(
            assert_type=dtypes.float16)
        y = layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float32)
        self.assertEqual(y.dtype, dtypes.float16)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(y), 1.)

  @parameterized.named_parameters(*TESTCASES)
  def test_layer_calling_tf_function(self, strategy_fn):
    x = constant_op.constant([1.])
    with strategy_fn().scope():
      with policy.policy_scope('mixed_float16'):
        layer = MultiplyLayerWithFunction(assert_type=dtypes.float16)
        y = layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float32)
        self.assertEqual(y.dtype, dtypes.float16)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(y), 1.)

  @parameterized.named_parameters(*TESTCASES)
  def test_layer_regularizer_runs_in_var_dtype(self, strategy_fn):
    x = constant_op.constant([1.])
    with strategy_fn().scope():
      with policy.policy_scope('mixed_float16'):
        # Test on MultiplyLayer
        layer = mp_test_util.MultiplyLayer(
            assert_type=dtypes.float16,
            regularizer=mp_test_util.IdentityRegularizer())
        layer(x)
        (regularizer_loss,) = layer.losses
        self.assertEqual(regularizer_loss.dtype, dtypes.float32)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(regularizer_loss), 1.)

        # Test on MultiplyLayerWithoutAutoCast
        layer = mp_test_util.MultiplyLayerWithoutAutoCast(
            assert_type=dtypes.float16,
            regularizer=mp_test_util.IdentityRegularizer())
        layer(x)
        (regularizer_loss,) = layer.losses
        self.assertEqual(regularizer_loss.dtype, dtypes.float32)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(regularizer_loss), 1.)

  @parameterized.named_parameters(*TESTCASES)
  def test_passing_policy_to_layer(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():
      # Passing a Policy to 'dtype' sets the policy for that layer.
      layer = mp_test_util.MultiplyLayer(
          assert_type=dtypes.float16, dtype=policy.Policy('mixed_float16'))
      # layer.dtype refers to the variable dtype
      self.assertEqual(layer.dtype, dtypes.float32)
      layer(x)
      self.assertEqual(layer.v.dtype, dtypes.float32)
      with policy.policy_scope('mixed_float16'):
        # Passing a Policy to dtype overrides the global Policy
        layer = mp_test_util.MultiplyLayer(
            assert_type=dtypes.float64, dtype=policy.Policy('float64'))
        self.assertEqual(layer.dtype_policy.name, 'float64')
        self.assertIsInstance(layer.dtype_policy, policy.Policy)
        self.assertEqual(layer.compute_dtype, dtypes.float64)
        self.assertEqual(layer.dtype, dtypes.float64)
        self.assertEqual(layer.variable_dtype, dtypes.float64)
        self.assertEqual(layer(x).dtype, dtypes.float64)
        self.assertEqual(layer.v.dtype, dtypes.float64)

  @parameterized.named_parameters(*TESTCASES)
  def test_gradient(self, strategy_fn):
    x = constant_op.constant([1.])
    with strategy_fn().scope() as strategy:
      with policy.policy_scope('mixed_float16'):
        layer = mp_test_util.MultiplyLayer(assert_type=dtypes.float16)
        # Learning rate is small enough that if applied to a float16 variable,
        # the variable will not change. So this tests the learning rate is not
        # applied to a float16 value, but instead the float32 variable.
        opt = gradient_descent.SGD(2**-14)

        def run_fn():
          with backprop.GradientTape() as tape:
            y = layer(x)
            # Divide by num_replicas_in_sync, as the effective total loss is the
            # sum of each of the replica's losses.
            y /= strategy.num_replicas_in_sync

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
        self.assertEqual(self.evaluate(layer.v), 1 - 2**-14)

  def _test_checkpointing_layer_weights(self, strategy_fn,
                                        mixed_prec_when_saving,
                                        mixed_prec_when_loading):
    # In this test, we potentially save with mixed precision enabled and load
    # with mixed precision disabled, or vice versa. This is possible because
    # variables are float32 regardless of whether mixed precision is enabled.
    save_policy = 'mixed_float16' if mixed_prec_when_saving else 'float32'
    load_policy = 'mixed_float16' if mixed_prec_when_loading else 'float32'
    save_input_dtype = 'float16' if mixed_prec_when_saving else 'float32'
    load_input_dtype = 'float16' if mixed_prec_when_loading else 'float32'

    # Create a layer and save a checkpoint.
    x = constant_op.constant([1.])
    with strategy_fn().scope():
      with policy.policy_scope(save_policy):
        layer = mp_test_util.MultiplyLayer(assert_type=save_input_dtype)
        layer(x)  # Build layer
    layer.set_weights([np.array(100.)])
    self.assertEqual(self.evaluate(layer(x)), 100.)
    checkpoint = trackable_utils.Checkpoint(layer=layer)
    prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    save_path = checkpoint.save(prefix)

    # Create a new layer and restore the checkpoint.
    x = constant_op.constant([1.])
    with strategy_fn().scope():
      with policy.policy_scope(load_policy):
        layer = mp_test_util.MultiplyLayer(assert_type=load_input_dtype)
        layer(x)  # Build layer
    layer.set_weights([np.array(200.)])
    self.assertEqual(self.evaluate(layer(x)), 200.)
    checkpoint = trackable_utils.Checkpoint(layer=layer)
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    self.assertEqual(layer.get_weights(), [100.])
    self.assertEqual(self.evaluate(layer(x)), 100.)

  @parameterized.named_parameters(*TESTCASES)
  def test_checkpointing_layer_weights(self, strategy_fn):
    with self.test_session():
      self._test_checkpointing_layer_weights(
          strategy_fn, mixed_prec_when_saving=True,
          mixed_prec_when_loading=True)
      self._test_checkpointing_layer_weights(
          strategy_fn, mixed_prec_when_saving=True,
          mixed_prec_when_loading=False)
      self._test_checkpointing_layer_weights(
          strategy_fn, mixed_prec_when_saving=False,
          mixed_prec_when_loading=True)

  @parameterized.named_parameters(*TESTCASES)
  def test_config(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():
      for layer, dtype in (
          (mp_test_util.MultiplyLayer(), 'float32'),
          (mp_test_util.MultiplyLayer(dtype='float64'), 'float64'),
          (mp_test_util.MultiplyLayer(dtype=policy.Policy('float64')),
           'float64')):
        config = layer.get_config()
        self.assertEqual(config['dtype'], dtype)
        self.assertIsInstance(config['dtype'], str)
        layer = mp_test_util.MultiplyLayer.from_config(config)
        self.assertEqual(layer.dtype, dtype)
        self.assertEqual(layer(x).dtype, dtype)
        self.assertEqual(layer.v.dtype, dtype)

      layer = mp_test_util.MultiplyLayer(dtype='mixed_float16')
      config = layer.get_config()
      self.assertEqual(config['dtype'],
                       {'class_name': 'Policy',
                        'config': {'name': 'mixed_float16'}})
      layer = mp_test_util.MultiplyLayer.from_config(config)
      self.assertEqual(layer.dtype, 'float32')
      self.assertEqual(layer(x).dtype, 'float16')
      self.assertEqual(layer.v.dtype, 'float32')
      config = layer.get_config()
      self.assertEqual(config['dtype'],
                       {'class_name': 'Policy',
                        'config': {'name': 'mixed_float16'}})

      layer = mp_test_util.MultiplyLayer(dtype=policy.Policy('_infer'))
      config = layer.get_config()
      self.assertIsNone(config['dtype'])
      layer = mp_test_util.MultiplyLayer.from_config(config)
      # If a layer is serialized with the "_infer" policy, when deserialized
      # into TF 2 it will have the global policy instead of "_infer". This is
      # because "_infer" is serialized into None, and passing dtype=None in
      # TensorFlow 2 indicates to use the global policy.
      self.assertEqual(layer.dtype, 'float32')
      self.assertEqual(layer(x).dtype, 'float32')
      self.assertEqual(layer.v.dtype, 'float32')

  @parameterized.named_parameters(*TESTCASES)
  def test_config_policy_v1(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():

      layer = mp_test_util.MultiplyLayer(dtype=policy.PolicyV1('mixed_float16',
                                                               loss_scale=None))
      config = layer.get_config()
      self.assertEqual(config['dtype'],
                       {'class_name': 'PolicyV1',
                        'config': {'name': 'mixed_float16',
                                   'loss_scale': None}})
      layer = mp_test_util.MultiplyLayer.from_config(config)
      self.assertEqual(layer.dtype, 'float32')
      self.assertEqual(layer(x).dtype, 'float16')
      self.assertEqual(layer.v.dtype, 'float32')
      # Restoring a PolicyV1 silently converts it to a Policy and drops the loss
      # scale.
      self.assertEqual(type(layer.dtype_policy), policy.Policy)
      config = layer.get_config()
      # The loss_scale is silently dropped
      self.assertEqual(config['dtype'],
                       {'class_name': 'Policy',
                        'config': {'name': 'mixed_float16'}})

      layer = mp_test_util.MultiplyLayer(dtype=policy.PolicyV1('float64',
                                                               loss_scale=2.))
      config = layer.get_config()
      self.assertEqual(config['dtype'],
                       {'class_name': 'PolicyV1',
                        'config': {'name': 'float64',
                                   'loss_scale': {
                                       'class_name': 'FixedLossScale',
                                       'config': {'loss_scale_value': 2.0}}}})
      layer = mp_test_util.MultiplyLayer.from_config(config)
      self.assertEqual(layer.dtype, 'float64')
      self.assertEqual(layer(x).dtype, 'float64')
      self.assertEqual(layer.v.dtype, 'float64')
      self.assertEqual(type(layer.dtype_policy), policy.Policy)
      config = layer.get_config()
      self.assertEqual(config['dtype'], 'float64')

      layer = mp_test_util.MultiplyLayer(dtype=policy.PolicyV1('_infer',
                                                               loss_scale=2.))
      config = layer.get_config()
      self.assertEqual(config['dtype'],
                       {'class_name': 'PolicyV1',
                        'config': {'name': '_infer',
                                   'loss_scale': {
                                       'class_name': 'FixedLossScale',
                                       'config': {'loss_scale_value': 2.0}}}})
      layer = mp_test_util.MultiplyLayer.from_config(config)
      self.assertEqual(layer.dtype, None)
      self.assertEqual(layer(x).dtype, 'float16')
      self.assertEqual(layer.v.dtype, 'float16')
      self.assertEqual(type(layer.dtype_policy), policy.Policy)
      config = layer.get_config()
      self.assertEqual(config['dtype'], 'float16')

  def test_delete_variable(self):
    layer = base_layer.Layer(dtype='mixed_float16')
    layer.x = layer.add_weight('x')
    self.assertEqual(layer.trainable_weights, [layer.x])
    del layer.x
    self.assertEqual(layer.trainable_weights, [])

  def test_build_and_call_layer_in_function(self):
    layer = mp_test_util.MultiplyLayer(dtype=policy.Policy('mixed_float16'))
    @def_function.function
    def f():
      return layer(1.)
    y = f()
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(y.dtype, 'float16')
    self.assertEqual(layer.v.dtype, 'float32')
    self.assertEqual(self.evaluate(y), 1.)

  def test_unsupported_strategy(self):
    strategy = create_central_storage_strategy()
    with strategy.scope(), self.assertRaisesRegex(
        ValueError, 'Mixed precision is not supported with the '
        'tf.distribute.Strategy: CentralStorageStrategy. Either '
        'stop using mixed precision by removing the use of the '
        '"mixed_float16" policy or use a different Strategy, e.g. '
        'a MirroredStrategy.'):
      mp_test_util.MultiplyLayer(dtype='mixed_float16')
    # Non-mixed policies are fine
    mp_test_util.MultiplyLayer(dtype=policy.Policy('float64'))

  def test_input_spec_dtype(self):
    # Test the InputSpec's dtype is compared against the inputs before the layer
    # casts them, not after.
    layer = mp_test_util.MultiplyLayer(dtype='float64')
    layer.input_spec = input_spec.InputSpec(dtype='float16')

    # Test passing Eager tensors
    x = array_ops.ones((2, 2), dtype='float16')
    layer(x)
    x = array_ops.ones((2, 2), dtype='float64')
    with self.assertRaisesRegex(
        ValueError, 'expected dtype=float16, found dtype=.*float64'):
      layer(x)

    # Test passing symbolic tensors
    x = layers.Input((2,), dtype='float16')
    y = layer(x)
    model = models.Model(x, y)
    model(array_ops.ones((2, 2)))

    x = layers.Input((2,), dtype='float64')
    with self.assertRaisesRegex(
        ValueError, 'expected dtype=float16, found dtype=.*float64'):
      # In TF2, the error is only raised when the model is run
      y = layer(x)
      model = models.Model(x, y)
      model(array_ops.ones((2, 2)))


if __name__ == '__main__':
  base_layer_utils.enable_v2_dtype_behavior()
  test.main()
