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

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.keras.mixed_precision.experimental import test_util as mp_test_util
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.tracking import util as trackable_utils


# Pylint's static analysis incorrectly believes many layers are non-callable, so
# we disable the lint error.
# pylint: disable=not-callable


class AddLayerWithoutAutoCast(mp_test_util.AddLayer):
  """Same as AddLayer, but does not use AutoCastVariables."""

  def build(self, _):
    dtype = self.dtype
    if dtype in ('float16', 'bfloat16'):
      dtype = 'float32'
    self.v = self.add_weight(
        'v', (),
        initializer='ones',
        dtype=dtype,
        experimental_autocast=False,
        regularizer=self._regularizer)
    self.built = True

  def call(self, inputs):
    self.assert_input_types(inputs)
    assert self.v.dtype in (dtypes.float32, dtypes.float64)
    return self._add(inputs, math_ops.cast(self.v, inputs.dtype))


class AddLayerWithFunction(mp_test_util.AddLayer):
  """Same as AddLayer, but _add is decorated with a tf.function."""

  @def_function.function
  def _add(self, x, y):
    return super(AddLayerWithFunction, self)._add(x, y)


# If called outside any strategy.scope() calls, this will return the default
# strategy.
default_strategy_fn = distribution_strategy_context.get_strategy


def create_mirrored_strategy():
  if context.num_gpus() >= 1:
    return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
  else:
    return mirrored_strategy.MirroredStrategy(['cpu:0'])


TESTCASES = ({
    'testcase_name': 'base',
    'strategy_fn': default_strategy_fn
}, {
    'testcase_name': 'distribute',
    'strategy_fn': create_mirrored_strategy
})


class KerasLayerTest(keras_parameterized.TestCase):
  """Test mixed precision with Keras layers."""

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_mixed_policies_(self, strategy_fn):
    for dtype in 'float16', 'bfloat16':
      x = constant_op.constant([1.])
      policy_name = 'mixed_' + dtype
      with strategy_fn().scope(), policy.policy_scope(policy_name):
        layer = mp_test_util.AddLayer(assert_type=dtype)
        self.assertEqual(layer.dtype, dtypes.float32)
        self.assertEqual(layer._dtype_policy._name, policy_name)
        y = layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float32)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(layer.dtype, dtypes.float32)
        self.assertEqual(layer._dtype_policy._name, policy_name)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(y), 2.)

  @test_util.run_in_graph_and_eager_modes
  def test_layer_with_int_variable(self):
    class LayerWithIntVar(base_layer.Layer):

      def build(self, _):
        self.v = self.add_weight('v', dtype='int32', trainable=False)

      def call(self, inputs):
        # Only float variables should be autocasted. This will fail if self.v is
        # autocasted to float32
        return math_ops.cast(inputs, 'int32') + self.v

    x = constant_op.constant([1.])
    layer = LayerWithIntVar(dtype=policy.Policy('mixed_float16'))
    self.assertEqual(layer(x).dtype, 'int32')

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_layer_with_non_autocast_variable(self, strategy_fn):
    x = constant_op.constant([1.])
    with strategy_fn().scope():
      with policy.policy_scope('mixed_float16'):
        layer = AddLayerWithoutAutoCast(assert_type=dtypes.float16)
        y = layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float32)
        self.assertEqual(y.dtype, dtypes.float16)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(y), 2.)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_layer_calling_tf_function(self, strategy_fn):
    x = constant_op.constant([1.])
    with strategy_fn().scope():
      with policy.policy_scope('mixed_float16'):
        layer = AddLayerWithFunction(assert_type=dtypes.float16)
        y = layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float32)
        self.assertEqual(y.dtype, dtypes.float16)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(y), 2.)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_layer_regularizer_runs_in_var_dtype(self, strategy_fn):
    x = constant_op.constant([1.])
    with strategy_fn().scope():
      with policy.policy_scope('mixed_float16'):
        # Test on AddLayer
        layer = mp_test_util.AddLayer(
            assert_type=dtypes.float16,
            regularizer=mp_test_util.IdentityRegularizer())
        layer(x)
        (regularizer_loss,) = layer.losses
        self.assertEqual(regularizer_loss.dtype, dtypes.float32)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(regularizer_loss), 1.)

        # Test on AddLayerWithoutAutoCast
        layer = AddLayerWithoutAutoCast(
            assert_type=dtypes.float16,
            regularizer=mp_test_util.IdentityRegularizer())
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
      layer = mp_test_util.AddLayer(
          assert_type=dtypes.float16, dtype=policy.Policy('mixed_float16'))
      # layer.dtype refers to the variable dtype
      self.assertEqual(layer.dtype, dtypes.float32)
      layer(x)
      self.assertEqual(layer.v.dtype, dtypes.float32)
      with policy.policy_scope('mixed_float16'):
        # Passing a Policy to dtype overrides the global Policy
        layer = mp_test_util.AddLayer(
            assert_type=dtypes.float64, dtype=policy.Policy('float64'))
        self.assertEqual(layer.dtype, 'float64')
        self.assertEqual(layer(x).dtype, dtypes.float64)
        self.assertEqual(layer.v.dtype, dtypes.float64)

  @test_util.run_in_graph_and_eager_modes
  def test_error_passing_policy_string_to_layer(self):
    with self.assertRaisesRegexp(
        TypeError, "Cannot convert value 'mixed_float16' to a "
                   "TensorFlow DType"):
      # This is not allowed, as otherwise a "mixed_float16" policy could be
      # created without an API call that has the name "experimental" in it.
      mp_test_util.AddLayer(dtype='mixed_float16')

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_gradient(self, strategy_fn):
    x = constant_op.constant([1.])
    with strategy_fn().scope() as strategy:
      with policy.policy_scope('mixed_float16'):
        layer = mp_test_util.AddLayer(assert_type=dtypes.float16)

        def run_fn():
          with backprop.GradientTape() as tape:
            y = layer(x)
            # Divide by num_replicas_in_sync, as the effective total loss is the
            # sum of each of the replica's losses.
            y /= strategy.num_replicas_in_sync

          # Learning rate is small enough that if applied to a float16 variable,
          # the variable will not change. So this tests the learning rate is not
          # applied to a float16 value, but instead the float32 variable.
          opt = gradient_descent.SGD(2**-14)
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
        layer = mp_test_util.AddLayer(assert_type=save_input_dtype)
        layer(x)  # Build layer
    layer.set_weights([np.array(100.)])
    self.assertEqual(self.evaluate(layer(x)), 101.)
    checkpoint = trackable_utils.Checkpoint(layer=layer)
    prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    save_path = checkpoint.save(prefix)

    # Create a new layer and restore the checkpoint.
    x = constant_op.constant([1.])
    with strategy_fn().scope():
      with policy.policy_scope(load_policy):
        layer = mp_test_util.AddLayer(assert_type=load_input_dtype)
        layer(x)  # Build layer
    layer.set_weights([np.array(200.)])
    self.assertEqual(self.evaluate(layer(x)), 201.)
    checkpoint = trackable_utils.Checkpoint(layer=layer)
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    self.assertEqual(layer.get_weights(), [100.])
    self.assertEqual(self.evaluate(layer(x)), 101.)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_checkpointing_layer_weights(self, strategy_fn):
    self._test_checkpointing_layer_weights(
        strategy_fn, mixed_prec_when_saving=True, mixed_prec_when_loading=True)
    self._test_checkpointing_layer_weights(
        strategy_fn, mixed_prec_when_saving=True, mixed_prec_when_loading=False)
    self._test_checkpointing_layer_weights(
        strategy_fn, mixed_prec_when_saving=False, mixed_prec_when_loading=True)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_config(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():
      for layer, dtype in (
          (mp_test_util.AddLayer(), 'float32'),
          (mp_test_util.AddLayer(dtype='float64'), 'float64'),
          (mp_test_util.AddLayer(dtype=policy.Policy('float64')), 'float64')):
        config = layer.get_config()
        self.assertEqual(config['dtype'], dtype)
        self.assertIsInstance(config['dtype'], str)
        layer = mp_test_util.AddLayer.from_config(config)
        self.assertEqual(layer.dtype, dtype)
        self.assertEqual(layer(x).dtype, dtype)
        self.assertEqual(layer.v.dtype, dtype)

      layer = mp_test_util.AddLayer(dtype=policy.Policy('mixed_float16'))
      config = layer.get_config()
      self.assertEqual(config['dtype'],
                       {'class_name': 'Policy',
                        'config': {'name': 'mixed_float16'}})
      layer = mp_test_util.AddLayer.from_config(config)
      self.assertEqual(layer.dtype, 'float32')
      self.assertEqual(layer(x).dtype, 'float16')
      self.assertEqual(layer.v.dtype, 'float32')

      layer = mp_test_util.AddLayer(dtype=policy.Policy('mixed_float16',
                                                        loss_scale=None))
      config = layer.get_config()
      self.assertEqual(config['dtype'],
                       {'class_name': 'Policy',
                        'config': {'name': 'mixed_float16',
                                   'loss_scale': None}})
      layer = mp_test_util.AddLayer.from_config(config)
      self.assertEqual(layer.dtype, 'float32')
      self.assertEqual(layer(x).dtype, 'float16')
      self.assertEqual(layer.v.dtype, 'float32')

      layer = mp_test_util.AddLayer(dtype=policy.Policy('float64',
                                                        loss_scale=2.))
      config = layer.get_config()
      self.assertEqual(config['dtype'],
                       {'class_name': 'Policy',
                        'config': {'name': 'float64',
                                   'loss_scale': {
                                       'class_name': 'FixedLossScale',
                                       'config': {'loss_scale_value': 2.0}}}})
      layer = mp_test_util.AddLayer.from_config(config)
      self.assertEqual(layer.dtype, 'float64')
      self.assertEqual(layer(x).dtype, 'float64')
      self.assertEqual(layer.v.dtype, 'float64')

      layer = mp_test_util.AddLayer(dtype=policy.Policy('infer'))
      config = layer.get_config()
      self.assertIsNone(config['dtype'])
      layer = mp_test_util.AddLayer.from_config(config)
      # If a layer is serialized with the "infer" policy, when deserialized into
      # TF 2 it will have the global policy instead of "infer". This is because
      # "infer" is serialized into None, and passing dtype=None in TensorFlow 2
      # indicates to use the global policy.
      self.assertEqual(layer.dtype, 'float32')
      self.assertEqual(layer(x).dtype, 'float32')
      self.assertEqual(layer.v.dtype, 'float32')

      layer = mp_test_util.AddLayer(dtype=policy.Policy('infer', loss_scale=2.))
      config = layer.get_config()
      self.assertEqual(config['dtype'],
                       {'class_name': 'Policy',
                        'config': {'name': 'infer',
                                   'loss_scale': {
                                       'class_name': 'FixedLossScale',
                                       'config': {'loss_scale_value': 2.0}}}})
      layer = mp_test_util.AddLayer.from_config(config)
      self.assertEqual(layer.dtype, None)
      self.assertEqual(layer(x).dtype, 'float16')
      self.assertEqual(layer.v.dtype, 'float16')

  @test_util.run_in_graph_and_eager_modes
  def test_delete_variable(self):
    layer = base_layer.Layer(dtype=policy.Policy('mixed_float16'))
    layer.x = layer.add_weight('x')
    self.assertEqual(layer.trainable_weights, [layer.x])
    del layer.x
    self.assertEqual(layer.trainable_weights, [])

  @test_util.run_in_graph_and_eager_modes
  def test_build_and_call_layer_in_function(self):
    layer = mp_test_util.AddLayer(dtype=policy.Policy('mixed_float16'))
    @def_function.function
    def f():
      return layer(1.)
    y = f()
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(y.dtype, 'float16')
    self.assertEqual(layer.v.dtype, 'float32')
    self.assertEqual(self.evaluate(y), 2.)


class KerasModelTest(keras_parameterized.TestCase):
  """Test mixed precision with Keras models."""

  def _skip_if_strategy_unsupported(self, strategy_fn, check_model_type=False):
    if (strategy_fn != default_strategy_fn and
        (testing_utils.should_run_eagerly() or
         (check_model_type and testing_utils.get_model_type() == 'subclass'))):
      self.skipTest('Non-default strategies are unsupported with subclassed '
                    'models or with passing run_eagerly=True to '
                    'Model.compile()')

  def _skip_if_save_format_unsupported(self, save_format):
    model_type = testing_utils.get_model_type()
    if save_format == 'h5' and model_type == 'subclass':
      self.skipTest('Saving subclassed models with the HDF5 format is '
                    'unsupported')
    if (save_format == 'tf' and model_type == 'subclass' and
        not testing_utils.should_run_tf_function()):
      self.skipTest('b/142352416: This combination of features is currently '
                    'broken.')
    if (save_format == 'tf' and model_type != 'subclass' and
        not context.executing_eagerly()):
      self.skipTest('b/134519980: This combination of features is currently '
                    'broken.')

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn
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
      }, {
          'testcase_name': 'get_config',
          'strategy_fn': create_mirrored_strategy,
          'get_config': True,
          'use_regularizer': True,
      }, {
          'testcase_name': 'saved_model',
          'strategy_fn': default_strategy_fn,
          'save_format': 'tf',
          'use_regularizer': True,
      }, {
          'testcase_name': 'saved_model_input_spec',
          'strategy_fn': default_strategy_fn,
          'save_format': 'tf',
          'use_regularizer': True,
          'use_input_spec': True,
      }, {
          'testcase_name': 'h5',
          'strategy_fn': default_strategy_fn,
          'save_format': 'h5',
          'use_regularizer': True,
      }, {
          'testcase_name': 'saved_model_distribute',
          'strategy_fn': create_mirrored_strategy,
          'save_format': 'tf',
          'use_regularizer': True,
      }, {
          'testcase_name': 'saved_model_input_spec_distribute',
          'strategy_fn': create_mirrored_strategy,
          'save_format': 'tf',
          'use_regularizer': True,
          'use_input_spec': True,
      }, {
          'testcase_name': 'h5_distribute',
          'strategy_fn': create_mirrored_strategy,
          'save_format': 'h5',
          'use_regularizer': True,
      }, {
          'testcase_name': 'norun_distributed',
          'strategy_fn': create_mirrored_strategy,
          'experimental_run_tf_function': False
      })
  def test_model(self,
                 strategy_fn,
                 use_operator=False,
                 use_regularizer=False,
                 policy_name='mixed_float16',
                 get_config=False,
                 save_format=None,
                 use_input_spec=False,
                 experimental_run_tf_function=True):
    self._skip_if_strategy_unsupported(strategy_fn, check_model_type=True)
    self._skip_if_save_format_unsupported(save_format)
    regularizer = (mp_test_util.IdentityRegularizer() if use_regularizer
                   else None)
    with strategy_fn().scope():
      # Pass loss_scale=None, as this test will fail if the DynamicLossScale
      # skips applying gradients for a step
      with policy.policy_scope(policy.Policy(policy_name, loss_scale=None)):
        layer = mp_test_util.AddLayer(
            assert_type=dtypes.float16,
            use_operator=use_operator,
            regularizer=regularizer,
            input_shape=(1,))
        if use_input_spec:
          layer.input_spec = input_spec.InputSpec(shape=(2, 1))
        cast_f32_layer = layers.Lambda(lambda x: math_ops.cast(x, 'float32'))
        model = testing_utils.get_model_from_layers(
            [layer, cast_f32_layer], input_shape=(1,),
            input_dtype=dtypes.float16)
        if get_config:
          config = model.get_config()
          model = model.__class__.from_config(
              config, custom_objects={'AddLayer': mp_test_util.AddLayer})
          (layer,) = (layer for layer in model.layers
                      if isinstance(layer, mp_test_util.AddLayer))

        def loss_fn(y_true, y_pred):
          del y_true
          return math_ops.reduce_mean(y_pred)

        # Learning rate is small enough that if applied to a float16 variable,
        # the variable will not change. So this tests the learning rate not
        # applied to a float16 value, but instead the float32 variable.
        opt = gradient_descent.SGD(2**-14)
        model.compile(
            opt,
            loss=loss_fn,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones((2, 1))
    y = np.ones((2, 1))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(2)
    model.fit(dataset)
    # Variable starts at 1, and should have gradient of 2 ** -14 subtracted
    # from it.
    expected = 1 - 2**-14
    if use_regularizer:
      # Regularizer adds another 2 ** -14 to the gradient.
      expected -= 2**-14
    self.assertEqual(backend.eval(layer.v), expected)

    if save_format:
      with generic_utils.CustomObjectScope(
          {'AddLayer': mp_test_util.AddLayer, 'loss_fn': loss_fn}):
        self._test_saving(model, dataset, save_format, use_regularizer)

  def _test_saving(self, model, dataset, save_format, use_regularizer):
    # Save and load model, asserting variable does not change
    save_path = os.path.join(self.get_temp_dir(), 'model')
    model.save(save_path, save_format=save_format)
    model = save.load_model(save_path)
    (layer,) = (layer for layer in model.layers
                if 'AddLayer' in layer.__class__.__name__)
    expected = 1 - 2**-14
    if use_regularizer:
      expected -= 2**-14
    self.assertEqual(backend.eval(layer.v), expected)

    # Continue training, and assert variable is correct value
    model.fit(dataset)
    new_expected = expected - 2 ** -14
    if use_regularizer:
      new_expected -= 2 ** -14
    self.assertEqual(backend.eval(layer.v), new_expected)

    # Load saved model again, and assert variable is previous value
    model = save.load_model(save_path)
    (layer,) = (layer for layer in model.layers
                if 'AddLayer' in layer.__class__.__name__)
    self.assertEqual(backend.eval(layer.v), expected)

    # Ensure various dtype-related aspects of the layer are correct
    self.assertEqual(layer.dtype, 'float32')
    self.assertEqual(layer._dtype_policy.name, 'mixed_float16')
    self.assertEqual(layer.v.dtype, 'float32')
    self.assertEqual(layer(np.ones((2, 1))).dtype, 'float16')

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'norun_distributed',
          'strategy_fn': create_mirrored_strategy,
          'experimental_run_tf_function': False,
      })
  def test_fixed_loss_scaling(self,
                              strategy_fn,
                              experimental_run_tf_function=True):
    # Note: We do not test mixed precision in this method, only loss scaling.
    self._skip_if_strategy_unsupported(strategy_fn)
    loss_scale = 8.
    batch_size = 4
    with strategy_fn().scope():
      x = layers.Input(shape=(1,), batch_size=batch_size)
      layer = mp_test_util.AddLayer()
      y = layer(x)

      # The gradient of 'y' at this point is 1. With loss scaling, the gradient
      # is 'loss_scale'. We divide by the batch size since the loss is averaged
      # across batch elements.
      expected_gradient = loss_scale / batch_size
      identity_with_grad_check_fn = (
          mp_test_util.create_identity_with_grad_check_fn([expected_gradient]))
      y = core.Lambda(identity_with_grad_check_fn)(y)
      model = models.Model(inputs=x, outputs=y)

      def loss_fn(y_true, y_pred):
        del y_true
        return math_ops.reduce_mean(y_pred)

      opt = gradient_descent.SGD(1.)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      model.compile(
          opt,
          loss=loss_fn,
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

    self.assertEqual(backend.eval(layer.v), 1)
    x = np.ones((batch_size, 1))
    y = np.ones((batch_size, 1))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    model.fit(dataset)
    # Variable starts at 1, and should have gradient of 1 subtracted from it.
    expected = 0
    self.assertEqual(backend.eval(layer.v), expected)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'loss_scaling',
          'strategy_fn': create_mirrored_strategy,
          'use_loss_scaling': True
      })
  def test_advanced_model(self, strategy_fn, use_loss_scaling=False):
    # The advanced model tests mixed-precision-related features that would occur
    # in a resnet50 model. It tests a model that has:
    #  * Multiple layers, some which use auto-cast variables and some which do
    #    not
    #  * Regularization on some variables and not others.
    #  * A fixed loss scale (if use_loss_scaling is True)

    self._skip_if_strategy_unsupported(strategy_fn)
    strategy = strategy_fn()
    if use_loss_scaling:
      loss_scale = 8.
    else:
      loss_scale = None
    learning_rate = 2**-14

    with strategy.scope():
      with policy.policy_scope(policy.Policy('mixed_float16',
                                             loss_scale=loss_scale)):
        x = layers.Input(shape=(1,), batch_size=2)
        layer1 = mp_test_util.AddLayer(
            assert_type=dtypes.float16,
            regularizer=mp_test_util.IdentityRegularizer(),
            use_operator=True)
        layer2 = AddLayerWithoutAutoCast(
            assert_type=dtypes.float16, use_operator=True)
        layer3 = mp_test_util.AddLayer(assert_type=dtypes.float16,
                                       use_operator=False)
        layer4 = AddLayerWithoutAutoCast(
            assert_type=dtypes.float16,
            regularizer=mp_test_util.IdentityRegularizer(),
            use_operator=False)
        y = layer1(x)
        y = layer2(y)
        y = layer3(y)
        y = layer4(y)
        if use_loss_scaling:
          # The gradient of 'y' at this point is 1. With loss scaling, the
          # gradient is 'loss_scale'. We divide by the batch size of 2 since the
          # loss is averaged across batch elements.
          expected_gradient = loss_scale / 2
          identity_with_grad_check_fn = (
              mp_test_util.create_identity_with_grad_check_fn(
                  expected_dtype=dtypes.float16,
                  expected_gradient=[expected_gradient]))
          y = core.Lambda(identity_with_grad_check_fn)(y)
        y = math_ops.cast(y, dtypes.float32)
        model = models.Model(inputs=x, outputs=y)

        def loss_fn(y_true, y_pred):
          self.assertEqual(y_true.dtype, dtypes.float32)
          self.assertEqual(y_pred.dtype, dtypes.float32)
          return math_ops.reduce_mean(y_pred)

        opt = gradient_descent.SGD(learning_rate)
        model.compile(
            opt,
            loss=loss_fn,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function())

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

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'pass_loss_scale_to_policy',
          'strategy_fn': create_mirrored_strategy,
          'pass_loss_scale_to_policy': True,
      }, {
          'testcase_name': 'get_config',
          'strategy_fn': create_mirrored_strategy,
          'get_config': True,
      }, {
          'testcase_name': 'get_config_and_pass_loss_scale_to_policy',
          'strategy_fn': create_mirrored_strategy,
          'get_config': True,
          'pass_loss_scale_to_policy': True,
      }, {
          'testcase_name': 'norun_distributed',
          'strategy_fn': create_mirrored_strategy,
          'experimental_run_tf_function': False,
      })
  def test_dynamic_loss_scaling(self,
                                strategy_fn,
                                pass_loss_scale_to_policy=False,
                                get_config=False,
                                experimental_run_tf_function=True):
    self._skip_if_strategy_unsupported(strategy_fn)
    strategy = strategy_fn()
    initial_loss_scale = 2.
    batch_size = 4
    loss_scale = loss_scale_module.DynamicLossScale(
        initial_loss_scale=initial_loss_scale, increment_period=2)
    expected_gradient = backend.variable([initial_loss_scale / batch_size],
                                         dtype=dtypes.float16)
    # If this variable is set to True, the model below will have NaN gradients
    have_nan_gradients = backend.variable(False, dtype=dtypes.bool)
    with strategy.scope():
      opt = gradient_descent.SGD(1.)
      if pass_loss_scale_to_policy:
        p = policy.Policy('mixed_float16', loss_scale=loss_scale)
      else:
        p = policy.Policy('mixed_float16', loss_scale=None)
        opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      with policy.policy_scope(p):
        x = layers.Input(
            shape=(1,), batch_size=batch_size, dtype=dtypes.float16)
        layer = mp_test_util.AddLayer(assert_type=dtypes.float16)
        y = layer(x)
        identity_with_nan_grads = (
            mp_test_util.create_identity_with_nan_gradients_fn(
                have_nan_gradients))
        y = core.Lambda(identity_with_nan_grads)(y)
        identity_with_grad_check_fn = (
            mp_test_util.create_identity_with_grad_check_fn(
                expected_dtype=dtypes.float16,
                expected_gradient=expected_gradient))
        y = core.Lambda(identity_with_grad_check_fn)(y)
        y = math_ops.cast(y, dtypes.float32)
        model = models.Model(inputs=x, outputs=y)
        if get_config:
          config = model.get_config()
          model = model.__class__.from_config(
              config, custom_objects={'AddLayer': mp_test_util.AddLayer})
          (layer,) = (layer for layer in model.layers
                      if isinstance(layer, mp_test_util.AddLayer))

        def loss_fn(y_true, y_pred):
          del y_true
          return math_ops.reduce_mean(y_pred)

        model.compile(
            opt,
            loss=loss_fn,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function())

    self.assertEqual(backend.eval(layer.v), 1)
    x = np.ones((batch_size, 1))
    y = np.ones((batch_size, 1))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    model.fit(dataset)
    # The variables starts with 1 and has a gradient of 1, so will go down by 1
    # each step.
    self.assertEqual(backend.eval(layer.v), 0)

    model.fit(dataset)
    self.assertEqual(backend.eval(layer.v), -1)

    # There have been two steps without NaNs, so the loss scale will double
    backend.set_value(expected_gradient,
                      backend.get_value(expected_gradient * 2))
    model.fit(dataset)
    self.assertEqual(backend.eval(layer.v), -2)

    # Next test with NaN gradients.
    backend.set_value(have_nan_gradients, True)
    model.fit(dataset)
    # Variable should not be updated
    self.assertEqual(backend.eval(layer.v), -2)

    # Test with finite gradients again
    backend.set_value(have_nan_gradients, False)
    # The loss scale will be halved due to the NaNs, so the gradient will also
    # be halved
    backend.set_value(expected_gradient,
                      backend.get_value(expected_gradient / 2))
    model.fit(dataset)
    self.assertEqual(backend.eval(layer.v), -3)

  @test_util.run_in_graph_and_eager_modes
  def test_loss_scale_optimizer_overrides_policy_loss_scale(self):
    with policy.policy_scope(policy.Policy('float32', loss_scale=10.)):
      opt = gradient_descent.SGD(1.)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale=5.)
      x = layers.Input(shape=(1,))
      y = mp_test_util.AddLayer()(x)
      model = models.Model(x, y)
      model.compile(opt, loss='mse')
      self.assertEqual(self.evaluate(model.optimizer.loss_scale()), 5.)

  @test_util.run_in_graph_and_eager_modes
  def test_pass_invalid_optimizer_with_loss_scaling(self):
    with policy.policy_scope(policy.Policy('float32', loss_scale=10.)):
      x = layers.Input(shape=(1,))
      y = mp_test_util.AddLayer()(x)
      model = models.Model(x, y)
      if context.executing_eagerly():
        error_msg = 'Use a `tf.keras` Optimizer instead'
      else:
        error_msg = 'optimizer" must be an instance of '
      with self.assertRaisesRegexp(ValueError, error_msg):
        model.compile(optimizers.SGD(1.), 'mse')

  @test_util.run_in_graph_and_eager_modes
  def test_functional_model_loss_dtype(self):
    with policy.policy_scope('float16'):
      x = layers.Input(shape=(1,))
      y = mp_test_util.AddLayer()(x)
      model = models.Model(x, y)
      model.add_loss(math_ops.cast(y, 'float32'))
      # The loss should not be casted to the policy's dtype.
      self.assertEqual(model.losses[0].dtype, 'float32')

  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn,
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'base_h5',
          'strategy_fn': default_strategy_fn,
          'h5': True,
      }, {
          'testcase_name': 'distribute_h5',
          'strategy_fn': create_mirrored_strategy,
          'h5': True,
      })
  @test_util.run_in_graph_and_eager_modes
  def test_save_weights_with_autocast_vars(self, strategy_fn, h5=False):
    with strategy_fn().scope():
      with policy.policy_scope('mixed_float16'):
        x = layers.Input(shape=(1,), batch_size=2)
        layer = mp_test_util.AddLayer(assert_type=dtypes.float16)
        y = layer(x)
        y = math_ops.cast(y, dtypes.float32)
        model = models.Model(inputs=x, outputs=y)

    model.set_weights([np.array(100.)])
    x = np.ones((2, 1))
    self.assertAllClose(backend.get_value(model(x)), x + 100.)
    suffix = '.h5' if h5 else ''
    weights_file = os.path.join(self.get_temp_dir(), 'weights' + suffix)
    model.save_weights(weights_file)

    model.set_weights([np.array(200.)])
    self.assertAllClose(backend.get_value(model(x)), x + 200.)
    model.load_weights(weights_file)
    self.assertAllClose(backend.get_value(model(x)), x + 100.)
    self.assertEqual(model.get_weights(), [np.array(100.)])

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn,
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'different_var_name',
          'strategy_fn': default_strategy_fn,
          'var_name': 'w'
      }, {
          'testcase_name': 'different_var_name_distribute',
          'strategy_fn': create_mirrored_strategy,
          'var_name': 'w'
      })
  def test_save_slot_variables_with_autocast_vars(self,
                                                  strategy_fn,
                                                  var_name='v'):
    self._skip_if_strategy_unsupported(strategy_fn)
    p = policy.Policy('mixed_float16', loss_scale=None)
    with strategy_fn().scope(), policy.policy_scope(p):
      x = layers.Input(shape=(2,), batch_size=2)
      # Having a var_name other than 'v' tests that a fixed bug (b/134713714)
      # does not reoccur. The bug was that a crash would occur when saving a
      # checkpoint where an AutoCastVariable with a slot variable would have a
      # different name than the layer attribute's name (layer.v in this case).
      layer = mp_test_util.AddLayer(assert_type=dtypes.float16,
                                    var_name=var_name)
      y = layer(x)
      y = math_ops.cast(y, dtypes.float32)
      model = models.Model(inputs=x, outputs=y)
      opt = gradient_descent.SGD(1., 1.)
      model.compile(
          optimizer=opt,
          loss='mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

    model.fit(np.zeros((2, 2)), np.zeros((2, 2)), batch_size=2)
    weights_file = os.path.join(self.get_temp_dir(), 'weights')
    model.save_weights(weights_file)
    saved_slot = backend.get_value(opt.get_slot(layer.v, 'momentum'))

    model.fit(np.zeros((2, 2)), np.zeros((2, 2)), batch_size=2)
    new_slot = backend.get_value(opt.get_slot(layer.v, 'momentum'))
    self.assertNotEqual(new_slot, saved_slot)

    model.load_weights(weights_file)
    restored_slot = backend.get_value(opt.get_slot(layer.v, 'momentum'))
    self.assertEqual(restored_slot, saved_slot)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(*TESTCASES)
  def test_save_weights_with_dynamic_loss_scaling(self, strategy_fn):
    self._skip_if_strategy_unsupported(strategy_fn)
    strategy = strategy_fn()
    if (isinstance(strategy, mirrored_strategy.MirroredStrategy) and
        not context.executing_eagerly()):
      # TODO(b/121381184): Enable running the test in this case.
      return

    # Create and run model.
    with strategy.scope():
      x = layers.Input(shape=(2,), batch_size=2, dtype=dtypes.float32)
      y = mp_test_util.AddLayer(assert_type=dtypes.float32)(x)
      model = models.Model(inputs=x, outputs=y)

      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=1., increment_period=2., multiplier=2.)
      opt = gradient_descent.SGD(1.)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      model.compile(
          optimizer=opt,
          loss='mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
    # Run for 3 steps (6 examples with a batch size of 2)
    model.fit(np.zeros((6, 2)), np.zeros((6, 2)), batch_size=2)
    self.assertEqual(backend.get_value(loss_scale()), 2)
    self.assertEqual(backend.get_value(loss_scale._num_good_steps), 1)

    # Save model weights.
    save_prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    model.save_weights(save_prefix)

    # Run model again for 1 step (2 examples with a batch size of 2)
    model.fit(np.zeros((2, 2)), np.zeros((2, 2)), batch_size=2)
    self.assertEqual(backend.get_value(loss_scale()), 4)
    self.assertEqual(backend.get_value(loss_scale._num_good_steps), 0)

    # Load model weights and ensure loss scale weights are restored.
    model.load_weights(save_prefix)
    self.assertEqual(backend.get_value(loss_scale()), 2)
    self.assertEqual(backend.get_value(loss_scale._num_good_steps), 1)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn,
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'base_h5',
          'strategy_fn': default_strategy_fn,
          'h5': True,
      }, {
          'testcase_name': 'distribute_h5',
          'strategy_fn': create_mirrored_strategy,
          'h5': True,
      })
  def test_save_model_with_dynamic_loss_scaling(self, strategy_fn, h5=False):
    self._skip_if_strategy_unsupported(strategy_fn)
    # TODO(reedwm): Support and test saving model with a mixed_[b]float16 policy
    # as well.
    strategy = strategy_fn()
    if (isinstance(strategy, mirrored_strategy.MirroredStrategy) and
        not context.executing_eagerly()):
      # TODO(b/121381184): Enable running the test in this case.
      return

    # Create and run model.
    with strategy.scope():
      x = layers.Input(shape=(2,), batch_size=2, dtype=dtypes.float32)
      y = mp_test_util.AddLayer()(x)
      model = models.Model(inputs=x, outputs=y)

      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=1., increment_period=2., multiplier=2.)
      opt = gradient_descent.SGD(1.)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      model.compile(
          optimizer=opt,
          loss='mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
    # Run for 3 steps (6 examples with a batch size of 2)
    model.fit(np.zeros((6, 2)), np.zeros((6, 2)), batch_size=2)
    self.assertEqual(backend.get_value(loss_scale()), 2)
    self.assertEqual(backend.get_value(loss_scale._num_good_steps), 1)
    (weight,) = model.trainable_weights
    orig_weight = backend.get_value(weight)

    # Save model weights.
    save_path = os.path.join(self.get_temp_dir(), 'model')
    model.save(save_path, save_format='h5' if h5 else 'tf')

    # Run model again for 1 step (2 examples with a batch size of 2)
    model.fit(np.zeros((2, 2)), np.zeros((2, 2)), batch_size=2)
    new_weight = backend.get_value(weight)
    self.assertNotEqual(new_weight, orig_weight)
    self.assertEqual(backend.get_value(loss_scale()), 4)
    self.assertEqual(backend.get_value(loss_scale._num_good_steps), 0)

    # Load model weights and ensure loss scale weights are restored.
    model = save.load_model(save_path,
                            custom_objects={'AddLayer': mp_test_util.AddLayer})
    loss_scale = model.optimizer.loss_scale
    (weight,) = model.trainable_weights
    loaded_weight = backend.get_value(weight)
    self.assertEqual(loaded_weight, orig_weight)
    # Currently the loss scale isn't always saved when the model is saved with
    # Model.save(). So we assert the loss scale either has the value when it was
    # saved, or the value it was initialized with.
    # TODO(reedwm): Always save/restore the loss scale with Model.save().
    self.assertIn(backend.get_value(loss_scale()), (1, 2))
    self.assertIn(backend.get_value(loss_scale._num_good_steps), (0, 1))


if __name__ == '__main__':
  base_layer_utils.enable_v2_dtype_behavior()
  test.main()
