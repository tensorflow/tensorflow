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
"""Tests for tf.keras models using tf.distribute.Strategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from tensorflow.python import keras
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import loss_reduction
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import rmsprop
from tensorflow.python.util import nest

_RANDOM_SEED = 1337
_TRAIN_SIZE = 200
_INPUT_SIZE = (10,)
_NUM_CLASS = 2

# Note: Please make sure the tests in this file are also covered in
# keras_backward_compat_test for features that are supported with both APIs.

# TODO(anjalisridhar): Add a decorator that will allow us to run these tests as
# part of the tf.keras unit tests suite.


def simple_sequential_model():
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(16, activation='relu', input_shape=_INPUT_SIZE))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(_NUM_CLASS, activation='softmax'))
  return model


def simple_subclassed_model(num_labels=_NUM_CLASS):

  class _SimpleMLP(keras.Model):

    def __init__(self, num_labels):
      super(_SimpleMLP, self).__init__()
      self.dense = keras.layers.Dense(num_labels)

    def call(self, inputs):
      return self.dense(inputs)

  return _SimpleMLP(num_labels)


def simple_multi_inputs_multi_outputs_model():
  input_a = keras.layers.Input(shape=(16,), name='input_a')
  input_b = keras.layers.Input(shape=(16,), name='input_b')

  merged = keras.layers.concatenate([input_a, input_b], name='merge')
  output_c = keras.layers.Dense(3, activation='softmax', name='dense_2')(merged)
  output_d = keras.layers.Dense(2, activation='softmax', name='dense_3')(merged)
  model = keras.models.Model(
      inputs=[input_a, input_b], outputs=[output_c, output_d])
  return model


def get_multi_inputs_multi_outputs_data():
  (a_train, c_train), (a_test, c_test) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=(16,),
      num_classes=3,
      random_seed=_RANDOM_SEED)
  (b_train, d_train), (b_test, d_test) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=(16,),
      num_classes=2,
      random_seed=_RANDOM_SEED)
  (m_train, _), (m_test, _) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=(8,),
      num_classes=2,
      random_seed=_RANDOM_SEED)

  c_train = np_utils.to_categorical(c_train)
  c_test = np_utils.to_categorical(c_test)
  d_train = np_utils.to_categorical(d_train)
  d_test = np_utils.to_categorical(d_test)

  train_data = {
      'input_a': a_train,
      'input_b': b_train,
      'input_m': m_train,
      'output_c': c_train,
      'output_d': d_train
  }
  test_data = {
      'input_a': a_test,
      'input_b': b_test,
      'input_m': m_test,
      'output_c': c_test,
      'output_d': d_test
  }

  return (train_data, test_data)


def batch_wrapper(dataset, batch_size, distribution, repeat=None):
  if repeat:
    dataset = dataset.repeat(repeat)
  # TPUs currently require fully defined input shapes, drop_remainder ensures
  # the input will have fully defined shapes.
  if isinstance(distribution,
                (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)):
    return dataset.batch(batch_size, drop_remainder=True)
  else:
    return dataset.batch(batch_size)


def get_model():
  x = keras.layers.Input(shape=(3,), name='input')
  y = keras.layers.Dense(4, name='dense')(x)
  model = keras.Model(x, y)
  return model


def get_sample_weights_model():
  x = keras.layers.Input(shape=(1,), name='input')
  y = keras.layers.Dense(
      1, kernel_initializer='ones', bias_initializer='zeros', name='dense')(
          x)
  model = keras.Model(x, y)
  return model


def get_dataset(distribution):
  inputs = np.zeros((10, 3), dtype=np.float32)
  targets = np.zeros((10, 4), dtype=np.float32)
  dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
  dataset = dataset.repeat(100)
  dataset = batch_wrapper(dataset, 10, distribution)
  return dataset


def get_predict_dataset(distribution):
  inputs = np.zeros((10, 3), dtype=np.float32)
  dataset = dataset_ops.Dataset.from_tensor_slices(inputs)
  dataset = dataset.repeat(100)
  dataset = batch_wrapper(dataset, 10, distribution)
  return dataset


def convert_numpy_to_dataset_with_unknown_cardinality(inputs, targets=None):
  if targets is not None:
    input_slices = (inputs, targets)
    dummy_op = (lambda inp, target: True)
  else:
    input_slices = inputs
    dummy_op = (lambda inp: True)

  original_dataset = (dataset_ops.Dataset.from_tensor_slices(input_slices))
  ds_with_unknown_cardinality = (
      original_dataset.filter(dummy_op).batch(10, drop_remainder=True))
  return ds_with_unknown_cardinality


def multi_input_output_model():
  a = keras.layers.Input(shape=(3,), name='input_a')
  b = keras.layers.Input(shape=(5,), name='input_b')
  # TODO(anjalisridhar): Change the output dimension of the second Dense layer
  # once the iterator output validation issue has been fixed.
  dense_1 = keras.layers.Dense(7, name='dense_1')
  dense_2 = keras.layers.Dense(7, name='dense_2')
  c = dense_1(a)
  d = dense_2(b)
  e = keras.layers.Dropout(0.5, name='dropout')(c)
  model = keras.models.Model([a, b], [d, e])
  return model


strategies_minus_default_minus_tpu = [
    strategy_combinations.one_device_strategy,
    strategy_combinations.one_device_strategy_gpu,
    strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
    strategy_combinations.mirrored_strategy_with_two_gpus,
    strategy_combinations.central_storage_strategy_with_gpu_and_cpu
]

strategies_minus_tpu = [
    strategy_combinations.default_strategy,
    strategy_combinations.one_device_strategy,
    strategy_combinations.one_device_strategy_gpu,
    strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
    strategy_combinations.mirrored_strategy_with_two_gpus,
    strategy_combinations.central_storage_strategy_with_gpu_and_cpu
]

tpu_strategies = [
    strategy_combinations.tpu_strategy,  # steps_per_run=2
    strategy_combinations.tpu_strategy_one_step
]

all_strategies = strategies_minus_tpu + tpu_strategies


def strategy_minus_tpu_combinations():
  return combinations.combine(
      distribution=strategies_minus_tpu, mode=['graph', 'eager'])


def tpu_strategy_combinations():
  return combinations.combine(
      distribution=tpu_strategies, mode=['graph', 'eager'])


def tpu_strategy_combinations_graph_only():
  return combinations.combine(distribution=tpu_strategies, mode=['graph'])


def all_strategy_combinations():
  return strategy_minus_tpu_combinations() + tpu_strategy_combinations()


def all_strategy_minus_default_and_tpu_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.one_device_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],
      mode=['graph', 'eager'])


def all_strategy_combinations_minus_default():
  return (all_strategy_minus_default_and_tpu_combinations() +
          tpu_strategy_combinations())


def strategy_and_optimizer_combinations():
  non_tpu_strategies = combinations.times(
      strategy_minus_tpu_combinations(),
      combinations.combine(
          optimizer=[
              strategy_combinations.adagrad_optimizer_v1_fn,
              strategy_combinations.adam_optimizer_v1_fn,
              strategy_combinations.gradient_descent_optimizer_v1_fn,
              strategy_combinations.rmsprop_optimizer_v1_fn,
              strategy_combinations.adadelta_optimizer_keras_v2_fn,
              strategy_combinations.adagrad_optimizer_keras_v2_fn,
              strategy_combinations.adam_optimizer_keras_v2_fn,
              strategy_combinations.adamax_optimizer_keras_v2_fn,
              strategy_combinations.gradient_descent_optimizer_keras_v2_fn,
              strategy_combinations.nadam_optimizer_keras_v2_fn,
              strategy_combinations.rmsprop_optimizer_keras_v2_fn,
              strategy_combinations.ftrl_optimizer_keras_v2_fn
          ]))
  tpu_strategies_graph = combinations.combine(
      distribution=tpu_strategies,
      mode=['graph'],
      optimizer=[
          strategy_combinations.adagrad_optimizer_v1_fn,
          strategy_combinations.adam_optimizer_v1_fn,
          strategy_combinations.gradient_descent_optimizer_v1_fn,
          strategy_combinations.rmsprop_optimizer_v1_fn,
          strategy_combinations.adagrad_optimizer_keras_v2_fn,
          strategy_combinations.adam_optimizer_keras_v2_fn,
          strategy_combinations.gradient_descent_optimizer_keras_v2_fn,
          strategy_combinations.rmsprop_optimizer_keras_v2_fn
      ])
  tpu_strategies_eager = combinations.combine(
      distribution=tpu_strategies,
      mode=['eager'],
      optimizer=[
          strategy_combinations.adagrad_optimizer_keras_v2_fn,
          strategy_combinations.adam_optimizer_keras_v2_fn,
          strategy_combinations.gradient_descent_optimizer_keras_v2_fn,
          strategy_combinations.rmsprop_optimizer_keras_v2_fn
      ])
  return non_tpu_strategies + tpu_strategies_eager + tpu_strategies_graph


class BatchCountingCB(keras.callbacks.Callback):

  def __init__(self):
    super(BatchCountingCB, self).__init__()
    self.train_begin_batches = []
    self.train_end_batches = []
    self.test_begin_batches = []
    self.test_end_batches = []
    self.predict_begin_batches = []
    self.predict_end_batches = []

  def on_train_batch_begin(self, batch, logs=None):
    self.train_begin_batches.append(batch)

  def on_train_batch_end(self, batch, logs=None):
    self.train_end_batches.append(batch)

  def on_test_batch_begin(self, batch, logs=None):
    self.test_begin_batches.append(batch)

  def on_test_batch_end(self, batch, logs=None):
    self.test_end_batches.append(batch)

  def on_predict_batch_begin(self, batch, logs=None):
    self.predict_begin_batches.append(batch)

  def on_predict_batch_end(self, batch, logs=None):
    self.predict_end_batches.append(batch)


class TestDistributionStrategyWithNumpyArrays(test.TestCase,
                                              parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_calculating_input_params_no_steps_no_batch_size(self, distribution):
    # Calculate the per_replica_batch_size scaling factor for strategies
    # that use per_core_batch_size
    replica_scale_factor = 1.0
    if not distributed_training_utils.global_batch_size_supported(distribution):
      replica_scale_factor = distribution.num_replicas_in_sync

    with self.cached_session():
      # Default global batch size 32 for input with 64 samples run in 2 steps
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, 64, steps=None, batch_size=None)
      self.assertEqual(batch_size, 32 // replica_scale_factor)
      self.assertEqual(steps, 2)

      # Computed global batch size 20 is lower than 32 if we pass less samples.
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, 20, steps=None, batch_size=None)
      self.assertEqual(batch_size, 20 // replica_scale_factor)
      self.assertEqual(steps, 1)

  @combinations.generate(all_strategy_combinations())
  def test_calculating_input_params_with_steps_no_batch_size(
      self, distribution):
    # Calculate the per_replica_batch_size scaling factor for strategies
    # that use per_core_batch_size
    replica_scale_factor = 1.0
    if not distributed_training_utils.global_batch_size_supported(distribution):
      replica_scale_factor = distribution.num_replicas_in_sync

    with self.cached_session():
      # Computed global batch size is correct for number of specified 1 step
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, 64, steps=1, batch_size=None)
      self.assertEqual(batch_size, 64 // replica_scale_factor)
      self.assertEqual(steps, 1)

      # Computed global batch size is correct for number of specified 2 steps
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, 64, steps=2, batch_size=None)
      self.assertEqual(batch_size, 32 // replica_scale_factor)
      self.assertEqual(steps, 2)

      # All samples can not be consumed in specified number of steps
      with self.assertRaisesRegexp(ValueError, 'not divisible by steps'):
        distributed_training_utils.get_input_params(
            distribution, 63, steps=2, batch_size=None)

      # This cases is different for different strategies due to the
      # difference in supported batch size being global or per-replica.
      if replica_scale_factor == 1:
        # Computed global batch size is correct even if not sharadable
        steps, batch_size = distributed_training_utils.get_input_params(
            distribution, 63, steps=3, batch_size=None)
        self.assertEqual(batch_size, 21)
        self.assertEqual(steps, 3)
      else:
        # Computed global batch size can not be sharded across replicas
        with self.assertRaisesRegexp(
            ValueError, 'could not be sharded evenly '
            'across the sync replicas'):
          distributed_training_utils.get_input_params(
              distribution, 63, steps=1, batch_size=None)

  @combinations.generate(all_strategy_combinations())
  def test_calculating_input_params_no_steps_with_batch_size(
      self, distribution):
    # Calculate the per_replica_batch_size scaling factor for strategies
    # that use per_core_batch_size
    replica_scale_factor = 1.0
    if not distributed_training_utils.global_batch_size_supported(distribution):
      replica_scale_factor = distribution.num_replicas_in_sync

    with self.cached_session():
      # Computed steps is correct for specified batch size
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, 64, steps=None, batch_size=16)
      self.assertEqual(batch_size, 16)
      self.assertEqual(steps, 4 // replica_scale_factor)

      # Computed steps is correct for specified batch size
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, 64, steps=None, batch_size=32)
      self.assertEqual(batch_size, 32)
      self.assertEqual(steps, 2 // replica_scale_factor)

  @combinations.generate(all_strategy_combinations())
  def test_calculating_input_params_with_steps_with_batch_size(
      self, distribution):
    with self.cached_session():
      # No change to steps and batch size if both specified and feasible
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, 64, steps=5, batch_size=3)
      self.assertEqual(batch_size, 3)
      self.assertEqual(steps, 5)

      # Number of samples is less than global batch size * steps
      with self.assertRaisesRegexp(ValueError, 'less than samples required'):
        distributed_training_utils.get_input_params(
            distribution, 64, steps=10, batch_size=13)

  @combinations.generate(all_strategy_combinations())
  def test_calling_model_with_numpy_arrays(self, distribution):
    with self.cached_session():
      with distribution.scope():
        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(0.001)
        model = get_model()
        loss = 'mse'
        metrics = ['mae']
        model.compile(
            optimizer,
            loss,
            metrics=metrics)

        inputs = np.zeros((64, 3), dtype=np.float32)
        targets = np.zeros((64, 4), dtype=np.float32)

        # Call fit with validation data
        model.fit(
            inputs,
            targets,
            epochs=1,
            batch_size=2,
            verbose=0,
            validation_data=(inputs, targets))

        # TODO(anjalisridhar): We need tests for when the batch size and steps
        # are smaller and results in a 0 batch_size and steps value.
        model.evaluate(inputs, targets)
        model.evaluate(inputs, targets, batch_size=8)

        model.predict(inputs)
        model.predict(inputs, batch_size=8)

  @combinations.generate(all_strategy_combinations())
  def test_calling_model_with_mixed_precision(self, distribution):
    if isinstance(distribution.extended,
                  parameter_server_strategy.ParameterServerStrategyExtended):
      self.skipTest('b/152097775')
    if isinstance(distribution,
                  (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)):
      policy_name = 'mixed_bfloat16'
    else:
      policy_name = 'mixed_float16'
    with self.cached_session(), \
         distribution.scope(), \
         policy.policy_scope(policy_name):
      optimizer_fn = gradient_descent_keras.SGD
      optimizer = optimizer_fn(0.001)
      x = keras.layers.Input(shape=(3,), name='input')
      y = keras.layers.Dense(4, name='dense')(x)
      y = keras.layers.Activation('softmax', dtype='float32')(y)
      model = keras.Model(x, y)
      loss = 'mse'
      metrics = ['mae']
      model.compile(
          optimizer,
          loss,
          metrics=metrics)

      # We need to pass float32 since TPUs do not support float64, even though
      # these arrays will immediately be casted to bfloat16 on TPUs. We also
      # cannot pass bfloat16, as Numpy does not support it.
      inputs = np.zeros((64, 3), dtype='float32')
      targets = np.zeros((64, 4), dtype='float32')

      model.fit(
          inputs,
          targets,
          epochs=1,
          batch_size=2,
          verbose=0,
          validation_data=(inputs, targets))

      model.evaluate(inputs, targets)
      model.evaluate(inputs, targets, batch_size=8)

      model.predict(inputs)
      model.predict(inputs, batch_size=8)

  @combinations.generate(all_strategy_combinations())
  def test_operator_overload_mixed_precision(self, distribution):
    # Regression test that tests a fixed bug does not reoccur. Adding an
    # AutoCastVariable to a tensor on a TPU, where the variable was the LHS of
    # the '+' operator, used to cause the gradient w.r.t. the variable to be
    # None.
    if isinstance(distribution.extended,
                  parameter_server_strategy.ParameterServerStrategyExtended):
      self.skipTest('b/152097775')

    if isinstance(distribution,
                  (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)):
      policy_name = 'mixed_bfloat16'
    else:
      policy_name = 'mixed_float16'

    class MyLayer(keras.layers.Layer):

      def build(self, _):
        self.v1 = self.add_weight('v', ())
        self.v2 = self.add_weight('v', ())

      def call(self, inp):
        inp += self.v1
        return self.v2 + inp

    with self.cached_session(), distribution.scope():
      layer = MyLayer(dtype=policy.Policy(policy_name))
      def run_fn():
        x = np.array([1.])
        with backprop.GradientTape() as tape:
          y = layer(x)
        grad_v1, grad_v2 = tape.gradient(y, [layer.v1, layer.v2])
        return grad_v1, grad_v2
      if context.executing_eagerly():
        run_fn = def_function.function(run_fn)

      grad_v1, grad_v2 = distribution.run(run_fn)
      self.assertIsNotNone(grad_v1)
      self.assertIsNotNone(grad_v2)

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.one_device_strategy] +
          tpu_strategies,
          mode=['graph', 'eager']))
  def test_optimizer_in_cross_replica_context_raises_error(self, distribution):

    with self.cached_session(), distribution.scope():
      model = keras.models.Sequential([keras.layers.Dense(1)])
      x = np.array([[1.]])
      with backprop.GradientTape() as tape:
        y = model(x)
      gradients = tape.gradient(y, model.trainable_variables)
      optimizer = gradient_descent_keras.SGD()

      with self.assertRaisesRegex(RuntimeError,
                                  'cannot be called in cross-replica context'):
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  @combinations.generate(all_strategy_combinations())
  def test_calling_model_with_nested_numpy_arrays(self, distribution):
    with self.cached_session():
      with distribution.scope():
        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(learning_rate=0.001)
        model = multi_input_output_model()
        loss = 'mse'
        model.compile(
            optimizer,
            loss)

      input_a_np = np.asarray(np.random.random((64, 3)), dtype=np.float32)
      input_b_np = np.asarray(np.random.random((64, 5)), dtype=np.float32)
      inputs = [input_a_np, input_b_np]

      output_d_np = np.asarray(np.random.random((64, 7)), dtype=np.float32)
      output_e_np = np.asarray(np.random.random((64, 7)), dtype=np.float32)
      targets = [output_d_np, output_e_np]

      # Call fit with validation data
      model.fit(inputs, targets, epochs=1, batch_size=8, verbose=0)

      # TODO(anjalisridhar): We need tests for when the batch size and steps are
      # smaller and results in a 0 batch_size and steps value.
      model.evaluate(inputs, targets)
      model.evaluate(inputs, targets, batch_size=8)

      model.predict(inputs)
      model.predict(inputs, batch_size=8)

  @combinations.generate(
      combinations.combine(
          distribution=strategies_minus_tpu,
          mode=['graph', 'eager']))
  def test_numpy_with_sample_weights(self, distribution):
    with self.cached_session(), distribution.scope():
      model = get_sample_weights_model()
      optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      model.compile(
          optimizer,
          loss)

      inputs = np.array([[0], [1], [2], [3]], np.float32)
      targets = np.array([[2], [4], [6], [8]], np.float32)
      sample_weights = np.array([0.25, 0.5, 0.75, 1], np.float32)

      result = model.evaluate(
          inputs,
          targets,
          batch_size=2,
          sample_weight=sample_weights,
          verbose=1)
      # The per sample loss is multipled by the corresponding sample weight. The
      # average of these weighted losses is the return value of the `evaluate`
      # call. For example, in the test above the average weighted loss is
      # calculated in the following manner:
      # batch_1 = (((2-0)^2) * 0.25 + ((4-1)^2) * 0.5) / 2 = 5.5 / 2 = 2.75
      # batch_2 = (((6-2)^2 * 0.75) + ((8-3)^2 * 1)) / 2 = 37 / 2 = 18.5
      # final result = (batch_1 + batch_2) / 2 = 10.625.
      # The first time we divide by number of input samples and the second time
      # we divide by number of steps/batches that the loss is aggregated over.
      self.assertAllClose(result, 10.625)

      # We now test without passing sample_weights:
      # batch_1 = ((2-0)^2) + ((4-1)^2) / 2 = 13 / 2 = 6.5
      # batch_2 = ((6-2)^2) + ((8-3)^2) / 2 = 41 / 2 = 20.5
      # final result = (batch_1 + batch_2) / 2 =  27 / 2 = 13.5
      result = model.evaluate(inputs, targets, batch_size=2, verbose=1)
      self.assertAllClose(result, 13.5)

  @combinations.generate(all_strategy_combinations())
  def test_flatten_predict_outputs(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = multi_input_output_model()
        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(learning_rate=0.001)
        loss = 'mse'
        model.compile(
            optimizer,
            loss)

      # We take 6 input samples with each input having a dimension of 3 or 5.
      input_a_np = np.asarray(np.random.random((6, 3)), dtype=np.float32)
      input_b_np = np.asarray(np.random.random((6, 5)), dtype=np.float32)
      inputs = [input_a_np, input_b_np]

      outs = model.predict(inputs)
      # `predict` a list that is equal in length to the number of model outputs.
      # In this test our model has two outputs and each element of `outs`
      # corresponds to all the samples of one of the model outputs.
      self.assertLen(outs, 2)
      # Each of the output samples have a dimension of 7. We should process all
      # the available input samples(6).
      self.assertAllEqual([6, 7], outs[0].shape)
      self.assertAllEqual([6, 7], outs[1].shape)

  @combinations.generate(
      combinations.times(tpu_strategy_combinations_graph_only(),
                         combinations.combine(batch_size=[4, 6])))
  def test_evaluate_with_partial_batch(self, distribution, batch_size):
    with self.cached_session():
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae', keras.metrics.CategoricalAccuracy()]

      with distribution.scope():
        model_with_ds_strategy = get_model()
        model_with_ds_strategy.compile(optimizer, loss, metrics=metrics)

      cpu_model = get_model()
      cpu_model.compile(optimizer, loss, metrics=metrics)

      x = np.random.random((10, 3)).astype('float32')
      y = np.random.random((10, 4)).astype('float32')

      # As sample size is 10, we batch by 4 so that the last batch is
      # a partial batch. Also `evaluate()` using numpy array as inputs without
      # distribution strategy uses entire sample as a single batch. As so,
      # we remove parameters `batch_size` and `steps`.
      cpu_model.set_weights(model_with_ds_strategy.get_weights())
      evaluate_ground_truth = cpu_model.evaluate(x, y)

      # We don't compare the loss as loss is currently not computed as metric
      # in Keras, the loss value is inaccurate for last partial batch due to
      # more weights for the last batch samples.
      steps = np.ceil(10.0 / batch_size)
      self.assertAllClose(
          model_with_ds_strategy.evaluate(
              x, y, batch_size=batch_size, steps=steps)[1:],
          evaluate_ground_truth[1:],
          atol=1e-5,
          rtol=1e-5)
      # Test that `steps` is inferred correctly when final partial batch exists.
      self.assertAllClose(
          model_with_ds_strategy.evaluate(x, y, batch_size=batch_size)[1:],
          evaluate_ground_truth[1:],
          atol=1e-5,
          rtol=1e-5)

  @combinations.generate(
      combinations.times(
          tpu_strategy_combinations_graph_only()))
  def test_predict_with_partial_batch(self, distribution):
    with self.cached_session():
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'

      with distribution.scope():
        model_with_ds_strategy = get_model()
        model_with_ds_strategy.compile(
            optimizer,
            loss)

      cpu_model = get_model()
      cpu_model.compile(optimizer, loss)

      inputs = np.random.random((10, 3)).astype(np.float32)

      # As sample size is 10, we batch by 4 so that the last batch is
      # a partial batch. Also `predict()` using numpy array as inputs without
      # distribution strategy uses entire sample as a single batch. As so,
      # we remove parameters `batch_size` and `steps`.
      cpu_model.set_weights(model_with_ds_strategy.get_weights())
      predict_ground_truth = cpu_model.predict(inputs)
      self.assertAllClose(
          model_with_ds_strategy.predict(inputs, batch_size=4, steps=3),
          predict_ground_truth,
          atol=1e-5,
          rtol=1e-5)
      # Test that `steps` is inferred correctly when final partial batch exists.
      self.assertAllClose(
          model_with_ds_strategy.predict(inputs, batch_size=4),
          predict_ground_truth,
          atol=1e-5,
          rtol=1e-5)

  @combinations.generate(tpu_strategy_combinations_graph_only())
  def test_no_target_model(self, distribution):
    with self.cached_session():
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)

      class MyLayer(keras.layers.Layer):

        def call(self, inputs, training=None):
          self.add_loss(math_ops.reduce_sum(inputs), inputs=True)
          return inputs

      with distribution.scope():
        model = keras.models.Sequential()
        model.add(
            keras.layers.Dense(16, activation='relu', input_shape=_INPUT_SIZE))
        model.add(MyLayer())
        model.add(keras.layers.Dense(_NUM_CLASS, activation='softmax'))

        model.compile(optimizer)
        inputs = np.zeros((20, 10), np.float32)

        model.fit(inputs, epochs=1, steps_per_epoch=2)
        model.predict(inputs, steps=1)
        model.evaluate(inputs, steps=1)

  @combinations.generate(
      combinations.times(
          tpu_strategy_combinations_graph_only()))
  def test_predict_multi_output_model_with_partial_batch(
      self, distribution):
    with self.cached_session():
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'

      with distribution.scope():
        model_with_ds_strategy = simple_multi_inputs_multi_outputs_model()
        model_with_ds_strategy.compile(
            optimizer,
            loss)

      cpu_model = simple_multi_inputs_multi_outputs_model()
      cpu_model.compile(optimizer, loss)

      input_data, _ = get_multi_inputs_multi_outputs_data()
      input_dict = {
          'input_a': input_data['input_a'],
          'input_b': input_data['input_b'],
      }

      # As sample size is 200, we batch by 18 so that the last batch is
      # a partial batch. Also `fit()` using numpy array as inputs without
      # distribution strategy uses entire sample as a single batch. As so,
      # we remove parameters `batch_size` and `steps`.
      cpu_model.set_weights(model_with_ds_strategy.get_weights())
      self.assertAllClose(
          model_with_ds_strategy.predict(input_dict, batch_size=18, steps=12),
          cpu_model.predict(input_dict),
          atol=1e-4,
          rtol=1e-4)

  @combinations.generate(all_strategy_combinations())
  def test_gradients_are_none(self, distribution):

    if not context.executing_eagerly():
      self.skipTest('None gradients are not supported in graph mode')

    class DenseWithExtraWeight(keras.layers.Dense):

      def build(self, input_shape):
        # Gradients w.r.t. extra_weights are None
        self.extra_weight_1 = self.add_weight('extra_weight_1', shape=(),
                                              initializer='ones')
        super(DenseWithExtraWeight, self).build(input_shape)
        self.extra_weight_2 = self.add_weight('extra_weight_2', shape=(),
                                              initializer='ones')

    with distribution.scope():
      model = keras.Sequential([DenseWithExtraWeight(4, input_shape=(4,))])
      model.compile('adam', 'mse')

    inputs = np.random.normal(size=(64, 4))
    targets = np.random.normal(size=(64, 4))
    old_kernel = model.get_weights()[1]
    model.fit(inputs, targets)
    new_kernel = model.get_weights()[1]
    self.assertNotAllEqual(old_kernel, new_kernel)


class TestDistributionStrategyWithDatasets(test.TestCase,
                                           parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_calling_model_on_same_dataset(self, distribution):
    with self.cached_session():
      with distribution.scope():
        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(0.001)
        model = get_model()
        loss = 'mse'
        metrics = ['mae', keras.metrics.CategoricalAccuracy()]
        model.compile(
            optimizer,
            loss,
            metrics=metrics)

      dataset = get_dataset(distribution)

      # Call fit with validation data
      model.fit(
          dataset,
          epochs=1,
          steps_per_epoch=2,
          verbose=0,
          validation_data=dataset,
          validation_steps=2)
      model.fit(
          dataset,
          epochs=1,
          steps_per_epoch=2,
          verbose=0,
          validation_data=dataset,
          validation_steps=2)
      model.predict(get_predict_dataset(distribution), steps=2)

  @combinations.generate(all_strategy_combinations())
  def test_model_interleaved_eval_same_as_direct_eval(
      self, distribution):
    with self.cached_session():
      with distribution.scope():
        optimizer_fn = gradient_descent_keras.SGD
        user_controlled_model = get_model()
        user_controlled_model.compile(
            optimizer_fn(0.001),
            loss='mse',
            metrics=['mae', keras.metrics.CategoricalAccuracy()])

        interleaved_model = get_model()
        interleaved_model.set_weights(user_controlled_model.get_weights())
        interleaved_model.compile(
            optimizer_fn(0.001),
            loss='mse',
            metrics=['mae', keras.metrics.CategoricalAccuracy()])

      dataset = get_dataset(distribution)

      # Call fit with validation interleaved
      interleaved_output = interleaved_model.fit(
          dataset,
          epochs=2,
          steps_per_epoch=2,
          verbose=1,
          validation_data=dataset,
          validation_steps=2,
          shuffle=False)

      # Manually control the validation running after each epoch.
      user_controlled_output = []
      for _ in range(2):
        user_controlled_model.fit(
            dataset, epochs=1, steps_per_epoch=2, verbose=1, shuffle=False)
        user_controlled_output.append(
            user_controlled_model.evaluate(dataset, steps=2))

      self.assertEqual(interleaved_output.history['val_loss'],
                       [x[0] for x in user_controlled_output])
      val_mean_absolute_error = interleaved_output.history.get(
          'val_mean_absolute_error')
      if not val_mean_absolute_error:
        # The name of the metric changed in TF2.0
        val_mean_absolute_error = interleaved_output.history['val_mae']
      self.assertEqual(val_mean_absolute_error,
                       [x[1] for x in user_controlled_output])
      self.assertEqual(interleaved_output.history['val_categorical_accuracy'],
                       [x[2] for x in user_controlled_output])

  @combinations.generate(all_strategy_combinations())
  def test_fit_with_tuple_and_dict_dataset_inputs(self, distribution):
    with self.cached_session():
      with distribution.scope():
        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(learning_rate=0.001)
        model = multi_input_output_model()
        loss = 'mse'
        metrics = ['mae', keras.metrics.CategoricalAccuracy()]
        model.compile(
            optimizer,
            loss,
            metrics=metrics)

      input_a_np = np.random.random((10, 3)).astype('float32')
      input_b_np = np.random.random((10, 5)).astype('float32')
      output_d_np = np.random.random((10, 7)).astype('float32')
      output_e_np = np.random.random((10, 7)).astype('float32')

      # Test with tuples
      dataset_tuple = dataset_ops.Dataset.from_tensor_slices(
          ((input_a_np, input_b_np), (output_d_np, output_e_np)))
      dataset_tuple = dataset_tuple.repeat(100)
      dataset_tuple = dataset_tuple.batch(10)

      model.fit(dataset_tuple, epochs=1, steps_per_epoch=2, verbose=1)

      # Test with dict
      dataset_dict = dataset_ops.Dataset.from_tensor_slices(({
          'input_a': input_a_np,
          'input_b': input_b_np
      }, (output_d_np, output_e_np)))
      dataset_dict = dataset_dict.repeat(100)
      dataset_dict = dataset_dict.batch(10)

      model.fit(dataset_dict, epochs=1, steps_per_epoch=2, verbose=1)

  @combinations.generate(all_strategy_combinations())
  def test_fit_with_dictionary_in_the_dataset_b135161171(
      self, distribution):

    if isinstance(distribution,
                  (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)):
      self.skipTest('b/142805125')

    def custom_loss(predict, label, weight):
      bce = keras.losses.binary_crossentropy(label, predict)
      return math_ops.reduce_mean(bce * weight)

    with self.cached_session():
      with distribution.scope():
        input_img = keras.layers.Input([64, 64, 3], name='img')
        input_lbl = keras.layers.Input([64, 64, 1], name='lbl')
        input_weight = keras.layers.Input([64, 64], name='weight')
        predict = keras.layers.Conv2D(2, [1, 1], padding='same')(input_img)
        loss_lambda = keras.layers.Lambda(
            lambda x: custom_loss(*x), name='my_loss')
        my_loss = loss_lambda([predict, input_lbl, input_weight])
        model = keras.models.Model(
            inputs=[input_img, input_lbl, input_weight],
            outputs=[predict, my_loss])
        model.add_loss(model.get_layer('my_loss').output)
        model.compile(
            optimizer='adam')

      if context.executing_eagerly():

        def map_fn(img, lbl, weight):
          inputs = {'img': img, 'lbl': lbl, 'weight': weight}
          return (inputs,)
      else:

        def map_fn(img, lbl, weight):
          inputs = {'img': img, 'lbl': lbl, 'weight': weight}
          return inputs, {}

      fake_imgs = np.ones([50, 64, 64, 3], dtype=np.float32)
      fake_lbls = np.ones([50, 64, 64, 1], dtype=np.float32)
      fake_weights = np.ones([50, 64, 64], dtype=np.float32)

      data = dataset_ops.Dataset.from_tensor_slices(
          (fake_imgs, fake_lbls, fake_weights)).map(map_fn).batch(10)

      model.fit(data)

  @combinations.generate(all_strategy_combinations())
  def test_fit_eval_and_predict_methods_on_dataset_without_steps(
      self, distribution):
    with self.cached_session():
      with distribution.scope():
        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(0.001)
        model = get_model()
        loss = 'mse'
        metrics = ['mae', keras.metrics.CategoricalAccuracy()]
        model.compile(
            optimizer,
            loss,
            metrics=metrics)

      inputs = np.zeros((1000, 3), dtype=np.float32)
      targets = np.zeros((1000, 4), dtype=np.float32)
      # steps/steps_per_epoch are calculated when using numpy arrays as
      # input data.
      fit_with_numpy = model.fit(
          inputs, targets, epochs=1, batch_size=10).history
      eval_with_numpy = model.evaluate(inputs, targets, batch_size=10)
      predict_with_numpy = model.predict(inputs, batch_size=10)

      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.batch(10, drop_remainder=True)
      fit_with_ds = model.fit(dataset, epochs=1).history
      eval_with_ds = model.evaluate(dataset)
      predict_dataset = dataset_ops.Dataset.from_tensor_slices(inputs)
      predict_dataset = predict_dataset.batch(10, drop_remainder=True)
      predict_with_ds = model.predict(predict_dataset)
      self.assertAllClose(fit_with_numpy, fit_with_ds, atol=1e-4, rtol=1e-4)
      self.assertAllClose(eval_with_numpy, eval_with_ds, atol=1e-4, rtol=1e-4)
      self.assertAllClose(
          predict_with_numpy, predict_with_ds, atol=1e-4, rtol=1e-4)

  @combinations.generate(all_strategy_combinations())
  def test_on_dataset_with_unknown_cardinality_without_steps(
      self, distribution, mode):

    if mode == 'graph' and isinstance(
        distribution, (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)):
      self.skipTest('partial batch not supported with TPU in graph mode.')

    with self.cached_session():
      with distribution.scope():
        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(0.001)
        model = get_model()
        loss = 'mse'
        metrics = ['mae', keras.metrics.CategoricalAccuracy()]
        model.compile(
            optimizer,
            loss,
            metrics=metrics)

      inputs = np.zeros((1000, 3), dtype=np.float32)
      targets = np.zeros((1000, 4), dtype=np.float32)
      # steps/steps_per_epoch are calculated when using numpy arrays as
      # input data.
      fit_with_numpy = model.fit(
          inputs, targets, epochs=1, batch_size=10).history
      fit_with_numpy_multiple_epochs = model.fit(
          inputs, targets, epochs=2, batch_size=10).history
      eval_with_numpy = model.evaluate(inputs, targets, batch_size=10)
      predict_with_numpy = model.predict(inputs, batch_size=10)

      dataset = convert_numpy_to_dataset_with_unknown_cardinality(
          inputs, targets)
      predict_dataset = convert_numpy_to_dataset_with_unknown_cardinality(
          inputs)

      self.assertEqual(
          keras.backend.get_value(cardinality.cardinality(dataset)),
          cardinality.UNKNOWN)
      self.assertEqual(
          keras.backend.get_value(cardinality.cardinality(predict_dataset)),
          cardinality.UNKNOWN)

      eval_with_ds = model.evaluate(dataset)
      predict_with_ds = model.predict(predict_dataset)
      self.assertAllClose(eval_with_numpy, eval_with_ds, atol=1e-4, rtol=1e-4)
      self.assertAllClose(
          predict_with_numpy, predict_with_ds, atol=1e-4, rtol=1e-4)

      fit_with_ds = model.fit(dataset, epochs=1).history
      fit_with_ds_multiple_epochs = model.fit(dataset, epochs=2).history
      self.assertAllClose(fit_with_numpy, fit_with_ds, atol=1e-4, rtol=1e-4)
      self.assertAllClose(
          fit_with_numpy_multiple_epochs,
          fit_with_ds_multiple_epochs,
          atol=1e-4,
          rtol=1e-4)

  @combinations.generate(tpu_strategy_combinations_graph_only())
  def test_on_dataset_with_unknown_cardinality(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = get_model()
        loss = 'mse'
        metrics = ['mae', keras.metrics.CategoricalAccuracy()]
        model.compile(
            gradient_descent.GradientDescentOptimizer(0.001),
            loss,
            metrics=metrics)

      inputs = np.zeros((1000, 3), dtype=np.float32)
      targets = np.zeros((1000, 4), dtype=np.float32)
      # steps/steps_per_epoch are calculated when using numpy arrays as
      # input data.
      eval_with_numpy = model.evaluate(inputs, targets, batch_size=10)
      predict_with_numpy = model.predict(inputs, batch_size=10)

      dataset = convert_numpy_to_dataset_with_unknown_cardinality(
          inputs, targets)
      predict_dataset = convert_numpy_to_dataset_with_unknown_cardinality(
          inputs)

      self.assertEqual(
          keras.backend.get_value(cardinality.cardinality(dataset)),
          cardinality.UNKNOWN)
      self.assertEqual(
          keras.backend.get_value(cardinality.cardinality(predict_dataset)),
          cardinality.UNKNOWN)

      eval_with_ds = model.evaluate(dataset, steps=100)
      predict_with_ds = model.predict(predict_dataset, steps=100)
      self.assertAllClose(eval_with_numpy, eval_with_ds, atol=1e-4, rtol=1e-4)
      self.assertAllClose(
          predict_with_numpy, predict_with_ds, atol=1e-4, rtol=1e-4)

      with self.assertRaisesRegexp(ValueError,
                                   'Number of steps could not be inferred'):
        model.fit(dataset, epochs=1)

  @combinations.generate(all_strategy_combinations())
  def test_fit_eval_and_predict_methods_on_dataset(
      self, distribution):
    with self.cached_session():
      with distribution.scope():
        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(0.001)
        model = get_model()
        loss = 'mse'
        metrics = ['mae', keras.metrics.CategoricalAccuracy()]
        model.compile(
            optimizer,
            loss,
            metrics=metrics)

      dataset = get_dataset(distribution)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)
      model.evaluate(dataset, steps=2, verbose=1)
      model.predict(get_predict_dataset(distribution), steps=2)

  @combinations.generate(strategy_and_optimizer_combinations())
  def test_fit_eval_and_predict_with_optimizer(self, distribution, optimizer):
    with self.cached_session():

      with distribution.scope():

        model = get_model()
        loss = 'mse'
        model.compile(
            optimizer(),
            loss)

      dataset = get_dataset(distribution)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)
      model.evaluate(dataset, steps=2, verbose=1)
      model.predict(get_predict_dataset(distribution), steps=2)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.one_device_strategy
          ],
          mode=['graph', 'eager']))
  def test_dataset_wrong_input_shape(self, distribution, mode):
    if mode == 'graph':
      self.skipTest(
          'TODO(b/120943676, b/120957836): Re-enable for graph once the '
          'validation code is restored.')
    with self.cached_session():
      with distribution.scope():
        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(learning_rate=0.001)
        model = get_model()
        loss = 'mse'
        model.compile(
            optimizer,
            loss)

      # Wrong input shape
      inputs = np.zeros((10, 5), dtype=np.float32)
      targets = np.zeros((10, 4), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      with self.assertRaisesRegexp(ValueError, 'incompatible with the layer'):
        model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu
          ],
          mode=['graph', 'eager']))
  def test_dataset_external_batch_input_validation(
      self, distribution):
    with self.cached_session():
      with distribution.scope():
        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(learning_rate=0.001)
        model = get_model()
        loss = 'mse'
        model.compile(
            optimizer,
            loss)

      # Batching is done outside tf.data's `batch`
      inputs = np.zeros((100, 10, 3), dtype=np.float32)
      targets = np.zeros((100, 10, 4), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_two_gpus
          ],
          mode=['graph', 'eager']))
  def test_learning_phase_value(self, distribution):
    # TODO(anjalisridhar): Modify this test to use Lambdas since we can compare
    # meaningful values. Currently we don't pass the learning phase if the
    # Lambda layer uses the learning phase.
    with self.cached_session():
      with distribution.scope():
        x = keras.layers.Input(shape=(1,), name='input')
        y = keras.layers.Dense(1, kernel_initializer='ones')(x)
        z = keras.layers.Dropout(0.9999)(y)
        model = keras.Model(x, z)
        initial_weights = model.get_weights()

        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(0.005)
        loss = 'mse'
        metrics = ['acc']
        model.compile(
            optimizer,
            loss,
            metrics=metrics)

      batch_size = 8
      if isinstance(distribution, mirrored_strategy.MirroredStrategy):
        # MirroredStrategy uses global batch size.
        batch_size = 8 * distribution.num_replicas_in_sync

      inputs = np.ones((10, 1), dtype=np.float32)
      targets = np.ones((10, 1), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat().batch(batch_size)
      hist = model.fit(dataset, epochs=1, steps_per_epoch=20, verbose=1)
      self.assertAlmostEqual(hist.history['acc'][0], 0, 0)

      with distribution.scope():
        model.set_weights(initial_weights)
      # TODO(psv/anjalisridhar): Enable these lines after we fix b/117431185.
      # evaluate_output = model.evaluate(dataset, steps=20)
      # self.assertAlmostEqual(evaluate_output[1], 1, 0)

      inputs = np.ones((10, 1), dtype=np.float32)
      predict_dataset = dataset_ops.Dataset.from_tensor_slices(inputs)

      predict_dataset = predict_dataset.repeat().batch(batch_size)
      output = model.predict(predict_dataset, steps=10)
      # `predict` runs for 10 steps
      ref_output = np.ones((160, 1), dtype=np.float32)
      self.assertArrayNear(output, ref_output, 1e-1)

  @combinations.generate(all_strategy_combinations())
  def testOptimizerWithCallbacks(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = get_model()
        optimizer = gradient_descent_keras.SGD(0.01)
        loss = 'mse'
        model.compile(
            optimizer,
            loss)

      dataset = get_dataset(distribution)

      def schedule(_):
        return 0.001

      model.fit(
          dataset,
          epochs=1,
          steps_per_epoch=2,
          verbose=0,
          callbacks=[keras.callbacks.LearningRateScheduler(schedule)])
      self.assertAllClose(0.001, keras.backend.get_value(model.optimizer.lr))

  @combinations.generate(
      combinations.times(tpu_strategy_combinations_graph_only(),
                         combinations.combine(batch_size=[4, 6])))
  def test_evaluate_with_dataset_with_partial_batch(self, distribution,
                                                    batch_size):
    with self.cached_session():
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae', keras.metrics.CategoricalAccuracy()]

      with distribution.scope():
        model_with_ds_strategy = get_model()
        model_with_ds_strategy.compile(optimizer, loss, metrics=metrics)

      cpu_model = get_model()
      cpu_model.compile(optimizer, loss, metrics=metrics)

      x = np.random.random((10, 3)).astype('float32')
      y = np.random.random((10, 4)).astype('float32')
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y))

      # As sample size is 10, we make the last batch a partial batch.
      cpu_model.set_weights(model_with_ds_strategy.get_weights())
      dataset_with_partial_batch = dataset.batch(batch_size)

      # We don't compare the loss as loss is currently not computed as metric
      # in Keras, the loss value is inaccurate for last partial batch due to
      # more weights for the last batch samples.
      steps = np.ceil(10.0 / batch_size)
      self.assertAllClose(
          model_with_ds_strategy.evaluate(
              dataset_with_partial_batch, steps=steps)[1:],
          cpu_model.evaluate(dataset_with_partial_batch, steps=steps)[1:],
          atol=1e-5,
          rtol=1e-5)
      self.assertAllClose(
          model_with_ds_strategy.evaluate(dataset_with_partial_batch)[1:],
          cpu_model.evaluate(dataset_with_partial_batch)[1:],
          atol=1e-5,
          rtol=1e-5)

  @combinations.generate(
      combinations.times(
          tpu_strategy_combinations_graph_only()))
  def test_predict_with_dataset_with_partial_batch(
      self, distribution):
    with self.cached_session():
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'

      with distribution.scope():
        model_with_ds_strategy = get_model()
        model_with_ds_strategy.compile(
            optimizer,
            loss)

      cpu_model = get_model()
      cpu_model.compile(optimizer, loss)

      inputs = np.random.random((10, 3)).astype(np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs))

      # As sample size is 10, we batch by 4 so that the last batch is
      # a partial batch.
      dataset_with_partial_batch = dataset.batch(4)
      cpu_model.set_weights(model_with_ds_strategy.get_weights())

      self.assertAllClose(
          model_with_ds_strategy.predict(dataset_with_partial_batch, steps=3),
          cpu_model.predict(dataset_with_partial_batch, steps=3),
          atol=1e-5,
          rtol=1e-5)

  @combinations.generate(
      combinations.times(
          tpu_strategy_combinations_graph_only()))
  def test_predict_multi_output_model_with_dataset_with_partial_batch(
      self, distribution):
    with self.cached_session():
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'

      with distribution.scope():
        model_with_ds_strategy = simple_multi_inputs_multi_outputs_model()
        model_with_ds_strategy.compile(
            optimizer,
            loss)

      cpu_model = simple_multi_inputs_multi_outputs_model()
      cpu_model.compile(optimizer, loss)

      input_data, _ = get_multi_inputs_multi_outputs_data()
      input_dict = {
          'input_a': input_data['input_a'],
          'input_b': input_data['input_b'],
      }

      dataset = dataset_ops.Dataset.from_tensor_slices(input_dict)

      # As sample size is 200, we batch by 18 using 12 steps per epoch so
      # that the last batch is a partial batch.
      dataset_with_partial_batch = dataset.batch(18)
      cpu_model.set_weights(model_with_ds_strategy.get_weights())

      self.assertAllClose(
          model_with_ds_strategy.predict(dataset_with_partial_batch, steps=12),
          cpu_model.predict(dataset_with_partial_batch, steps=12),
          atol=1e-4,
          rtol=1e-4)

  @combinations.generate(all_strategy_combinations_minus_default())
  def test_match_model_input_matches_with_dataset_tensors(self, distribution):

    def _create_model_input_output_tensors():
      input_a = keras.layers.Input(shape=(16,), name='z_input_sorted_last')
      input_b = keras.layers.Input(shape=(32,), name='a_input_sorted_first')
      intermediate_a = keras.layers.Dense(10)(input_a)
      intermediate_b = keras.layers.Dense(10)(input_b)
      merged = keras.layers.Add()([intermediate_a, intermediate_b])
      output = keras.layers.Dense(2)(merged)
      return input_a, input_b, output

    input_dict = {
        'z_input_sorted_last': np.random.rand(32, 16).astype(np.float32),
        'a_input_sorted_first': np.random.rand(32, 32).astype(np.float32)
    }
    target = np.ones((32, 2), dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((input_dict, target))
    dataset = dataset.batch(4, drop_remainder=True)

    with self.cached_session():
      with distribution.scope():
        input_a, input_b, output = _create_model_input_output_tensors()
        # `input_a`, which has input name that comes last in alphanumeric
        # order, is the first input of the model input layers. If tensors
        # from `input_dict` is blindly flattened and passed to model
        # inputs incorrectly, this would result in `input_a` input layer
        # matching with tensor `a_input_sorted_first` and would result in
        # shape mismatch.
        model_with_array_input = keras.models.Model(
            inputs=[input_a, input_b], outputs=output)
        model_with_array_input.compile('sgd', 'mse')
        model_weights = model_with_array_input.get_weights()
        model_with_array_input_fit = model_with_array_input.fit(
            dataset, steps_per_epoch=1, epochs=1).history

        input_a, input_b, output = _create_model_input_output_tensors()
        model_with_dict_input = keras.models.Model(
            inputs={
                'z_input_sorted_last': input_a,
                'a_input_sorted_first': input_b,
            },
            outputs=output)
        model_with_dict_input.compile('sgd', 'mse')
        model_with_dict_input.set_weights(model_weights)
        model_with_dict_input_fit = model_with_dict_input.fit(
            dataset, steps_per_epoch=1, epochs=1).history
        self.assertAllClose(
            model_with_dict_input_fit,
            model_with_array_input_fit,
            atol=1e-4,
            rtol=1e-4)

  @combinations.generate(
      combinations.combine(
          distribution=strategies_minus_tpu,
          mode=['graph', 'eager']))
  def test_dataset_with_sample_weights(self, distribution):
    with self.cached_session(), distribution.scope():
      model = get_sample_weights_model()
      optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      model.compile(
          optimizer,
          loss)

      inputs = np.array([[0], [1], [2], [3]], np.float32)
      targets = np.array([[2], [4], [6], [8]], np.float32)
      sample_weights = np.array([0.25, 0.5, 0.75, 1], np.float32)
      ds = dataset_ops.Dataset.from_tensor_slices(
          (inputs, targets, sample_weights)).batch(2)
      result = model.evaluate(ds, verbose=1)
      # The per sample loss is multipled by the corresponding sample weight. The
      # average of these weighted losses is the return value of the `evaluate`
      # call. For example, in the test above the average weighted loss is
      # calculated in the following manner:
      # batch_1 = (((2-0)^2) * 0.25 + ((4-1)^2) * 0.5) / 2 = 5.5 / 2 = 2.75
      # batch_2 = (((6-2)^2 * 0.75) + ((8-3)^2 * 1)) / 2 = 37 / 2 = 18.5
      # final result = (batch_1 + batch_2) / 2 = 10.625.
      # The first time we divide by number of input samples and the second time
      # we divide by number of steps/batches that the loss is aggregated over.
      self.assertAllClose(result, 10.625)

      # We now test without passing sample_weights:
      # batch_1 = ((2-0)^2) + ((4-1)^2) / 2 = 13 / 2 = 6.5
      # batch_2 = ((6-2)^2) + ((8-3)^2) / 2 = 41 / 2 = 20.5
      # final result = (batch_1 + batch_2) / 2 =  27 / 2 = 13.5
      ds = dataset_ops.Dataset.from_tensor_slices((inputs, targets)).batch(2)
      result = model.evaluate(ds, verbose=1)
      self.assertAllClose(result, 13.5)


class TestRegularizerLoss(test.TestCase, parameterized.TestCase):

  class IdentityRegularizer(keras.regularizers.Regularizer):

    def __call__(self, x):
      return array_ops.identity(x)

  class AddLayer(keras.layers.Layer):

    def build(self, _):
      self.v = self.add_weight(
          'v', (),
          initializer='ones',
          regularizer=TestRegularizerLoss.IdentityRegularizer())

    def call(self, inputs):
      return inputs + self.v

  @staticmethod
  def loss_fn(_, y_pred):
    return math_ops.reduce_mean(y_pred)

  @combinations.generate(
      combinations.times(
          strategy_combinations.all_strategy_combinations_minus_default()))
  def test_regularizer_loss(self, distribution):
    batch_size = 2
    if not distributed_training_utils.global_batch_size_supported(distribution):
      batch_size //= distribution.num_replicas_in_sync

      # Given an input x, which is always 1, and variable v, this model computes
      # Loss=x+v+regularizer_loss, where regularizer_loss=v and the variable is
      # initialized to 1. Therefore, this model computes Loss=1+2v, and so the
      # gradient dLoss/dv = 2. This gradient of 2 is averaged over all examples
      # in a batch and then multiplied by the learning rate of 1. As a result,
      # the model update for one batch should subtract 2 from v, resulting in v
      # being -1. If the regularizer loss is not scaled correctly by number of
      # replicas, the variable value will be incorrect when number of replicas
      # >1. For e.g. it will be -2 if num replicas = 2.
    with distribution.scope():
      x = keras.layers.Input(shape=(1,), batch_size=batch_size)
      y = TestRegularizerLoss.AddLayer()(x)
      model = keras.models.Model(inputs=x, outputs=y)
      opt = gradient_descent_keras.SGD(1.)
      model.compile(
          opt,
          loss=TestRegularizerLoss.loss_fn)
      model.fit(
          x=np.array([[1.], [1.]], dtype=np.float32),
          y=np.array([[1.], [1.]], dtype=np.float32),
          batch_size=batch_size)
      v = model.get_weights()[0]
      self.assertEqual(-1.0, v)


class TestDistributionStrategyWithKerasModels(test.TestCase,
                                              parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_distribution_strategy_on_sequential_model(
      self, distribution):
    with distribution.scope():
      optimizer_fn = gradient_descent_keras.SGD
      optimizer = optimizer_fn(learning_rate=0.001)
      model = simple_sequential_model()
      loss = 'mse'
      model.compile(
          optimizer,
          loss)

      inputs = np.zeros((20, 10), np.float32)
      targets = np.zeros((20, 2), np.float32)

    model.fit(inputs, targets, epochs=1, batch_size=10)
    model.predict(inputs, batch_size=10)
    model.evaluate(inputs, targets, batch_size=10)

  @combinations.generate(all_strategy_combinations())
  def test_distribution_strategy_on_functional_model(
      self, distribution):
    with distribution.scope():
      optimizer_fn = gradient_descent_keras.SGD
      optimizer = optimizer_fn(learning_rate=0.001)
      model = get_model()
      loss = 'mse'
      model.compile(
          optimizer,
          loss)

      inputs = np.zeros((64, 3), dtype=np.float32)
      targets = np.zeros((64, 4), dtype=np.float32)

    model.fit(inputs, targets, epochs=1)
    model.predict(inputs)
    model.evaluate(inputs, targets)

  @combinations.generate(
      combinations.combine(distribution=all_strategies, mode=['eager']))
  def test_distributed_dataset(self, distribution):
    with distribution.scope():

      class CBCounter(keras.callbacks.Callback):

        def __init__(self):
          self.epochs = 0
          self.train_batches = 0
          self.test_batches = 0

        def on_epoch_end(self, batch, logs=None):
          self.epochs += 1

        def on_train_batch_end(self, batch, logs=None):
          self.train_batches += 1

        def on_test_batch_end(self, batch, logs=None):
          self.test_batches += 1

      model = keras.Sequential([keras.layers.Dense(1)])
      model.compile('sgd', 'mse')
      cb_counter = CBCounter()

      x, y = np.ones((100, 10)), np.ones((100, 1))
      ds = dataset_ops.DatasetV2.from_tensor_slices((x, y))
      ds = ds.batch(10).repeat(2)
      ds = distribution.experimental_distribute_dataset(ds)

      val_ds = dataset_ops.DatasetV2.from_tensor_slices((x, y))
      val_ds = val_ds.batch(20)
      val_ds = distribution.experimental_distribute_dataset(val_ds)

      model.fit(
          ds,
          steps_per_epoch=10,
          validation_data=val_ds,
          validation_steps=5,
          epochs=2,
          callbacks=[cb_counter])

      self.assertEqual(cb_counter.train_batches, 20)
      self.assertEqual(cb_counter.test_batches, 10)
      self.assertEqual(cb_counter.epochs, 2)

      # Check for `steps_per_epoch`.
      if distribution.num_replicas_in_sync > 1:
        with self.assertRaisesRegexp(ValueError,
                                     'distributed dataset, you must specify'):
          model.fit(ds, epochs=2)

  @combinations.generate(
      combinations.combine(distribution=all_strategies, mode=['eager']))
  def test_distributed_datasets_from_function(self, distribution):
    with distribution.scope():

      class CBCounter(keras.callbacks.Callback):

        def __init__(self):
          self.epochs = 0
          self.train_batches = 0
          self.test_batches = 0

        def on_epoch_end(self, batch, logs=None):
          self.epochs += 1

        def on_train_batch_end(self, batch, logs=None):
          self.train_batches += 1

        def on_test_batch_end(self, batch, logs=None):
          self.test_batches += 1

      model = keras.Sequential([keras.layers.Dense(1)])
      model.compile('sgd', 'mse')
      cb_counter = CBCounter()

      def make_dataset(_):
        x, y = np.ones((100, 10)), np.ones((100, 1))
        ds = dataset_ops.DatasetV2.from_tensor_slices((x, y))
        ds = ds.batch(5).repeat()
        return ds

      ds = distribution.experimental_distribute_datasets_from_function(
          make_dataset)
      val_ds = distribution.experimental_distribute_datasets_from_function(
          make_dataset)

      model.fit(
          ds,
          steps_per_epoch=10,
          validation_data=val_ds,
          validation_steps=5,
          epochs=2,
          callbacks=[cb_counter])

      self.assertEqual(cb_counter.train_batches, 20)
      self.assertEqual(cb_counter.test_batches, 10)
      self.assertEqual(cb_counter.epochs, 2)

      # Check for `steps_per_epoch`.
      if distribution.num_replicas_in_sync > 1:
        with self.assertRaisesRegexp(ValueError,
                                     'distributed dataset, you must specify'):
          model.fit(ds, epochs=2)

  @combinations.generate(
      combinations.combine(distribution=all_strategies, mode=['eager']))
  def test_host_training_loop(self, distribution):
    with distribution.scope():
      inputs = keras.Input((10, 10, 3))
      x = keras.layers.Conv2D(3, kernel_size=3)(inputs)
      x = keras.layers.Flatten()(x)
      outputs = keras.layers.Dense(1)(x)
      model = keras.Model(inputs, outputs)

    model.compile('sgd', 'mse', experimental_steps_per_execution=10)

    bc = BatchCountingCB()
    x, y = np.ones((100, 10, 10, 3)), np.ones((100, 1))
    model.fit(x, y, batch_size=2, epochs=1, callbacks=[bc])
    self.assertEqual(bc.train_begin_batches, [0, 10, 20, 30, 40])
    self.assertEqual(bc.train_end_batches, [9, 19, 29, 39, 49])

    model.evaluate(x, y, batch_size=2, callbacks=[bc])
    self.assertEqual(bc.test_begin_batches, [0, 10, 20, 30, 40])
    self.assertEqual(bc.test_end_batches, [9, 19, 29, 39, 49])

    model.predict(x, batch_size=2, callbacks=[bc])
    self.assertEqual(bc.predict_begin_batches, [0, 10, 20, 30, 40])
    self.assertEqual(bc.predict_end_batches, [9, 19, 29, 39, 49])

  @combinations.generate(
      combinations.combine(distribution=all_strategies, mode=['eager']))
  def test_host_training_loop_last_partial_execution(self, distribution):
    with distribution.scope():
      inputs = keras.Input(10)
      outputs = keras.layers.Dense(1)(inputs)
      model = keras.Model(inputs, outputs)

    model.compile('sgd', 'mse', experimental_steps_per_execution=20)

    bc = BatchCountingCB()
    x, y = np.ones((100, 10)), np.ones((100, 1))
    model.fit(x, y, batch_size=2, epochs=1, callbacks=[bc])
    self.assertEqual(bc.train_begin_batches, [0, 20, 40])
    self.assertEqual(bc.train_end_batches, [19, 39, 49])

    model.evaluate(x, y, batch_size=2, callbacks=[bc])
    self.assertEqual(bc.test_begin_batches, [0, 20, 40])
    self.assertEqual(bc.test_end_batches, [19, 39, 49])

    model.predict(x, batch_size=2, callbacks=[bc])
    self.assertEqual(bc.predict_begin_batches, [0, 20, 40])
    self.assertEqual(bc.predict_end_batches, [19, 39, 49])

  @combinations.generate(
      combinations.combine(distribution=all_strategies, mode=['eager']))
  def test_host_training_loop_dataset_unknown_size(self, distribution):
    with distribution.scope():
      inputs = keras.Input(10)
      outputs = keras.layers.Dense(1)(inputs)
      model = keras.Model(inputs, outputs)

    model.compile('sgd', 'mse', experimental_steps_per_execution=20)

    x, y = np.ones((100, 10)), np.ones((100, 1))
    ds = dataset_ops.DatasetV2.from_tensor_slices((x, y)).batch(2)
    ds = ds.filter(lambda *args, **kwargs: True)  # Makes the size UNKNOWN.
    bc = BatchCountingCB()

    with self.assertRaisesRegexp(ValueError, 'steps_per_execution'):
      model.fit(ds, epochs=2, callbacks=[bc])

    train_ds = ds.repeat(2)
    model.fit(train_ds, steps_per_epoch=50, epochs=2, callbacks=[bc])
    self.assertEqual(bc.train_begin_batches, [0, 20, 40, 0, 20, 40])
    self.assertEqual(bc.train_end_batches, [19, 39, 49, 19, 39, 49])

    with self.assertRaisesRegexp(ValueError, 'steps_per_execution'):
      model.evaluate(ds, callbacks=[bc])

    test_ds = ds.repeat(2)
    model.evaluate(test_ds, steps=50, callbacks=[bc])
    self.assertEqual(bc.test_begin_batches, [0, 20, 40])
    self.assertEqual(bc.test_end_batches, [19, 39, 49])

    predict_ds = ds.repeat(2)
    model.predict(predict_ds, steps=50, callbacks=[bc])
    self.assertEqual(bc.predict_begin_batches, [0, 20, 40])
    self.assertEqual(bc.predict_end_batches, [19, 39, 49])

  @combinations.generate(
      combinations.combine(distribution=all_strategies, mode=['eager']))
  def test_host_training_loop_truncate_to_epoch(self, distribution):
    with distribution.scope():
      inputs = keras.Input(10)
      outputs = keras.layers.Dense(1)(inputs)
      model = keras.Model(inputs, outputs)

    model.compile('sgd', 'mse', experimental_steps_per_execution=500)

    x, y = np.ones((100, 10)), np.ones((100, 1))
    bc = BatchCountingCB()
    model.fit(x, y, batch_size=2, epochs=2, callbacks=[bc])
    self.assertEqual(bc.train_begin_batches, [0, 0])
    self.assertEqual(bc.train_end_batches, [49, 49])

    x, y = np.ones((50, 10)), np.ones((50, 1))
    model.evaluate(x, y, batch_size=2, callbacks=[bc])
    self.assertEqual(bc.test_begin_batches, [0])
    self.assertEqual(bc.test_end_batches, [24])

    x = np.ones((50, 10))
    model.predict(x, batch_size=2, callbacks=[bc])
    self.assertEqual(bc.predict_begin_batches, [0])
    self.assertEqual(bc.predict_end_batches, [24])

  @combinations.generate(
      combinations.times(
          all_strategy_combinations_minus_default()))
  def test_distribution_strategy_one_dimensional(self, distribution):
    with distribution.scope():
      inp = keras.layers.Input(shape=(10,))
      out = keras.layers.Dense(3, activation='softmax')(inp)
      model = keras.Model(inputs=[inp], outputs=[out])
      model.compile(
          optimizer='rmsprop',
          loss='sparse_categorical_crossentropy',
          metrics=['sparse_categorical_accuracy'])

      x = np.random.random((64, 10)).astype('float32')
      y = np.random.randint(3, size=64)

      model.fit(x, y, epochs=1, steps_per_epoch=2)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_two_gpus
          ],
          mode=['graph', 'eager'],
          reduction=[
              loss_reduction.ReductionV2.AUTO,
              loss_reduction.ReductionV2.SUM_OVER_BATCH_SIZE,
              loss_reduction.ReductionV2.SUM
          ]))
  def test_distribution_strategy_with_loss_reduction_types(
      self, distribution, reduction):
    np.random.seed(_RANDOM_SEED)

    def _get_model():
      inputs = keras.Input((10,))
      x1 = keras.layers.Dense(10, kernel_initializer='zeros')(inputs)
      x2 = keras.layers.Dense(10, kernel_initializer='zeros')(x1)
      outputs = keras.layers.Dense(1, kernel_initializer='zeros')(x2)
      model = keras.Model(inputs, outputs)
      return model

    x = np.random.random((64, 10))
    y = np.random.random((64, 1))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(32)

    model = _get_model()
    model.compile(
        'sgd', loss=keras.losses.MeanSquaredError(reduction=reduction))
    history = model.fit(dataset, steps_per_epoch=2, epochs=1, shuffle=False)

    with distribution.scope():
      ds_model = _get_model()
      ds_model.compile(
          'sgd',
          loss=keras.losses.MeanSquaredError(reduction=reduction))
      ds_history = ds_model.fit(
          dataset, steps_per_epoch=2, epochs=1, shuffle=False)
    self.assertArrayNear(history.history['loss'], ds_history.history['loss'],
                         1e-5)

  @combinations.generate(
      combinations.times(
          all_strategy_combinations_minus_default()))
  def test_distribution_strategy_with_symbolic_add_loss(
      self, mode, distribution):

    def _make_model_with_add_loss():
      inputs = keras.Input((10,))
      x1 = keras.layers.Dense(10, kernel_initializer='zeros')(inputs)
      x2 = keras.layers.Dense(10, kernel_initializer='zeros')(x1)
      outputs = keras.layers.Dense(1, kernel_initializer='zeros')(x2)
      model = keras.Model(inputs, outputs)
      model.add_loss(math_ops.reduce_mean(x1))
      model.add_loss(math_ops.reduce_mean(outputs))
      return model

    x = np.ones((64, 10)).astype('float32')

    model = _make_model_with_add_loss()
    model.compile('sgd')
    history = model.fit(x, epochs=1)

    with distribution.scope():
      ds_model = _make_model_with_add_loss()
      ds_model.compile(
          'sgd')
      ds_history = ds_model.fit(x, epochs=1)

    self.assertAllClose(history.history, ds_history.history)

  # TODO(omalleyt): Investigate flakiness and re-enable.
  @combinations.generate(all_strategy_minus_default_and_tpu_combinations())
  def DISABLED_test_distribution_strategy_with_callable_add_loss(
      self, distribution):

    def _make_model():
      inputs = keras.Input((10,))
      x1 = keras.layers.Dense(10, kernel_initializer='zeros')(inputs)
      x2 = keras.layers.Dense(10, kernel_initializer='zeros')(x1)
      d = keras.layers.Dense(1, kernel_initializer='zeros')
      outputs = d(x2)
      model = keras.Model(inputs, outputs)
      model.add_loss(lambda: 100. * math_ops.reduce_mean(d.kernel))
      return model

    x = np.ones((64, 10)).astype('float32')
    y = np.ones((64, 1)).astype('float32')

    model = _make_model()
    self.assertLen(model.losses, 1)

    model.compile('sgd', 'mse')
    history = model.fit(x, y, steps_per_epoch=2, epochs=1)

    with distribution.scope():
      ds_model = _make_model()
      self.assertLen(ds_model.losses, 1)
      ds_model.compile('sgd', 'mse')
      ds_history = ds_model.fit(x, y, steps_per_epoch=2, epochs=1)

    self.assertAllClose(history.history, ds_history.history)

  @combinations.generate(
      combinations.times(
          all_strategy_minus_default_and_tpu_combinations()))
  def test_distribution_strategy_with_add_metric_in_call(
      self, distribution):

    class Bias(keras.layers.Layer):

      def build(self, input_shape):
        self.bias = self.add_weight(name='bias', initializer='zeros', shape=())

      def call(self, inputs):
        self.add_metric(
            math_ops.reduce_mean(inputs), name='bias', aggregation='mean')
        return inputs + self.bias

    def _make_model_with_add_metric():
      inputs = keras.Input((10,))
      x1 = keras.layers.Dense(10, kernel_initializer='zeros')(inputs)
      x2 = Bias()(x1)
      outputs = keras.layers.Dense(1, kernel_initializer='zeros')(x2)
      model = keras.Model(inputs, outputs)
      return model

    x = np.ones((64, 10)).astype('float32')
    y = np.ones((64, 1)).astype('float32')

    model = _make_model_with_add_metric()
    self.assertLen(model.metrics, 1)

    model.compile('sgd', 'mse')
    history = model.fit(
        x, y, validation_data=(x, y), validation_steps=2, epochs=2)

    with distribution.scope():
      ds_model = _make_model_with_add_metric()
      self.assertLen(ds_model.metrics, 1)
      ds_model.compile(
          'sgd',
          'mse')
      ds_history = ds_model.fit(
          x, y, validation_data=(x, y), validation_steps=2, epochs=2)
      # includes stateful loss metric in eager.
      metrics_len = 2 if context.executing_eagerly() else 1
      self.assertLen(ds_model.metrics, metrics_len)

    self.assertAllClose(history.history, ds_history.history)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.one_device_strategy_gpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_two_gpus,
          ],
          mode=['eager']))
  def test_distribution_strategy_with_add_metric_object(
      self, distribution):

    class Bias(keras.layers.Layer):

      def build(self, input_shape):
        self.bias = self.add_weight(name='bias', initializer='zeros', shape=())
        self.mean = keras.metrics.Mean(name='mean')

      def call(self, inputs):
        self.add_metric(self.mean(inputs))
        return inputs + self.bias

    def _make_model_with_add_metric_object():
      inputs = keras.Input((10,))
      x1 = keras.layers.Dense(10, kernel_initializer='zeros')(inputs)
      x2 = Bias()(x1)
      outputs = keras.layers.Dense(1, kernel_initializer='zeros')(x2)
      model = keras.Model(inputs, outputs)
      return model

    x = np.ones((64, 10)).astype('float32')
    y = np.ones((64, 1)).astype('float32')

    model = _make_model_with_add_metric_object()
    self.assertLen(model.metrics, 1)

    model.compile('sgd', 'mse')
    history = model.fit(
        x, y, validation_data=(x, y), validation_steps=2, epochs=2)

    with distribution.scope():
      ds_model = _make_model_with_add_metric_object()
      self.assertLen(ds_model.metrics, 1)
      ds_model.compile(
          'sgd',
          'mse')
      ds_history = ds_model.fit(
          x, y, validation_data=(x, y), validation_steps=2, epochs=2)
      # includes stateful loss metric in eager.
      metrics_len = 2 if context.executing_eagerly() else 1
      self.assertLen(ds_model.metrics, metrics_len)

    self.assertAllClose(history.history, ds_history.history)

  @combinations.generate(
      # TODO(phillypham): Why does validation_steps > 1 not work on TPUs?
      combinations.times(
          all_strategy_minus_default_and_tpu_combinations()))
  def test_distribution_strategy_with_add_metric_outside_call(
      self, distribution):

    def _make_model_with_add_metric():
      inputs = keras.Input((10,))
      x1 = keras.layers.Dense(10, kernel_initializer='zeros')(inputs)
      outputs = keras.layers.Dense(1, kernel_initializer='zeros')(x1)
      model = keras.Model(inputs, outputs)
      model.add_metric(
          math_ops.reduce_mean(x1), name='mid_mean', aggregation='mean')
      return model

    x = np.ones((64, 10)).astype('float32')
    y = np.ones((64, 1)).astype('float32')

    model = _make_model_with_add_metric()
    self.assertLen(model.metrics, 1)

    model.compile('sgd', 'mse')
    history = model.fit(
        x, y, validation_data=(x, y), validation_steps=2, epochs=2)

    with distribution.scope():
      ds_model = _make_model_with_add_metric()
      self.assertLen(ds_model.metrics, 1)
      ds_model.compile(
          'sgd',
          'mse')
      ds_history = ds_model.fit(
          x, y, validation_data=(x, y), validation_steps=2, epochs=2)
      # includes stateful loss metric in eager.
      metrics_len = 2 if context.executing_eagerly() else 1
      self.assertLen(ds_model.metrics, metrics_len)

    self.assertAllClose(history.history, ds_history.history)

  @combinations.generate(
      combinations.combine(
          distribution=strategies_minus_tpu,
          mode=['eager']))
  def test_sparse_tensor_outputs(self, distribution):

    class ToSparse(keras.layers.Layer):
      """Create a sparse tensor based on a given dense tensor."""

      def call(self, inputs):
        indices = array_ops.where_v2(math_ops.not_equal(inputs, 0))
        values = array_ops.gather_nd(inputs, indices)
        shape = array_ops.shape(inputs, out_type='int64')
        return sparse_tensor.SparseTensor(indices, values, dense_shape=shape)

    model = keras.Sequential([ToSparse()])

    # Define some input data with additional padding.
    input_data = np.array([[1, 0, 0], [2, 3, 0]])
    output = model.predict(input_data, batch_size=2)

    expected_indices = np.array([[0, 0], [1, 0], [1, 1]])
    expected_values = np.array([1, 2, 3])
    expected_dense_shape = np.array([2, 3])

    self.assertAllEqual(output.indices, expected_indices)
    self.assertAllEqual(output.values, expected_values)
    self.assertAllEqual(output.dense_shape, expected_dense_shape)

  @combinations.generate(
      combinations.combine(
          distribution=strategies_minus_tpu,
          mode=['eager']))
  def test_ragged_tensor_outputs(self, distribution):

    class ToRagged(keras.layers.Layer):
      """Create a ragged tensor based on a given dense tensor."""

      def __init__(self, padding, ragged_rank=1, **kwargs):
        super(ToRagged, self).__init__(**kwargs)
        self._padding = padding
        self._ragged_rank = ragged_rank

      def call(self, inputs):
        return ragged_tensor.RaggedTensor.from_tensor(
            inputs, padding=self._padding, ragged_rank=self._ragged_rank)

    model = keras.Sequential([ToRagged(padding=0)])

    # Define some input data with additional padding.
    input_data = np.array([[1, 0, 0], [2, 3, 0]])
    output = model.predict(input_data, batch_size=2)

    expected_values = [[1], [2, 3]]
    self.assertAllEqual(expected_values, output)

  @combinations.generate(
      combinations.combine(
          distribution=strategies_minus_default_minus_tpu + tpu_strategies,
          mode=['eager']))
  def test_correctness_of_add_loss_with_merge_call(self, distribution):
    batch_size = 32

    def _get_model():
      inputs = keras.layers.Input(shape=(1,))
      labels = keras.layers.Input(shape=(1,))
      x = keras.layers.Dense(10, activation='relu')(inputs)
      y = keras.layers.Dense(1)(x)
      model = keras.models.Model([inputs, labels], y)
      model.add_loss(keras.losses.mean_squared_error(labels, y))
      return model

    def _get_data():
      x_train = np.random.rand(64, 1)
      y_train = 3 * x_train
      x_train = x_train.astype('float32')
      y_train = y_train.astype('float32')
      dataset = dataset_ops.DatasetV2.from_tensor_slices((x_train, y_train))
      dataset = dataset.batch(batch_size)
      return dataset

    with distribution.scope():
      model = _get_model()
      optimizer = gradient_descent_keras.SGD(0.2)

      @def_function.function
      def train_step(dist_inputs):

        def step_fn(inputs):
          with backprop.GradientTape() as tape:
            logits = model(inputs)

            # Invoke a merge_call()
            distribution_strategy_context.get_replica_context().merge_call(
                lambda d: None)

            # Verify that there is only one loss on the model.
            assert len(model.losses) == 1
            loss_from_model = math_ops.reduce_sum(
                model.losses) * 1.0 / batch_size

            # Compute loss in this loop.
            loss = keras.losses.mean_squared_error(inputs[1], logits)
            loss = nn.compute_average_loss(loss, global_batch_size=batch_size)

            # Verify that the loss computed in this loop is equivalent to the
            # loss from the model that was added via add_loss.
            check_ops.assert_equal(loss, loss_from_model)

          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(grads, model.trainable_variables))
          return loss

        per_replica_losses = distribution.run(step_fn, args=(dist_inputs,))
        return distribution.reduce(
            reduce_util.ReduceOp.SUM, per_replica_losses, axis=None)

      dataset = distribution.experimental_distribute_dataset(_get_data())
      for _ in range(2):
        for x in dataset:
          train_step(x)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_unimplemented_parameter_server_strategy(self):
    cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=2)
    cluster_resolver = SimpleClusterResolver(
        cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
        task_type='worker',
        task_id=1,
        num_accelerators={'GPU': 0})
    distribution = parameter_server_strategy.ParameterServerStrategy(
        cluster_resolver)

    self.assertIsInstance(distribution,
                          (parameter_server_strategy.ParameterServerStrategyV1,
                           parameter_server_strategy.ParameterServerStrategy))

    with self.assertRaisesRegexp(NotImplementedError,
                                 'ParameterServerStrategy*'):
      with distribution.scope():
        model = simple_sequential_model()
        optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
        loss = 'mse'
        model.compile(optimizer, loss)


# Models to exercise inserting ancillary layers with add_loss and add_metric.
def _functional_with_add_loss_and_metric(input_shape, num_classes, l1, l2):
  inputs = keras.Input(input_shape, name='images')
  x = keras.layers.Conv2D(32, kernel_size=5, activation='relu')(inputs)
  x = keras.layers.MaxPooling2D(pool_size=2)(x)
  x = keras.layers.Conv2D(64, kernel_size=5, activation='relu')(x)
  x = keras.layers.MaxPooling2D(pool_size=2)(x)
  # Apply L2 regularization to embedding. Use a mix of TensorFlow ops and layers
  # to exercise all code paths.
  x = keras.layers.Flatten(name='embedding')(x)
  l2_loss = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(x), -1))
  # Apply L1 regularization to next layer.
  x = keras.layers.Dense(1024, activation='relu', name='sparse_embedding')(x)
  l1_loss = keras.layers.Lambda(
      lambda x: math_ops.reduce_mean(math_ops.reduce_sum(x, -1)),
      name='l1_loss')(
          x)
  outputs = keras.layers.Dense(num_classes, name='logits')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  # Weight regularization terms.
  model.add_loss(keras.layers.Lambda(lambda x: x * l2)(l2_loss))
  model.add_metric(l2_loss, aggregation='mean', name='l2_loss')
  model.add_loss(l1_loss * l1)
  model.add_metric(l1_loss, aggregation='mean', name='l1_loss')
  return model


def _sequential_with_add_loss_and_metric(input_shape, num_classes, l1, l2):
  model = keras.Sequential([
      keras.layers.Conv2D(
          32, kernel_size=5, activation='relu', input_shape=input_shape),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Conv2D(64, kernel_size=5, activation='relu'),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Flatten(name='embedding'),
      keras.layers.Dense(1024, activation='relu', name='sparse_embedding'),
      keras.layers.Dense(num_classes, name='logits'),
  ])
  # Extract layer outputs, add regularization terms, and rescale the metric.
  # Use a mix of TensorFlow ops and layers to exercise all code paths.
  x = model.get_layer('sparse_embedding').get_output_at(-1)
  l1_loss = l1 * math_ops.reduce_mean(math_ops.reduce_sum(x, -1))
  model.add_loss(l1_loss)
  model.add_metric(
      keras.layers.Lambda(lambda x: math_ops.divide(x, l1))(l1_loss),
      aggregation='mean',
      name='l1_loss')
  x = model.get_layer('embedding').get_output_at(-1)
  l2_loss = keras.layers.Lambda(
      lambda x: l2 * math_ops.reduce_mean(math_ops.reduce_sum(x * x, -1)),
      name='l2_loss')(
          x)
  model.add_loss(l2_loss)
  model.add_metric(l2_loss / l2, aggregation='mean', name='l2_loss')
  return model


def _functional_with_layer_reuse(input_shape, num_classes, l1, l2):
  base_model = keras.Sequential([
      keras.layers.Conv2D(
          32, kernel_size=5, activation='relu', input_shape=input_shape),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Conv2D(64, kernel_size=5, activation='relu'),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Flatten(),
      keras.layers.Dense(1024, activation='relu'),
      keras.layers.Dense(num_classes, name='logits'),
  ])
  inputs = keras.Input(input_shape, name='images')
  logits = base_model(inputs)
  model = keras.Model(inputs=inputs, outputs=logits)
  # Reuse sequential layer and create new nodes.
  zero_logits = base_model(array_ops.zeros_like(inputs))
  one_logits = base_model(array_ops.ones_like(inputs))
  # L2 loss.
  l2_loss = math_ops.reduce_mean(
      math_ops.reduce_sum(math_ops.square(logits - zero_logits), -1))
  model.add_loss(l2_loss * l2)
  model.add_metric(l2_loss, aggregation='mean', name='l2_loss')
  # L1 loss.
  l1_loss = math_ops.reduce_mean(
      math_ops.reduce_sum(math_ops.abs(logits - one_logits), -1))
  model.add_loss(l1_loss * l1)
  model.add_metric(l1_loss, aggregation='mean', name='l1_loss')
  return model


class TestDistributionStrategyWithMultipleAddLossAndMetricCalls(
    test.TestCase, parameterized.TestCase):
  """Tests complex models with multiple add loss and metric calls."""

  @combinations.generate(
      combinations.times(
          all_strategy_combinations_minus_default(),
          combinations.combine(
              model_fn=[
                  _functional_with_add_loss_and_metric,
                  _sequential_with_add_loss_and_metric,
                  _functional_with_layer_reuse,
              ],
              l1=[0.01],
              l2=[0.1])))
  def test_fit_and_evaluate(self, distribution, model_fn, l1, l2):
    # Make fake MNIST-like image data.
    np.random.seed(_RANDOM_SEED)
    dataset = dataset_ops.DatasetV2.from_tensor_slices(
        (np.random.uniform(size=(64, 28, 28, 1)).astype(np.float32),
         np.random.randint(0, 10, size=(64,))))
    dataset = dataset.shuffle(64).batch(
        8 * distribution.num_replicas_in_sync, drop_remainder=True)
    # Make model with distribution strategy and initialize with dataset shape.
    input_shape = dataset_ops.get_structure(dataset)[0].shape[1:]
    with distribution.scope():
      model = model_fn(input_shape, 10, l1, l2)
      model.compile(
          optimizer=keras.optimizers.adam_v2.Adam(1e-4),
          loss=keras.losses.SparseCategoricalCrossentropy(
              from_logits=True,
              reduction=loss_reduction.ReductionV2.SUM_OVER_BATCH_SIZE),
          metrics=[
              keras.metrics.SparseCategoricalAccuracy(),
              keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
          ])
    # Non-eager training doesn't support steps_per_epoch=None.
    for unused_epoch in range(2):
      model.fit(dataset)
    results = dict(zip(model.metrics_names, model.evaluate(dataset)))
    # Sanity checks.
    self.assertBetween(results['sparse_categorical_accuracy'], 0.02, 1.)
    self.assertGreater(results['l2_loss'], 0.)
    self.assertGreater(results['l1_loss'], 0.)
    # Assert correctness of the loss calculation and updating of metrics.
    self.assertNear(
        results['l1_loss'] * l1 + results['l2_loss'] * l2 +
        results['sparse_categorical_crossentropy'], results['loss'], 1e-6)


class DeterministicModel(keras.Model):
  """Deterministic Model that always outputs the same initial result.

  It verifies the `call` method is run inside the same distribution
  strategy that the model was initially passed.
  """

  def __init__(self, strategy):
    super(DeterministicModel, self).__init__()
    self.x = None
    self.strategy = strategy

  def build(self, input_shape):
    self.x = variables.Variable(array_ops.ones(shape=()))

  def call(self, inputs, training=None, mask=None):
    active_strategy = distribution_strategy_context.get_strategy()
    if active_strategy is not self.strategy:
      raise ValueError('Model must execute call w/ the original strategy')
    return self.x * inputs


class TestModelCapturesStrategy(test.TestCase, parameterized.TestCase):
  """Tests that model creation captures the strategy."""

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=['eager']))
  def test_fit_and_evaluate(self, distribution):
    dataset = dataset_ops.DatasetV2.from_tensor_slices(
        (array_ops.ones(shape=(64,)), array_ops.ones(shape=(64,))))
    dataset = dataset.batch(8 * distribution.num_replicas_in_sync)
    # Make model with distribution strategy
    with distribution.scope():
      model = DeterministicModel(distribution)
      optimizer = keras.optimizers.adam_v2.Adam(1e-4)

    # Compile & evaluate the model outside of the distribution strategy scope
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=['binary_accuracy'])

    # Call `optimizer.iterations` out of strategy scope.
    self.assertEqual(model.optimizer.iterations.numpy(), 0)

    # Non-eager training doesn't support steps_per_epoch=None.
    for unused_epoch in range(2):
      model.fit(dataset)

    results = model.evaluate(dataset)
    results = dict(zip(model.metrics_names, results))

    # Check that the metrics have a result we expect
    self.assertEqual(results['binary_accuracy'], 1.0)
    self.assertAllClose(results['loss'], 0.0)

    # Assert that all metric/optimizer/model variables were made in the
    # distribution strategy (Test that compile uses the captured
    # distribution strategy)
    metric_vars = nest.flatten(
        [metric.variables for metric in model.metrics])
    for var in metric_vars:
      self.assertTrue(distribution.extended.variable_created_in_scope(var))
    for var in model.optimizer._weights:
      self.assertTrue(distribution.extended.variable_created_in_scope(var))
    for var in model.variables:
      self.assertTrue(distribution.extended.variable_created_in_scope(var))

    # Make sure the metric must be created in the same scope as the model:
    # This shouldn't raise any validation errors
    with distribution.scope():
      metric = keras.metrics.BinaryAccuracy()
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[metric])

    # This should raise an error because the metric is constructed
    # outside of the scope, and not by compile
    if distribution_strategy_context.has_strategy():
      with self.assertRaisesRegexp(
          ValueError, 'All metrics must be created in'):
        model.compile(
            optimizer=keras.optimizers.adam_v2.Adam(1e-4),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.BinaryAccuracy()])

if __name__ == '__main__':
  base_layer_utils.enable_v2_dtype_behavior()
  test.main()
