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
"""Tests for tf.keras models using DistributionStrategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import tpu_strategy
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import test
from tensorflow.python.framework import random_seed
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import distributed_training_utils
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras
from tensorflow.python.ops.parsing_ops import gen_parsing_ops
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import rmsprop

_RANDOM_SEED = 1337
_TRAIN_SIZE = 200
_INPUT_SIZE = (10,)
_NUM_CLASS = 2


# TODO(anjalisridhar): Add a decorator that will allow us to run these tests as
# part of the tf.keras unit tests suite.
def simple_sequential_model():
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(16, activation='relu', input_shape=_INPUT_SIZE))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(_NUM_CLASS, activation='softmax'))
  return model


def simple_functional_model():
  a = keras.layers.Input(shape=_INPUT_SIZE)
  b = keras.layers.Dense(16, activation='relu')(a)
  b = keras.layers.Dropout(0.1)(b)
  b = keras.layers.Dense(_NUM_CLASS, activation='softmax')(b)
  model = keras.models.Model(inputs=[a], outputs=[b])
  return model


def multi_inputs_multi_outputs_model():
  input_a = keras.layers.Input(shape=(16,), name='input_a')
  input_b = keras.layers.Input(shape=(16,), name='input_b')
  input_m = keras.layers.Input(shape=(8,), dtype='string', name='input_m')
  dense = keras.layers.Dense(8, name='dense_1')

  interm_a = dense(input_a)
  # Read m
  interm_m = keras.layers.Lambda(gen_parsing_ops.string_to_number)(input_m)
  interm_s = keras.layers.Lambda(lambda k: k[0] * k[1])([interm_m, interm_a])
  interm_b = dense(input_b)
  merged = keras.layers.concatenate([interm_s, interm_b], name='merge')
  output_c = keras.layers.Dense(3, activation='softmax', name='dense_2')(merged)
  output_d = keras.layers.Dense(2, activation='softmax', name='dense_3')(merged)
  model = keras.models.Model(
      inputs=[input_a, input_b, input_m], outputs=[output_c, output_d])
  model.compile(
      loss='categorical_crossentropy',
      optimizer=gradient_descent.GradientDescentOptimizer(0.001),
      metrics={
          'dense_2': 'categorical_accuracy',
          'dense_3': 'categorical_accuracy'
      })
  return model


def get_ds_train_input_fn():
  np.random.seed(_RANDOM_SEED)
  (x_train, y_train), _ = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=_INPUT_SIZE,
      num_classes=_NUM_CLASS)
  y_train = keras.utils.to_categorical(y_train)

  dataset = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
  dataset = dataset.batch(32)
  return dataset


def get_ds_test_input_fn():
  np.random.seed(_RANDOM_SEED)
  _, (x_test, y_test) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=_INPUT_SIZE,
      num_classes=_NUM_CLASS)
  y_test = keras.utils.to_categorical(y_test)

  dataset = dataset_ops.Dataset.from_tensor_slices((x_test, y_test))
  dataset = dataset.batch(32)
  return dataset


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

  c_train = keras.utils.to_categorical(c_train)
  c_test = keras.utils.to_categorical(c_test)
  d_train = keras.utils.to_categorical(d_train)
  d_test = keras.utils.to_categorical(d_test)

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
  if isinstance(distribution, tpu_strategy.TPUStrategy):
    return dataset.batch(batch_size, drop_remainder=True)
  else:
    return dataset.batch(batch_size)


def get_model():
  x = keras.layers.Input(shape=(3,), name='input')
  y = keras.layers.Dense(4, name='dense')(x)
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


def get_correctness_test_inputs(use_numpy, use_validation_data,
                                with_distribution,
                                x_train, y_train, x_predict):
  """Generates the inputs for correctness check when enable Keras with DS."""
  training_epochs = 2
  global_batch_size = 64
  batch_size = global_batch_size
  # TODO(b/118776054): Use global batch size for Keras/DS support.
  use_per_core_batch_size = (
      with_distribution and
      not distributed_training_utils.global_batch_size_supported(
          with_distribution))
  if use_per_core_batch_size:
    batch_size //= with_distribution.num_replicas_in_sync

  if use_numpy:
    training_inputs = {
        'batch_size': batch_size,
        'x': x_train,
        'y': y_train,
        'epochs': training_epochs,
        'shuffle': False,
    }

    if use_validation_data:
      eval_inputs = None
      training_inputs['validation_data'] = (x_train, y_train)
    else:
      eval_inputs = {
          'batch_size': batch_size,
          'x': x_train,
          'y': y_train,
      }
    predict_inputs = {
        'x': np.array(x_predict, dtype=np.float32),
    }
  else:
    # For dataset inputs, we do not pass batch_size to
    # keras.fit/evaluate/predict. The batch size is part of the dataset.
    train_dataset = dataset_ops.Dataset.from_tensor_slices(
        (x_train, y_train))
    x = batch_wrapper(
        train_dataset, batch_size, with_distribution, repeat=training_epochs)

    training_inputs = {
        'batch_size': None,
        'x': x,
        'y': None,
        'epochs': training_epochs,
        'shuffle': False,
        'steps_per_epoch': len(x_train) // global_batch_size,
    }
    if use_validation_data:
      eval_inputs = None  # Remove the eval_inputs
      eval_dataset = dataset_ops.Dataset.from_tensor_slices(
          (x_train, y_train))
      x = batch_wrapper(eval_dataset, batch_size, with_distribution)
      training_inputs['validation_data'] = x
      training_inputs['validation_steps'] = 5
    else:
      eval_inputs = {
          'batch_size': None,
          'x': x,
          'y': None,
          'steps': 20,
      }

    predict_batch_size = len(x_predict)
    if use_per_core_batch_size:
      predict_batch_size //= with_distribution.num_replicas_in_sync
    predict_dataset = dataset_ops.Dataset.from_tensor_slices(x_predict)
    predict_dataset = batch_wrapper(predict_dataset,
                                    predict_batch_size, with_distribution)
    predict_inputs = {
        'steps': 1,
        'x': predict_dataset,
    }

  return training_inputs, eval_inputs, predict_inputs


strategies_minus_tpu = [
    combinations.default_strategy,
    combinations.one_device_strategy,
    combinations.mirrored_strategy_with_gpu_and_cpu,
    combinations.mirrored_strategy_with_two_gpus,
    combinations.core_mirrored_strategy_with_gpu_and_cpu,
    combinations.core_mirrored_strategy_with_two_gpus]

tpu_strategies = [
    combinations.tpu_strategy,  # steps_per_run=2
    combinations.tpu_strategy_one_step]


def strategy_minus_tpu_combinations():
  return combinations.combine(
      distribution=strategies_minus_tpu,
      mode=['graph', 'eager'])


def tpu_strategy_combinations():
  return combinations.combine(
      distribution=tpu_strategies,
      mode=['graph'])


def all_strategy_combinations():
  return strategy_minus_tpu_combinations() + tpu_strategy_combinations()


def strategy_and_optimizer_combinations():
  # TODO(b/122372746): Uncomment optimizers after they pass tests.
  return combinations.times(
      all_strategy_combinations(),
      combinations.combine(optimizer=[
          combinations.adagrad_optimizer_v1_fn,
          # combinations.adagrad_optimizer_keras_v2_fn,
          combinations.adam_optimizer_v1_fn,
          combinations.adam_optimizer_keras_v2_fn,
          combinations.gradient_descent_optimizer_v1_fn,
          combinations.gradient_descent_optimizer_keras_v2_fn,
          combinations.rmsprop_optimizer_v1_fn,
          # combinations.rmsprop_optimizer_keras_v2_fn
      ]))


def strategy_and_input_combinations():
  return (
      combinations.times(
          combinations.combine(distribution=strategies_minus_tpu),
          combinations.combine(mode=['graph'],
                               use_numpy=[True, False],
                               use_validation_data=[True, False])
          + combinations.combine(mode=['eager'],
                                 use_numpy=[False],
                                 use_validation_data=[False])) +
      combinations.times(
          combinations.combine(distribution=tpu_strategies),
          combinations.combine(mode=['graph'],
                               use_numpy=[True, False],
                               use_validation_data=[True, False])))


def strategy_for_numpy_input_combinations():
  return combinations.combine(
      distribution=strategies_minus_tpu + tpu_strategies,
      mode=['graph'])


class TestDistributionStrategyWithNumpyArrays(test.TestCase,
                                              parameterized.TestCase):

  @combinations.generate(strategy_for_numpy_input_combinations())
  def test_calling_model_with_numpy_arrays(self, distribution):
    with self.cached_session():
      model = get_model()

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics, distribute=distribution)

      inputs = np.zeros((64, 3), dtype=np.float32)
      targets = np.zeros((64, 4), dtype=np.float32)

      # Call fit with validation data
      model.fit(inputs, targets, epochs=1, batch_size=2, verbose=0,
                validation_data=(inputs, targets))

      # TODO(anjalisridhar): We need tests for when the batch size and steps are
      # smaller and results in a 0 batch_size and steps value.
      model.evaluate(inputs, targets)
      # with steps
      model.evaluate(inputs, targets, steps=2)
      # with batch_size
      model.evaluate(inputs, targets, batch_size=8)

      model.predict(inputs)
      # with steps
      model.predict(inputs, steps=2)
      # with batch_size
      model.predict(inputs, batch_size=8)

  @combinations.generate(strategy_for_numpy_input_combinations())
  def test_calling_model_with_nested_numpy_arrays(self, distribution):
    with self.cached_session():
      model = multi_input_output_model()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.001)
      loss = 'mse'
      model.compile(optimizer, loss, distribute=distribution)

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
      # with steps
      model.evaluate(inputs, targets, steps=2)
      # with batch_size
      model.evaluate(inputs, targets, batch_size=8)

      model.predict(inputs)
      # with steps
      model.predict(inputs, steps=2)
      # with batch_size
      model.predict(inputs, batch_size=8)

  @combinations.generate(combinations.combine(
      distribution=strategies_minus_tpu, mode=['graph']))
  def test_numpy_with_sample_weights(self, distribution):
    model = get_model()
    optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
    loss = 'mse'
    model.compile(optimizer, loss, distribute=distribution)

    inputs = np.zeros((20, 3), np.float32)
    targets = np.zeros((20, 4), np.float32)
    sample_weights = np.ones((20), np.float32)

    model.fit(inputs, targets, sample_weight=sample_weights, epochs=1,
              steps_per_epoch=2, verbose=1)

  @combinations.generate(strategy_for_numpy_input_combinations())
  def test_flatten_predict_outputs(self, distribution):
    with self.cached_session():
      model = multi_input_output_model()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.001)
      loss = 'mse'
      model.compile(optimizer, loss, distribute=distribution)

      # We take 6 input samples with each input having a dimension of 3 or 5.
      input_a_np = np.asarray(np.random.random((6, 3)), dtype=np.float32)
      input_b_np = np.asarray(np.random.random((6, 5)), dtype=np.float32)
      inputs = [input_a_np, input_b_np]

      outs = model.predict(inputs, steps=1)
      # `predict` a list that is equal in length to the number of model outputs.
      # In this test our model has two outputs and each element of `outs`
      # corresponds to all the samples of one of the model outputs.
      self.assertLen(outs, 2)
      # Each of the output samples have a dimension of 7. We should process all
      # the available input samples(6).
      self.assertAllEqual([6, 7], outs[0].shape)
      self.assertAllEqual([6, 7], outs[1].shape)


class TestDistributionStrategyWithDatasets(test.TestCase,
                                           parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_calling_model_on_same_dataset(self, distribution):
    with self.cached_session():
      model = get_model()

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae', keras.metrics.CategoricalAccuracy()]
      model.compile(optimizer, loss, metrics=metrics, distribute=distribution)

      dataset = get_dataset(distribution)

      # Call fit with validation data
      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                validation_data=dataset, validation_steps=2)
      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                validation_data=dataset, validation_steps=2)
      model.predict(get_predict_dataset(distribution), steps=2)

  @combinations.generate(all_strategy_combinations())
  def test_model_interleaved_eval_same_as_direct_eval(self, distribution):
    with self.cached_session():
      user_controlled_model = get_model()
      user_controlled_model.compile(
          gradient_descent.GradientDescentOptimizer(0.001),
          loss='mse',
          metrics=['mae', keras.metrics.CategoricalAccuracy()],
          distribute=distribution)

      interleaved_model = get_model()
      interleaved_model.set_weights(user_controlled_model.get_weights())
      interleaved_model.compile(
          gradient_descent.GradientDescentOptimizer(0.001),
          loss='mse',
          metrics=['mae', keras.metrics.CategoricalAccuracy()],
          distribute=distribution)

      dataset = get_dataset(distribution)

      # Call fit with validation interleaved
      interleaved_output = interleaved_model.fit(
          dataset, epochs=2, steps_per_epoch=2, verbose=1,
          validation_data=dataset, validation_steps=2, shuffle=False)

      # Manually control the validation running after each epoch.
      user_controlled_output = []
      for _ in range(2):
        user_controlled_model.fit(
            dataset, epochs=1, steps_per_epoch=2, verbose=1, shuffle=False)
        user_controlled_output.append(
            user_controlled_model.evaluate(dataset, steps=2))

      self.assertEqual(interleaved_output.history['val_loss'],
                       [x[0] for x in user_controlled_output])
      self.assertEqual(interleaved_output.history['val_mean_absolute_error'],
                       [x[1] for x in user_controlled_output])
      self.assertEqual(interleaved_output.history['val_categorical_accuracy'],
                       [x[2] for x in user_controlled_output])

  # TODO(priyag): Enable this test for TPU. Currently tuples/dict don't work
  # as clone_model's input_tensors argument only seems to accept list and not
  # tuples or dict.

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph', 'eager']))
  def test_fit_with_tuple_and_dict_dataset_inputs(self, distribution):
    with self.cached_session():
      model = multi_input_output_model()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae', keras.metrics.CategoricalAccuracy()]
      model.compile(optimizer, loss, metrics=metrics, distribute=distribution)

      input_a_np = np.random.random((10, 3))
      input_b_np = np.random.random((10, 5))
      output_d_np = np.random.random((10, 7))
      output_e_np = np.random.random((10, 7))

      # Test with tuples
      dataset_tuple = dataset_ops.Dataset.from_tensor_slices((
          (input_a_np, input_b_np), (output_d_np, output_e_np)))
      dataset_tuple = dataset_tuple.repeat(100)
      dataset_tuple = dataset_tuple.batch(10)

      model.fit(dataset_tuple, epochs=1, steps_per_epoch=2, verbose=1)

      # Test with dict
      dataset_dict = dataset_ops.Dataset.from_tensor_slices((
          {'input_a': input_a_np, 'input_b': input_b_np},
          (output_d_np, output_e_np)))
      dataset_dict = dataset_dict.repeat(100)
      dataset_dict = dataset_dict.batch(10)

      model.fit(dataset_dict, epochs=1, steps_per_epoch=2, verbose=1)

  @combinations.generate(all_strategy_combinations())
  def test_fit_eval_and_predict_methods_on_dataset(self, distribution):
    with self.cached_session():
      model = get_model()

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae', keras.metrics.CategoricalAccuracy()]
      model.compile(optimizer, loss, metrics=metrics, distribute=distribution)

      dataset = get_dataset(distribution)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)
      model.evaluate(dataset, steps=2, verbose=1)
      model.predict(get_predict_dataset(distribution), steps=2)

  @combinations.generate(strategy_and_optimizer_combinations())
  def test_fit_eval_and_predict_with_optimizer(self, distribution, optimizer):
    with self.cached_session():
      model = get_model()

      loss = 'mse'
      model.compile(optimizer(), loss, distribute=distribution)

      dataset = get_dataset(distribution)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)
      model.evaluate(dataset, steps=2, verbose=1)
      model.predict(get_predict_dataset(distribution), steps=2)

  @combinations.generate(strategy_minus_tpu_combinations())
  def test_dataset_with_sample_weights(self, distribution):
    model = get_model()
    optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
    loss = 'mse'
    model.compile(optimizer, loss, distribute=distribution)

    inputs = np.zeros((10, 3), np.float32)
    targets = np.zeros((10, 4), np.float32)
    sample_weights = np.ones((10), np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets,
                                                      sample_weights))
    dataset = dataset.repeat()
    dataset = dataset.batch(10)

    model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)
    model.evaluate(dataset, steps=2, verbose=1)
    model.predict(dataset, steps=2)

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph', 'eager']))
  # TODO(b/120943676, b/120957836): Re-enable once the validation code is
  # restored.
  def DISABLED_test_dataset_wrong_input_shape(self, distribution):
    with self.cached_session():
      model = get_model()

      optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      model.compile(optimizer, loss, distribute=distribution)

      # Wrong input shape
      inputs = np.zeros((10, 5), dtype=np.float32)
      targets = np.zeros((10, 4), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      with self.assertRaisesRegexp(ValueError,
                                   'expected input to have shape'):
        model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0)

  @combinations.generate(combinations.combine(
      distribution=[combinations.mirrored_strategy_with_gpu_and_cpu],
      mode=['graph', 'eager']))
  # TODO(b/120943676, b/120957836): Re-enable once the validation code is
  # restored.
  def DISABLED_test_dataset_no_batch_input_validation(self, distribution):
    with self.cached_session():
      model = get_model()

      optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      model.compile(optimizer, loss, distribute=distribution)

      # User forgets to batch the dataset
      inputs = np.zeros((10, 3), dtype=np.float32)
      targets = np.zeros((10, 4), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)

      with self.assertRaisesRegexp(ValueError, 'expected input to have shape'):
        model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0)

  @combinations.generate(combinations.combine(
      distribution=[combinations.tpu_strategy_one_step],
      mode=['graph']))
  def test_dataset_input_shape_fully_defined(self, distribution):
    with self.cached_session():
      model = get_model()

      optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      model.compile(optimizer, loss, distribute=distribution)

      dataset = get_dataset(distribution)
      # Input shapes are not fully known. Batch dimension is unknown as we are
      # not using the drop_remainder argument.
      dataset = dataset.repeat(100).batch(10)

      with self.assertRaisesRegexp(ValueError, 'requires fully defined shapes'):
        model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0)

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.mirrored_strategy_with_two_gpus,
          combinations.core_mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_two_gpus],
      mode=['graph', 'eager']))
  def test_learning_phase_value(self, distribution):
    # TODO(anjalisridhar): Modify this test to use Lambdas since we can compare
    # meaningful values. Currently we don't pass the learning phase if the
    # Lambda layer uses the learning phase.
    with self.cached_session():
      x = keras.layers.Input(shape=(1,), name='input')
      y = keras.layers.Dense(1, kernel_initializer='ones')(x)
      z = keras.layers.Dropout(0.9999)(y)
      model = keras.Model(x, z)
      initial_weights = model.get_weights()

      optimizer = gradient_descent.GradientDescentOptimizer(0.005)
      loss = 'mse'
      metrics = ['acc']
      model.compile(optimizer, loss, metrics=metrics, distribute=distribution)

      batch_size = 8
      if isinstance(distribution, mirrored_strategy.CoreMirroredStrategy):
        # CoreMirroredStrategy uses global batch size.
        batch_size = 8 * distribution.num_replicas_in_sync

      inputs = np.ones((10, 1), dtype=np.float32)
      targets = np.ones((10, 1), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat().batch(batch_size)
      hist = model.fit(dataset, epochs=1, steps_per_epoch=20, verbose=1)
      self.assertAlmostEqual(hist.history['acc'][0], 0, 0)

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

  @combinations.generate(strategy_minus_tpu_combinations())
  def testOptimizerWithCallbacks(self, distribution):
    with self.cached_session():
      model = get_model()

      optimizer = gradient_descent_keras.SGD(0.01)
      loss = 'mse'
      model.compile(optimizer, loss, distribute=distribution)

      dataset = get_dataset(distribution)

      def schedule(_):
        return 0.001

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                callbacks=[keras.callbacks.LearningRateScheduler(schedule)])
      grouped_models = distribution.unwrap(model._distributed_model)
      with distribution.scope():
        for m in grouped_models:
          self.assertAllClose(0.001, keras.backend.get_value(
              m.optimizer.lr), atol=1e-05, rtol=1e-05)


class TestDistributionStrategyErrorCases(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph', 'eager']))
  def test_unsupported_features(self, distribution):
    with self.cached_session():
      model = get_model()

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics, distribute=distribution)

      dataset = get_dataset(distribution)

      # Test with validation split
      with self.assertRaisesRegexp(
          ValueError, '`validation_split` argument is not '
                      'supported when input `x` is a dataset or a '
                      'dataset iterator.+'):
        model.fit(dataset,
                  epochs=1, steps_per_epoch=2, verbose=0,
                  validation_split=0.5, validation_steps=2)

      # Test with sample weight.
      sample_weight = np.random.random((10,))
      with self.assertRaisesRegexp(
          ValueError, '`sample_weight` argument is not supported when input '
                      '`x` is a dataset or a dataset iterator.'):
        model.fit(
            dataset,
            epochs=1,
            steps_per_epoch=2,
            verbose=0,
            sample_weight=sample_weight)

      # Test with not specifying the `steps` argument.
      with self.assertRaisesRegexp(
          ValueError, 'the `steps_per_epoch` argument'):
        model.fit(dataset, epochs=1, verbose=0)
      with self.assertRaisesRegexp(ValueError,
                                   'the `steps` argument'):
        model.evaluate(dataset, verbose=0)

      with self.assertRaisesRegexp(ValueError,
                                   'the `steps` argument'):
        model.predict(dataset, verbose=0)

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph', 'eager']))
  def test_calling_with_unsupported_predefined_callbacks(self, distribution):
    with self.cached_session():
      model = get_model()

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics, distribute=distribution)

      dataset = get_dataset(distribution)

      def schedule(_):
        return 0.001
      with self.assertRaisesRegexp(ValueError,
                                   'You must specify a Keras Optimizer V2 when '
                                   'using'):
        model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                  callbacks=[keras.callbacks.LearningRateScheduler(schedule)])

      with self.assertRaisesRegexp(ValueError,
                                   'You must specify a Keras Optimizer V2 when '
                                   'using'):
        model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                  callbacks=[keras.callbacks.ReduceLROnPlateau()])


class TestDistributionStrategyWithLossMasking(test.TestCase,
                                              parameterized.TestCase):

  # TODO(priyag): Enable all strategies for this test. Currently it does not
  # work for TPU due to some invalid datatype.
  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph', 'eager']))
  def test_masking(self, distribution):
    with self.cached_session():
      np.random.seed(1337)
      x = np.array([[[1], [1]], [[0], [0]]])
      model = keras.models.Sequential()
      model.add(keras.layers.Masking(mask_value=0, input_shape=(2, 1)))
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(1, kernel_initializer='one')))
      model.compile(loss='mse',
                    optimizer=gradient_descent.GradientDescentOptimizer(0.01),
                    distribute=distribution)
      y = np.array([[[1], [1]], [[1], [1]]])
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)
      hist = model.fit(x=dataset, epochs=1, steps_per_epoch=2)
      self.assertEqual(hist.history['loss'][0], 0)


class TestDistributionStrategyWithNormalizationLayer(
    test.TestCase, parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_batchnorm_correctness(self, distribution):
    with self.cached_session():
      model = keras.models.Sequential()
      norm = keras.layers.BatchNormalization(input_shape=(10,), momentum=0.8)
      model.add(norm)
      model.compile(loss='mse',
                    optimizer=gradient_descent.GradientDescentOptimizer(0.01),
                    distribute=distribution)

      # centered on 5.0, variance 10.0
      x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
      x = x.astype('float32')
      dataset = dataset_ops.Dataset.from_tensor_slices((x, x))
      dataset = dataset.repeat(100)
      dataset = batch_wrapper(dataset, 32, distribution)

      predict_dataset = dataset_ops.Dataset.from_tensor_slices(x)
      predict_dataset = predict_dataset.repeat(100)
      predict_dataset = batch_wrapper(predict_dataset, 32, distribution)

      model.fit(dataset, epochs=4, verbose=0, steps_per_epoch=10)
      out = model.predict(predict_dataset, steps=2)
      out -= keras.backend.eval(norm.beta)
      out /= keras.backend.eval(norm.gamma)
      np.testing.assert_allclose(out.mean(), 0.0, atol=1e-1)
      np.testing.assert_allclose(out.std(), 1.0, atol=1e-1)


class TestDistributionStrategyCorrectness(test.TestCase,
                                          parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_metric_correctness(self, distribution):
    with self.cached_session():
      keras.backend.set_image_data_format('channels_last')
      num_samples = 10000

      x_train = np.random.randint(0, 2, num_samples)
      x_train = np.reshape(x_train, (num_samples, 1))
      y_train = x_train
      x_train = x_train.astype('float32')
      y_train = y_train.astype('float32')

      # Create identity model.
      model = keras.Sequential()
      model.add(
          keras.layers.Dense(1, input_shape=(1,), kernel_initializer='ones'))
      model.compile(
          loss=keras.losses.mean_squared_error,
          optimizer=gradient_descent.GradientDescentOptimizer(0.5),
          metrics=[keras.metrics.BinaryAccuracy()],
          distribute=distribution)

      batch_size = 64
      if not distributed_training_utils.global_batch_size_supported(
          distribution):
        batch_size //= distribution.num_replicas_in_sync
      train_dataset = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
      train_dataset = batch_wrapper(train_dataset, batch_size, distribution)

      history = model.fit(x=train_dataset, epochs=2, steps_per_epoch=10)
      self.assertEqual(history.history['binary_accuracy'], [1.0, 1.0])

  @combinations.generate(all_strategy_combinations())
  def test_eval_metrics_correctness(self, distribution):
    with self.cached_session():
      model = keras.Sequential()
      model.add(
          keras.layers.Dense(
              3, activation='relu', input_dim=4, kernel_initializer='ones'))
      model.add(
          keras.layers.Dense(
              1, activation='sigmoid', kernel_initializer='ones'))
      model.compile(
          loss='mae',
          metrics=['accuracy', keras.metrics.BinaryAccuracy()],
          optimizer=gradient_descent.GradientDescentOptimizer(0.001),
          distribute=distribution)

      # verify correctness of stateful and stateless metrics.
      x = np.ones((100, 4)).astype('float32')
      y = np.ones((100, 1)).astype('float32')
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).repeat()
      dataset = batch_wrapper(dataset, 4, distribution)
      outs = model.evaluate(dataset, steps=10)
      self.assertEqual(outs[1], 1.)
      self.assertEqual(outs[2], 1.)

      y = np.zeros((100, 1)).astype('float32')
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).repeat()
      dataset = batch_wrapper(dataset, 4, distribution)
      outs = model.evaluate(dataset, steps=10)
      self.assertEqual(outs[1], 0.)
      self.assertEqual(outs[2], 0.)

  @combinations.generate(strategy_and_input_combinations())
  def test_correctness(self, distribution, use_numpy, use_validation_data):
    with self.cached_session():
      default_tolerance = 1e-5
      tol_table = {}

      if isinstance(distribution, (
          mirrored_strategy.MirroredStrategy,
          mirrored_strategy.CoreMirroredStrategy,
          distribute_lib._DefaultDistributionStrategy)):  # pylint: disable=protected-access
        # TODO(b/119257215): Weights are not exactly the same, so use larger
        # tolerance for now. Predict should be related to weights.
        tol_table = {
            'weights_1': 1e-4,
            'weights_2': 1e-4,
            'predict_result_1': 1e-4,
        }

      keras.backend.set_image_data_format('channels_last')
      np.random.seed(_RANDOM_SEED)
      random_seed.set_random_seed(_RANDOM_SEED)

      # Train, eval, and predict datasets are created with the same input numpy
      # arrays.
      # TODO(xiejw): Change this back to 10000, once we support final partial
      # batch.
      num_samples = 9984
      x_train = np.random.rand(num_samples, 1)
      y_train = 3 * x_train
      x_train = x_train.astype('float32')
      y_train = y_train.astype('float32')
      x_predict = [[1.], [2.], [3.], [4.]]

      # The model is built once and the initial weights are saved.
      # This is used to initialize the model for both the distribution and
      # non-distribution run. In addition, we add few non-linear layers to make
      # it non-trivial.
      def _create_model():
        model = keras.Sequential()
        model.add(keras.layers.Dense(10, activation='relu', input_shape=(1,)))
        model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(1))
        return model

      model = _create_model()
      initial_weights = model.get_weights()
      del model  # avoid accident usage.

      def fit_eval_and_predict(with_distribution=None):
        model = _create_model()
        # We have initialized the model to the same weight for the distribution
        # and non-distribution run.
        model.set_weights(initial_weights)
        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=gradient_descent_keras.SGD(0.5),
            metrics=['mse'],
            distribute=with_distribution)

        training_inputs, eval_inputs, predict_inputs = (
            get_correctness_test_inputs(use_numpy, use_validation_data,
                                        with_distribution,
                                        x_train, y_train, x_predict))

        result = {}
        result['training_history_1'] = model.fit(**training_inputs).history

        if eval_inputs is not None:
          result['eval_result_1'] = model.evaluate(**eval_inputs)

        result['weights_1'] = model.get_weights()
        result['predict_result_1'] = model.predict(**predict_inputs)

        # Train and eval again to mimic user's flow.

        result['training_history_2'] = model.fit(**training_inputs).history

        if eval_inputs is not None:
          result['eval_result_2'] = model.evaluate(**eval_inputs)

        result['weights_2'] = model.get_weights()

        return result

      results_with_ds = fit_eval_and_predict(with_distribution=distribution)
      results_without_ds = fit_eval_and_predict(with_distribution=None)

      # Verify that the weights, training history, eval results, predict outputs
      # are the same within some limits of tolerance.
      for key in results_with_ds:
        if (key.startswith('training_history') and
            isinstance(distribution, tpu_strategy.TPUStrategy) and
            distribution.extended.steps_per_run > 1):
          # TODO(b/119894254): Enable this test for all cases once the
          # underlying bug is fixed.
          continue

        tolerance = tol_table.get(key, default_tolerance)

        self.assertAllClose(
            results_with_ds[key],
            results_without_ds[key],
            atol=tolerance,
            rtol=tolerance,
            msg='Fail to assert {}.'.format(key))


if __name__ == '__main__':
  test.main()
