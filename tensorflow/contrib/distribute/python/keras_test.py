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

import collections
import os
import tempfile
from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import tpu_strategy
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import values
from tensorflow.python.eager import test
from tensorflow.python.estimator import keras as keras_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import distributed_training_utils
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras
from tensorflow.python.ops.parsing_ops import gen_parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import rmsprop

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


def simple_functional_model():
  a = keras.layers.Input(shape=_INPUT_SIZE)
  b = keras.layers.Dense(16, activation='relu')(a)
  b = keras.layers.Dropout(0.1)(b)
  b = keras.layers.Dense(_NUM_CLASS, activation='softmax')(b)
  model = keras.models.Model(inputs=[a], outputs=[b])
  return model


def simple_multi_inputs_multi_outputs_model():
  input_a = keras.layers.Input(shape=(16,), name='input_a')
  input_b = keras.layers.Input(shape=(16,), name='input_b')

  merged = keras.layers.concatenate([input_a, input_b], name='merge')
  output_c = keras.layers.Dense(3, activation='softmax', name='dense_2')(merged)
  output_d = keras.layers.Dense(2, activation='softmax', name='dense_3')(merged)
  model = keras.models.Model(
      inputs=[input_a, input_b], outputs=[output_c, output_d])
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


def all_strategy_combinations_minus_default():
  strategy_minus_default_combinations = combinations.combine(
      distribution=[
          combinations.one_device_strategy,
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.mirrored_strategy_with_two_gpus,
          combinations.core_mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_two_gpus],
      mode=['graph', 'eager'])
  return strategy_minus_default_combinations + tpu_strategy_combinations()


def strategy_and_optimizer_combinations():
  return combinations.times(
      all_strategy_combinations(),
      combinations.combine(optimizer=[
          combinations.adagrad_optimizer_v1_fn,
          combinations.adagrad_optimizer_keras_v2_fn,
          combinations.adam_optimizer_v1_fn,
          combinations.adam_optimizer_keras_v2_fn,
          combinations.gradient_descent_optimizer_v1_fn,
          combinations.gradient_descent_optimizer_keras_v2_fn,
          combinations.rmsprop_optimizer_v1_fn,
          combinations.rmsprop_optimizer_keras_v2_fn
      ]))


def strategy_for_numpy_input_combinations():
  return combinations.combine(
      distribution=strategies_minus_tpu + tpu_strategies,
      mode=['graph'])


class TestEstimatorDistributionStrategy(test_util.TensorFlowTestCase,
                                        parameterized.TestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(),
                                  'keras_mirrored_strategy_test')
    gfile.MakeDirs(self._base_dir)
    self._config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED, model_dir=self._base_dir)

  def tearDown(self):
    writer_cache.FileWriterCache.clear()
    if os.path.isdir(self._base_dir):
      gfile.DeleteRecursively(self._base_dir)

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.mirrored_strategy_with_two_gpus,
          combinations.core_mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_two_gpus],
      mode=['graph']))
  def test_train_functional_with_distribution_strategy(self, distribution):
    keras_model = simple_functional_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        metrics=[keras.metrics.CategoricalAccuracy()],
        optimizer=rmsprop.RMSPropOptimizer(learning_rate=0.01))
    config = run_config_lib.RunConfig(tf_random_seed=_RANDOM_SEED,
                                      model_dir=self._base_dir,
                                      train_distribute=distribution,
                                      eval_distribute=distribution)
    with self.cached_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, config=config)
      before_eval_results = est_keras.evaluate(
          input_fn=get_ds_test_input_fn, steps=1)
      est_keras.train(input_fn=get_ds_train_input_fn, steps=_TRAIN_SIZE / 16)
      after_eval_results = est_keras.evaluate(input_fn=get_ds_test_input_fn,
                                              steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

    writer_cache.FileWriterCache.clear()
    gfile.DeleteRecursively(self._config.model_dir)

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.mirrored_strategy_with_two_gpus,
          combinations.core_mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_two_gpus],
      mode=['graph']))
  def test_train_sequential_with_distribution_strategy(self, distribution):
    keras_model = simple_sequential_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        metrics=[keras.metrics.CategoricalAccuracy()],
        optimizer=rmsprop.RMSPropOptimizer(learning_rate=0.01))
    config = run_config_lib.RunConfig(tf_random_seed=_RANDOM_SEED,
                                      model_dir=self._base_dir,
                                      train_distribute=distribution)
    with self.cached_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, config=config)
      before_eval_results = est_keras.evaluate(
          input_fn=get_ds_test_input_fn, steps=1)
      est_keras.train(input_fn=get_ds_train_input_fn, steps=_TRAIN_SIZE / 16)
      after_eval_results = est_keras.evaluate(input_fn=get_ds_test_input_fn,
                                              steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

    writer_cache.FileWriterCache.clear()
    gfile.DeleteRecursively(self._config.model_dir)

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph']))
  def test_multi_inputs_multi_outputs_with_input_fn_as_dict(self, distribution):
    train_data, test_data = get_multi_inputs_multi_outputs_data()

    def train_input_fn():
      input_dict = {
          'input_a': train_data['input_a'],
          'input_b': train_data['input_b'],
          'input_m': train_data['input_m'].astype(np.str)
      }
      output_dict = {
          'dense_2': train_data['output_c'],
          'dense_3': train_data['output_d']
      }
      return dataset_ops.Dataset.from_tensor_slices((input_dict,
                                                     output_dict)).batch(16)

    def eval_input_fn():
      input_dict = {
          'input_a': test_data['input_a'],
          'input_b': test_data['input_b'],
          'input_m': test_data['input_m'].astype(np.str)
      }
      output_dict = {
          'dense_2': test_data['output_c'],
          'dense_3': test_data['output_d']
      }
      return dataset_ops.Dataset.from_tensor_slices((input_dict,
                                                     output_dict)).batch(16)

    self.do_test_multi_inputs_multi_outputs_with_input_fn(
        distribution, train_input_fn, eval_input_fn)

  def do_test_multi_inputs_multi_outputs_with_input_fn(
      self, distribution, train_input_fn, eval_input_fn):
    config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED,
        model_dir=self._base_dir,
        train_distribute=distribution)
    with self.cached_session():
      model = multi_inputs_multi_outputs_model()
      est_keras = keras_lib.model_to_estimator(keras_model=model, config=config)
      baseline_eval_results = est_keras.evaluate(
          input_fn=eval_input_fn, steps=1)
      est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
      eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
      self.assertLess(eval_results['loss'], baseline_eval_results['loss'])

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph']))
  def test_keras_optimizer_with_distribution_strategy(self, distribution):
    keras_model = simple_sequential_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.rmsprop(lr=0.01))

    config = run_config_lib.RunConfig(tf_random_seed=_RANDOM_SEED,
                                      model_dir=self._base_dir,
                                      train_distribute=distribution)
    with self.cached_session():
      est_keras = keras_lib.model_to_estimator(keras_model=keras_model,
                                               config=config)
      with self.assertRaisesRegexp(ValueError,
                                   'Only TensorFlow native optimizers are '
                                   'supported with DistributionStrategy.'):
        est_keras.train(input_fn=get_ds_train_input_fn, steps=_TRAIN_SIZE / 16)

    writer_cache.FileWriterCache.clear()
    gfile.DeleteRecursively(self._config.model_dir)


class TestDistributionStrategyWithNumpyArrays(test.TestCase,
                                              parameterized.TestCase):

  @combinations.generate(strategy_for_numpy_input_combinations())
  def test_calculating_input_params_no_steps_no_batch_size(self, distribution):
    # Calculate the per_replica_batch_size scaling factor for strategies
    # that use per_core_batch_size
    replica_scale_factor = 1.0
    if not distributed_training_utils.global_batch_size_supported(distribution):
      replica_scale_factor = distribution.num_replicas_in_sync

    with self.cached_session():
      # Input samples of different sizes
      input_20_samples = np.zeros((20, 3), dtype=np.float32)
      input_63_samples = np.zeros((63, 3), dtype=np.float32)
      input_64_samples = np.zeros((64, 3), dtype=np.float32)

      # Default global batch size 32 for input with 64 samples run in 2 steps
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, input_64_samples, steps=None, batch_size=None)
      self.assertEqual(batch_size, 32 // replica_scale_factor)
      self.assertEqual(steps, 2)

      # Computed global batch size 20 is lower than 32 if we pass less samples.
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, input_20_samples, steps=None, batch_size=None)
      self.assertEqual(batch_size, 20 // replica_scale_factor)
      self.assertEqual(steps, 1)

      #  Default global batch size 32 cannot be used with 63 samples.
      with self.assertRaisesRegexp(ValueError, 'not divisible by batch size'):
        distributed_training_utils.get_input_params(
            distribution, input_63_samples, steps=None, batch_size=None)

  @combinations.generate(strategy_for_numpy_input_combinations())
  def test_calculating_input_params_with_steps_no_batch_size(self,
                                                             distribution):
    # Calculate the per_replica_batch_size scaling factor for strategies
    # that use per_core_batch_size
    replica_scale_factor = 1.0
    if not distributed_training_utils.global_batch_size_supported(distribution):
      replica_scale_factor = distribution.num_replicas_in_sync

    with self.cached_session():
      # Input samples of different sizes
      input_63_samples = np.zeros((63, 3), dtype=np.float32)
      input_64_samples = np.zeros((64, 3), dtype=np.float32)

      # Computed global batch size is correct for number of specified 1 step
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, input_64_samples, steps=1, batch_size=None)
      self.assertEqual(batch_size, 64 // replica_scale_factor)
      self.assertEqual(steps, 1)

      # Computed global batch size is correct for number of specified 2 steps
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, input_64_samples, steps=2, batch_size=None)
      self.assertEqual(batch_size, 32 // replica_scale_factor)
      self.assertEqual(steps, 2)

      # All samples can not be consumed in specified number of steps
      with self.assertRaisesRegexp(ValueError, 'not divisible by steps'):
        distributed_training_utils.get_input_params(
            distribution, input_63_samples, steps=2, batch_size=None)

      # This cases is different for different strategies due to the
      # difference in supported batch size being global or per-replica.
      if replica_scale_factor == 1:
        # Computed global batch size is correct even if not sharadable
        steps, batch_size = distributed_training_utils.get_input_params(
            distribution, input_63_samples, steps=3, batch_size=None)
        self.assertEqual(batch_size, 21)
        self.assertEqual(steps, 3)
      else:
        # Computed global batch size can not be sharded across replicas
        with self.assertRaisesRegexp(ValueError, 'could not be sharded evenly '
                                     'across the sync replicas'):
          distributed_training_utils.get_input_params(
              distribution, input_63_samples, steps=1, batch_size=None)

  @combinations.generate(strategy_for_numpy_input_combinations())
  def test_calculating_input_params_no_steps_with_batch_size(self,
                                                             distribution):
    # Calculate the per_replica_batch_size scaling factor for strategies
    # that use per_core_batch_size
    replica_scale_factor = 1.0
    if not distributed_training_utils.global_batch_size_supported(distribution):
      replica_scale_factor = distribution.num_replicas_in_sync

    with self.cached_session():
      input_64_samples = np.zeros((64, 3), dtype=np.float32)

      # Computed steps is correct for specified batch size
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, input_64_samples, steps=None, batch_size=16)
      self.assertEqual(batch_size, 16)
      self.assertEqual(steps, 4 // replica_scale_factor)

      # Computed steps is correct for specified batch size
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, input_64_samples, steps=None, batch_size=32)
      self.assertEqual(batch_size, 32)
      self.assertEqual(steps, 2 // replica_scale_factor)

      # Number of samples is not divisible by the global batch size
      with self.assertRaisesRegexp(ValueError, 'not divisible by batch size'):
        distributed_training_utils.get_input_params(
            distribution, input_64_samples, steps=None, batch_size=20)

      # Number of samples is not divisible by the global batch size
      with self.assertRaisesRegexp(ValueError, 'not divisible by batch size'):
        distributed_training_utils.get_input_params(
            distribution, input_64_samples, steps=None, batch_size=3)

  @combinations.generate(strategy_for_numpy_input_combinations())
  def test_calculating_input_params_with_steps_with_batch_size(self,
                                                               distribution):
    with self.cached_session():
      input_64_samples = np.zeros((64, 3), dtype=np.float32)

      # No change to steps and batch size if both specified and feasible
      steps, batch_size = distributed_training_utils.get_input_params(
          distribution, input_64_samples, steps=5, batch_size=3)
      self.assertEqual(batch_size, 3)
      self.assertEqual(steps, 5)

      # Number of samples is less than global batch size * steps
      with self.assertRaisesRegexp(ValueError, 'less than samples required'):
        distributed_training_utils.get_input_params(
            distribution, input_64_samples, steps=10, batch_size=13)

  @combinations.generate(strategy_for_numpy_input_combinations())
  def test_calling_model_with_numpy_arrays(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = get_model()
        optimizer = gradient_descent.GradientDescentOptimizer(0.001)
        loss = 'mse'
        metrics = ['mae']
        model.compile(optimizer, loss, metrics=metrics)

        inputs = np.zeros((64, 3), dtype=np.float32)
        targets = np.zeros((64, 4), dtype=np.float32)

        # Call fit with validation data
        model.fit(inputs, targets, epochs=1, batch_size=2, verbose=0,
                  validation_data=(inputs, targets))

        # TODO(anjalisridhar): We need tests for when the batch size and steps
        # are smaller and results in a 0 batch_size and steps value.
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
      with distribution.scope():
        model = multi_input_output_model()
        optimizer = gradient_descent.GradientDescentOptimizer(
            learning_rate=0.001)
        loss = 'mse'
        model.compile(optimizer, loss)

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
    with self.cached_session():
      with distribution.scope():
        model = get_model()
        optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
        loss = 'mse'
        model.compile(optimizer, loss)

      inputs = np.zeros((20, 3), np.float32)
      targets = np.zeros((20, 4), np.float32)
      sample_weights = np.ones((20), np.float32)

      model.fit(inputs, targets, sample_weight=sample_weights, epochs=1,
                steps_per_epoch=2, verbose=1)

  @combinations.generate(strategy_for_numpy_input_combinations())
  def test_flatten_predict_outputs(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = multi_input_output_model()
        optimizer = gradient_descent.GradientDescentOptimizer(
            learning_rate=0.001)
        loss = 'mse'
        model.compile(optimizer, loss)

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

  @combinations.generate(tpu_strategy_combinations())
  def test_predict_with_partial_batch(self, distribution):
    with self.cached_session():
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'

      with distribution.scope():
        model_with_ds_strategy = get_model()
        model_with_ds_strategy.compile(optimizer, loss)

      cpu_model = get_model()
      cpu_model.compile(optimizer, loss)

      inputs = np.zeros((10, 3), dtype=np.float32)

      # As sample size is 10, we batch by 4 so that the last batch is
      # a partial batch. Also `fit()` using numpy array as inputs without
      # distribution strategy uses entire sample as a single batch. As so,
      # we remove parameters `batch_size` and `steps`.
      cpu_model.set_weights(model_with_ds_strategy.get_weights())
      self.assertAllClose(
          model_with_ds_strategy.predict(inputs, batch_size=4, steps=3),
          cpu_model.predict(inputs),
          atol=1e-5, rtol=1e-5)

  @combinations.generate(tpu_strategy_combinations())
  def test_predict_multi_output_model_with_partial_batch(
      self, distribution):
    with self.cached_session():
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'

      with distribution.scope():
        model_with_ds_strategy = simple_multi_inputs_multi_outputs_model()
        model_with_ds_strategy.compile(optimizer, loss)

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
          atol=1e-4, rtol=1e-4)


class TestDistributionStrategyWithDatasets(test.TestCase,
                                           parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_calling_model_on_same_dataset(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = get_model()
        optimizer = gradient_descent.GradientDescentOptimizer(0.001)
        loss = 'mse'
        metrics = ['mae', keras.metrics.CategoricalAccuracy()]
        model.compile(optimizer, loss, metrics=metrics)

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
      with distribution.scope():
        user_controlled_model = get_model()
        user_controlled_model.compile(
            gradient_descent.GradientDescentOptimizer(0.001),
            loss='mse',
            metrics=['mae', keras.metrics.CategoricalAccuracy()])

        interleaved_model = get_model()
        interleaved_model.set_weights(user_controlled_model.get_weights())
        interleaved_model.compile(
            gradient_descent.GradientDescentOptimizer(0.001),
            loss='mse',
            metrics=['mae', keras.metrics.CategoricalAccuracy()])

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
      with distribution.scope():
        model = multi_input_output_model()
        optimizer = gradient_descent.GradientDescentOptimizer(
            learning_rate=0.001)
        loss = 'mse'
        metrics = ['mae', keras.metrics.CategoricalAccuracy()]
        model.compile(optimizer, loss, metrics=metrics)

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

  # TODO(b/122743976): Include TPUStrategy for this test as well once
  # step inference is supported.
  @combinations.generate(strategy_minus_tpu_combinations())
  def test_fit_eval_and_predict_methods_on_dataset_without_steps(
      self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = get_model()
        optimizer = gradient_descent.GradientDescentOptimizer(0.001)
        loss = 'mse'
        metrics = ['mae', keras.metrics.CategoricalAccuracy()]
        model.compile(optimizer, loss, metrics=metrics)

      inputs = np.zeros((1000, 3), dtype=np.float32)
      targets = np.zeros((1000, 4), dtype=np.float32)
      # steps/steps_per_epoch are calculated when using numpy arrays as
      # input data.
      fit_with_numpy = model.fit(inputs, targets, epochs=1,
                                 batch_size=10).history
      eval_with_numpy = model.evaluate(inputs, targets, batch_size=10)
      predict_with_numpy = model.predict(inputs, batch_size=10)

      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.batch(10)
      fit_with_ds = model.fit(dataset, epochs=1).history
      eval_with_ds = model.evaluate(dataset)
      predict_dataset = dataset_ops.Dataset.from_tensor_slices(inputs)
      predict_dataset = predict_dataset.batch(10, drop_remainder=True)
      predict_with_ds = model.predict(predict_dataset)
      self.assertAllClose(
          fit_with_numpy, fit_with_ds, atol=1e-4, rtol=1e-4)
      self.assertAllClose(
          eval_with_numpy, eval_with_ds, atol=1e-4, rtol=1e-4)
      self.assertAllClose(
          predict_with_numpy, predict_with_ds, atol=1e-4, rtol=1e-4)

  @combinations.generate(all_strategy_combinations())
  def test_fit_eval_and_predict_methods_on_dataset(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = get_model()
        optimizer = gradient_descent.GradientDescentOptimizer(0.001)
        loss = 'mse'
        metrics = ['mae', keras.metrics.CategoricalAccuracy()]
        model.compile(optimizer, loss, metrics=metrics)

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
        model.compile(optimizer(), loss)

      dataset = get_dataset(distribution)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)
      model.evaluate(dataset, steps=2, verbose=1)
      model.predict(get_predict_dataset(distribution), steps=2)

  @combinations.generate(strategy_minus_tpu_combinations())
  def test_dataset_with_sample_weights(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = get_model()
        optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
        loss = 'mse'
        model.compile(optimizer, loss)

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
      with distribution.scope():
        model = get_model()
        optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
        loss = 'mse'
        model.compile(optimizer, loss)

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
      with distribution.scope():
        model = get_model()
        optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
        loss = 'mse'
        model.compile(optimizer, loss)

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
      with distribution.scope():
        model = get_model()
        optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
        loss = 'mse'
        model.compile(optimizer, loss)

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
      with distribution.scope():
        x = keras.layers.Input(shape=(1,), name='input')
        y = keras.layers.Dense(1, kernel_initializer='ones')(x)
        z = keras.layers.Dropout(0.9999)(y)
        model = keras.Model(x, z)
        initial_weights = model.get_weights()

        optimizer = gradient_descent.GradientDescentOptimizer(0.005)
        loss = 'mse'
        metrics = ['acc']
        model.compile(optimizer, loss, metrics=metrics)

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
        model.compile(optimizer, loss)

      dataset = get_dataset(distribution)

      def schedule(_):
        return 0.001

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                callbacks=[keras.callbacks.LearningRateScheduler(schedule)])
      self.assertAllClose(0.001, keras.backend.get_value(model.optimizer.lr))

  @combinations.generate(tpu_strategy_combinations())
  def test_predict_with_dataset_with_partial_batch(self, distribution):
    with self.cached_session():
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'

      with distribution.scope():
        model_with_ds_strategy = get_model()
        model_with_ds_strategy.compile(optimizer, loss)

      cpu_model = get_model()
      cpu_model.compile(optimizer, loss)

      inputs = np.zeros((10, 3), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs))

      # As sample size is 10, we batch by 4 so that the last batch is
      # a partial batch.
      dataset_with_partial_batch = dataset.batch(4)
      cpu_model.set_weights(model_with_ds_strategy.get_weights())

      self.assertAllClose(
          model_with_ds_strategy.predict(dataset_with_partial_batch, steps=3),
          cpu_model.predict(dataset_with_partial_batch, steps=3),
          atol=1e-5, rtol=1e-5)

  @combinations.generate(tpu_strategy_combinations())
  def test_predict_multi_output_model_with_dataset_with_partial_batch(
      self, distribution):
    with self.cached_session():
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'

      with distribution.scope():
        model_with_ds_strategy = simple_multi_inputs_multi_outputs_model()
        model_with_ds_strategy.compile(optimizer, loss)

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
          atol=1e-4, rtol=1e-4)


class Counter(keras.callbacks.Callback):
  """Counts the number of times each callback method was run.

  Attributes:
    method_counts: dict. Contains the counts of time  each callback method was
      run.
  """

  def __init__(self):
    self.method_counts = collections.defaultdict(int)
    methods_to_count = [
        'on_batch_begin', 'on_batch_end', 'on_epoch_begin', 'on_epoch_end',
        'on_predict_batch_begin', 'on_predict_batch_end', 'on_predict_begin',
        'on_predict_end', 'on_test_batch_begin', 'on_test_batch_end',
        'on_test_begin', 'on_test_end', 'on_train_batch_begin',
        'on_train_batch_end', 'on_train_begin', 'on_train_end'
    ]
    for method_name in methods_to_count:
      setattr(self, method_name,
              self.wrap_with_counts(method_name, getattr(self, method_name)))

  def wrap_with_counts(self, method_name, method):

    def _call_and_count(*args, **kwargs):
      self.method_counts[method_name] += 1
      return method(*args, **kwargs)

    return _call_and_count


class TestDistributionStrategyWithCallbacks(test.TestCase,
                                            parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_callbacks_in_fit(self, distribution):
    with distribution.scope():
      model = get_model()
      model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

    dataset = get_dataset(distribution)
    counter = Counter()

    epochs = 2
    steps_per_epoch = 5
    validation_steps = 3

    model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=0,
        validation_data=dataset,
        validation_steps=validation_steps,
        callbacks=[counter])

    if isinstance(distribution, tpu_strategy.TPUStrategy):
      # TPU Strategy can have multi step training, from extended.steps_per_run
      # if steps_per_run = 1, then num_batch_call_per_epoch = steps_per_epoch
      steps_per_run = distribution.extended.steps_per_run
      num_batch_call_per_epoch = steps_per_epoch // steps_per_run
      if steps_per_epoch % steps_per_run:
        num_batch_call_per_epoch += 1
    else:
      num_batch_call_per_epoch = steps_per_epoch

    self.assertDictEqual(
        counter.method_counts, {
            'on_batch_begin': epochs * num_batch_call_per_epoch,
            'on_batch_end': epochs * num_batch_call_per_epoch,
            'on_epoch_begin': epochs,
            'on_epoch_end': epochs,
            'on_test_batch_begin': epochs * validation_steps,
            'on_test_batch_end': epochs * validation_steps,
            'on_test_begin': epochs,
            'on_test_end': epochs,
            'on_train_batch_begin': epochs * num_batch_call_per_epoch,
            'on_train_batch_end': epochs * num_batch_call_per_epoch,
            'on_train_begin': 1,
            'on_train_end': 1
        })

  @combinations.generate(all_strategy_combinations())
  def test_callbacks_in_eval(self, distribution):
    with distribution.scope():
      model = get_model()
      model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

    dataset = get_dataset(distribution)
    counter = Counter()

    model.evaluate(dataset, steps=5, callbacks=[counter])

    self.assertDictEqual(
        counter.method_counts, {
            'on_test_batch_begin': 5,
            'on_test_batch_end': 5,
            'on_test_begin': 1,
            'on_test_end': 1
        })

  @combinations.generate(all_strategy_combinations())
  def test_callbacks_in_predict(self, distribution):
    with distribution.scope():
      model = get_model()
      model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

    dataset = get_dataset(distribution)
    counter = Counter()

    model.predict(get_predict_dataset(dataset), steps=5, callbacks=[counter])

    self.assertDictEqual(
        counter.method_counts, {
            'on_predict_batch_begin': 5,
            'on_predict_batch_end': 5,
            'on_predict_begin': 1,
            'on_predict_end': 1
        })


class TestDistributionStrategyErrorCases(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph', 'eager']))
  def test_validating_dataset_input_tensors_with_shape_mismatch(self,
                                                                distribution):
    with self.cached_session():
      a = constant_op.constant([1, 2], shape=(1, 2))
      b = constant_op.constant([[1, 2], [1, 2]], shape=(2, 2))
      device_map = values.ReplicaDeviceMap(('/device:CPU:0', '/device:GPU:0'))
      x = values.DistributedValues(device_map, (a, b))
      y = values.DistributedValues(device_map, (a, a))
      # Removed device and input tensor shape details from the error message
      # since the order of the device and the corresponding input tensor shape
      # is not deterministic over different runs.
      with self.assertRaisesRegexp(ValueError,
                                   'Input tensor shapes do not match for '
                                   'distributed tensor inputs '
                                   'DistributedValues:.+'):
        with distribution.scope():
          distributed_training_utils.validate_distributed_dataset_inputs(
              distribution, x, y)

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph', 'eager']))
  def test_validating_dataset_input_tensors_with_dtype_mismatch(self,
                                                                distribution):
    with self.cached_session():
      a = constant_op.constant([1, 2], shape=(1, 2), dtype=dtypes.int32)
      b = constant_op.constant([1, 2], shape=(1, 2), dtype=dtypes.float64)
      device_map = values.ReplicaDeviceMap(('/device:CPU:0', '/device:GPU:0'))
      x = values.DistributedValues(device_map, (a, b))
      y = values.DistributedValues(device_map, (a, a))
      # Removed device and input tensor dtype details from the error message
      # since the order of the device and the corresponding input tensor dtype
      # is not deterministic over different runs.
      with self.assertRaisesRegexp(ValueError,
                                   'Input tensor dtypes do not match for '
                                   'distributed tensor inputs '
                                   'DistributedValues:.+'):
        with distribution.scope():
          distributed_training_utils.validate_distributed_dataset_inputs(
              distribution, x, y)

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph', 'eager']))
  def test_unsupported_features(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = get_model()
        optimizer = gradient_descent.GradientDescentOptimizer(0.001)
        loss = 'mse'
        metrics = ['mae']
        model.compile(optimizer, loss, metrics=metrics)

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

      # Test with not specifying the `steps` argument for dataset with infinite
      # cardinality.
      dataset = dataset.repeat()
      with self.assertRaisesRegexp(ValueError, 'When passing an infinitely '
                                   'repeating dataset, you must specify the '
                                   '`steps_per_epoch` argument'):
        model.fit(dataset, epochs=1, verbose=0)
      with self.assertRaisesRegexp(ValueError, 'When passing an infinitely '
                                   'repeating dataset, you must specify the '
                                   '`steps` argument'):
        model.evaluate(dataset, verbose=0)

      with self.assertRaisesRegexp(ValueError, 'When passing an infinitely '
                                   'repeating dataset, you must specify the '
                                   '`steps` argument'):
        model.predict(dataset, verbose=0)

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph', 'eager']))
  def test_calling_with_unsupported_predefined_callbacks(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = get_model()
        optimizer = gradient_descent.GradientDescentOptimizer(0.001)
        loss = 'mse'
        metrics = ['mae']
        model.compile(optimizer, loss, metrics=metrics)

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
      with distribution.scope():
        model = keras.models.Sequential()
        model.add(keras.layers.Masking(mask_value=0, input_shape=(2, 1)))
        model.add(
            keras.layers.TimeDistributed(
                keras.layers.Dense(1, kernel_initializer='one')))
        model.compile(loss='mse',
                      optimizer=gradient_descent.GradientDescentOptimizer(0.01))
      y = np.array([[[1], [1]], [[1], [1]]])
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)
      hist = model.fit(x=dataset, epochs=1, steps_per_epoch=2)
      self.assertEqual(hist.history['loss'][0], 0)


class TestDistributionStrategyWithNormalizationLayer(
    test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.times(
      all_strategy_combinations(),
      combinations.combine(fused=[True, False])))
  def test_batchnorm_correctness(self, distribution, fused):
    with self.cached_session():
      with distribution.scope():
        model = keras.models.Sequential()
        norm = keras.layers.BatchNormalization(
            input_shape=(10,), momentum=0.8, fused=fused)
        model.add(norm)
        model.compile(loss='mse',
                      optimizer=gradient_descent.GradientDescentOptimizer(0.01))

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


class TestDistributionStrategySaveLoadWeights(test.TestCase,
                                              parameterized.TestCase):

  @combinations.generate(all_strategy_combinations_minus_default())
  def test_save_load_h5(self, distribution):
    with self.cached_session():
      dataset = get_dataset(distribution)
      with distribution.scope():
        model = get_model()
        model.compile(gradient_descent_keras.SGD(0.01), 'mse')
        model.fit(dataset, epochs=1, steps_per_epoch=1)

        weights_file = tempfile.mktemp('.h5')
        model.save_weights(weights_file)

        model_2 = get_model()
        model_2.compile(gradient_descent_keras.SGD(0.01), 'mse')
        model_2.load_weights(weights_file)
        model_2.predict(get_predict_dataset(distribution), steps=2)
        model_2.fit(dataset, epochs=1, steps_per_epoch=1)

  @combinations.generate(all_strategy_combinations_minus_default())
  def test_save_load_checkpointable(self, distribution):
    # TODO(sourabhbajaj): Test fails with optimizer v2 without h5
    with self.cached_session():
      dataset = get_dataset(distribution)
      with distribution.scope():
        model = get_model()
        model.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')
        model.fit(dataset, epochs=1, steps_per_epoch=1)

        weights_file = tempfile.mktemp()
        model.save_weights(weights_file)

        model_2 = get_model()
        model_2.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')
        model_2.load_weights(weights_file)
        model_2.predict(get_predict_dataset(distribution), steps=2)
        model_2.fit(dataset, epochs=1, steps_per_epoch=1)


class TestDistributionStrategyValidation(test.TestCase,
                                         parameterized.TestCase):

  @combinations.generate(all_strategy_combinations_minus_default())
  def test_layer_outside_scope(self, distribution):
    with self.cached_session():
      with self.assertRaisesRegexp(
          ValueError, 'was not created in the distribution strategy'):
        x = keras.layers.Input(shape=(3,), name='input')
        y = keras.layers.Dense(4, name='dense')(x)
        with distribution.scope():
          model = keras.Model(x, y)
          optimizer = gradient_descent.GradientDescentOptimizer(0.001)
          loss = 'mse'
          metrics = ['mae', keras.metrics.CategoricalAccuracy()]
          model.compile(optimizer, loss, metrics=metrics)

  @combinations.generate(all_strategy_combinations_minus_default())
  def test_model_outside_scope(self, distribution):
    with self.cached_session():
      with self.assertRaisesRegexp(
          ValueError, 'was not created in the distribution strategy'):
        x = keras.layers.Input(shape=(3,), name='input')
        y = keras.layers.Dense(4, name='dense')(x)
        model = keras.Model(x, y)
        with distribution.scope():
          optimizer = gradient_descent.GradientDescentOptimizer(0.001)
          loss = 'mse'
          metrics = ['mae', keras.metrics.CategoricalAccuracy()]
          model.compile(optimizer, loss, metrics=metrics)


if __name__ == '__main__':
  test.main()
