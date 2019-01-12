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
"""Correctness tests for tf.keras using DistributionStrategy."""
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
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import random_seed
from tensorflow.python.keras.engine import distributed_training_utils

_RANDOM_SEED = 1337
_EVAL_STEPS = 20
_GLOBAL_BATCH_SIZE = 64

# Note: Please make sure the tests in this file are also covered in
# keras_backward_compat_test for features that are supported with both APIs.


all_strategies = [
    combinations.default_strategy,
    combinations.one_device_strategy,
    combinations.mirrored_strategy_with_gpu_and_cpu,
    combinations.mirrored_strategy_with_two_gpus,
    combinations.core_mirrored_strategy_with_gpu_and_cpu,
    combinations.core_mirrored_strategy_with_two_gpus,
    combinations.tpu_strategy,  # steps_per_run=2
    combinations.tpu_strategy_one_step,
]


def all_strategy_combinations_with_eager_and_graph_modes():
  return combinations.combine(distribution=all_strategies,
                              mode=['graph', 'eager'])


def all_strategy_combinations_with_graph_mode():
  return combinations.combine(distribution=all_strategies, mode=['graph'])


def all_strategy_and_input_config_combinations():
  return (
      combinations.times(
          combinations.combine(distribution=all_strategies),
          combinations.combine(mode=['graph', 'eager'],
                               use_numpy=[True, False],
                               use_validation_data=[True, False])))


class MaybeDistributionScope(object):
  """Provides a context allowing no distribution strategy."""

  def __init__(self, distribution):
    self._distribution = distribution
    self._scope = None

  def __enter__(self):
    if self._distribution:
      self._scope = self._distribution.scope()
      self._scope.__enter__()

  def __exit__(self, exc_type, value, traceback):
    if self._distribution:
      self._scope.__exit__(exc_type, value, traceback)
      self._scope = None


def batch_wrapper(dataset, batch_size, distribution, repeat=None):
  if repeat:
    dataset = dataset.repeat(repeat)
  # TPUs currently require fully defined input shapes, drop_remainder ensures
  # the input will have fully defined shapes.
  if isinstance(distribution, tpu_strategy.TPUStrategy):
    return dataset.batch(batch_size, drop_remainder=True)
  else:
    return dataset.batch(batch_size)


def get_batch_size(global_batch_size, distribution):
  batch_size = global_batch_size
  # TODO(b/118776054): Use global batch size for Keras/DS support.
  use_per_core_batch_size = (
      distribution and
      not distributed_training_utils.global_batch_size_supported(distribution))
  if use_per_core_batch_size:
    batch_size //= distribution.num_replicas_in_sync
  return batch_size


def get_correctness_test_inputs(use_numpy, use_validation_data,
                                with_distribution, x_train, y_train, x_predict):
  """Generates the inputs for correctness check when enable Keras with DS."""
  training_epochs = 2
  global_batch_size = _GLOBAL_BATCH_SIZE
  batch_size = get_batch_size(global_batch_size, with_distribution)

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
    if len(x_train) < _GLOBAL_BATCH_SIZE * _EVAL_STEPS:
      # Currently, we cannot detech the size of a dataset. So, the eval steps is
      # hard coded.
      raise ValueError('x_train must have at least '
                       '_GLOBAL_BATCH_SIZE * _EVAL_STEPS samples')
    # For dataset inputs, we do not pass batch_size to
    # keras.fit/evaluate/predict. The batch size is part of the dataset.
    train_dataset = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
    x = batch_wrapper(train_dataset, batch_size, with_distribution,
                      repeat=training_epochs)

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
      eval_dataset = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
      x = batch_wrapper(eval_dataset, batch_size, with_distribution)
      training_inputs['validation_data'] = x
      training_inputs['validation_steps'] = 5
    else:
      eval_inputs = {
          'batch_size': None,
          'x': x,
          'y': None,
          'steps': _EVAL_STEPS,
      }

    predict_batch_size = get_batch_size(len(x_predict), with_distribution)
    predict_dataset = dataset_ops.Dataset.from_tensor_slices(x_predict)
    predict_dataset = batch_wrapper(predict_dataset, predict_batch_size,
                                    with_distribution)
    predict_inputs = {
        'steps': 1,
        'x': predict_dataset,
    }

  return training_inputs, eval_inputs, predict_inputs


def fit_eval_and_predict(
    initial_weights, input_fn, model_fn, distribution=None):
  """Generates results for fit/predict/evaluate for given model."""
  model = model_fn(initial_weights=initial_weights, distribution=distribution)
  training_inputs, eval_inputs, predict_inputs = input_fn(distribution)

  result = {}
  result['training_history_1'] = model.fit(**training_inputs).history

  if eval_inputs is not None:
    result['eval_result_1'] = model.evaluate(**eval_inputs)

  result['weights_1'] = model.get_weights()

  if predict_inputs is not None:
    result['predict_result_1'] = model.predict(**predict_inputs)

  # Train and eval again to mimic user's flow.

  result['training_history_2'] = model.fit(**training_inputs).history

  if eval_inputs is not None:
    result['eval_result_2'] = model.evaluate(**eval_inputs)

  result['weights_2'] = model.get_weights()

  return result


def compare_results(results_with_ds, results_without_ds, distribution,
                    testcase):
  """Compares results of model compiled with/without distribution strategy."""

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

  for key in results_with_ds:
    if (key.startswith('training_history') and
        isinstance(distribution, tpu_strategy.TPUStrategy) and
        distribution.extended.steps_per_run > 1):
      # TODO(b/119894254): Enable this test for all cases once the
      # underlying bug is fixed.
      continue

    tolerance = tol_table.get(key, default_tolerance)

    testcase.assertAllClose(
        results_with_ds[key],
        results_without_ds[key],
        atol=tolerance,
        rtol=tolerance,
        msg='Fail to assert {}.'.format(key))


def should_skip_tpu_with_eager(distribution):
  return (context.executing_eagerly() and
          isinstance(distribution, tpu_strategy.TPUStrategy))


class LearningRateBatchScheduler(keras.callbacks.Callback):
  """Scheduler that dynamically sets the learning rate of model."""

  def __init__(self, update_freq=None):
    self._update_freq = update_freq

  def on_batch_begin(self, batch, logs=None):
    if self._update_freq and batch % self._update_freq != 0:
      return

    # To avoid divergence, limit the value range.
    lr = 0.001 * (batch % 10)
    keras.backend.set_value(self.model.optimizer.lr, lr)


class TestDistributionStrategyCorrectnessBase(test.TestCase,
                                              parameterized.TestCase):
  """Model agnostic testing infra to test correctness of Keras models."""

  def set_up_test_config(self, use_numpy=False,
                         use_validation_data=False,
                         with_batch_norm=False):
    self.use_numpy = use_numpy
    self.use_validation_data = use_validation_data
    self.with_batch_norm = with_batch_norm

    keras.backend.set_image_data_format('channels_last')
    np.random.seed(_RANDOM_SEED)
    random_seed.set_random_seed(_RANDOM_SEED)

  def get_data(self):
    num_samples = 10000
    x_train = np.random.randint(0, 2, num_samples)
    x_train = np.reshape(x_train, (num_samples, 1))
    y_train = x_train
    return (x_train.astype('float32'), y_train.astype('float32'), None)

  def get_model(self, distribution=None):
    raise NotImplementedError

  def skip_unsupported_test_configuration(self, distribution):
    if should_skip_tpu_with_eager(distribution):
      self.skipTest('TPUStrategy does not support eager mode now.')

    if context.executing_eagerly() and self.use_numpy:
      self.skipTest('Numpy as inputs is not supported with strategy in eager.')

    if context.executing_eagerly() and self.use_validation_data:
      self.skipTest('TODO(hongjunchoi): Add test logic for using validation '
                    'data for eager execution.')
    return

  def run_correctness_test(self,
                           distribution,
                           use_numpy,
                           use_validation_data,
                           with_batch_norm=False):
    with self.cached_session():
      self.set_up_test_config(use_numpy, use_validation_data, with_batch_norm)
      self.skip_unsupported_test_configuration(distribution)

      # Train, eval, and predict datasets are created with the same input numpy
      # arrays.
      x_train, y_train, x_predict = self.get_data()

      # The model is built once and the initial weights are saved.
      # This is used to initialize the model for both the distribution and
      # non-distribution run.
      model = self.get_model()
      initial_weights = model.get_weights()

      def input_fn(dist):
        return get_correctness_test_inputs(
            use_numpy, use_validation_data, dist, x_train, y_train, x_predict)

      results_with_ds = fit_eval_and_predict(
          initial_weights, input_fn=input_fn, model_fn=self.get_model,
          distribution=distribution)
      results_without_ds = fit_eval_and_predict(
          initial_weights, input_fn=input_fn, model_fn=self.get_model,
          distribution=None)

      # First, special case, for multi-replica distributed training, batch norm
      # is not aggregated globally. So it is expected to have different weights.
      if (self.with_batch_norm and
          distribution.num_replicas_in_sync > 1):
        with self.assertRaises(AssertionError):
          compare_results(results_with_ds, results_without_ds, distribution,
                          testcase=self)
      else:
        compare_results(results_with_ds, results_without_ds, distribution,
                        testcase=self)

  def run_dynamic_lr_test(self, distribution):
    with self.cached_session():
      self.set_up_test_config()
      self.skip_unsupported_test_configuration(distribution)

      x_train, y_train, _ = self.get_data()
      model = self.get_model()
      initial_weights = model.get_weights()
      update_freq = None

      if (isinstance(distribution, tpu_strategy.TPUStrategy) and
          distribution.extended.steps_per_run > 1):
        # For TPUStrategy with steps_per_run > 1, the callback is not invoked
        # every step. So, to compare the CPU/TPU, we let the CPU to behave the
        # same as TPU.
        update_freq = distribution.extended.steps_per_run

      def input_fn(dist):
        """Generates training test given test configuration."""
        training_epochs = 2
        global_batch_size = 64
        batch_size = get_batch_size(global_batch_size, dist)

        training_inputs = {
            'batch_size': batch_size,
            'x': x_train,
            'y': y_train,
            'epochs': training_epochs,
            'shuffle': False,
            'callbacks': [LearningRateBatchScheduler(update_freq)],
            'validation_data': (x_train, y_train)
        }
        # In this test case, we do not care eval and predict.
        eval_inputs, predict_inputs = None, None
        return training_inputs, eval_inputs, predict_inputs

      results_with_ds = fit_eval_and_predict(
          initial_weights, input_fn=input_fn, model_fn=self.get_model,
          distribution=distribution)
      results_without_ds = fit_eval_and_predict(
          initial_weights, input_fn=input_fn, model_fn=self.get_model,
          distribution=None)
      compare_results(results_with_ds, results_without_ds, distribution,
                      testcase=self)


if __name__ == '__main__':
  test.main()
