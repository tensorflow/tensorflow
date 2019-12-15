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
"""Base class for testing saving/loading with DS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import model_combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import test
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.saved_model import saved_model

_RANDOM_SEED = 1337
_DEFAULT_FUNCTION_KEY = 'serving_default'

_TOLERANCE = 1e-30
# TPU uses bfloat16 for computation in hardware underlying, so it has less
# precision than CPU/GPU.
_TPU_TOLERANCE = 1e-7

PREDICT_STEPS = 1

simple_models = [
    model_combinations.simple_functional_model,
    model_combinations.simple_sequential_model,

    # TODO(b/131715604): figure out why subclass model does not work
    # model_combinations.simple_subclass_model,
]


strategies = [
    strategy_combinations.default_strategy,
    strategy_combinations.one_device_strategy,
    strategy_combinations.one_device_strategy_gpu,
    strategy_combinations.mirrored_strategy_with_one_cpu,
    strategy_combinations.mirrored_strategy_with_one_gpu,
    strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
    strategy_combinations.mirrored_strategy_with_two_gpus,
    strategy_combinations.tpu_strategy
]


def is_tpu_strategy(distribution):
  return (distribution is not None and
          distribution.__class__.__name__.startswith('TPUStrategy'))


def get_tolerance(save_distribution, restore_distribution):
  if is_tpu_strategy(save_distribution) or is_tpu_strategy(
      restore_distribution):
    return _TPU_TOLERANCE
  return _TOLERANCE


def simple_models_with_strategies():
  return combinations.combine(
      model_and_input=simple_models,
      distribution=strategies,
      mode=['eager'],
      experimental_run_tf_function=[True, False])


def simple_models_with_strategy_pairs():
  return combinations.combine(
      model_and_input=simple_models,
      distribution_for_saving=strategies,
      distribution_for_restoring=strategies,
      mode=['eager'],
      experimental_run_tf_function=[True, False])


def tfmodule_models_with_strategies():
  return combinations.combine(
      model_and_input=[model_combinations.simple_tfmodule_model],
      distribution=strategies,
      mode=['eager'],
      experimental_run_tf_function=[True])


def tfmodule_models_with_strategy_pairs():
  return combinations.combine(
      model_and_input=[model_combinations.simple_tfmodule_model],
      distribution_for_saving=strategies,
      distribution_for_restoring=strategies,
      mode=['eager'],
      experimental_run_tf_function=[True])


def load_and_run_with_saved_model_api(distribution, saved_dir, predict_dataset,
                                      output_name):
  """Loads a saved_model using tf.saved_model API, and runs it."""
  func = saved_model.load(saved_dir)
  if distribution:
    dist_predict_dataset = distribution.experimental_distribute_dataset(
        predict_dataset)
    per_replica_predict_data = next(iter(dist_predict_dataset))
    result = distribution.experimental_run_v2(
        func.signatures[_DEFAULT_FUNCTION_KEY],
        args=(per_replica_predict_data,))
    result = result[output_name]

    # Convert the per_replica value to a list, then concatenate them
    reduced = distribution.experimental_local_results(result)
    concat = array_ops.concat(reduced, 0)
    return concat
  else:
    result = func.signatures[_DEFAULT_FUNCTION_KEY](next(iter(predict_dataset)))
    return result[output_name]


class TestSavedModelBase(test.TestCase, parameterized.TestCase):
  """Base class for testing saving/loading with DS."""

  def setUp(self):
    np.random.seed(_RANDOM_SEED)
    random_seed.set_random_seed(_RANDOM_SEED)
    self._root_dir = 'base'
    super(TestSavedModelBase, self).setUp()

  def _save_model(self, model, saved_dir):
    """Save the given model to the given saved_dir.

    This method needs to be implemeted by the subclasses.

    Args:
      model: a keras model object to save.
      saved_dir: a string representing the path to save the keras model
    """
    raise NotImplementedError('must be implemented in descendants')

  def _load_and_run_model(self, distribution, saved_dir, predict_dataset,
                          output_name, experimental_run_tf_function):
    """Load the model and run 1 step of predict with it.

    This method must be implemented by the subclasses.

    Args:
      distribution: the distribution strategy used to load the model. None if no
        distribution strategy is used
      saved_dir: the string representing the path where the model is saved.
      predict_dataset: the data used to do the predict on the model for
        cross_replica context.
      output_name: the string representing the name of the output layer of the
        model.
      experimental_run_tf_function: Whether to use the single execution path
        for models.
    """

    raise NotImplementedError('must be implemented in descendants')

  def _train_model(self, model, x_train, y_train, batch_size):
    training_dataset = dataset_ops.Dataset.from_tensor_slices(
        (x_train, y_train))
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.batch(batch_size)

    # Train the model for 1 epoch
    model.fit(x=training_dataset, epochs=1, steps_per_epoch=100)

  def _predict_with_model(self, distribution, model, predict_dataset):
    return model.predict(predict_dataset, steps=PREDICT_STEPS)

  def _get_predict_dataset(self, x_predict, batch_size):
    predict_dataset = dataset_ops.Dataset.from_tensor_slices(x_predict)
    predict_dataset = predict_dataset.repeat()
    predict_dataset = predict_dataset.batch(batch_size)
    return predict_dataset

  def run_test_save_no_strategy_restore_strategy(self, model_and_input,
                                                 distribution,
                                                 experimental_run_tf_function):
    """Save a model without DS, and restore it with DS."""

    saved_dir = os.path.join(self.get_temp_dir(), '0')

    model, output_name = model_and_input.get_model(
        experimental_run_tf_function=experimental_run_tf_function)
    x_train, y_train, x_predict = model_and_input.get_data()
    batch_size = model_and_input.get_batch_size()
    predict_dataset = self._get_predict_dataset(x_predict, batch_size)

    self._train_model(model, x_train, y_train, batch_size)
    result_before_save = self._predict_with_model(None, model, predict_dataset)

    self._save_model(model, saved_dir)

    with distribution.scope():
      result_after_save = self._load_and_run_model(
          distribution=distribution,
          saved_dir=saved_dir,
          predict_dataset=predict_dataset,
          output_name=output_name,
          experimental_run_tf_function=experimental_run_tf_function)

    tolerance = get_tolerance(None, distribution)
    self.assertAllClose(result_before_save, result_after_save, atol=tolerance)

  def run_test_save_strategy_restore_no_strategy(self, model_and_input,
                                                 distribution, save_in_scope,
                                                 experimental_run_tf_function):
    """Save a model with DS, and restore it without DS."""

    saved_dir = os.path.join(self.get_temp_dir(), '1')

    with distribution.scope():
      model, output_name = model_and_input.get_model(
          experimental_run_tf_function=experimental_run_tf_function)
      x_train, y_train, x_predict = model_and_input.get_data()
      batch_size = model_and_input.get_batch_size()

      self._train_model(model, x_train, y_train, batch_size)
      predict_dataset = self._get_predict_dataset(x_predict, batch_size)
      result_before_save = self._predict_with_model(
          distribution, model, predict_dataset)

    if save_in_scope:
      with distribution.scope():
        self._save_model(model, saved_dir)
    else:
      self._save_model(model, saved_dir)

    load_result = self._load_and_run_model(
        distribution=None,
        saved_dir=saved_dir,
        predict_dataset=predict_dataset,
        output_name=output_name,
        experimental_run_tf_function=experimental_run_tf_function)

    tolerance = get_tolerance(distribution, None)
    self.assertAllClose(result_before_save, load_result, atol=tolerance)

  def run_test_save_strategy_restore_strategy(self, model_and_input,
                                              distribution_for_saving,
                                              distribution_for_restoring,
                                              save_in_scope,
                                              experimental_run_tf_function):
    """Save a model with DS, and restore it with potentially different DS."""
    saved_dir = os.path.join(self.get_temp_dir(), '2')

    with distribution_for_saving.scope():
      model, output_name = model_and_input.get_model(
          experimental_run_tf_function=experimental_run_tf_function)
      x_train, y_train, x_predict = model_and_input.get_data()
      batch_size = model_and_input.get_batch_size()

      self._train_model(model, x_train, y_train, batch_size)
      predict_dataset = self._get_predict_dataset(x_predict, batch_size)
      result_before_save = self._predict_with_model(
          distribution_for_saving, model, predict_dataset)

    if save_in_scope:
      with distribution_for_saving.scope():
        self._save_model(model, saved_dir)
    else:
      self._save_model(model, saved_dir)

    with distribution_for_restoring.scope():

      load_result = self._load_and_run_model(
          distribution=distribution_for_restoring,
          saved_dir=saved_dir,
          predict_dataset=predict_dataset,
          output_name=output_name,
          experimental_run_tf_function=experimental_run_tf_function)

    tolerance = get_tolerance(distribution_for_saving,
                              distribution_for_restoring)
    self.assertAllClose(result_before_save, load_result, atol=tolerance)
