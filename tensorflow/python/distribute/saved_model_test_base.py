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

_RANDOM_SEED = 1337
_IN_SCOPE_SAVE_DIR = 'in_scope/'
_OUT_OF_SCOPE_SAVE_DIR = 'out_of_scope/'

PREDICT_STEPS = 1

simple_models = [
    model_combinations.simple_functional_model,
    model_combinations.simple_sequential_model,

    # TODO(b/131715604): figure out why subclass model does not work
    # model_combinations.simple_subclass_model,
]


strategies_minus_tpu = [
    # TODO(b/132702156): include default strategy
    strategy_combinations.one_device_strategy,
    strategy_combinations.one_device_strategy_gpu,
    strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
    strategy_combinations.mirrored_strategy_with_two_gpus
]


def get_strategy_cross_product():
  result = []
  for strategy_1 in strategies_minus_tpu:
    for strategy_2 in strategies_minus_tpu:
      result.append(combinations.NamedDistributionPair(strategy_1, strategy_2))

  return result


def simple_models_with_strategies():
  return combinations.combine(
      model_and_input=simple_models,
      distribution=strategies_minus_tpu,
      mode=['eager'])


def simple_models_with_strategy_pairs():
  return combinations.combine(
      model_and_input=simple_models,
      distribution_pair=get_strategy_cross_product(),
      mode=['eager'])


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
                          output_name):
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
    """

    raise NotImplementedError('must be implemented in descendants')

  def _train_model(self, model, x_train, y_train, batch_size):
    training_dataset = dataset_ops.Dataset.from_tensor_slices(
        (x_train, y_train))
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.batch(batch_size)

    # Train the model for 1 epoch
    model.fit(x=training_dataset, epochs=1, steps_per_epoch=100)

  def _get_predict_dataset(self, x_predict, batch_size):
    predict_dataset = dataset_ops.Dataset.from_tensor_slices(x_predict)
    predict_dataset = predict_dataset.repeat()
    predict_dataset = predict_dataset.batch(batch_size)
    return predict_dataset

  def run_test_save_no_strategy_restore_strategy(self, model_and_input,
                                                 distribution):
    """Save a model without DS, and restore it with DS."""

    saved_dir = os.path.join(self.get_temp_dir(), self._root_dir,
                             'test_save_no_dist_restore_dist')

    model, output_name = model_and_input.get_model()
    x_train, y_train, x_predict = model_and_input.get_data()
    batch_size = model_and_input.get_batch_size()

    self._train_model(model, x_train, y_train, batch_size)
    predict_dataset = self._get_predict_dataset(x_predict, batch_size)
    result_before_save = model.predict(predict_dataset, steps=PREDICT_STEPS)

    self._save_model(model, saved_dir)

    with distribution.scope():
      result_after_save = self._load_and_run_model(
          distribution=distribution,
          saved_dir=saved_dir,
          predict_dataset=predict_dataset,
          output_name=output_name)

    self.assertAllEqual(result_before_save, result_after_save)

  def run_test_save_strategy_restore_no_strategy(self, model_and_input,
                                                 distribution):
    """Save a model with DS, and restore it without DS."""

    saved_dir = os.path.join(self.get_temp_dir(), self._root_dir,
                             'test_save_no_dist_restore_dist')
    saved_dir_in_scope = os.path.join(saved_dir, _IN_SCOPE_SAVE_DIR)
    saved_dir_out_of_scope = os.path.join(saved_dir, _OUT_OF_SCOPE_SAVE_DIR)

    with distribution.scope():
      model, output_name = model_and_input.get_model()
      x_train, y_train, x_predict = model_and_input.get_data()
      batch_size = model_and_input.get_batch_size()

      self._train_model(model, x_train, y_train, batch_size)
      predict_dataset = self._get_predict_dataset(x_predict, batch_size)
      result_before_save = model.predict(predict_dataset, steps=PREDICT_STEPS)

      # save the model both in and out of the DS scope
      self._save_model(model, saved_dir_in_scope)
    self._save_model(model, saved_dir_out_of_scope)

    result_load_from_save_in_scope = self._load_and_run_model(
        distribution=None,
        saved_dir=saved_dir_in_scope,
        predict_dataset=predict_dataset,
        output_name=output_name)
    result_load_from_save_out_of_scope = self._load_and_run_model(
        distribution=None,
        saved_dir=saved_dir_out_of_scope,
        predict_dataset=predict_dataset,
        output_name=output_name)

    self.assertAllEqual(result_before_save, result_load_from_save_in_scope)
    self.assertAllEqual(result_before_save, result_load_from_save_out_of_scope)

  def run_test_save_strategy_restore_strategy(self, model_and_input,
                                              distribution_pair):
    """Save a model with DS, and restore it with potentially different DS."""

    combinations.maybe_skip_test(self, distribution_pair.is_tpu_required,
                                 distribution_pair.num_gpus_required)

    saved_dir = os.path.join(self.get_temp_dir(), self._root_dir,
                             'test_save_dist_restore_dist')
    saved_dir_in_scope = os.path.join(saved_dir, _IN_SCOPE_SAVE_DIR)
    saved_dir_out_of_scope = os.path.join(saved_dir, _OUT_OF_SCOPE_SAVE_DIR)

    dist_for_save = distribution_pair.strategy_1
    dist_for_restore = distribution_pair.strategy_2

    with dist_for_save.scope():
      model, output_name = model_and_input.get_model()
      x_train, y_train, x_predict = model_and_input.get_data()
      batch_size = model_and_input.get_batch_size()

      self._train_model(model, x_train, y_train, batch_size)
      predict_dataset = self._get_predict_dataset(x_predict, batch_size)
      result_before_save = model.predict(predict_dataset, steps=PREDICT_STEPS)

      # save the model both in and out of the DS scope
      self._save_model(model, saved_dir_in_scope)
    self._save_model(model, saved_dir_out_of_scope)

    with dist_for_restore.scope():

      result_load_from_save_in_scope = self._load_and_run_model(
          distribution=dist_for_restore,
          saved_dir=saved_dir_in_scope,
          predict_dataset=predict_dataset,
          output_name=output_name)
      result_load_from_save_out_of_scope = self._load_and_run_model(
          distribution=dist_for_restore,
          saved_dir=saved_dir_out_of_scope,
          predict_dataset=predict_dataset,
          output_name=output_name)

    self.assertAllEqual(result_before_save, result_load_from_save_in_scope)
    self.assertAllEqual(result_before_save, result_load_from_save_out_of_scope)
