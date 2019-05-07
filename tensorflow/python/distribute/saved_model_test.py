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
"""Test saved_model with distribution strategies."""

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
from tensorflow.python.saved_model import saved_model

_RANDOM_SEED = 1337
_DEFAULT_FUNCTION_KEY = 'serving_default'
_IN_SCOPE_SAVE_DIR = 'in_scope/'
_OUT_OF_SCOPE_SAVE_DIR = 'out_of_scope/'

simple_models = [
    model_combinations.simple_functional_model,
    model_combinations.simple_sequential_model,

    # TODO(b/131715604): figure out why subclass model does not work
    # model_combinations.simple_subclass_model,
]


def get_strategy_cross_product():
  result = []
  for strategy_1 in strategy_combinations.strategies_minus_tpu:
    for strategy_2 in strategy_combinations.strategies_minus_tpu:
      result.append(combinations.NamedDistributionPair(strategy_1, strategy_2))

  return result


def simple_models_with_strategies():
  return combinations.combine(
      model_and_input=simple_models,
      distribution=strategy_combinations.strategies_minus_tpu,
      mode=['eager'])


class TestSavedModel(test.TestCase, parameterized.TestCase):

  def setUp(self):
    np.random.seed(_RANDOM_SEED)
    random_seed.set_random_seed(_RANDOM_SEED)
    super(TestSavedModel, self).setUp()

  def _train_model(self, model, x_train, y_train, batch_size):
    training_dataset = dataset_ops.Dataset.from_tensor_slices(
        (x_train, y_train))
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.batch(batch_size)

    # Train the model for 1 step
    model.fit(x=training_dataset, epochs=1, steps_per_epoch=1)

  def _load_and_run_model(self, saved_dir, x_predict):
    func = saved_model.load(saved_dir)
    return func.signatures[_DEFAULT_FUNCTION_KEY](x_predict)

  def _get_predict_dataset(self, x_predict, batch_size):
    predict_dataset = dataset_ops.Dataset.from_tensor_slices(x_predict)
    predict_dataset = predict_dataset.batch(batch_size)
    return predict_dataset

  @combinations.generate(simple_models_with_strategies())
  def test_save_no_dist_restore_dist(self, model_and_input, distribution):
    """Save a model without DS, and restore it with DS."""

    self.skipTest('Loading model with DS is not supported yet')

    saved_dir = os.path.join(self.get_temp_dir(),
                             'test_save_no_dist_restore_dist')

    model, output_name = model_and_input.get_model()
    x_train, y_train, x_predict = model_and_input.get_data()
    batch_size = model_and_input.get_batch_size()

    self._train_model(model, x_train, y_train, batch_size)
    predict_dataset = self._get_predict_dataset(x_predict, batch_size)
    result_before_save = model.predict(predict_dataset)

    saved_model.save(model, saved_dir)

    with distribution.scope():
      predict_dataset = distribution.experimental_distribute_dataset(
          predict_dataset)
      actual_data = next(iter(predict_dataset))
      result_after_save = self._load_and_run_model(saved_dir, actual_data)

    self.assertAllEqual(result_before_save, result_after_save[output_name])

  @combinations.generate(simple_models_with_strategies())
  def test_save_dist_restore_no_dist(self, model_and_input, distribution):
    """Save a model with DS, and restore it without DS."""

    self.skipTest('Saving model with DS is not supported yet')

    saved_dir = os.path.join(self.get_temp_dir(),
                             'test_save_no_dist_restore_dist')
    saved_dir_in_scope = os.path.join(saved_dir, _IN_SCOPE_SAVE_DIR)
    saved_dir_out_of_scope = os.path.join(saved_dir, _OUT_OF_SCOPE_SAVE_DIR)

    with distribution.scope():
      model, output_name = model_and_input.get_model()
      x_train, y_train, x_predict = model_and_input.get_data()
      batch_size = model_and_input.get_batch_size()

      self._train_model(model, x_train, y_train, batch_size)
      predict_dataset = self._get_predict_dataset(x_predict, batch_size)
      result_before_save = model.predict(predict_dataset)

      # save the model both in and out of the DS scope
      saved_model.save(model, saved_dir_in_scope)
    saved_model.save(model, saved_dir_out_of_scope)

    actual_data = next(iter(predict_dataset))
    result_load_from_save_in_scope = self._load_and_run_model(
        saved_dir_in_scope, actual_data)
    result_load_from_save_out_of_scope = self._load_and_run_model(
        saved_dir_out_of_scope, actual_data)

    self.assertAllEqual(result_before_save,
                        result_load_from_save_in_scope[output_name])
    self.assertAllEqual(result_before_save,
                        result_load_from_save_out_of_scope[output_name])

  @combinations.generate(
      combinations.combine(
          model_and_input=simple_models,
          distribution_pair=get_strategy_cross_product(),
          mode=['eager']))
  def test_save_dist_restore_dist(self, model_and_input, distribution_pair):
    """Save a model with DS, and restore it with potentially different DS."""

    self.skipTest('Saving model with DS is not supported yet')

    combinations.maybe_skip_test(self, distribution_pair.is_tpu_required,
                                 distribution_pair.num_gpus_required)

    saved_dir = os.path.join(self.get_temp_dir(), 'test_save_dist_restore_dist')
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
      result_before_save = model.predict(predict_dataset)

      # save the model both in and out of the DS scope
      saved_model.save(model, saved_dir_in_scope)
    saved_model.save(model, saved_dir_out_of_scope)

    with dist_for_restore.scope():
      predict_dataset = dist_for_restore.experimental_distribute_dataset(
          predict_dataset)
      actual_data = next(iter(predict_dataset))

      result_load_from_save_in_scope = self._load_and_run_model(
          saved_dir_in_scope, actual_data)
      result_load_from_save_out_of_scope = self._load_and_run_model(
          saved_dir_out_of_scope, actual_data)

    self.assertAllEqual(result_before_save,
                        result_load_from_save_in_scope[output_name])
    self.assertAllEqual(result_before_save,
                        result_load_from_save_out_of_scope[output_name])


if __name__ == '__main__':
  test.main()
