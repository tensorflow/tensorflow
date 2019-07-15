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
"""Tests for saving and loading using keras experimental APIs with DS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import saved_model_test_base as test_base
from tensorflow.python.eager import test
from tensorflow.python.keras.saving import saved_model_experimental as saved_model


class KerasExperimentalSaveLoadTest(test_base.TestSavedModelBase):

  def setUp(self):
    self._root_dir = 'keras_experimental_save_load'
    super(KerasExperimentalSaveLoadTest, self).setUp()

  def _save_model(self, model, saved_dir):
    saved_model.export_saved_model(model, saved_dir)

  def _load_and_run_model(self, distribution, saved_dir, predict_dataset,
                          output_name):
    restored_keras_model = saved_model.load_from_saved_model(saved_dir)
    return restored_keras_model.predict(
        predict_dataset, steps=test_base.PREDICT_STEPS)

  @combinations.generate(test_base.simple_models_with_strategies())
  def test_save_no_strategy_restore_strategy(self, model_and_input,
                                             distribution):
    self.run_test_save_no_strategy_restore_strategy(model_and_input,
                                                    distribution)

  @combinations.generate(
      combinations.times(test_base.simple_models_with_strategies(),
                         combinations.combine(save_in_scope=[True, False])))
  def test_save_strategy_restore_no_strategy(self, model_and_input,
                                             distribution, save_in_scope):
    self.run_test_save_strategy_restore_no_strategy(model_and_input,
                                                    distribution, save_in_scope)

  @combinations.generate(
      combinations.times(test_base.simple_models_with_strategy_pairs(),
                         combinations.combine(save_in_scope=[True, False])))
  def test_save_strategy_restore_strategy(self, model_and_input,
                                          distribution_for_saving,
                                          distribution_for_restoring,
                                          save_in_scope):
    self.run_test_save_strategy_restore_strategy(model_and_input,
                                                 distribution_for_saving,
                                                 distribution_for_restoring,
                                                 save_in_scope)


if __name__ == '__main__':
  test.main()
