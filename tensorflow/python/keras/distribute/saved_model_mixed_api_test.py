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
"""Tests for saving and loading with mixed APIs with distribution strategies.

For saving, Keras's export_saved_model() API is used; and for loading,
saved_model's load() API is used. Keras's export_save_model() when used with
`serving_only` parameter equals to True should be the same as using
tf.saved_model.save().
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.distribute import saved_model_test_base as test_base
from tensorflow.python.keras.saving import save
from tensorflow.python.platform import test

_DEFAULT_FUNCTION_KEY = 'serving_default'


@testing_utils.run_all_without_tensor_float_32(
    'Uses Dense layers, which call matmul')
class SavedModelSaveAndLoadTest(test_base.TestSavedModelBase):

  def setUp(self):
    self._root_dir = 'saved_model_save_load'
    super(SavedModelSaveAndLoadTest, self).setUp()

  def _save_model(self, model, saved_dir):
    save.save_model(model, saved_dir, save_format='tf')

  def _load_and_run_model(self,
                          distribution,
                          saved_dir,
                          predict_dataset,
                          output_name='output_1'):
    return test_base.load_and_run_with_saved_model_api(distribution, saved_dir,
                                                       predict_dataset,
                                                       output_name)

  @ds_combinations.generate(test_base.simple_models_with_strategies())
  def test_save_no_strategy_restore_strategy(self, model_and_input,
                                             distribution):
    self.run_test_save_no_strategy_restore_strategy(
        model_and_input, distribution)

  @ds_combinations.generate(
      combinations.times(test_base.simple_models_with_strategies(),
                         combinations.combine(save_in_scope=[True, False])))
  def test_save_strategy_restore_no_strategy(self, model_and_input,
                                             distribution, save_in_scope):
    self.run_test_save_strategy_restore_no_strategy(
        model_and_input, distribution, save_in_scope)

  @ds_combinations.generate(
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
  ops.enable_eager_execution()
  test.main()
