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
"""Tests for saving and loading using tf's saved_model APIs with DS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import saved_model_test_base as test_base
from tensorflow.python.eager import test
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.saved_model import saved_model


class SavedModelKerasModelTest(test_base.TestSavedModelBase):

  def setUp(self):
    self._root_dir = 'saved_model_save_load'
    super(SavedModelKerasModelTest, self).setUp()

  def _save_model(self, model, saved_dir):
    saved_model.save(model, saved_dir)

  def _load_and_run_model(self, distribution, saved_dir, predict_dataset,
                          output_name, experimental_run_tf_function):
    return test_base.load_and_run_with_saved_model_api(distribution, saved_dir,
                                                       predict_dataset,
                                                       output_name)

  @combinations.generate(test_base.simple_models_with_strategies())
  def test_save_no_strategy_restore_strategy(self, model_and_input,
                                             distribution,
                                             experimental_run_tf_function):
    self.run_test_save_no_strategy_restore_strategy(
        model_and_input, distribution, experimental_run_tf_function)

  @combinations.generate(
      combinations.times(test_base.simple_models_with_strategies(),
                         combinations.combine(save_in_scope=[True, False])))
  def test_save_strategy_restore_no_strategy(self, model_and_input,
                                             distribution, save_in_scope,
                                             experimental_run_tf_function):
    if save_in_scope:
      # TODO(b/134703272): Unskip this test when fixed.
      self.skipTest(('Saving model within tf.distribute.Strategy scope is not ',
                     'supported.'))
    self.run_test_save_strategy_restore_no_strategy(
        model_and_input, distribution, save_in_scope,
        experimental_run_tf_function)

  @combinations.generate(
      combinations.times(test_base.simple_models_with_strategy_pairs(),
                         combinations.combine(save_in_scope=[True, False])))
  def test_save_strategy_restore_strategy(self, model_and_input,
                                          distribution_for_saving,
                                          distribution_for_restoring,
                                          save_in_scope,
                                          experimental_run_tf_function):
    if save_in_scope:
      # TODO(b/134703272): Unskip this test when fixed.
      self.skipTest(('Saving model within tf.distribute.Strategy scope is not ',
                     'supported.'))
    self.run_test_save_strategy_restore_strategy(model_and_input,
                                                 distribution_for_saving,
                                                 distribution_for_restoring,
                                                 save_in_scope,
                                                 experimental_run_tf_function)


class SavedModelTFModuleTest(test_base.TestSavedModelBase):

  def setUp(self):
    self._root_dir = 'saved_model_save_load'
    super(SavedModelTFModuleTest, self).setUp()

  def _train_model(self, model, x_train, y_train, batch_size):
    pass

  def _predict_with_model(self, distribution, model, predict_dataset):
    if distribution:
      dist_predict_dataset = distribution.experimental_distribute_dataset(
          predict_dataset)
      per_replica_predict_data = next(iter(dist_predict_dataset))
      result = distribution.experimental_run_v2(
          model, args=(per_replica_predict_data,))
      # Convert the per_replica value to a list, then concatenate them
      reduced = distribution.experimental_local_results(result)
      concat = array_ops.concat(reduced, 0)
      return concat
    else:
      return model(next(iter(predict_dataset)))

  def _save_model(self, model, saved_dir):
    call = model.__call__.get_concrete_function(tensor_spec.TensorSpec(None))
    saved_model.save(model, saved_dir, signatures=call)

  def _load_and_run_model(self, distribution, saved_dir, predict_dataset,
                          output_name, experimental_run_tf_function):
    del output_name, experimental_run_tf_function
    model = saved_model.load(saved_dir)
    return self._predict_with_model(distribution, model, predict_dataset)

  @combinations.generate(test_base.tfmodule_models_with_strategies())
  def test_save_no_strategy_restore_strategy(self, model_and_input,
                                             distribution,
                                             experimental_run_tf_function):
    self.run_test_save_no_strategy_restore_strategy(
        model_and_input, distribution, experimental_run_tf_function)

  @combinations.generate(
      combinations.times(test_base.tfmodule_models_with_strategies(),
                         combinations.combine(save_in_scope=[True, False])))
  def test_save_strategy_restore_no_strategy(
      self, model_and_input, distribution, save_in_scope,
      experimental_run_tf_function):
    if save_in_scope:
      # TODO(b/134703272): Unskip this test when fixed.
      self.skipTest(('Saving model within tf.distribute.Strategy scope is not ',
                     'supported.'))
    self.run_test_save_strategy_restore_no_strategy(
        model_and_input, distribution, save_in_scope,
        experimental_run_tf_function)

  @combinations.generate(
      combinations.times(test_base.tfmodule_models_with_strategy_pairs(),
                         combinations.combine(save_in_scope=[True, False])))
  def test_save_strategy_restore_strategy(self, model_and_input,
                                          distribution_for_saving,
                                          distribution_for_restoring,
                                          save_in_scope,
                                          experimental_run_tf_function):
    if save_in_scope:
      # TODO(b/134703272): Unskip this test when fixed.
      self.skipTest(('Saving model within tf.distribute.Strategy scope is not ',
                     'supported.'))
    self.run_test_save_strategy_restore_strategy(model_and_input,
                                                 distribution_for_saving,
                                                 distribution_for_restoring,
                                                 save_in_scope,
                                                 experimental_run_tf_function)


if __name__ == '__main__':
  test.main()
