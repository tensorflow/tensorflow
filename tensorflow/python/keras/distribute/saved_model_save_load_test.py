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

import os
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.distribute import model_combinations
from tensorflow.python.keras.distribute import saved_model_test_base as test_base
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load_options as load_options_lib
from tensorflow.python.saved_model import save_options as save_options_lib
from tensorflow.python.saved_model import saved_model


@testing_utils.run_all_without_tensor_float_32(
    'Uses Dense layers, which call matmul')
class SavedModelKerasModelTest(test_base.TestSavedModelBase):

  def setUp(self):
    self._root_dir = 'saved_model_save_load'
    super(SavedModelKerasModelTest, self).setUp()

  def _save_model(self, model, saved_dir):
    saved_model.save(model, saved_dir)

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

  @ds_combinations.generate(
      combinations.times(test_base.simple_models_with_strategies(),
                         combinations.combine(save_in_scope=[True, False])))
  def test_no_variable_device_placement(self, model_and_input, distribution,
                                        save_in_scope):
    saved_dir = self.run_test_save_strategy(model_and_input, distribution,
                                            save_in_scope)
    func = saved_model.load(saved_dir)
    concrete_function = func.signatures[test_base._DEFAULT_FUNCTION_KEY]
    for f in concrete_function.graph.as_graph_def().library.function:
      for n in f.node_def:
        if n.op == 'ReadVariableOp':
          self.assertEmpty(n.device)


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
      result = distribution.run(model, args=(per_replica_predict_data,))
      # Convert the per_replica value to a list, then concatenate them
      reduced = distribution.experimental_local_results(result)
      concat = array_ops.concat(reduced, 0)
      return concat
    else:
      return model(next(iter(predict_dataset)))

  def _save_model(self, model, saved_dir):
    call = model.__call__.get_concrete_function(tensor_spec.TensorSpec(None))
    saved_model.save(model, saved_dir, signatures=call)

  def _load_and_run_model(self,
                          distribution,
                          saved_dir,
                          predict_dataset,
                          output_name='output_1'):
    del output_name
    model = saved_model.load(saved_dir)
    return self._predict_with_model(distribution, model, predict_dataset)

  @ds_combinations.generate(test_base.tfmodule_models_with_strategies())
  def test_save_no_strategy_restore_strategy(self, model_and_input,
                                             distribution):
    self.run_test_save_no_strategy_restore_strategy(
        model_and_input, distribution)

  @ds_combinations.generate(
      combinations.times(test_base.tfmodule_models_with_strategies(),
                         combinations.combine(save_in_scope=[True, False])))
  def test_save_strategy_restore_no_strategy(
      self, model_and_input, distribution, save_in_scope):
    self.run_test_save_strategy_restore_no_strategy(
        model_and_input, distribution, save_in_scope)

  @ds_combinations.generate(
      combinations.times(test_base.tfmodule_models_with_strategy_pairs(),
                         combinations.combine(save_in_scope=[True, False])))
  def test_save_strategy_restore_strategy(self, model_and_input,
                                          distribution_for_saving,
                                          distribution_for_restoring,
                                          save_in_scope):
    self.run_test_save_strategy_restore_strategy(model_and_input,
                                                 distribution_for_saving,
                                                 distribution_for_restoring,
                                                 save_in_scope)

  @ds_combinations.generate(
      combinations.combine(
          model_and_input=[model_combinations.simple_tfmodule_model],
          distribution=test_base.strategies +
          [strategy_combinations.cloud_tpu_strategy]))
  def test_save_load_io_device(self, model_and_input, distribution):
    saved_dir = os.path.join(self.get_temp_dir(), 'io_device')
    with distribution.scope():
      model = model_and_input.get_model()
      x_train, y_train, _ = model_and_input.get_data()
      batch_size = model_and_input.get_batch_size()
      self._train_model(model, x_train, y_train, batch_size)
    call = model.__call__.get_concrete_function(tensor_spec.TensorSpec(None))
    save_options = save_options_lib.SaveOptions(
        experimental_io_device='/job:localhost')
    saved_model.save(model, saved_dir, signatures=call, options=save_options)
    load_options = load_options_lib.LoadOptions(
        experimental_io_device='/job:localhost')
    # Check that the model can be loaded and training continued without error.
    with distribution.scope():
      loaded_model = saved_model.load(saved_dir, options=load_options)
      self._train_model(loaded_model, x_train, y_train, batch_size)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
