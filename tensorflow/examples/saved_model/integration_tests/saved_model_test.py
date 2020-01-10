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
"""SavedModel integration tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow.examples.saved_model.integration_tests import distribution_strategy_utils as ds_utils
from tensorflow.examples.saved_model.integration_tests import integration_scripts as scripts
from tensorflow.python.distribute import combinations


class SavedModelTest(scripts.TestCase, parameterized.TestCase):

  def __init__(self, method_name="runTest", has_extra_deps=False):
    super(SavedModelTest, self).__init__(method_name)
    self.has_extra_deps = has_extra_deps

  def skipIfMissingExtraDeps(self):
    """Skip test if it requires extra dependencies.

    b/132234211: The extra dependencies are not available in all environments
    that run the tests, e.g. "tensorflow_hub" is not available from tests
    within "tensorflow" alone. Those tests are instead run by another
    internal test target.
    """
    if not self.has_extra_deps:
      self.skipTest("Missing extra dependencies")

  def test_text_rnn(self):
    export_dir = self.get_temp_dir()
    self.assertCommandSucceeded("export_text_rnn_model", export_dir=export_dir)
    self.assertCommandSucceeded("use_text_rnn_model", model_dir=export_dir)

  def test_rnn_cell(self):
    export_dir = self.get_temp_dir()
    self.assertCommandSucceeded("export_rnn_cell", export_dir=export_dir)
    self.assertCommandSucceeded("use_rnn_cell", model_dir=export_dir)

  def test_text_embedding_in_sequential_keras(self):
    self.skipIfMissingExtraDeps()
    export_dir = self.get_temp_dir()
    self.assertCommandSucceeded(
        "export_simple_text_embedding", export_dir=export_dir)
    self.assertCommandSucceeded(
        "use_model_in_sequential_keras", model_dir=export_dir)

  def test_text_embedding_in_dataset(self):
    export_dir = self.get_temp_dir()
    self.assertCommandSucceeded(
        "export_simple_text_embedding", export_dir=export_dir)
    self.assertCommandSucceeded(
        "use_text_embedding_in_dataset", model_dir=export_dir)

  TEST_MNIST_CNN_GENERATE_KWARGS = dict(
      combinations=(
          combinations.combine(
              # Test all combinations with tf.saved_model.save().
              # Test all combinations using tf.keras.models.save_model()
              # for both the reusable and the final full model.
              use_keras_save_api=True,
              named_strategy=list(ds_utils.named_strategies.values()),
              retrain_flag_value=["true", "false"],
              regularization_loss_multiplier=[None, 2],  # Test for b/134528831.
          ) + combinations.combine(
              # Test few critcial combinations with raw tf.saved_model.save(),
              # including export of a reusable SavedModel that gets assembled
              # manually, including support for adjustable hparams.
              use_keras_save_api=False,
              named_strategy=None,
              retrain_flag_value=["true", "false"],
              regularization_loss_multiplier=[None, 2],  # Test for b/134528831.
          )),
      test_combinations=(combinations.NamedGPUCombination(),
                         combinations.NamedTPUCombination()))

  @combinations.generate(**TEST_MNIST_CNN_GENERATE_KWARGS)
  def test_mnist_cnn(self, use_keras_save_api, named_strategy,
                     retrain_flag_value, regularization_loss_multiplier):

    self.skipIfMissingExtraDeps()

    fast_test_mode = True
    temp_dir = self.get_temp_dir()
    feature_extrator_dir = os.path.join(temp_dir, "mnist_feature_extractor")
    full_model_dir = os.path.join(temp_dir, "full_model")

    self.assertCommandSucceeded(
        "export_mnist_cnn",
        fast_test_mode=fast_test_mode,
        export_dir=feature_extrator_dir,
        use_keras_save_api=use_keras_save_api)

    use_kwargs = dict(fast_test_mode=fast_test_mode,
                      input_saved_model_dir=feature_extrator_dir,
                      retrain=retrain_flag_value,
                      output_saved_model_dir=full_model_dir,
                      use_keras_save_api=use_keras_save_api)
    if named_strategy:
      use_kwargs["strategy"] = str(named_strategy)
    if regularization_loss_multiplier is not None:
      use_kwargs[
          "regularization_loss_multiplier"] = regularization_loss_multiplier
    self.assertCommandSucceeded("use_mnist_cnn", **use_kwargs)

    self.assertCommandSucceeded(
        "deploy_mnist_cnn",
        fast_test_mode=fast_test_mode,
        saved_model_dir=full_model_dir)


if __name__ == "__main__":
  scripts.MaybeRunScriptInstead()
  tf.test.main()
