# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for `DatasetCreator` with `Model.fit` across usages and strategies."""
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.framework import test_util
from tensorflow.python.keras.distribute import dataset_creator_model_fit_test_base as test_base
from tensorflow.python.keras.distribute import strategy_combinations


# TODO(rchao): Investigate why there cannot be single worker and multi worker
# PS strategies running in the same shard.
@ds_combinations.generate(
    combinations.combine(
        strategy=strategy_combinations.all_strategies +
        strategy_combinations.multi_worker_mirrored_strategies +
        strategy_combinations.parameter_server_strategies_multi_worker,
        mode="eager"))
class DatasetCreatorModelFitTest(test_base.DatasetCreatorModelFitTestBase):

  def setUp(self):
    super().setUp()
    if test_util.is_xla_enabled():
      self.skipTest("model.optimizer.iterations values is not as expected "
                    "with XLA: b/184384487")

  def testModelFit(self, strategy):
    model = self._model_fit(strategy)
    self.assertEqual(model.optimizer.iterations, 100)

  def testModelFitWithLookupLayer(self, strategy):
    model = self._model_fit(strategy, use_lookup_layer=True)
    self.assertEqual(model.optimizer.iterations, 100)

  def testModelFitWithNormalizationLayer(self, strategy):
    model = self._model_fit(strategy, with_normalization_layer=True)
    self.assertEqual(model.optimizer.iterations, 100)

  def testModelFitWithStepsPerExecution(self, strategy):
    model = self._model_fit(strategy, steps_per_execution=10)
    self.assertEqual(model.optimizer.iterations, 100)

  def testModelFitWithNoStepsPerEpoch(self, strategy):
    with self.assertRaisesRegex(
        ValueError, "When using a "
        "`tf.keras.utils.experimental.DatasetCreator`, `steps_per_epoch`, "
        "`validation_steps` or `steps` argument must be provided in "
        "`Model.fit` or `Model.evaluate`."):
      self._model_fit(strategy, steps_per_epoch=None)

  def testModelEvaluate(self, strategy):
    self._model_evaluate(strategy)
    self.assertGreaterEqual(self._metric.result(), 0.0)

  def testModelEvaluateWithNormalizationLayer(self, strategy):
    self._model_evaluate(strategy, with_normalization_layer=True)
    self.assertGreaterEqual(self._metric.result(), 0.0)

  def testModelEvaluateWithStepsPerExecution(self, strategy):
    self._model_evaluate(strategy, steps_per_execution=10)
    self.assertGreaterEqual(self._metric.result(), 0.0)

  def testModelEvaluateWithNoStepsPerEpoch(self, strategy):
    with self.assertRaisesRegex(
        ValueError, "When using a "
        "`tf.keras.utils.experimental.DatasetCreator`, `steps_per_epoch`, "
        "`validation_steps` or `steps` argument must be provided in "
        "`Model.fit` or `Model.evaluate`."):
      self._model_evaluate(strategy, steps=None)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
