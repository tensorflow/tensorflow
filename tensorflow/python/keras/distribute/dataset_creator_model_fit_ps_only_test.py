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
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import callbacks as callbacks_lib
from tensorflow.python.keras.distribute import dataset_creator_model_fit_test_base as test_base
from tensorflow.python.keras.distribute import strategy_combinations
from tensorflow.python.platform import gfile


@ds_combinations.generate(
    combinations.combine(
        strategy=strategy_combinations.parameter_server_strategies_multi_worker,
        mode="eager"))
class DatasetCreatorModelFitParameterServerStrategyOnlyTest(
    test_base.DatasetCreatorModelFitTestBase):

  def testModelFitWithRunEagerly(self, strategy):
    with self.assertRaisesRegex(
        ValueError, "When using `Model` with `ParameterServerStrategy`, "
        "`run_eagerly` is not supported."):
      self._model_fit(strategy, run_eagerly=True)

  def testModelFitWithDatasetInstance(self, strategy):
    with self.assertRaisesRegex(
        NotImplementedError,
        "Only `tf.keras.utils.experimental.DatasetCreator` input is supported "
        "with `ParameterServerStrategy` at this time. Please see "
        "`tf.keras.utils.experimental.DatasetCreator` class docstring for "
        "more information."):
      self._model_fit(
          strategy, x=dataset_ops.DatasetV2.from_tensor_slices([1, 1]))

  def testModelPredict(self, strategy):
    model, _ = self._model_compile(strategy)
    with self.assertRaisesRegex(
        NotImplementedError, "`model.predict` is not yet supported with "
        "`ParameterServerStrategy`."):
      model.predict(x=dataset_ops.DatasetV2.from_tensor_slices([1, 1]))

  def testClusterCoordinatorSingleInstance(self, strategy):
    model = self._model_fit(strategy)
    strategy = model.distribute_strategy
    self.assertIs(strategy._cluster_coordinator,
                  coordinator_lib.ClusterCoordinator(strategy))

  def testModelFitErrorOnBatchLevelCallbacks(self, strategy):

    class BatchLevelCallback(callbacks_lib.Callback):

      def on_train_batch_end(self, batch, logs=None):
        pass

    with self.assertRaisesRegex(ValueError,
                                "Batch-level `Callback`s are not supported"):
      callbacks = [BatchLevelCallback()]
      self._model_fit(strategy, callbacks=callbacks)

  def testModelFitCallbackSupportsTFLogs(self, strategy):

    class MyCallback(callbacks_lib.Callback):

      def __init__(self):
        super(MyCallback, self).__init__()
        # Fetches the RemoteValues if necessary.
        self._supports_tf_logs = True

      def on_train_batch_end(self, batch, logs=None):
        assert isinstance(logs, coordinator_lib.RemoteValue)

    my_callback = MyCallback()
    callbacks = [my_callback]
    self._model_fit(strategy, callbacks=callbacks)

  def testModelFitVerbosity(self, strategy):

    class MyCallback(callbacks_lib.Callback):
      pass

    my_callback = MyCallback()
    callbacks = [my_callback]
    self._model_fit(strategy, callbacks=callbacks)
    # PSStrategy should default to epoch-level logging.
    self.assertEqual(my_callback.params["verbose"], 2)

  def testModelFitTensorBoardEpochLevel(self, strategy):
    log_dir = self.get_temp_dir()
    callbacks = [callbacks_lib.TensorBoard(log_dir)]
    self._model_fit(strategy, callbacks=callbacks)
    self.assertTrue(gfile.Exists(log_dir))
    files = gfile.ListDirectory(log_dir)
    self.assertGreaterEqual(len(files), 1)

  def testModelEvaluateWithDatasetInstance(self, strategy):
    with self.assertRaisesRegex(
        NotImplementedError,
        "Only `tf.keras.utils.experimental.DatasetCreator` input is supported "
        "with `ParameterServerStrategy` at this time. Please see "
        "`tf.keras.utils.experimental.DatasetCreator` class docstring for more "
        "information."):
      self._model_evaluate(
          strategy,
          validation_data=dataset_ops.DatasetV2.from_tensor_slices([1, 1]))

  def testModelEvaluateErrorOnBatchLevelCallbacks(self, strategy):

    class BatchLevelCallback(callbacks_lib.Callback):

      def on_train_batch_end(self, batch, logs=None):
        pass

    with self.assertRaisesRegex(ValueError,
                                "Batch-level `Callback`s are not supported"):
      callbacks = [BatchLevelCallback()]
      self._model_evaluate(strategy, callbacks=callbacks)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
