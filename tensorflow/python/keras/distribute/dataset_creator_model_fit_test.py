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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from absl.testing import parameterized
import numpy as np
from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.framework import config
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import callbacks as callbacks_lib
from tensorflow.python.keras.distribute import multi_worker_testing_utils
from tensorflow.python.keras.distribute import strategy_combinations
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.layers import core as core_layers
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class DatasetCreatorModelFitTestBase(test.TestCase, parameterized.TestCase):

  def _model_compile(self,
                     strategy,
                     steps_per_execution=1,
                     run_eagerly=False,
                     with_normalization_layer=False):

    class ResultAssertingCallback(callbacks_lib.Callback):

      def __init__(self):
        self._prev_epoch = -1
        self._loss_to_compare_against = 2  # Empirical initial value

      def on_epoch_end(self, epoch, logs=None):
        logging.info("testModelFit: epoch=%r, logs=%r", epoch, logs)
        if epoch <= self._prev_epoch:
          raise RuntimeError("Epoch is supposed to be larger than previous.")
        self._prev_epoch = epoch
        is_loss_float = (
            logs.get("loss", None) is not None and
            isinstance(logs["loss"], (float, np.floating)))
        if not is_loss_float:
          raise RuntimeError("loss is supposed to be in the logs and float.")
        if epoch == 0 or epoch == 9:
          # Making sure the loss of first epoch is below 1, and that of last
          # epoch is smaller than the first epoch.
          if logs["loss"] > self._loss_to_compare_against:
            raise RuntimeError(
                "loss at epoch {} is larger than previous.".format(epoch))
          self._loss_to_compare_against = logs["loss"]

      def on_train_end(self, logs=None):
        if self._prev_epoch != 9:
          raise RuntimeError("Unexpected last epoch: {}".format(
              self._prev_epoch))

    # TODO(b/182193218): Use ParameterServerStrategy as a proper strategy
    # combination.
    if strategy == "ParameterServerStrategy":
      gpu_devices = config.list_physical_devices("GPU")
      if len(gpu_devices) > 1:
        self.skipTest("b/178452835: Multi-GPUs not supported in "
                      "ParameterServerStrategy.")
      strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
          multi_worker_testing_utils.make_parameter_server_cluster(3, 2),
          variable_partitioner=sharded_variable.FixedShardsPartitioner(2))

    with strategy.scope():
      model = sequential.Sequential([core_layers.Dense(10)])
      if with_normalization_layer:
        norm = keras.layers.BatchNormalization(
            axis=-1, input_shape=(4, 4, 3), momentum=0.8)
        model.add(norm)

    model.compile(
        gradient_descent.SGD(),
        loss="mse",
        steps_per_execution=steps_per_execution,
        run_eagerly=run_eagerly)
    return model, [ResultAssertingCallback()]

  def _model_fit(self,
                 strategy,
                 steps_per_execution=1,
                 validation_data=None,
                 x=None,
                 steps_per_epoch=10,
                 run_eagerly=False,
                 with_normalization_layer=False):
    model, callbacks = self._model_compile(strategy, steps_per_execution,
                                           run_eagerly,
                                           with_normalization_layer)

    def dataset_fn(input_context):
      del input_context
      x = random_ops.random_uniform((10, 10))
      y = random_ops.random_uniform((10,))
      return dataset_ops.DatasetV2.from_tensor_slices(
          (x, y)).shuffle(10).repeat().batch(2)

    x = x or dataset_creator.DatasetCreator(dataset_fn)

    model.fit(
        x,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        verbose=0,
        callbacks=callbacks,
        validation_data=validation_data)
    return model


@ds_combinations.generate(
    combinations.combine(
        strategy=strategy_combinations.all_strategies +
        strategy_combinations.multi_worker_mirrored_strategies +
        ["ParameterServerStrategy"],
        mode="eager"))
class DatasetCreatorModelFitTest(DatasetCreatorModelFitTestBase):

  def testModelFit(self, strategy):
    model = self._model_fit(strategy)
    self.assertEqual(model.optimizer.iterations, 100)
    return model

  def testModelFitWithNormalizationLayer(self, strategy):
    model = self._model_fit(strategy, with_normalization_layer=True)
    self.assertEqual(model.optimizer.iterations, 100)

  def testModelFitWithStepsPerExecution(self, strategy):
    model = self._model_fit(strategy, steps_per_execution=10)
    self.assertEqual(model.optimizer.iterations, 100)


@ds_combinations.generate(
    combinations.combine(strategy=["ParameterServerStrategy"], mode="eager"))
class DatasetCreatorModelFitParameterServerStrategyOnlyTest(
    DatasetCreatorModelFitTestBase):

  def testModelFitWithNoStepsPerEpoch(self, strategy):
    with self.assertRaisesRegex(
        ValueError, "`steps_per_epoch` must be specified with "
        "`ParameterServerStrategy`."):
      self._model_fit(strategy, steps_per_epoch=None)

  def testModelFitWithRunEagerly(self, strategy):
    with self.assertRaisesRegex(
        ValueError, "When using `Model` with `ParameterServerStrategy`, "
        "`run_eagerly` is not supported."):
      self._model_fit(strategy, run_eagerly=True)

  def testModelFitWithValidationData(self, strategy):
    with self.assertRaisesRegex(
        NotImplementedError, "Evaluation in `model.fit` with "
        "`ParameterServerStrategy` is not yet supported."):
      self._model_fit(
          strategy,
          validation_data=dataset_ops.DatasetV2.from_tensor_slices([1, 1]))

  def testModelFitWithDatasetInstance(self, strategy):
    with self.assertRaisesRegex(
        NotImplementedError, "Only `DatasetCreator` input is supported in "
        "`ParameterServerStrategy` at this time."):
      self._model_fit(
          strategy, x=dataset_ops.DatasetV2.from_tensor_slices([1, 1]))

  def testModelEvaluate(self, strategy):
    model, _ = self._model_compile(strategy)
    with self.assertRaisesRegex(
        NotImplementedError, "`model.evaluate` is not yet supported with "
        "`ParameterServerStrategy`."):
      model.evaluate(x=dataset_ops.DatasetV2.from_tensor_slices([1, 1]))

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


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
