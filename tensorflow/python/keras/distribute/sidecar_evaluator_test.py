# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test covering sidecar_evaluator.py."""

import enum
import os

from absl import logging
from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras.distribute import sidecar_evaluator as sidecar_evaluator_lib
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.tracking import util as tracking_util

_BATCH_SIZE = 32


class TestModel(keras.Model):

  def __init__(self):
    super().__init__(name='test_model')
    self.dense = keras.layers.Dense(10)

  def call(self, inputs):
    return self.dense(inputs)


class DictMetric(keras.metrics.MeanSquaredError):

  def result(self):
    res = super().result()
    return {'mean_squared_error_1': res, 'mean_squared_error_2': res}


class ModelType(enum.Enum):
  SEQUENTIAL = 'sequential'
  SUBCLASS = 'subclass'


def _test_model_builder(model_type: ModelType, compile_model, build_model):
  if model_type == ModelType.SEQUENTIAL:
    model = keras.Sequential([keras.layers.Dense(10)])
  elif model_type == ModelType.SUBCLASS:
    model = TestModel()

  if compile_model:
    model.compile(
        gradient_descent.SGD(),
        loss='mse',
        metrics=[keras.metrics.CategoricalAccuracy(),
                 DictMetric()])
  if build_model:
    model.build((None, 32))

  return model


class SidecarEvaluatorTest(test.TestCase, parameterized.TestCase):

  def assertSummaryEventsWritten(self, log_dir):
    # Asserts summary files do get written when log_dir is provided.
    summary_files = file_io.list_directory_v2(log_dir)
    self.assertNotEmpty(
        summary_files, 'Summary should have been written and '
        'log_dir should not be empty.')

    # Asserts the content of the summary file.
    event_pb_written = False
    event_tags = []
    for summary_file in summary_files:
      for event_pb in summary_iterator.summary_iterator(
          os.path.join(log_dir, summary_file)):
        if event_pb.step > 0:
          self.assertEqual(event_pb.step, 32)
          event_tags.append(event_pb.summary.value[0].tag)
          event_pb_written = True
    self.assertCountEqual(event_tags, [
        'evaluation_categorical_accuracy_vs_iterations',
        'evaluation_loss_vs_iterations',
        'evaluation_mean_squared_error_1_vs_iterations',
        'evaluation_mean_squared_error_2_vs_iterations',
    ])

    # Verifying at least one non-zeroth step is written to summary.
    self.assertTrue(event_pb_written)

  def assertModelsSameVariables(self, model_a, model_b):
    # Check both have the same number of variables.
    self.assertEqual(len(model_a.variables), len(model_b.variables))

    # Check variable values to be equal.
    for var_a, var_b in zip(model_a.variables, model_b.variables):
      self.assertAllEqual(var_a.numpy(), var_b.numpy())

  @ds_combinations.generate(
      combinations.combine(
          mode=['eager'], model_type=[ModelType.SEQUENTIAL,
                                      ModelType.SUBCLASS]))
  def testIterationsNotSavedWillRaiseError(self, model_type):
    model = _test_model_builder(
        model_type=model_type, compile_model=False, build_model=True)

    checkpoint_dir = self.get_temp_dir()
    checkpoint = tracking_util.Checkpoint(model=model)
    checkpoint_manager = checkpoint_management.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=2)
    checkpoint_manager.save()

    sidecar_evaluator = sidecar_evaluator_lib.SidecarEvaluator(
        model, data=None, checkpoint_dir=checkpoint_dir)
    with self.assertRaisesRegex(
        RuntimeError, '`iterations` cannot be loaded '
        'from the checkpoint file.'):
      sidecar_evaluator.start()

  @ds_combinations.generate(
      combinations.combine(
          mode=['eager'], model_type=[ModelType.SEQUENTIAL,
                                      ModelType.SUBCLASS]))
  def testModelNotBuiltRaiseError(self, model_type):
    model = _test_model_builder(
        model_type=model_type, compile_model=False, build_model=False)

    checkpoint_dir = self.get_temp_dir()
    checkpoint = tracking_util.Checkpoint(model=model)
    checkpoint_manager = checkpoint_management.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=2)
    checkpoint_manager.save()

    sidecar_evaluator = sidecar_evaluator_lib.SidecarEvaluator(
        model, data=None, checkpoint_dir=checkpoint_dir)
    with self.assertRaisesRegex(AssertionError, 'Nothing to load.'):
      sidecar_evaluator.start()

  @ds_combinations.generate(
      combinations.combine(
          mode=['eager'],
          model_type=[ModelType.SEQUENTIAL, ModelType.SUBCLASS],
          build_model=[True, False]))
  def testSidecarEvaluatorOutputsSummary(self, model_type, build_model):
    # Create a model with synthetic data, and fit for one epoch.
    model = _test_model_builder(
        model_type=model_type, compile_model=True, build_model=False)
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    dataset = dataset_ops.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    model.fit(dataset, epochs=1)

    # Save a checkpoint.
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'ckpt')
    log_dir = os.path.join(self.get_temp_dir(), 'summary')
    logging.info('checkpoint_dir = %s, log_dir = %s', checkpoint_dir, log_dir)
    checkpoint = tracking_util.Checkpoint(
        model=model, optimizer=model.optimizer)
    checkpoint_manager = checkpoint_management.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=2)
    logging.info('Checkpoint manager saved to: %s', checkpoint_manager.save())
    self.assertNotEmpty(
        file_io.list_directory_v2(checkpoint_dir),
        'Checkpoint should have been written and '
        'checkpoint_dir should not be empty.')

    # Create a new model used for evaluation.
    eval_model = _test_model_builder(
        model_type=model_type, compile_model=True, build_model=build_model)
    # Have a sidecar_evaluator evaluate once.
    sidecar_evaluator = sidecar_evaluator_lib.SidecarEvaluator(
        eval_model,
        data=dataset,
        checkpoint_dir=checkpoint_dir,
        max_evaluations=1,
        callbacks=[keras.callbacks.TensorBoard(log_dir=log_dir)])
    sidecar_evaluator.start()
    # Eval model has been restored to the same state as the original model, so
    # their weights should match. If not, restoration of the model didn't
    # work.
    self.assertModelsSameVariables(model, eval_model)

    self.assertSummaryEventsWritten(os.path.join(log_dir, 'validation'))

  @ds_combinations.generate(
      combinations.combine(
          mode=['eager'],
          model_type=[ModelType.SEQUENTIAL, ModelType.SUBCLASS],
          build_model=[True, False]))
  def testSidecarEvaluatorOutputsSummarySavedWithCallback(
      self, model_type, build_model):
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'checkpoints')
    log_dir = os.path.join(self.get_temp_dir(), 'summary')
    # Create a model with synthetic data, and fit for one epoch.
    model = _test_model_builder(
        model_type=model_type, compile_model=True, build_model=False)
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    dataset = dataset_ops.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(_BATCH_SIZE)
    save_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'ckpt-{epoch}'),
        save_weights_only=True)
    model.fit(dataset, epochs=1, callbacks=[save_callback])
    self.assertNotEmpty(
        file_io.list_directory_v2(checkpoint_dir),
        'Checkpoint should have been written and '
        'checkpoint_dir should not be empty.')

    # Create a new model used for evaluation.
    eval_model = _test_model_builder(
        model_type=model_type, compile_model=True, build_model=build_model)
    # Have an sidecar_evaluator evaluate once.
    sidecar_evaluator = sidecar_evaluator_lib.SidecarEvaluator(
        eval_model,
        data=dataset,
        checkpoint_dir=checkpoint_dir,
        max_evaluations=1,
        callbacks=[keras.callbacks.TensorBoard(log_dir=log_dir)])
    with self.assertLogs() as cm:
      sidecar_evaluator.start()

    metrics_logging = [
        line for line in cm.output if 'End of evaluation' in line
    ]
    self.assertLen(metrics_logging, 1)
    expected_logged_metrics = [
        'loss', 'categorical_accuracy', 'mean_squared_error_1',
        'mean_squared_error_2'
    ]
    for metric_name in expected_logged_metrics:
      self.assertRegex(metrics_logging[0], f'{metric_name}=')

    # Eval model has been restored to the same state as the original model, so
    # their weights should match. If not, restoration of the model didn't
    # work.
    self.assertModelsSameVariables(model, eval_model)

    # check the iterations is restored.
    self.assertEqual(sidecar_evaluator._iterations.numpy(), _BATCH_SIZE)

    self.assertSummaryEventsWritten(os.path.join(log_dir, 'validation'))


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
