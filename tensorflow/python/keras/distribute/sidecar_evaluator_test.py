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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging
import numpy as np
from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.distribute import sidecar_evaluator as sidecar_evaluator_lib
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.tracking import util as tracking_util


class SidecarEvaluatorTest(test.TestCase):

  def testIterationsNotSavedWillRaiseError(self):
    model = keras.Sequential([keras.layers.Dense(10)])

    checkpoint_dir = self.get_temp_dir()
    checkpoint = tracking_util.Checkpoint(model=model)
    checkpoint_manager = checkpoint_management.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=2)
    checkpoint_manager.save()

    sidecar_evaluator = sidecar_evaluator_lib.SidecarEvaluator(
        model, data=None, checkpoint_dir=checkpoint_dir, log_dir=None)
    with self.assertRaisesRegexp(
        RuntimeError, '`iterations` cannot be loaded '
        'from the checkpoint file.'):
      sidecar_evaluator.start()

  def testSidecarEvaluatorOutputsSummary(self):
    # Create a model with synthetic data, and fit for one epoch.
    model = keras.models.Sequential([keras.layers.Dense(10)])
    model.compile(
        gradient_descent.SGD(),
        loss='mse',
        metrics=keras.metrics.CategoricalAccuracy())
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

    # Have an sidecar_evaluator evaluate once.
    sidecar_evaluator_lib.SidecarEvaluator(
        model,
        data=dataset,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        max_evaluations=1).start()

    # Asserts summary files do get written when log_dir is provided.
    summary_files = file_io.list_directory_v2(log_dir)
    self.assertNotEmpty(
        file_io.list_directory_v2(checkpoint_dir),
        'Checkpoint should have been written and '
        'checkpoint_dir should not be empty.')
    self.assertNotEmpty(
        summary_files, 'Summary should have been written and '
        'log_dir should not be empty.')

    # Asserts the content of the summary file.
    event_pb_written = False
    for event_pb in summary_iterator.summary_iterator(
        os.path.join(log_dir, summary_files[0])):
      if event_pb.step > 0:
        self.assertEqual(event_pb.step, 32)
        self.assertEqual(event_pb.summary.value[0].tag, 'categorical_accuracy')
        event_pb_written = True

    # Verifying at least one non-zeroth step is written to summary.
    self.assertTrue(event_pb_written)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
