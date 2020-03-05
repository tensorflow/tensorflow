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
"""Tests for Keras callbacks in multi-worker training with TF2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from absl.testing import parameterized

from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.distribute import multi_worker_testing_utils
from tensorflow.python.keras.distribute import multi_worker_training_state as training_state
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test


def _model_setup(test_obj, file_format):
  """Set up a MNIST Keras model for testing purposes.

  This function builds a MNIST Keras model and returns relevant information
  for testing.

  Args:
    test_obj: The `TestCase` testing object.
    file_format: File format for checkpoints. 'tf' or 'h5'.

  Returns:
    A tuple of (model, saving_filepath, train_ds, steps) where train_ds is
    the training dataset.
  """
  batch_size = 64
  steps = 2
  with collective_strategy.CollectiveAllReduceStrategy().scope():
    # TODO(b/142509827): In rare cases this errors out at C++ level with the
    # "Connect failed" error message.
    train_ds, _ = multi_worker_testing_utils.mnist_synthetic_dataset(
        batch_size, steps)
    model = multi_worker_testing_utils.get_mnist_model((28, 28, 1))
  # Pass saving_filepath from the parent thread to ensure every worker has the
  # same filepath to save.
  saving_filepath = os.path.join(test_obj.get_temp_dir(),
                                 'checkpoint.' + file_format)
  return model, saving_filepath, train_ds, steps


class KerasCallbackMultiProcessTest(parameterized.TestCase, test.TestCase):

  @combinations.generate(
      combinations.combine(
          mode=['eager'],
          file_format=['h5', 'tf'],
          save_weights_only=[True, False]))
  def test_model_checkpoint_saves_on_chief_but_not_otherwise(
      self, file_format, mode, save_weights_only):

    def proc_model_checkpoint_saves_on_chief_but_not_otherwise(
        test_obj, file_format):

      model, saving_filepath, train_ds, steps = _model_setup(
          test_obj, file_format)
      num_epoch = 2
      extension = os.path.splitext(saving_filepath)[1]

      # Incorporate type/index information and thread id in saving_filepath to
      # ensure every worker has a unique path. Note that in normal use case the
      # saving_filepath will be the same for all workers, but we use different
      # ones here just to test out chief saves checkpoint but non-chief doesn't.
      saving_filepath = os.path.join(
          test_obj.get_temp_dir(), 'checkpoint_%s_%d%s' %
          (test_base.get_task_type(), test_base.get_task_index(), extension))

      # The saving_filepath shouldn't exist at the beginning (as it's unique).
      test_obj.assertFalse(training_state.checkpoint_exists(saving_filepath))

      model.fit(
          x=train_ds,
          epochs=num_epoch,
          steps_per_epoch=steps,
          validation_data=train_ds,
          validation_steps=steps,
          callbacks=[
              callbacks.ModelCheckpoint(
                  filepath=saving_filepath, save_weights_only=save_weights_only)
          ])

      # If it's chief, the model should be saved; if not, the model shouldn't.
      test_obj.assertEqual(
          training_state.checkpoint_exists(saving_filepath),
          test_base.is_chief())

    multi_process_runner.run(
        proc_model_checkpoint_saves_on_chief_but_not_otherwise,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self, file_format))

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_model_checkpoint_works_with_same_file_path(self, mode):

    def proc_model_checkpoint_works_with_same_file_path(
        test_obj, saving_filepath):
      model, _, train_ds, steps = _model_setup(test_obj, file_format='')
      num_epoch = 2

      # The saving_filepath shouldn't exist at the beginning (as it's unique).
      test_obj.assertFalse(file_io.file_exists(saving_filepath))

      model.fit(
          x=train_ds,
          epochs=num_epoch,
          steps_per_epoch=steps,
          callbacks=[callbacks.ModelCheckpoint(filepath=saving_filepath)])

      test_obj.assertTrue(file_io.file_exists(saving_filepath))

    saving_filepath = os.path.join(self.get_temp_dir(), 'checkpoint')

    multi_process_runner.run(
        proc_model_checkpoint_works_with_same_file_path,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self, saving_filepath))

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_tensorboard_saves_on_chief_but_not_otherwise(self, mode):

    def proc_tensorboard_saves_on_chief_but_not_otherwise(test_obj):
      model, _, train_ds, steps = _model_setup(test_obj, file_format='')
      num_epoch = 2

      # Incorporate type/index information and thread id in saving_filepath to
      # ensure every worker has a unique path. Note that in normal use case the
      # saving_filepath will be the same for all workers, but we use different
      # ones here just to test out chief saves summaries but non-chief doesn't.
      saving_filepath = os.path.join(
          test_obj.get_temp_dir(), 'logfile_%s_%d' %
          (test_base.get_task_type(), test_base.get_task_index()))

      # The saving_filepath shouldn't exist at the beginning (as it's unique).
      test_obj.assertFalse(file_io.file_exists(saving_filepath))

      model.fit(
          x=train_ds,
          epochs=num_epoch,
          steps_per_epoch=steps,
          callbacks=[callbacks.TensorBoard(log_dir=saving_filepath)])

      # If it's chief, the summaries should be saved in the filepath; if not,
      # the directory should be empty (although created). Using
      # `file_io.list_directory()` since the directory may be created at this
      # point.
      test_obj.assertEqual(
          bool(file_io.list_directory(saving_filepath)), test_base.is_chief())

    multi_process_runner.run(
        proc_tensorboard_saves_on_chief_but_not_otherwise,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self,))

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_tensorboard_can_still_save_to_temp_even_if_it_exists(self, mode):

    def proc_tensorboard_can_still_save_to_temp_even_if_it_exists(test_obj):
      model, _, train_ds, steps = _model_setup(test_obj, file_format='')
      num_epoch = 2

      saving_filepath = os.path.join(test_obj.get_temp_dir(),
                                     'logfile_%s' % (test_base.get_task_type()))

      saving_filepath_for_temp = os.path.join(saving_filepath, 'workertemp_1')
      os.mkdir(saving_filepath)
      os.mkdir(saving_filepath_for_temp)

      # Verifies that even if `saving_filepath_for_temp` exists, tensorboard
      # can still save to temporary directory.
      test_obj.assertTrue(file_io.file_exists(saving_filepath_for_temp))

      model.fit(
          x=train_ds,
          epochs=num_epoch,
          steps_per_epoch=steps,
          callbacks=[callbacks.TensorBoard(log_dir=saving_filepath)])

    multi_process_runner.run(
        proc_tensorboard_can_still_save_to_temp_even_if_it_exists,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self,))

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_tensorboard_works_with_same_file_path(self, mode):

    def proc_tensorboard_works_with_same_file_path(test_obj, saving_filepath):
      model, _, train_ds, steps = _model_setup(test_obj, file_format='')
      num_epoch = 2

      # The saving_filepath shouldn't exist at the beginning (as it's unique).
      test_obj.assertFalse(file_io.file_exists(saving_filepath))

      multi_process_runner.barrier().wait()

      model.fit(
          x=train_ds,
          epochs=num_epoch,
          steps_per_epoch=steps,
          callbacks=[callbacks.TensorBoard(log_dir=saving_filepath)])

      multi_process_runner.barrier().wait()

      test_obj.assertTrue(file_io.list_directory(saving_filepath))

    saving_filepath = os.path.join(self.get_temp_dir(), 'logfile')

    multi_process_runner.run(
        proc_tensorboard_works_with_same_file_path,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self, saving_filepath))

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_early_stopping(self, mode):

    def proc_early_stopping(test_obj):

      class EpochCounterCallback(callbacks.Callback):

        def on_epoch_begin(self, epoch, logs):
          self.last_epoch = epoch

      model, _, train_ds, steps = _model_setup(test_obj, file_format='')
      epoch_counter_cbk = EpochCounterCallback()
      cbks = [
          callbacks.EarlyStopping(
              monitor='loss', min_delta=0.05, patience=1, verbose=1),
          epoch_counter_cbk
      ]

      # Empirically, it is expected that `model.fit()` terminates around the
      # 22th epoch. Asserting that it should have been stopped before the 50th
      # epoch to avoid flakiness and be more predictable.
      model.fit(x=train_ds, epochs=100, steps_per_epoch=steps, callbacks=cbks)
      test_obj.assertLess(epoch_counter_cbk.last_epoch, 50)

    multi_process_runner.run(
        proc_early_stopping,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self,))


if __name__ == '__main__':
  multi_process_runner.test_main(barrier_parties=2)
