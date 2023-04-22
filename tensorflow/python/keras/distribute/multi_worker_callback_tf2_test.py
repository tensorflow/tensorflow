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

import json
import os

from absl.testing import parameterized

from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.distribute import distributed_file_utils
from tensorflow.python.keras.distribute import multi_worker_testing_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test


def checkpoint_exists(filepath):
  """Returns whether the checkpoint `filepath` refers to exists."""
  if filepath.endswith('.h5'):
    return file_io.file_exists_v2(filepath)
  tf_saved_model_exists = file_io.file_exists_v2(filepath)
  tf_weights_only_checkpoint_exists = file_io.file_exists_v2(
      filepath + '.index')
  return tf_saved_model_exists or tf_weights_only_checkpoint_exists


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


def get_tf_config_task():
  return json.loads(os.environ['TF_CONFIG'])['task']


def get_tf_config_cluster_spec():
  return json.loads(os.environ['TF_CONFIG'])['cluster']


def get_task_type():
  return get_tf_config_task()['type']


def get_task_index():
  return get_tf_config_task()['index']


def is_chief():
  return ('chief' not in get_tf_config_cluster_spec() and
          get_task_type() == 'worker' and get_task_index() == 0)


class KerasCallbackMultiProcessTest(parameterized.TestCase, test.TestCase):

  @ds_combinations.generate(
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
      task_config = get_tf_config_task()
      saving_filepath = os.path.join(
          test_obj.get_temp_dir(), 'checkpoint_%s_%d%s' %
          (task_config['type'], task_config['index'], extension))

      # The saving_filepath shouldn't exist at the beginning (as it's unique).
      test_obj.assertFalse(checkpoint_exists(saving_filepath))

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
      test_obj.assertEqual(checkpoint_exists(saving_filepath), is_chief())

      # If it's chief, the model should be saved (`write_filepath` should
      # simply return `saving_filepath`); if not, i.e. for non-chief workers,
      # the temporary path generated by `write_filepath` should no longer
      # contain the checkpoint that has been deleted.
      test_obj.assertEqual(
          checkpoint_exists(
              distributed_file_utils.write_filepath(
                  saving_filepath, model._distribution_strategy)), is_chief())

    multi_process_runner.run(
        proc_model_checkpoint_saves_on_chief_but_not_otherwise,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self, file_format))

  @ds_combinations.generate(combinations.combine(mode=['eager']))
  def test_model_checkpoint_works_with_same_file_path(self, mode):

    def proc_model_checkpoint_works_with_same_file_path(
        test_obj, saving_filepath):
      model, _, train_ds, steps = _model_setup(test_obj, file_format='')
      num_epoch = 2

      # The saving_filepath shouldn't exist at the beginning (as it's unique).
      test_obj.assertFalse(file_io.file_exists_v2(saving_filepath))

      model.fit(
          x=train_ds,
          epochs=num_epoch,
          steps_per_epoch=steps,
          callbacks=[callbacks.ModelCheckpoint(filepath=saving_filepath)])

      test_obj.assertTrue(file_io.file_exists_v2(saving_filepath))

    saving_filepath = os.path.join(self.get_temp_dir(), 'checkpoint')

    multi_process_runner.run(
        proc_model_checkpoint_works_with_same_file_path,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self, saving_filepath))

  @ds_combinations.generate(combinations.combine(mode=['eager']))
  def test_backupandrestore_checkpoint_works_with_interruption(self, mode):

    class InterruptingCallback(callbacks.Callback):

      def on_epoch_begin(self, epoch, logs=None):
        if epoch == 2:
          raise RuntimeError('Interrupting!')

    class AssertCallback(callbacks.Callback):

      def on_epoch_begin(self, epoch, logs=None):
        # the interruption happened on epoch 2 as specified in
        # InterruptingCallback, so the initial epoch after restart will begin
        # at 2.
        assert epoch > 1

    def proc_model_checkpoint_works_with_same_file_path(test_obj,
                                                        saving_filepath):
      model, _, train_ds, steps = _model_setup(test_obj, file_format='')
      num_epoch = 4

      # The saving_filepath shouldn't exist at the beginning (as it's unique).
      test_obj.assertFalse(file_io.file_exists_v2(saving_filepath))
      bar_dir = os.path.join(os.path.dirname(saving_filepath), 'backup')

      try:
        model.fit(
            x=train_ds,
            epochs=num_epoch,
            steps_per_epoch=steps,
            callbacks=[
                callbacks.ModelCheckpoint(filepath=saving_filepath),
                callbacks.BackupAndRestore(backup_dir=bar_dir),
                InterruptingCallback()
            ])
      except RuntimeError as e:
        if 'Interrupting!' not in str(e):
          raise

      multi_process_runner.get_barrier().wait()
      backup_filepath = os.path.join(bar_dir, 'chief', 'checkpoint')
      test_obj.assertTrue(file_io.file_exists_v2(backup_filepath))
      test_obj.assertTrue(file_io.file_exists_v2(saving_filepath))

      model.fit(
          x=train_ds,
          epochs=num_epoch,
          steps_per_epoch=steps,
          callbacks=[
              callbacks.ModelCheckpoint(filepath=saving_filepath),
              callbacks.BackupAndRestore(backup_dir=bar_dir),
              AssertCallback()
          ])
      multi_process_runner.get_barrier().wait()
      test_obj.assertFalse(file_io.file_exists_v2(backup_filepath))
      test_obj.assertTrue(file_io.file_exists_v2(saving_filepath))

    saving_filepath = os.path.join(self.get_temp_dir(), 'checkpoint')

    multi_process_runner.run(
        proc_model_checkpoint_works_with_same_file_path,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self, saving_filepath))

  @ds_combinations.generate(combinations.combine(mode=['eager']))
  def test_tensorboard_saves_on_chief_but_not_otherwise(self, mode):

    def proc_tensorboard_saves_on_chief_but_not_otherwise(test_obj):
      model, _, train_ds, steps = _model_setup(test_obj, file_format='')
      num_epoch = 2

      # Incorporate type/index information and thread id in saving_filepath to
      # ensure every worker has a unique path. Note that in normal use case the
      # saving_filepath will be the same for all workers, but we use different
      # ones here just to test out chief saves summaries but non-chief doesn't.
      task_config = get_tf_config_task()
      saving_filepath = os.path.join(
          test_obj.get_temp_dir(),
          'logfile_%s_%d' % (task_config['type'], task_config['index']))

      # The saving_filepath shouldn't exist at the beginning (as it's unique).
      test_obj.assertFalse(file_io.file_exists_v2(saving_filepath))

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
          bool(file_io.list_directory_v2(saving_filepath)), is_chief())

    multi_process_runner.run(
        proc_tensorboard_saves_on_chief_but_not_otherwise,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self,))

  @ds_combinations.generate(combinations.combine(mode=['eager']))
  def test_tensorboard_can_still_save_to_temp_even_if_it_exists(self, mode):

    def proc_tensorboard_can_still_save_to_temp_even_if_it_exists(test_obj):
      model, _, train_ds, steps = _model_setup(test_obj, file_format='')
      num_epoch = 2

      saving_filepath = os.path.join(
          test_obj.get_temp_dir(),
          'logfile_%s' % (get_tf_config_task()['type']))

      saving_filepath_for_temp = os.path.join(saving_filepath, 'workertemp_1')
      os.mkdir(saving_filepath)
      os.mkdir(saving_filepath_for_temp)

      # Verifies that even if `saving_filepath_for_temp` exists, tensorboard
      # can still save to temporary directory.
      test_obj.assertTrue(file_io.file_exists_v2(saving_filepath_for_temp))

      model.fit(
          x=train_ds,
          epochs=num_epoch,
          steps_per_epoch=steps,
          callbacks=[callbacks.TensorBoard(log_dir=saving_filepath)])

    multi_process_runner.run(
        proc_tensorboard_can_still_save_to_temp_even_if_it_exists,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self,))

  @ds_combinations.generate(combinations.combine(mode=['eager']))
  def test_tensorboard_works_with_same_file_path(self, mode):

    def proc_tensorboard_works_with_same_file_path(test_obj, saving_filepath):
      model, _, train_ds, steps = _model_setup(test_obj, file_format='')
      num_epoch = 2

      # The saving_filepath shouldn't exist at the beginning (as it's unique).
      test_obj.assertFalse(file_io.file_exists_v2(saving_filepath))

      multi_process_runner.get_barrier().wait()

      model.fit(
          x=train_ds,
          epochs=num_epoch,
          steps_per_epoch=steps,
          callbacks=[callbacks.TensorBoard(log_dir=saving_filepath)])

      multi_process_runner.get_barrier().wait()

      test_obj.assertTrue(file_io.list_directory_v2(saving_filepath))

    saving_filepath = os.path.join(self.get_temp_dir(), 'logfile')

    multi_process_runner.run(
        proc_tensorboard_works_with_same_file_path,
        cluster_spec=test_base.create_cluster_spec(num_workers=2),
        args=(self, saving_filepath))

  @ds_combinations.generate(combinations.combine(mode=['eager']))
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
  multi_process_runner.test_main()
