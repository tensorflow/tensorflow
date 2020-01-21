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
"""Tests for Keras callbacks in multi-worker training with TF1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import tempfile
import threading

from absl.testing import parameterized

from tensorflow.python import keras
from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.distribute import multi_worker_testing_utils
from tensorflow.python.keras.distribute import multi_worker_training_state as training_state
from tensorflow.python.platform import test


def get_strategy_object(strategy_cls):
  if strategy_cls == mirrored_strategy.MirroredStrategy:
    return strategy_cls(mirrored_strategy.all_local_devices())
  else:
    # CollectiveAllReduceStrategy and ParameterServerStrategy.
    return strategy_cls()


def generate_callback_test_function(custom_callable):
  """Generic template for callback tests using mnist synthetic dataset."""

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          strategy_cls=[collective_strategy.CollectiveAllReduceStrategy],
          required_gpus=[0, 1],
          file_format=['h5', 'tf']))
  def test_template(self, strategy_cls, file_format):
    num_workers = 2
    num_epoch = 2

    cluster_spec = test_base.create_cluster_spec(num_workers=num_workers)
    self._barrier = dc._Barrier(2)

    def _independent_worker_fn(*args, **kwargs):  # pylint: disable=unused-argument
      """Simulates an Independent Worker inside of a thread."""
      with test.mock.patch.object(dc, '_run_std_server',
                                  self._make_mock_run_std_server()):
        strategy = get_strategy_object(strategy_cls)
        batch_size = 64
        steps = 2
        train_ds, _ = multi_worker_testing_utils.mnist_synthetic_dataset(
            batch_size, steps)
        with strategy.scope():
          model = multi_worker_testing_utils.get_mnist_model((28, 28, 1))

        custom_callable(
            model,
            self,
            train_ds,
            num_epoch,
            steps,
            strategy,
            saving_filepath=kwargs['saving_filepath'],
            barrier=kwargs['barrier'],
            threading_local=kwargs['threading_local'])

    # Pass saving_filepath from the parent thread to ensure every worker has the
    # same fileapth to save.
    saving_filepath = os.path.join(self.get_temp_dir(),
                                   'checkpoint.' + file_format)
    barrier = dc._Barrier(2)
    threading_local = threading.local()
    threads = self.run_multiple_tasks_in_threads(
        _independent_worker_fn,
        cluster_spec,
        saving_filepath=saving_filepath,
        barrier=barrier,
        threading_local=threading_local)
    self.assertFalse(training_state.checkpoint_exists(saving_filepath))

    threads_to_join = []
    strategy = get_strategy_object(strategy_cls)
    if strategy.extended.experimental_between_graph:
      for ts in threads.values():
        threads_to_join.extend(ts)
    else:
      threads_to_join = [threads['worker'][0]]
    self.join_independent_workers(threads_to_join)

  return test_template


class KerasMultiWorkerCallbackTest(test_base.IndependentWorkerTestBase,
                                   parameterized.TestCase):
  """KerasMultiWorkerCallbackTest for TF1.

  TODO(rchao): Migrate all tests in this class to
  `multi_worker_callback_tf2_test`.
  """

  # The callables of the actual testing content to be run go below.
  @staticmethod
  def callableForTestChiefOnlyCallback(model, test_obj, train_ds, num_epoch,
                                       steps, strategy, saving_filepath,
                                       **kwargs):

    class ChiefOnly(keras.callbacks.Callback):

      def __init__(self):
        self._chief_worker_only = True
        self.filtered_correctly = True

      def on_train_begin(self, logs):
        if not multi_worker_util.is_chief():
          # Non-chief workers shouldn't run this callback.
          self.filtered_correctly = False

    cb = ChiefOnly()
    model.fit(
        x=train_ds, epochs=num_epoch, steps_per_epoch=steps, callbacks=[cb])

    test_obj.assertTrue(cb.filtered_correctly)

  @staticmethod
  def callableForTestModelCheckpointSavesOnChiefButNotOtherwise(
      model, test_obj, train_ds, num_epoch, steps, strategy, saving_filepath,
      **kwargs):

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
        callbacks=[callbacks.ModelCheckpoint(filepath=saving_filepath)])

    # If it's chief, the model should be saved; if not, the model shouldn't.
    test_obj.assertEqual(
        training_state.checkpoint_exists(saving_filepath), test_base.is_chief())

  @staticmethod
  def initialFitting(test_obj, model, train_ds, num_epoch, steps,
                     saving_filepath):
    # The saving_filepath shouldn't exist at the beginning.
    test_obj.assertFalse(training_state.checkpoint_exists(saving_filepath))

    model.fit(
        x=train_ds,
        epochs=num_epoch,
        steps_per_epoch=steps,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=saving_filepath, save_weights_only=True)
        ])

    # The saving_filepath should exist after fitting with callback. Both chief
    # and non-chief worker should both see it exists (which was saved only by
    # chief).
    test_obj.assertTrue(training_state.checkpoint_exists(saving_filepath))

    history_after_one_more_epoch = model.fit(
        x=train_ds, epochs=1, steps_per_epoch=steps)

    # The saving_filepath should continue to exist (if it did) after fitting
    # without callback.
    test_obj.assertTrue(training_state.checkpoint_exists(saving_filepath))

    return saving_filepath, history_after_one_more_epoch

  @staticmethod
  def callableForTestLoadWeightFromModelCheckpoint(model, test_obj, train_ds,
                                                   num_epoch, steps, strategy,
                                                   saving_filepath, **kwargs):
    filepaths = []
    real_mkstemp = tempfile.mkstemp
    def mocked_mkstemp():
      # Only non-chief should call tempfile.mkstemp() inside fit() in sync
      # training.
      assert not test_base.is_chief()
      file_handle, temp_file_name = real_mkstemp()
      extension = os.path.splitext(saving_filepath)[1]
      temp_filepath = temp_file_name + extension
      filepaths.append(temp_filepath)
      return file_handle, temp_file_name

    # Mock tempfile.mkstemp() so the filepaths can be stored and verified later.
    with test.mock.patch.object(tempfile, 'mkstemp', mocked_mkstemp):
      saving_filepath, history_after_one_more_epoch = \
          KerasMultiWorkerCallbackTest.initialFitting(
              test_obj, model, train_ds, num_epoch, steps, saving_filepath)

      with strategy.scope():
        model.load_weights(saving_filepath)

      history_after_loading_weight_and_one_more_epoch = model.fit(
          x=train_ds, epochs=1, steps_per_epoch=steps)

      test_obj.assertAllClose(
          history_after_one_more_epoch.history,
          history_after_loading_weight_and_one_more_epoch.history,
          rtol=5e-5)

    # Verify the temp files are indeed removed (no trace left behind).
    for filepath in filepaths:
      assert not training_state.checkpoint_exists(filepath)

  @staticmethod
  def callableForTestModelRestoreCallback(model, test_obj, train_ds, num_epoch,
                                          steps, strategy, saving_filepath,
                                          **kwargs):

    saving_filepath, history_after_one_more_epoch = \
        KerasMultiWorkerCallbackTest.initialFitting(
            test_obj, model, train_ds, num_epoch, steps, saving_filepath)

    # The model should get restored to the weights previously saved, by
    # adding a ModelCheckpoint callback (which results in a
    # _ModelRestoreCallback being added), with load_weights_on_restart=True.
    history_after_model_restoring_and_one_more_epoch = model.fit(
        x=train_ds,
        epochs=1,
        steps_per_epoch=steps,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=saving_filepath,
                save_weights_only=True,
                load_weights_on_restart=True)
        ])

    # Asserting the history one epoch after initial fitting and one epoch after
    # restoring are closed.
    test_obj.assertAllClose(
        history_after_one_more_epoch.history,
        history_after_model_restoring_and_one_more_epoch.history,
        rtol=5e-5)

    history_one_more_epoch_without_model_restoring = model.fit(
        x=train_ds, epochs=1, steps_per_epoch=steps)

    # Ensuring training for another epoch gives different result.
    test_obj.assertNotAllClose(
        history_after_model_restoring_and_one_more_epoch.history,
        history_one_more_epoch_without_model_restoring.history,
        rtol=5e-5)

  @staticmethod
  def callableForTestBackupModelRemoved(model, test_obj, train_ds, num_epoch,
                                        steps, strategy, saving_filepath,
                                        **kwargs):

    # `barrier` object needs to be passed in from parent
    # thread so both threads refer to the same object.
    barrier = kwargs['barrier']

    num_epoch = 3

    # Testing the backup filepath `multi_worker_training_state` uses.
    _, backup_filepath = training_state._get_backup_filepath(saving_filepath)

    # The backup_filepath shouldn't exist at the beginning.
    test_obj.assertFalse(training_state.checkpoint_exists(backup_filepath))

    # Callback to verify that the backup file exists in the middle of training.
    class BackupFilepathVerifyingCallback(callbacks.Callback):

      def on_epoch_begin(self, epoch, logs=None):
        if epoch > 1:
          # Asserting that after the first two epochs, the backup file should
          # exist.
          test_obj.assertTrue(training_state.checkpoint_exists(backup_filepath))

    model.fit(
        x=train_ds,
        epochs=num_epoch,
        steps_per_epoch=steps,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=saving_filepath, save_weights_only=True),
            BackupFilepathVerifyingCallback()
        ])

    # Sync on the two threads so we make sure the backup file is removed before
    # we move on.
    barrier.wait()

    # The back up file should not exist at successful exit of `model.fit()`.
    test_obj.assertFalse(training_state.checkpoint_exists(backup_filepath))

  @staticmethod
  def callableForTestBackupModelNotRemovedIfInterrupted(model, test_obj,
                                                        train_ds, num_epoch,
                                                        steps, strategy,
                                                        saving_filepath,
                                                        **kwargs):

    # `barrier` object needs to be passed in from parent
    # thread so both threads refer to the same object.
    barrier = kwargs['barrier']

    num_epoch = 4

    # Testing the backup filepath `multi_worker_training_state` uses.
    _, backup_filepath = training_state._get_backup_filepath(saving_filepath)

    # The backup_filepath shouldn't exist at the beginning.
    test_obj.assertFalse(training_state.checkpoint_exists(backup_filepath))

    # Callback to interrupt in the middle of training.
    class InterruptingCallback(callbacks.Callback):

      def on_epoch_begin(self, epoch, logs=None):
        if epoch == 2:
          raise RuntimeError('Interrupting!')

    try:
      model.fit(
          x=train_ds,
          epochs=num_epoch,
          steps_per_epoch=steps,
          callbacks=[
              callbacks.ModelCheckpoint(
                  filepath=saving_filepath, save_weights_only=True),
              InterruptingCallback()
          ])
    except RuntimeError as e:
      if 'Interrupting!' not in e.message:
        raise

    # Sync on the two threads.
    barrier.wait()

    # The back up file should exist after interruption of `model.fit()`.
    test_obj.assertTrue(training_state.checkpoint_exists(backup_filepath))

  @staticmethod
  def callableForTestUnmatchedModelFile(model, test_obj, train_ds, num_epoch,
                                        steps, strategy, saving_filepath,
                                        **kwargs):

    # The saving_filepath shouldn't exist at the beginning.
    test_obj.assertFalse(training_state.checkpoint_exists(saving_filepath))

    model.fit(
        x=train_ds,
        epochs=num_epoch,
        steps_per_epoch=steps,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=saving_filepath, save_weights_only=True)
        ])

    (train_ds, _), (_, _) = testing_utils.get_test_data(
        train_samples=10, test_samples=10, input_shape=(3,), num_classes=2)

    # Switch to a model of different structure.
    with strategy.scope():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(5, input_dim=3, activation='relu'))
      model.add(keras.layers.Dense(2, activation='softmax'))
      model.compile(
          loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    test_obj.assertTrue(training_state.checkpoint_exists(saving_filepath))

    if saving_filepath.endswith('.tf'):
      test_obj.skipTest('Loading mismatched TF checkpoint would cause Fatal '
                        'Python error: Aborted. Skipping.')

    # Unmatched format. Should raise ValueError.
    with test_obj.assertRaisesRegexp(ValueError, 'Error loading file from'):
      model.fit(
          x=train_ds,
          epochs=num_epoch,
          batch_size=8,
          callbacks=[
              callbacks.ModelCheckpoint(
                  filepath=saving_filepath,
                  save_weights_only=True,
                  load_weights_on_restart=True)
          ])

  @staticmethod
  def callableForTestReduceLROnPlateau(model, test_obj, train_ds, num_epoch,
                                       steps, strategy, saving_filepath,
                                       **kwargs):

    cbks = [
        callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            min_delta=1,
            patience=1,
            cooldown=5,
            verbose=1)
    ]

    # It is expected that the learning rate would drop by `factor` within
    # 3 epochs with `min_delta=1`.
    model.fit(x=train_ds, epochs=3, steps_per_epoch=steps, callbacks=cbks)
    test_obj.assertAllClose(
        float(K.get_value(model.optimizer.lr)), 0.0001, atol=1e-8)

    # It is expected that the learning rate would drop by another `factor`
    # within 3 epochs with `min_delta=1`.
    model.fit(x=train_ds, epochs=3, steps_per_epoch=steps, callbacks=cbks)
    test_obj.assertAllClose(
        float(K.get_value(model.optimizer.lr)), 0.00001, atol=1e-8)

  @staticmethod
  def callableForTestEarlyStopping(model, test_obj, train_ds, num_epoch, steps,
                                   strategy, saving_filepath, **kwargs):

    class EpochCounterCallback(callbacks.Callback):

      def on_epoch_begin(self, epoch, logs):
        self.last_epoch = epoch

    epoch_counter_cbk = EpochCounterCallback()
    cbks = [
        callbacks.EarlyStopping(
            monitor='loss', min_delta=0.05, patience=1, verbose=1),
        epoch_counter_cbk
    ]

    # Empirically, it is expected that `model.fit()` would terminate around the
    # 22th epoch. Asserting that it should have been stopped before the 50th
    # epoch to avoid flakiness and be more predictable.
    model.fit(x=train_ds, epochs=100, steps_per_epoch=steps, callbacks=cbks)
    test_obj.assertLess(epoch_counter_cbk.last_epoch, 50)

  @staticmethod
  def callableForTestLearningRateScheduler(model, test_obj, train_ds, num_epoch,
                                           steps, strategy, saving_filepath,
                                           **kwargs):

    cbks = [
        callbacks.LearningRateScheduler(
            schedule=lambda x: 1. / (1. + x), verbose=1)
    ]

    # It is expected that with `epochs=2`, the learning rate would drop to
    # 1 / (1 + 2) = 0.5.
    model.fit(x=train_ds, epochs=2, steps_per_epoch=steps, callbacks=cbks)
    test_obj.assertAllClose(
        float(K.get_value(model.optimizer.lr)), 0.5, atol=1e-8)

    # It is expected that with `epochs=4`, the learning rate would drop to
    # 1 / (1 + 4) = 0.25.
    model.fit(x=train_ds, epochs=4, steps_per_epoch=steps, callbacks=cbks)
    test_obj.assertAllClose(
        float(K.get_value(model.optimizer.lr)), 0.25, atol=1e-8)

  # pylint: disable=g-doc-args
  @staticmethod
  def callableForTestIntermediateDirForFTAreRemoved(model, test_obj, train_ds,
                                                    num_epoch, steps, strategy,
                                                    saving_filepath, **kwargs):
    """Testing that the temporary directory are removed.

    Some temporary directories are created for the purpose of fault tolerance.
    This test ensures that such directories should have been removed at the time
    `model.fit()` finishes successfully.
    """

    # `threading_local` and `barrier` objects have to be passed in from parent
    # thread so both threads refer to the same object.
    threading_local = kwargs['threading_local']
    barrier = kwargs['barrier']

    # Two threads will each has one copy of `temp_dirs_supposed_to_be_removed`
    # list.
    threading_local.temp_dirs_supposed_to_be_removed = []

    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=saving_filepath,
            save_weights_only=True,
            load_weights_on_restart=True),
    ]

    # Keep the references to the real function objects.
    real_os_path_join = os.path.join
    real_tempfile_mkdtemp = tempfile.mkdtemp

    # Make a `os.path.join` wrapper, which will be patched onto the real
    # function, so the temporary directories can be tracked.
    def wrapper_os_path_join(path, *paths):
      join_result = real_os_path_join(path, *paths)
      if len(paths) == 1 and paths[0] == 'backup':
        threading_local.temp_dirs_supposed_to_be_removed.append(join_result)
      return join_result

    # Likewise for `tempfile.mkdtemp`.
    def wrapper_tempfile_mkdtemp():
      result = real_tempfile_mkdtemp()
      threading_local.temp_dirs_supposed_to_be_removed.append(result)
      return result

    # Now the two threads must sync here: if they are out of sync, one thread
    # can go ahead and patch `os.path.join` while the other has not even
    # assigned the real `os.path.join` to `real_os_path_join`. If this happened,
    # the "real" `os.path.join` the slower thread would see is actually the
    # wrapper of the other.
    barrier.wait()

    # Note that `os.path.join` will respect the second patch (there are two
    # patches because of the two threads). Both threads will refer to the same
    # copy of `wrapper_os_path_join` because of the `barrier` preceding
    # `model.fit()`. Likewise for `wrapper_tempfile_mkdtemp`.
    os.path.join = wrapper_os_path_join
    tempfile.mkdtemp = wrapper_tempfile_mkdtemp

    barrier.wait()
    model.fit(
        x=train_ds,
        epochs=num_epoch,
        steps_per_epoch=steps,
        callbacks=callbacks_list)

    # Sync before un-patching to prevent either thread from accessing the real
    # functions. Also to make sure `model.fit()` is done on both threads (so we
    # can safely assert the directories are removed).
    barrier.wait()
    os.path.join = real_os_path_join
    tempfile.mkdtemp = real_tempfile_mkdtemp

    # There should be directory (names) that are supposed to be removed.
    test_obj.assertTrue(threading_local.temp_dirs_supposed_to_be_removed)
    for temp_dir_supposed_to_be_removed in (
        threading_local.temp_dirs_supposed_to_be_removed):
      # They should have been removed and thus don't exist.
      test_obj.assertFalse(os.path.exists(temp_dir_supposed_to_be_removed))

  # The actual testing methods go here.
  test_chief_only_callback = generate_callback_test_function(
      callableForTestChiefOnlyCallback.__func__)
  test_model_checkpoint_saves_on_chief_but_not_otherwise = \
      generate_callback_test_function(
          callableForTestModelCheckpointSavesOnChiefButNotOtherwise.__func__)
  test_load_weight_from_model_checkpoint = generate_callback_test_function(
      callableForTestLoadWeightFromModelCheckpoint.__func__)
  test_model_restore_callback = generate_callback_test_function(
      callableForTestModelRestoreCallback.__func__)
  test_unmatched_model_file = generate_callback_test_function(
      callableForTestUnmatchedModelFile.__func__)
  test_reduce_lr_on_plateau = generate_callback_test_function(
      callableForTestReduceLROnPlateau.__func__)
  test_early_stopping = generate_callback_test_function(
      callableForTestEarlyStopping.__func__)
  test_learning_rate_scheduler = generate_callback_test_function(
      callableForTestLearningRateScheduler.__func__)
  test_intermediate_dir_for_ft_are_removed = generate_callback_test_function(
      callableForTestIntermediateDirForFTAreRemoved.__func__)
  test_backup_model_removed = generate_callback_test_function(
      callableForTestBackupModelRemoved.__func__)
  test_backup_model_not_removed_if_interrupted = \
      generate_callback_test_function(
          callableForTestBackupModelNotRemovedIfInterrupted.__func__)


if __name__ == '__main__':
  with test.mock.patch.object(sys, 'exit', os._exit):
    test.main()
