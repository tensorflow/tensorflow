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
"""Tests Keras multi worker callbacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import sys
import tempfile
import threading

from absl.testing import parameterized

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import keras
from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.distribute import multi_worker_testing_utils
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
          required_gpus=[0, 1]))
  def test_template(self, strategy_cls):
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
            saving_filepath=kwargs['saving_filepath'])

    # Pass saving_filepath from the parent thread to ensure every worker has the
    # same fileapth to save.
    saving_filepath = os.path.join(self.get_temp_dir(), 'checkpoint.h5')
    threads = self.run_multiple_tasks_in_threads(
        _independent_worker_fn, cluster_spec, saving_filepath=saving_filepath)
    if os.path.exists(saving_filepath):
      os.remove(saving_filepath)

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

  # The callables of the actual testing content to be run go below.
  @staticmethod
  def callableForTestChiefOnlyCallback(model, test_obj, train_ds, num_epoch,
                                       steps, strategy, saving_filepath):

    class ChiefOnly(keras.callbacks.Callback):

      def __init__(self):
        self._chief_worker_only = True
        self.filtered_correctly = True

      def on_train_begin(self, logs):
        if not dc_context.get_current_worker_context().is_chief:
          # Non-chief workers shouldn't run this callback.
          self.filtered_correctly = False

    cb = ChiefOnly()
    model.fit(
        x=train_ds, epochs=num_epoch, steps_per_epoch=steps, callbacks=[cb])

    test_obj.assertTrue(cb.filtered_correctly)

  @staticmethod
  def callableForTestModelCheckpointSavesOnChiefButNotOtherwise(
      model, test_obj, train_ds, num_epoch, steps, strategy, saving_filepath):
    # Incorporate type/index information and thread id in saving_filepath to
    # ensure every worker has a unique path. Note that in normal use case the
    # saving_filepath will be the same for all workers, but we use different
    # ones here just to test out chief saves checkpoint but non-chief doesn't.
    saving_filepath = os.path.join(
        test_obj.get_temp_dir(), 'checkpoint_%s_%d' %
        (test_base.get_task_type(), test_base.get_task_index()))

    # The saving_filepath shouldn't exist at the beginning (as it's unique).
    test_obj.assertFalse(os.path.exists(saving_filepath))

    model.fit(
        x=train_ds,
        epochs=num_epoch,
        steps_per_epoch=steps,
        callbacks=[callbacks.ModelCheckpoint(filepath=saving_filepath)])

    # If it's chief, the model should be saved; if not, the model shouldn't.
    test_obj.assertEqual(os.path.exists(saving_filepath), test_base.is_chief())

  @staticmethod
  def initialFitting(test_obj, model, train_ds, num_epoch, steps,
                     saving_filepath):
    # The saving_filepath shouldn't exist at the beginning.
    test_obj.assertFalse(os.path.exists(saving_filepath))

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
    test_obj.assertTrue(os.path.exists(saving_filepath))

    history_after_one_more_epoch = model.fit(
        x=train_ds, epochs=1, steps_per_epoch=steps)

    # The saving_filepath should continue to exist (if it did) after fitting
    # without callback.
    test_obj.assertTrue(os.path.exists(saving_filepath))

    return saving_filepath, history_after_one_more_epoch

  @staticmethod
  def callableForTestLoadWeightFromModelCheckpoint(model, test_obj, train_ds,
                                                   num_epoch, steps, strategy,
                                                   saving_filepath):
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
          history_after_loading_weight_and_one_more_epoch.history)

    # Verify the temp files are indeed removed (no trace left behind).
    for filepath in filepaths:
      assert not os.path.exists(filepath)

  @staticmethod
  def callableForTestModelRestoreCallback(model, test_obj, train_ds, num_epoch,
                                          steps, strategy, saving_filepath):

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
        history_after_model_restoring_and_one_more_epoch.history)

    history_one_more_epoch_without_model_restoring = model.fit(
        x=train_ds, epochs=1, steps_per_epoch=steps)

    # Ensuring training for another epoch gives different result.
    test_obj.assertNotAllClose(
        history_after_model_restoring_and_one_more_epoch.history,
        history_one_more_epoch_without_model_restoring.history)

  @staticmethod
  def callableForTestUnmatchedModelFile(model, test_obj, train_ds, num_epoch,
                                        steps, strategy, saving_filepath):

    # The saving_filepath shouldn't exist at the beginning.
    test_obj.assertFalse(os.path.exists(saving_filepath))

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

    test_obj.assertTrue(os.path.exists(saving_filepath))

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

  class PreemptionAtBatchBoundarySimulatingCallback(callbacks.Callback):
    """Callback to simulate preemtion at batch boundary."""

    def on_epoch_begin(self, epoch, logs=None):
      self._current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
      if self._current_epoch == 1 and batch == 1 and not test_base.is_chief():
        # Simulate preemtion at the start of second batch of second epoch.
        raise RuntimeError('Preemption!')

    def on_batch_end(self, batch, logs=None):
      assert self._current_epoch < 1 or batch < 1

    def on_epoch_end(self, epoch, logs=None):
      assert epoch < 1

  class PreemptionAtEpochBoundarySimulatingCallback(callbacks.Callback):
    """Callback to simulate preemtion at epoch boundary."""

    def on_epoch_begin(self, epoch, logs=None):
      if epoch == 1 and not test_base.is_chief():
        # Simulate preemtion at the start of second epoch.
        raise RuntimeError('Preemption!')

    def on_epoch_end(self, epoch, logs=None):
      assert epoch < 1

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          strategy_cls=[collective_strategy.CollectiveAllReduceStrategy],
          required_gpus=[0, 1],
          file_format=['h5'],  # TODO(rchao): Support TF format.
          preemption_callback=[
              PreemptionAtEpochBoundarySimulatingCallback,
              PreemptionAtBatchBoundarySimulatingCallback
          ]))
  def testFaultToleranceInSyncStrategy(self, strategy_cls, file_format,
                                       preemption_callback):
    """Test fault-tolerance with multi-threading using sync dist-strat.

    This test simulates multi-worker training that is interrupted by a
    preemption, by having two threads, each of which represents a chief and a
    non-chief worker, where the non-chief raises an error in the middle of
    training loop. Upon excepting the error, a new thread with a new cluster
    spec is created to simulate the recovered non-chief worker. Meanwhile, the
    chief worker cannot proceed and hangs since the non-chief worker has
    crashed. To simulate a restart of the chief, a new thread has been prepared
    to run to take over chief with the help of a condition variable. It is
    expected that after the restart of both chief and non-chief workers, the
    training continues from the epoch they previously failed at. The test
    concludes by verifying the preemption-interrupted training can finish with
    the same loss and accuracy had the preemption not occurred.

    Arguments:
      strategy_cls: The strategy class to use.
      file_format: `h5` or `tf`.
      preemption_callback: The callback to simulate preemption.
    """

    def _independent_worker_fn(*args, **kwargs):  # pylint: disable=unused-argument
      with test.mock.patch.object(dc, '_run_std_server',
                                  self._make_mock_run_std_server()):
        # Condition variable that blocks the thread that represents the
        # restarted chief.
        cv = kwargs.get('cv', None)
        # `before_restart` is True for the threads that represent the original
        # chief and non-chief worker, and False for threads that represent the
        # restarted chief and non-chief workers.
        before_restart = kwargs['before_restart']
        if kwargs['new_chief']:
          # `new_chief` is only True for the restarted chief thread. It waits
          # until non-chief is preempted and restarted to simulate the causality
          # where chief's restart results from non-chief's failure.
          cv.acquire()
          while not hasattr(cv, 'preempted'):
            cv.wait()
          cv.release()

        # Model building under strategy scope. Following is the code we expect
        # the user runs on every worker.
        strategy = get_strategy_object(strategy_cls)
        batch_size = 64
        steps = 3
        train_ds, _ = multi_worker_testing_utils.mnist_synthetic_dataset(
            batch_size, steps)
        with strategy.scope():
          model = multi_worker_testing_utils.get_mnist_model((28, 28, 1))

        # Function to start a new thread. This will be called twice in the
        # following code: one represents the restart of the non-chief, and one
        # represents the restart of the chief as a result of the restart of the
        # non-chief (so the training can continue in sync).
        def start_new_thread(new_chief=False):
          new_thread_tf_config = json.loads(os.environ['TF_CONFIG'])
          new_thread_tf_config['cluster']['worker'] = kwargs['reserved_ports']
          return self._run_task_in_thread(
              task_fn=_independent_worker_fn,
              cluster_spec=None,
              task_type=None,
              task_id=None,
              tf_config=new_thread_tf_config,
              before_restart=False,
              cv=cv,
              new_chief=new_chief)

        if test_base.is_chief() and before_restart:
          # Chief to start a new thread (that will be blocked by a condition
          # variable until the non-chief's new thread is started). The thread
          # for (recovered) chief is started before entering `fit()` because
          # the original chief thread will eventually hang and be ignored.
          start_new_thread(new_chief=True)

        try:

          class CkptSavedEpochAssertingCallback(callbacks.Callback):

            def __init__(self, test_obj):
              super(CkptSavedEpochAssertingCallback, self).__init__()
              self.test_obj = test_obj

            def on_epoch_begin(self, epoch, logs=None):
              # `_ckpt_saved_epoch` attribute is set at the end of every epoch.
              self.test_obj.assertEqual(self.model._ckpt_saved_epoch is None,
                                        epoch == 0)

          callbacks_list = [
              callbacks.ModelCheckpoint(
                  filepath=saving_filepath,
                  save_weights_only=True,
                  load_weights_on_restart=True),
              CkptSavedEpochAssertingCallback(self)
          ]
          if before_restart:
            callbacks_list.append(preemption_callback())

          self.assertIsNone(model._ckpt_saved_epoch)
          history = model.fit(
              x=train_ds,
              epochs=num_epoch,
              steps_per_epoch=steps,
              callbacks=callbacks_list)
          self.assertIsNone(model._ckpt_saved_epoch)

          # `history` of the training result is collected to be compared against
          # each other. It is expected that the training results (loss and
          # accuracy`) are the same with or without preemption.
          self._histories.append(history.history)

        except RuntimeError:
          # pylint: disable=g-assert-in-except
          self.assertTrue(before_restart)
          # Reset the barrier so the new threads simulating recovery can
          # continue.
          self._barrier._counter = 0
          self._barrier._flag = False

          # Now that the non-chief has been preempted, it notifies the thread
          # that simulates the restarted chief to start so they can be back in
          # sync.
          cv.acquire()
          cv.preempted = True
          cv.notify()
          cv.release()

          # At this point we should discard the original non-chief thread, and
          # start the new thread that simulates the restarted non-chief, hence
          # joining the thread and return.
          self.join_independent_workers([start_new_thread()])
          return

        # Successful end of a `fit()` call.
        self._successful_thread_ends += 1
        self.assertFalse(before_restart)

    # Common parameters
    num_workers = 2
    num_epoch = 3
    # History list storing the results for preemption and no preemption cases.
    self._histories = []
    # Pass `saving_filepath` from the parent thread to ensure every worker has
    # the same filepath to save.
    saving_filepath = os.path.join(self.get_temp_dir(),
                                   'checkpoint.' + file_format)
    strategy = get_strategy_object(strategy_cls)

    # Case 1: Training for `num_epoch` without preemptions.
    cluster_spec = test_base.create_cluster_spec(num_workers=num_workers)
    self._barrier = dc._Barrier(2)
    self._successful_thread_ends = 0
    threads = self.run_multiple_tasks_in_threads(
        _independent_worker_fn,
        cluster_spec,
        saving_filepath=saving_filepath,
        before_restart=False,
        new_chief=False)
    if os.path.exists(saving_filepath):
      os.remove(saving_filepath)
    threads_to_join = []
    if strategy.extended.experimental_between_graph:
      for ts in threads.values():
        threads_to_join.extend(ts)
    else:
      threads_to_join = [threads['worker'][0]]
    self.join_independent_workers(threads_to_join)
    self.assertEqual(self._successful_thread_ends, 2)

    # Case 2: Training for `num_epoch` epoch with preemptions.
    # The preemption is simulated at both epoch boundary and batch boundary.
    cluster_spec = test_base.create_cluster_spec(num_workers=num_workers)
    cv = threading.Condition()
    self._barrier = dc._Barrier(2)
    # Ports reserved for new threads simulating recovery.
    reserved_ports = [
        'localhost:%s' % test_base.pick_unused_port()
        for _ in range(num_workers)
    ]
    self._successful_thread_ends = 0
    threads = self.run_multiple_tasks_in_threads(
        _independent_worker_fn,
        cluster_spec,
        saving_filepath=saving_filepath,
        reserved_ports=reserved_ports,
        before_restart=True,
        cv=cv,
        new_chief=False)
    if os.path.exists(saving_filepath):
      os.remove(saving_filepath)
    threads_to_join = []
    if strategy.extended.experimental_between_graph:
      # Only join the non-chief thread since the first thread for chief will
      # eventually hang and be ignored.
      threads_to_join = [threads['worker'][1]]
    else:
      threads_to_join = [threads['worker'][0]]
    self.join_independent_workers(threads_to_join)
    self.assertEqual(self._successful_thread_ends, 2)

    def assert_all_elements_are_identical(list_to_check):
      first_item = list_to_check[0]
      for item in list_to_check[1:]:
        self.assertAllClose(first_item, item, rtol=1e-5, atol=1e-5)

    # Important: the results from preemption interrupted and non-interrupted
    # cases should give the same final results.
    assert_all_elements_are_identical(
        [history['acc'][-1] for history in self._histories])
    assert_all_elements_are_identical(
        [history['loss'][-1] for history in self._histories])
    # The length of `self._histories` would be num_workers * num_runs (3).
    self.assertLen(self._histories, 4)

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


if __name__ == '__main__':
  with test.mock.patch.object(sys, 'exit', os._exit):
    test.main()
