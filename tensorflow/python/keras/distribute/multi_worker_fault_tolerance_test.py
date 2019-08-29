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
"""Tests Keras multi worker fault tolerance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import signal
import sys
import tempfile
import threading
from absl.testing import parameterized
from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.distribute import multi_worker_testing_utils
from tensorflow.python.keras.distribute import multi_worker_training_state as training_state
from tensorflow.python.platform import test


def get_strategy_object(strategy_cls):
  if strategy_cls == mirrored_strategy.MirroredStrategy:
    return strategy_cls(mirrored_strategy.all_local_devices())
  else:
    # CollectiveAllReduceStrategy and ParameterServerStrategy.
    return strategy_cls()


class KerasMultiWorkerFaultToleranceTest(test_base.IndependentWorkerTestBase,
                                         parameterized.TestCase):

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

  # TODO(rchao): Add tests for checking 0th and 2nd epoch boundary.
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
          # Eager runtime unfortunately cannot be tested with multi-threading.
          # TODO(rchao): Add test to use multi-process for eager mode after
          # b/132095481 is resolved.
          mode=['graph'],
          strategy_cls=[collective_strategy.CollectiveAllReduceStrategy],
          required_gpus=[0, 1],
          file_format=['h5', 'tf'],
          preemption_callback=[
              PreemptionAtEpochBoundarySimulatingCallback,
              PreemptionAtBatchBoundarySimulatingCallback
          ],
          # FT should work regardless of `ModelCheckpoint`'s parameters.
          save_weights_only=[True, False],
          load_weights_on_restart=[True, False],
      ))
  def testFaultToleranceInSyncStrategy(self, strategy_cls, file_format,
                                       preemption_callback, save_weights_only,
                                       load_weights_on_restart):
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

    TODO(rchao): Add test to check preemption on chief (possibly using multi
    processes).

    TODO(rchao): Add test to check fault-tolerance with multiple `model.fit()`.

    Arguments:
      strategy_cls: The strategy class to use.
      file_format: `h5` or `tf`.
      preemption_callback: The callback to simulate preemption.
      save_weights_only: The argument for `model.fit()`'s `save_weights_only`.
      load_weights_on_restart: The argument for `model.fit()`'s
        `load_weights_on_restart`.
    """

    def _independent_worker_fn(*args, **kwargs):  # pylint: disable=unused-argument
      with test.mock.patch.object(dc, '_run_std_server',
                                  self._make_mock_run_std_server()):
        # `before_restart` is True for the threads that represent the original
        # chief and non-chief worker, and False for threads that represent the
        # restarted chief and non-chief workers.
        before_restart = kwargs['before_restart']

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
        def start_new_thread(new_chief):
          new_thread_tf_config = json.loads(os.environ['TF_CONFIG'])

          # Update the ports in new chief and new worker threads.
          new_thread_tf_config['cluster']['worker'] = kwargs['reserved_ports']

          # Since both new chief and new worker threads are started from the
          # worker thread, we need to overwrite the tf config task index.
          new_thread_tf_config['task']['index'] = 0 if new_chief else 1
          return self._run_task_in_thread(
              task_fn=_independent_worker_fn,
              cluster_spec=None,
              task_type=None,
              task_id=None,
              tf_config=new_thread_tf_config,
              before_restart=False,
              new_chief=new_chief)

        try:

          class CkptSavedEpochAssertingCallback(callbacks.Callback):

            def __init__(self, test_obj):
              super(CkptSavedEpochAssertingCallback, self).__init__()
              self.test_obj = test_obj

            def on_epoch_begin(self, epoch, logs=None):
              # `_ckpt_saved_epoch` attribute is set at the end of every epoch.
              self.test_obj.assertEqual(
                  K.eval(self.model._ckpt_saved_epoch) ==
                  training_state.CKPT_SAVED_EPOCH_UNUSED_VALUE, epoch == 0)

          callbacks_list = [
              callbacks.ModelCheckpoint(
                  filepath=saving_filepath,
                  save_weights_only=save_weights_only,
                  load_weights_on_restart=load_weights_on_restart),
              CkptSavedEpochAssertingCallback(self)
          ]
          if before_restart:
            callbacks_list.append(preemption_callback())

          self.assertFalse(hasattr(model, training_state.CKPT_SAVED_EPOCH))
          history = model.fit(
              x=train_ds,
              epochs=num_epoch,
              steps_per_epoch=steps,
              callbacks=callbacks_list)
          self.assertFalse(hasattr(model, training_state.CKPT_SAVED_EPOCH))

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

          # At this point we block the original non-chief thread, and
          # start the new threads that simulate the restarted chief and
          # non-chief, joining the threads and return.
          new_chief_thread = start_new_thread(new_chief=True)
          new_worker_thread = start_new_thread(new_chief=False)
          self.join_independent_workers([new_chief_thread, new_worker_thread])
          return

        # Successful end of a `fit()` call.
        with self._lock:
          self._successful_thread_ends += 1
        self.assertFalse(before_restart)

    # Common parameters
    num_workers = 2
    num_epoch = 3
    # History list storing the results for preemption and no preemption cases.
    self._histories = []
    # Lock required to prevent race condition between two threads.
    self._lock = threading.Lock()
    strategy = get_strategy_object(strategy_cls)

    def handler(signum, frame):
      del signum, frame
      # `session.run()` within `model.fit()` can time out. Skipping it as it
      # doesn't represent the failure of this test.
      self.skipTest('Skipping test due to `session.run()` timeout.')

    signal.signal(signal.SIGALRM, handler)
    # Alarming within 5 min before the test timeouts and fails.
    signal.alarm(240)

    def get_saving_dir_and_filepath():
      saving_dir = tempfile.mkdtemp(prefix=self.get_temp_dir())
      saving_filepath = os.path.join(saving_dir, 'checkpoint.' + file_format)
      return saving_dir, saving_filepath

    # Case 1: Training for `num_epoch` without preemptions.
    cluster_spec = test_base.create_cluster_spec(num_workers=num_workers)
    self._barrier = dc._Barrier(2)
    self._successful_thread_ends = 0
    # Get a new temporary filepath to save the checkpoint to.
    saving_dir, saving_filepath = get_saving_dir_and_filepath()
    threads = self.run_multiple_tasks_in_threads(
        _independent_worker_fn,
        cluster_spec,
        # Pass `saving_filepath` from the parent thread to ensure every worker
        # has the same filepath to save.
        saving_filepath=saving_filepath,
        before_restart=False,
        new_chief=False)
    threads_to_join = []
    if strategy.extended.experimental_between_graph:
      for ts in threads.values():
        threads_to_join.extend(ts)
    else:
      threads_to_join = [threads['worker'][0]]
    self.join_independent_workers(threads_to_join)

    # `self.test_skipped_reason` could be set when a non-main thread attempts
    # to skip the test.
    # `multi_worker_test_base.skip_if_grpc_server_cant_be_started()` is an
    # example of where this can be set. Since raising `SkipTest` in a non-main
    # thread doesn't actually skip the test, we check if the test should be
    # skipped here once we have joined the threads.
    if getattr(self, 'test_skipped_reason', None) is not None:
      self.skipTest(self.test_skipped_reason)

    self.assertTrue(
        training_state.remove_checkpoint_if_exists(saving_dir, saving_filepath))
    self.assertEqual(self._successful_thread_ends, 2)

    # Case 2: Training for `num_epoch` epoch with preemptions.
    # The preemption is simulated at both epoch boundary and batch boundary.
    cluster_spec = test_base.create_cluster_spec(num_workers=num_workers)
    self._barrier = dc._Barrier(2)
    # Ports reserved for new threads simulating recovery.
    reserved_ports = [
        'localhost:%s' % test_base.pick_unused_port()
        for _ in range(num_workers)
    ]
    self._successful_thread_ends = 0
    # Get a new temporary filepath to save the checkpoint to.
    saving_dir, saving_filepath = get_saving_dir_and_filepath()
    threads = self.run_multiple_tasks_in_threads(
        _independent_worker_fn,
        cluster_spec,
        # Pass `saving_filepath` from the parent thread to ensure every worker
        # has the same filepath to save.
        saving_filepath=saving_filepath,
        reserved_ports=reserved_ports,
        before_restart=True,
        new_chief=False)
    threads_to_join = []
    if strategy.extended.experimental_between_graph:
      # Only join the non-chief thread since the first thread for chief will
      # eventually hang and be ignored.
      threads_to_join = [threads['worker'][1]]
    else:
      threads_to_join = [threads['worker'][0]]
    self.join_independent_workers(threads_to_join)
    if getattr(self, 'test_skipped_reason', None) is not None:
      self.skipTest(self.test_skipped_reason)

    self.assertTrue(
        training_state.remove_checkpoint_if_exists(saving_dir, saving_filepath))
    self.assertEqual(self._successful_thread_ends, 2)

    def assert_all_elements_are_identical(list_to_check):
      first_item = list_to_check[0]
      for item in list_to_check[1:]:
        self.assertAllClose(first_item, item, rtol=2e-5, atol=1e-5)

    # Important: the results from preemption interrupted and non-interrupted
    # cases should give the same final results.
    assert_all_elements_are_identical(
        [history['acc'][-1] for history in self._histories])
    assert_all_elements_are_identical(
        [history['loss'][-1] for history in self._histories])
    # The length of `self._histories` would be num_workers * num_runs (3).
    self.assertLen(self._histories, 4)

    # Results from case 1 should have 3 full epochs.
    self.assertLen(self._histories[0]['acc'], 3)
    # Results from case 2 should only have 2 full epochs because it restarted at
    # epoch 1.
    self.assertLen(self._histories[-1]['acc'], 2)


if __name__ == '__main__':
  with test.mock.patch.object(sys, 'exit', os._exit):
    test.main()
