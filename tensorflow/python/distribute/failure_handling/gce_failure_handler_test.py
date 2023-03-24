# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for GCE specifics of PreemptionCheckpointHandler."""
import os
import random
import re
import sys
import threading
import time
import urllib

from absl.testing import parameterized

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import one_device_strategy as one_device_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.failure_handling import failure_handling
from tensorflow.python.distribute.failure_handling import failure_handling_util
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.module import module
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib


try:
  import dill  # pylint:disable=g-import-not-at-top

  _REGISTER_DECORATOR = dill.register
except ImportError:
  _REGISTER_DECORATOR = lambda fn, *_: fn

mock = test.mock


CLUSTER_SIZE = 4
EPOCHS_TO_RUN = 5
STEPS_PER_EPOCH = 6
_PEER_WATCHER_THREAD_PREFIX = 'PeerTerminationWatcher'
_LOCAL_WATCHER_THREAD_PREFIX = 'WorkerTerminationSignalWatcher'
MAX_WAIT_TIME = 30


def _is_oss():
  """Returns whether the test is run under OSS."""
  return len(sys.argv) >= 1 and 'bazel' in sys.argv[0]


def _make_checkpoint_manager(checkpoint, checkpoint_dir, cluster_resolver):

  if not cluster_resolver or not cluster_resolver.cluster_spec().as_dict() or (
      multi_worker_util.is_chief(
          cluster_spec=cluster_resolver.cluster_spec(),
          task_type=cluster_resolver.task_type,
          task_id=cluster_resolver.task_id)):
    return checkpoint_management.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=1)
  else:
    return checkpoint_management.CheckpointManager(
        checkpoint,
        directory=failure_handling._non_chief_checkpoint_dir(
            checkpoint_dir, cluster_resolver.task_id),
        max_to_keep=1)


def raise_if_not_all_exit(grace_period, mpr):
  """Wait for all cluster to exit with a time out."""
  waiting_time = 0
  exit_process_count = 0
  # This addition to mitigate the fact that our step time is too short in test
  while exit_process_count != CLUSTER_SIZE and waiting_time < max(
      grace_period + 15, MAX_WAIT_TIME):
    exit_process_count = 0
    for worker_id in range(CLUSTER_SIZE):
      if not mpr.process_exists('worker', worker_id):
        exit_process_count += 1
    waiting_time += 1
    time.sleep(1)

  if waiting_time == max(grace_period + 5, 40):
    raise RuntimeError('Waited long but at least one worker still exist. '
                       'Considering size of our model, this should not'
                       ' happen.')


class GceFailureHandlingTest(test.TestCase, parameterized.TestCase):
  """Integration test for PreemptionCheckpointHandler."""

  def worker_fn(
      self,
      checkpoint_dir,
      cluster_spec,
      input_arg,
      strategy_option,
      maintenance_event=None,
      training_finished=None,
      frequent_send=False,
      training_restarted=None,
      termination_config=failure_handling.TerminationConfig(grace_period=0),
      api_wrapping_train=True,
  ):
    if strategy_option == 'MS':
      strategy = mirrored_strategy.MirroredStrategy()
    elif strategy_option == 'OneDevice':
      if config.list_physical_devices('GPU'):
        strategy = one_device_lib.OneDeviceStrategy(device='/gpu:0')
      else:
        strategy = one_device_lib.OneDeviceStrategy(device='/cpu:0')
    else:
      strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()

    def mock_termination_watcher_function_gce(*args, **kwargs):
      del args, kwargs
      if not frequent_send:
        time.sleep(1)
        if (not maintenance_event.is_set()) and (random.randrange(0, 7) == 5):
          maintenance_event.set()
          logging.info('Termination notice available.')
          return True

      elif frequent_send and not maintenance_event.is_set():
        logging.info('Termination notice available.')
        return True

      return False

    with mock.patch.object(
        failure_handling_util, 'termination_watcher_function_gce',
        mock_termination_watcher_function_gce), mock.patch.object(
            failure_handling_util, 'detect_platform',
            lambda: failure_handling_util.PlatformDevice.GCE_GPU):

      class Model(module.Module):

        def __init__(self):
          self.v = variables_lib.Variable(
              0.,
              synchronization=variables_lib.VariableSynchronization.ON_WRITE,
              aggregation=variables_lib.VariableAggregation.SUM)

        @def_function.function(input_signature=[])
        def __call__(self):
          return self.v.read_value()

      with strategy.scope():
        model = Model()
        fh_ckpt = tracking_util.Checkpoint(model=model)

        if input_arg == 'checkpoint':
          checkpoint_or_manager = fh_ckpt
        else:
          checkpoint_or_manager = _make_checkpoint_manager(
              fh_ckpt, checkpoint_dir, strategy.cluster_resolver)
        preemption_handler = (
            failure_handling.PreemptionCheckpointHandler(
                strategy.cluster_resolver, checkpoint_or_manager,
                checkpoint_dir, termination_config))

      def distributed_train_step(current_epoch, current_step):

        @def_function.function
        def train_step():
          model.v.assign_add(constant_op.constant(1.))

        strategy.run(train_step)

        if current_step == STEPS_PER_EPOCH - 1:
          logging.info('epoch %d finished', current_epoch)

      logging.info('Start training at %d',
                   preemption_handler.total_run_calls)

      # If the training process has been restarted, verify that the expected
      # number of checkpoints have been written.
      # We also want to check training_finished, because there's a corner case
      # where the signal is sent quite late and training finishes before the
      # grace period ends.
      if training_restarted.is_set() and not training_finished.is_set():
        logging.info(gfile.ListDirectory(checkpoint_dir))
        match_group = [
            re.search(r'.*ckpt-(\d+).index', a_file)
            for a_file in gfile.ListDirectory(checkpoint_dir)
        ]
        checkpoint_index = [
            a_match.group(1) for a_match in match_group if a_match
        ]
        self.assertNotEmpty(checkpoint_index)

        if api_wrapping_train:
          if termination_config.grace_period > 0:
            # Two checkpoints were saved for the extended grace period.
            self.assertEqual(
                max([int(ckpt_index) for ckpt_index in checkpoint_index]), 2)
          else:
            self.assertEqual(
                max([int(ckpt_index) for ckpt_index in checkpoint_index]), 1)

        else:
          # Test if arguments to _save_checkpoint_if_preempted are passed
          # successfully.
          self.assertEqual(
              max([int(ckpt_index) for ckpt_index in checkpoint_index]),
              preemption_handler.total_run_calls)

      for epoch in range(preemption_handler.total_run_calls // STEPS_PER_EPOCH,
                         EPOCHS_TO_RUN):

        for step in range(preemption_handler.total_run_calls % STEPS_PER_EPOCH,
                          STEPS_PER_EPOCH):

          # Testing two different APIs to save checkpoint.
          if api_wrapping_train:
            preemption_handler.run(distributed_train_step, epoch, step)

          else:
            preemption_handler._save_checkpoint_if_preempted(
                checkpoint_number=preemption_handler.total_run_calls)
            distributed_train_step(epoch, step)

      logging.info('Training finished.')
      training_finished.set()

      self.assertEqual(
          model.v.numpy(),
          strategy.num_replicas_in_sync * EPOCHS_TO_RUN * STEPS_PER_EPOCH)

      running_threads = test_util.get_running_threads()
      if test_util.has_thread(_PEER_WATCHER_THREAD_PREFIX,
                              running_threads) and test_util.has_thread(
                                  _LOCAL_WATCHER_THREAD_PREFIX,
                                  running_threads):
        try:
          # Explicitly call __del__ since making it None and gc.collect does
          # not invoke __del__ here.
          preemption_handler.__del__()

          time.sleep(2)

          running_threads = test_util.get_running_threads()
          self.assertFalse(
              test_util.has_thread(_LOCAL_WATCHER_THREAD_PREFIX,
                                   running_threads))
          self.assertFalse(
              test_util.has_thread(_PEER_WATCHER_THREAD_PREFIX,
                                   running_threads))

        except urllib.error.URLError as e:
          if 'Temporary failure in name resolution' in e.message:
            # This is caused by a weird flakiness that mock.patch does not
            # correctly patch failure_handling_util.request_compute_metadata, a
            # real request is attempted, and an error is hit in
            # failure_handling_util.request_compute_metadata
            logging.warning('Hit a mock issue.')
            return

  @combinations.generate(
      combinations.combine(
          input_arg=['checkpoint', 'manager'],
          strategy_option=[
              'MS',
              'OneDevice',
              'MWMS_local',
              'MWMS_multi_worker',
          ],
      )
  )
  def test_basic_run(self, input_arg, strategy_option):
    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt/')

    if strategy_option == 'MWMS_multi_worker':
      has_chief = False
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief,
          num_workers=CLUSTER_SIZE)
      maintenance_event = multi_process_runner.manager().Event()
      training_finished = multi_process_runner.manager().Event()
      training_restarted = multi_process_runner.manager().Event()

      mpr = multi_process_runner.MultiProcessRunner(
          self.worker_fn,
          cluster_spec,
          args=(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                maintenance_event, training_finished, False,
                training_restarted),
          rpc_layer=rpc_layer,
          return_output=True,
          dependence_on_chief=has_chief)

      logging.info('Cluster starting.')
      mpr.start()

      raise_if_not_all_exit(0, mpr)

      if not training_finished.is_set():
        logging.info('restarting workers')
        training_restarted.set()
        for worker_id in range(CLUSTER_SIZE):
          mpr.start_single_process('worker', worker_id, cluster_spec)
        logging.info('workers restarted')

      mpr.join(timeout=250)
      self.assertTrue(training_finished.is_set())

    else:
      maintenance_event = threading.Event()
      training_finished = threading.Event()
      training_restarted = threading.Event()

      cluster_spec = server_lib.ClusterSpec({})
      caught_exit = False
      try:
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, False,
                       training_restarted)
      except SystemExit as exit_error:
        caught_exit = True
        # We cannot use assertRaise instead, since termination is not always
        # triggered.
        self.assertEqual(exit_error.code, 143)  # pylint:disable=g-assert-in-except

      if maintenance_event.is_set() and not training_finished.is_set():
        self.assertTrue(caught_exit)

        logging.info('restarting workers')
        training_restarted.set()
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, False,
                       training_restarted)

      self.assertTrue(training_finished.is_set())

  @combinations.generate(
      combinations.combine(
          grace_period=[0, 7],
          input_arg=['checkpoint', 'manager'],
          strategy_option=[
              'MS',
              'OneDevice',
              'MWMS_local',
              'MWMS_multi_worker',
          ],
          api_wrapping_train=[True, False]
      ))
  def test_multiple_workers_preempted_consecutively(self, grace_period,
                                                    input_arg, strategy_option,
                                                    api_wrapping_train):

    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt/')
    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    termination_config = failure_handling.TerminationConfig(
        grace_period=grace_period)

    if strategy_option == 'MWMS_multi_worker':
      has_chief = False
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief,
          num_workers=CLUSTER_SIZE)
      has_chief = False
      maintenance_event = multi_process_runner.manager().Event()
      training_finished = multi_process_runner.manager().Event()
      training_restarted = multi_process_runner.manager().Event()
      mpr = multi_process_runner.MultiProcessRunner(
          self.worker_fn,
          cluster_spec,
          args=(checkpoint_dir, cluster_spec, input_arg, maintenance_event,
                strategy_option, training_finished, True, training_restarted,
                termination_config),
          kwargs={'api_wrapping_train': api_wrapping_train},
          rpc_layer=rpc_layer,
          return_output=True,
          dependence_on_chief=has_chief)

      logging.info('Cluster starting.')
      mpr.start()

      raise_if_not_all_exit(grace_period, mpr)

      maintenance_event.set()
      logging.info('restarting workers')
      training_restarted.set()
      for worker_id in range(CLUSTER_SIZE):
        mpr.start_single_process('worker', worker_id, cluster_spec)
      logging.info('workers restarted')

      mpr.join(timeout=250)
      self.assertTrue(training_finished.is_set())

    else:
      maintenance_event = threading.Event()
      training_finished = threading.Event()
      training_restarted = threading.Event()

      cluster_spec = server_lib.ClusterSpec({})

      caught_exit = False
      try:
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, True,
                       training_restarted, termination_config)
      except SystemExit as exit_error:
        caught_exit = True
        # We cannot use assertRaise instead, since termination is not always
        # triggered.
        self.assertEqual(exit_error.code, 143)  # pylint:disable=g-assert-in-except

      if maintenance_event.is_set() and not training_finished.is_set():
        self.assertTrue(caught_exit)

        logging.info('restarting workers')
        training_restarted.set()
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, True,
                       training_restarted, termination_config)

  @combinations.generate(
      combinations.combine(
          input_arg=['checkpoint', 'manager'],
          strategy_option=[
              'MS',
              'OneDevice',
              'MWMS_local',
              'MWMS_multi_worker',
          ],
          api_wrapping_train=[True, False]))
  def test_grace_period_continue_training(self, input_arg, strategy_option,
                                          api_wrapping_train):
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt/')
    grace_period = 7
    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    termination_config = failure_handling.TerminationConfig(
        grace_period=grace_period)

    if strategy_option == 'multi_worker':
      has_chief = False
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief,
          num_workers=CLUSTER_SIZE)

      checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt/')
      maintenance_event = multi_process_runner.manager().Event()
      training_finished = multi_process_runner.manager().Event()
      training_restarted = multi_process_runner.manager().Event()

      mpr = multi_process_runner.MultiProcessRunner(
          self.worker_fn,
          cluster_spec,
          args=(checkpoint_dir, cluster_spec, input_arg, maintenance_event,
                strategy_option, training_finished, False, training_restarted,
                termination_config),
          kwargs={'api_wrapping_train': api_wrapping_train},
          rpc_layer=rpc_layer,
          return_output=True,
          dependence_on_chief=has_chief)

      logging.info('Cluster starting.')
      mpr.start()

      while (not maintenance_event.is_set()) and (
          not training_finished.is_set()):
        time.sleep(1)

      raise_if_not_all_exit(grace_period, mpr)

      if not training_finished.is_set():
        logging.info('restarting workers')
        training_restarted.set()
        for worker_id in range(CLUSTER_SIZE):
          mpr.start_single_process('worker', worker_id, cluster_spec)
        logging.info('workers restarted')

      mpr.join(timeout=250)

      self.assertTrue(training_finished.is_set())

    else:
      maintenance_event = threading.Event()
      training_finished = threading.Event()
      training_restarted = threading.Event()

      cluster_spec = server_lib.ClusterSpec({})
      caught_exit = False
      try:
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, False,
                       training_restarted, termination_config)
      except SystemExit as exit_error:
        caught_exit = True
        # We cannot use assertRaise instead, since termination is not always
        # triggered.
        self.assertEqual(exit_error.code, 143)  # pylint:disable=g-assert-in-except

      if maintenance_event.is_set() and not training_finished.is_set():
        self.assertTrue(caught_exit)

        logging.info('restarting workers')
        training_restarted.set()
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, False,
                       training_restarted, termination_config)
        self.assertTrue(training_finished.is_set())


@_REGISTER_DECORATOR(GceFailureHandlingTest)
def _save_test_case(pickler, obj):
  def reconstruct(*args, **kwargs):
    del args, kwargs
    return GceFailureHandlingTest()

  return pickler.save_reduce(reconstruct, (), obj=obj)


if __name__ == '__main__':
  test_util.main()
