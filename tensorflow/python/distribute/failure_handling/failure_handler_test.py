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
"""Tests for PreemptionCheckpointHandler."""
import os
import random
import re
import signal
import sys
import threading
import time

from absl.testing import parameterized

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.failure_handling import failure_handling
from tensorflow.python.distribute.failure_handling import failure_handling_util
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.module import module
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib


mock = test.mock


CLUSTER_SIZE = 4
EPOCHS_TO_RUN = 8
STEPS_PER_EPOCH = 15
MAX_WAIT_TIME = 40


def _is_oss():
  """Returns whether the test is run under OSS."""
  return len(sys.argv) >= 1 and 'bazel' in sys.argv[0]


def _make_checkpoint_manager(checkpoint, checkpoint_dir, cluster_resolver):
  if not cluster_resolver.cluster_spec().as_dict() or (
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


class PreemptionCheckpointTest(test.TestCase, parameterized.TestCase):
  """Integration test for PreemptionCheckpointHandler."""

  def _maybe_trigger_a_preemption(self, training_started_event,
                                  trigger_it=False):
    if not training_started_event:
      return
    clear_events = [
        event for event in training_started_event if not event.is_set()
    ]
    if clear_events:
      if trigger_it:
        logging.info('Set preemption signal')
        clear_events[0].set()
      elif random.randrange(0, 9) > 6:
        clear_events[0].set()

  def worker_fn(self,
                checkpoint_dir,
                cluster_spec,
                input_arg='checkpoint',
                training_started_event=None,
                raise_app_error_on_worker=None,
                training_restarted=None,
                training_finished=None,
                termination_config=failure_handling.TerminationConfig(),
                api_wrapping_train=True):

    strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()

    class Model(module.Module):

      def __init__(self):
        self.v = variables_lib.Variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_WRITE,
            aggregation=variables_lib.VariableAggregation.SUM)

      @def_function.function(input_signature=[])
      def __call__(self):
        return self.v.read_value()

    with mock.patch.object(failure_handling_util, 'on_gcp', lambda: False):

      with strategy.scope():
        model = Model()
        # Named it fh_ckpt because it'd be better that the user have their
        # regular checkpoint separate from the checkpoint for
        # PreemptionCheckpointHandler, since we will create CheckpointManager
        # to manage the checkpoint and only one CheckpointManager should be
        # active in a particular directory at a time.
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
          if cluster_spec and (
              distribution_strategy_context.get_distribution_strategy(
              ).cluster_resolver.task_id == raise_app_error_on_worker):
            raise errors_impl.ResourceExhaustedError(
                node_def=None, op=None, message='Running out of resources')

          model.v.assign_add(constant_op.constant(1.))

        strategy.run(train_step)

        if current_step == STEPS_PER_EPOCH - 1:
          logging.info('epoch %d finished', current_epoch)

      logging.info('Start training at %d',
                   preemption_handler.total_run_calls)

      # If the training process has been restarted, verify that the expected
      # number of checkpoints have been written.
      # we also want to check training_finished, because there's a corner case
      # where the signal is sent quite late and training finishes before the
      # grace period ends.
      if training_restarted and training_restarted.is_set(
      ) and not training_finished.is_set():
        logging.info('training restarted')
        match_group = [
            re.search(r'.*ckpt-(\d+).index', a_file)
            for a_file in gfile.ListDirectory(checkpoint_dir)
        ]
        checkpoint_index = [
            a_match.group(1) for a_match in match_group if a_match
        ]
        if getattr(termination_config, 'grace_period', 0):
          # Two checkpoints were saved for the extended grace period.
          self.assertEqual(int(checkpoint_index[0]), 2)
        else:
          self.assertEqual(int(checkpoint_index[0]), 1)

      for epoch in range(
          preemption_handler.total_run_calls // STEPS_PER_EPOCH,
          EPOCHS_TO_RUN):

        for step in range(
            preemption_handler.total_run_calls % STEPS_PER_EPOCH,
            STEPS_PER_EPOCH):
          if api_wrapping_train:
            preemption_handler.run(distributed_train_step, epoch, step)
          else:
            preemption_handler._save_checkpoint_if_preempted()
            distributed_train_step(epoch, step)
        # Add some randomness to when preemption actually happens. We should
        # trigger it for sure if the training is coming to an end and it hasn't
        # been triggered yet.
        if epoch >= EPOCHS_TO_RUN - 2:
          trigger_it = True
        else:
          trigger_it = False

        self._maybe_trigger_a_preemption(training_started_event, trigger_it)

      training_finished.set()

      logging.info('Training finished.')

      self.assertEqual(
          model.v.numpy(),
          strategy.num_replicas_in_sync * EPOCHS_TO_RUN * STEPS_PER_EPOCH)

  @combinations.generate(
      combinations.combine(
          input_arg=['checkpoint', 'manager'],
          mwms_mode=['local', 'multi_worker'],
          api_wrapping_train=[True, False]))
  def test_preemption_checkpointing(self, input_arg, mwms_mode,
                                    api_wrapping_train):
    has_chief = False

    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt')

    if mwms_mode == 'multi_worker':
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief,
          num_workers=CLUSTER_SIZE)
      training_started_event = multi_process_runner.manager().Event()
      training_restarted = multi_process_runner.manager().Event()
      training_finished = multi_process_runner.manager().Event()

      mpr = multi_process_runner.MultiProcessRunner(
          self.worker_fn,
          cluster_spec,
          args=(checkpoint_dir, cluster_spec, input_arg,
                [training_started_event
                ], None, training_restarted, training_finished),
          kwargs={'api_wrapping_train': api_wrapping_train},
          rpc_layer=rpc_layer,
          return_output=True,
          dependence_on_chief=has_chief)

      logging.info('Cluster starting.')
      mpr.start()
      while not training_started_event.is_set():
        time.sleep(1)

      logging.info('sending sigterm')
      killed_worker = random.randrange(0, CLUSTER_SIZE)
      os.kill(mpr.get_process_id('worker', killed_worker), signal.SIGTERM)

      logging.info('sigterm sent')
      raise_if_not_all_exit(0, mpr)

      logging.info('restarting workers')
      training_restarted.set()
      for worker_id in range(CLUSTER_SIZE):
        mpr.start_single_process('worker', worker_id, cluster_spec)
      logging.info('workers restarted')

      mpr.join(timeout=270)

    else:
      cluster_spec = server_lib.ClusterSpec({})

      training_started_event = threading.Event()
      training_restarted = threading.Event()
      training_finished = threading.Event()

      def sending_sigterm(training_started_event):
        while not training_started_event.is_set():
          time.sleep(1)
        logging.info('sending sigterm')
        training_started_event.set()
        os.kill(os.getpid(), signal.SIGTERM)

      preemption_sender_thread = threading.Thread(
          target=sending_sigterm, args=(training_started_event,))
      preemption_sender_thread.start()

      caught_exit = False
      try:
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg,
                       [training_started_event], None, training_restarted,
                       training_finished)

      except SystemExit as exit_error:
        caught_exit = True
        # We cannot use assertRaise instead, since termination is not always
        # triggered.
        self.assertEqual(exit_error.code, 42)  # pylint: disable=g-assert-in-except

      preemption_sender_thread.join(10)
      if not training_finished.is_set():
        self.assertTrue(caught_exit)

        logging.info('restarting workers')
        training_restarted.set()
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg,
                       [training_started_event], None, training_restarted,
                       training_finished)

  def test_error_propagation(self):
    error_worker = random.randint(0, CLUSTER_SIZE)
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=False, num_workers=CLUSTER_SIZE)
    checkpoint_dir = self.get_temp_dir()

    def assert_raise_error():
      # Asserts that an error raised during a training step on one of the worker
      # is caught on all workers.
      with self.assertRaises(errors_impl.ResourceExhaustedError) as error:
        self.worker_fn(
            checkpoint_dir,
            cluster_spec,
            raise_app_error_on_worker=error_worker)
      self.assertIn('Running out of resources', str(error.exception))

    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    mpr = multi_process_runner.MultiProcessRunner(
        assert_raise_error,
        cluster_spec,
        rpc_layer=rpc_layer,
        return_output=True,
        dependence_on_chief=False)

    logging.info('Cluster starting.')
    mpr.start()
    mpr.join(timeout=250)

  @combinations.generate(
      combinations.combine(input_arg=['checkpoint', 'manager'],
                           mwms_mode=['local', 'multi_worker'],))
  def test_grace_period_continue_training(self, input_arg, mwms_mode):
    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt')

    if mwms_mode == 'multi_worker':
      grace_period = 5
      termination_config = failure_handling.TerminationConfig(
          grace_period=grace_period)
      has_chief = False
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief,
          num_workers=CLUSTER_SIZE)
      training_started_event = multi_process_runner.manager().Event()
      training_restarted = multi_process_runner.manager().Event()
      training_finished = multi_process_runner.manager().Event()

      mpr = multi_process_runner.MultiProcessRunner(
          self.worker_fn,
          cluster_spec,
          args=(checkpoint_dir, cluster_spec, input_arg,
                [training_started_event], None, training_restarted,
                training_finished, termination_config),
          rpc_layer=rpc_layer,
          return_output=True,
          dependence_on_chief=has_chief)

      logging.info('Cluster starting.')
      mpr.start()
      while not training_started_event.is_set():
        time.sleep(1)

      killed_worker = random.randrange(0, CLUSTER_SIZE)
      logging.info('sending SIGTERM')
      os.kill(mpr.get_process_id('worker', killed_worker), signal.SIGTERM)
      logging.info('SIGTERM sent')

      raise_if_not_all_exit(grace_period, mpr)

      logging.info('restarting workers')
      training_restarted.set()
      for worker_id in range(CLUSTER_SIZE):
        mpr.start_single_process('worker', worker_id, cluster_spec)
      logging.info('workers restarted')

      mpr.join(timeout=250)

    else:
      # This is because single worker trains super fast with regards to the size
      # of "model" here. With a longer grace period, the training just finishes
      # within the grace period so we can't verify the exit behavior.
      grace_period = 1
      termination_config = failure_handling.TerminationConfig(
          grace_period=grace_period)
      cluster_spec = server_lib.ClusterSpec({})

      training_started_event = threading.Event()
      training_restarted = threading.Event()
      training_finished = threading.Event()
      def sending_sigterm(training_started_event):
        while not training_started_event.is_set():
          time.sleep(1)
        logging.info('sending sigterm')
        training_started_event.set()
        os.kill(os.getpid(), signal.SIGTERM)

      preemption_sender_thread = threading.Thread(
          target=sending_sigterm, args=(training_started_event,))
      preemption_sender_thread.start()

      caught_exit = False
      try:
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg,
                       [training_started_event], None, training_restarted,
                       training_finished, termination_config)

      except SystemExit as exit_error:
        caught_exit = True
        # We cannot use assertRaise instead, since termination is not always
        # triggered.
        self.assertEqual(exit_error.code, 42)  # pylint: disable=g-assert-in-except

      preemption_sender_thread.join(10)
      if not training_finished.is_set():
        self.assertTrue(caught_exit)

        logging.info('restarting workers')
        training_restarted.set()
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg,
                       [training_started_event], None, training_restarted,
                       training_finished, termination_config)

  def test_passed_in_save_fn(self):
    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt')
    gfile.MakeDirs(checkpoint_dir)

    save_fn = lambda: print('Do nothing')
    termination_config = failure_handling.TerminationConfig(
        save_fn=save_fn)

    has_chief = False
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=has_chief,
        num_workers=CLUSTER_SIZE)
    training_started_event = multi_process_runner.manager().Event()

    mpr = multi_process_runner.MultiProcessRunner(
        self.worker_fn,
        cluster_spec,
        args=(checkpoint_dir, cluster_spec, 'checkpoint',
              [training_started_event], None, None, None, termination_config),
        rpc_layer=rpc_layer,
        return_output=True,
        dependence_on_chief=has_chief)

    logging.info('Cluster starting.')
    mpr.start()
    while not training_started_event.is_set():
      time.sleep(1)

    killed_worker = random.randrange(0, CLUSTER_SIZE)
    logging.info('sending SIGTERM')
    os.kill(mpr.get_process_id('worker', killed_worker), signal.SIGTERM)
    logging.info('SIGTERM sent')

    # 5 is the grace period length
    raise_if_not_all_exit(5, mpr)

    match_group = [
        re.search(r'.*ckpt-(\d+).index', a_file)
        for a_file in gfile.ListDirectory(checkpoint_dir)
    ]

    # By default, as tested by other test cases, checkpoint will be saved.
    # This passed in save_fn skips it.
    self.assertEmpty(match_group)


if __name__ == '__main__':
  test_util.main()
