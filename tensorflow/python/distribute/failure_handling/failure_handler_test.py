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
"""Tests for WorkerPreemptionHandler."""
import os
import random
import re
import signal
import sys
import time

from absl.testing import parameterized

# pylint:disable=g-direct-tensorflow-import

from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.failure_handling import failure_handling
from tensorflow.python.distribute.failure_handling import gce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.module import module
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import util as tracking_util


mock = test.mock


CLUSTER_SIZE = 4
EPOCHS_TO_RUN = 8
STEPS_PER_EPOCH = 15


def _is_oss():
  """Returns whether the test is run under OSS."""
  return len(sys.argv) >= 1 and 'bazel' in sys.argv[0]


def _enable_coordination_service(cluster_spec):

  if context.context().coordination_service is None:
    coordination_service = 'standalone'
    coordinated_jobs = ['chief', 'worker']
    context.context().configure_coordination_service(
        service_type=coordination_service,
        service_leader=multi_worker_util.coordination_leader(
            cluster_spec),
        coordinated_jobs=coordinated_jobs)


class PreemptionCheckpointTest(test.TestCase, parameterized.TestCase):
  """Integration test for WorkerPreemptionHandler."""

  def _mwms_write_checkpoint_dir(self, checkpoint_dir, cluster_spec, task_type,
                                 task_id):
    dirpath = os.path.dirname(checkpoint_dir)
    base = os.path.basename(checkpoint_dir)
    if not multi_worker_util.is_chief(
        cluster_spec=cluster_spec, task_type=task_type, task_id=task_id):
      base_dirpath = 'workertemp_' + str(task_id)
      dirpath = os.path.join(dirpath, base_dirpath)
      gfile.MakeDirs(dirpath)
    return os.path.join(dirpath, base)

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
                training_started_event=None,
                raise_app_error_on_worker=None,
                training_restarted=None,
                training_finished=None,
                termination_config=failure_handling.TerminationConfig()):

    _enable_coordination_service(cluster_spec)
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

    with mock.patch.object(gce_util, 'on_gcp', lambda: False):

      with strategy.scope():
        model = Model()
        # Named it fh_ckpt because it'd be better that the user have their
        # regular checkpoint separate from the checkpoint for
        # WorkerPreemptionHandler, since we will create CheckpointManager
        # to manage the checkpoint and only one CheckpointManager should be
        # active in a particular directory at a time.
        fh_ckpt = tracking_util.Checkpoint(model=model)

        worker_preemption_watcher = failure_handling.WorkerPreemptionHandler(
            strategy.cluster_resolver, fh_ckpt, checkpoint_dir,
            termination_config)

      def distributed_train_step(current_epoch, current_step):

        @def_function.function
        def train_step():
          if distribution_strategy_context.get_distribution_strategy(
          ).cluster_resolver.task_id == raise_app_error_on_worker:
            raise errors_impl.ResourceExhaustedError(
                node_def=None, op=None, message='Running out of resources')

          model.v.assign_add(constant_op.constant(1.))

        strategy.run(train_step)

        if current_step == STEPS_PER_EPOCH - 1:
          logging.info('epoch %d finished', current_epoch)

      logging.info('Start training at %d',
                   worker_preemption_watcher.total_runs)

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
        if getattr(termination_config, 'time_till_termination', 0):
          # Two checkpoints were saved for the extended grace period.
          self.assertEqual(int(checkpoint_index[0]), 2)
        else:
          self.assertEqual(int(checkpoint_index[0]), 1)

      for epoch in range(
          worker_preemption_watcher.total_runs // STEPS_PER_EPOCH,
          EPOCHS_TO_RUN):

        for step in range(
            worker_preemption_watcher.total_runs % STEPS_PER_EPOCH,
            STEPS_PER_EPOCH):
          worker_preemption_watcher.run(distributed_train_step, epoch, step)
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

  def test_preemption_checkpointing(self):
    has_chief = False
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=has_chief,
        num_workers=CLUSTER_SIZE)
    training_started_event = multi_process_runner.manager().Event()
    training_restarted = multi_process_runner.manager().Event()
    training_finished = multi_process_runner.manager().Event()

    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt')

    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    mpr = multi_process_runner.MultiProcessRunner(
        self.worker_fn,
        cluster_spec,
        args=(checkpoint_dir, cluster_spec, [training_started_event], None,
              training_restarted, training_finished),
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
    time.sleep(5)

    logging.info('restarting workers')
    training_restarted.set()
    for worker_id in range(CLUSTER_SIZE):
      mpr.start_single_process('worker', worker_id, cluster_spec)
    logging.info('workers restarted')

    mpr.join(timeout=270)

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

  def test_grace_period_continue_training(self):
    grace_period = 5
    has_chief = False
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=has_chief,
        num_workers=CLUSTER_SIZE)
    training_started_event = multi_process_runner.manager().Event()
    training_restarted = multi_process_runner.manager().Event()
    training_finished = multi_process_runner.manager().Event()
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt')

    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    termination_config = failure_handling.TerminationConfig(
        time_till_termination=grace_period)
    mpr = multi_process_runner.MultiProcessRunner(
        self.worker_fn,
        cluster_spec,
        args=(checkpoint_dir, cluster_spec, [training_started_event], None,
              training_restarted, training_finished, termination_config),
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

    # wait for all cluster within the given grace period (plus a buffer since
    # our per-step time here is too small)
    waiting_time = 0
    exit_process_count = 0
    while exit_process_count != CLUSTER_SIZE and waiting_time < grace_period + 10:
      exit_process_count = 0
      for worker_id in range(CLUSTER_SIZE):
        if not mpr.process_exists('worker', worker_id):
          exit_process_count += 1
      waiting_time += 1
      time.sleep(1)

    if waiting_time == grace_period + 10:
      raise RuntimeError('Waited exceeding grace period. ')

    logging.info('restarting workers')
    training_restarted.set()
    for worker_id in range(CLUSTER_SIZE):
      mpr.start_single_process('worker', worker_id, cluster_spec)
    logging.info('workers restarted')

    mpr.join(timeout=250)


if __name__ == '__main__':
  test_util.main()
