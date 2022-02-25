# Lint as: python3
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
"""Tests for CoordinatedCheckpointManager."""
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
EPOCHS_TO_RUN = 15
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
  """Integration test for CoordinatedCheckpointManager."""

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
        clear_events[0].set()
      elif random.randrange(0, 9) > 6:
        clear_events[0].set()

  def worker_fn(self,
                checkpoint_dir,
                cluster_spec,
                training_started_event=None,
                raise_app_error_on_worker=None):

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
        # CoordinatedCheckpointManager, since we will create CheckpointManager
        # to manage the checkpoint and only one CheckpointManager should be
        # active in a particular directory at a time.
        fh_ckpt = tracking_util.Checkpoint(model=model)

        failure_handler = failure_handling.CoordinatedCheckpointManager(
            strategy.cluster_resolver, fh_ckpt, checkpoint_dir)

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

      logging.info('Restored training at %d', failure_handler.total_runs)
      for epoch in range(failure_handler.total_runs // STEPS_PER_EPOCH,
                         EPOCHS_TO_RUN):

        for step in range(failure_handler.total_runs % STEPS_PER_EPOCH,
                          STEPS_PER_EPOCH):
          failure_handler.run(distributed_train_step, epoch, step)
        # Add some randomness to when preemption actually happens. We should
        # trigger it for sure if the training is coming to an end and it hasn't
        # been triggered yet.
        if epoch >= EPOCHS_TO_RUN - 2:
          trigger_it = True
        else:
          trigger_it = False

        self._maybe_trigger_a_preemption(training_started_event, trigger_it)

      self.assertEqual(
          model.v.numpy(),
          strategy.num_replicas_in_sync * EPOCHS_TO_RUN * STEPS_PER_EPOCH)

  def test_preemption_checkpointing(self):
    has_chief = False
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=has_chief,
        num_workers=CLUSTER_SIZE)
    training_started_event = multi_process_runner.manager().Event()

    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt')

    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    mpr = multi_process_runner.MultiProcessRunner(
        self.worker_fn,
        cluster_spec,
        args=(checkpoint_dir, cluster_spec, [training_started_event]),
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
    for worker_id in range(CLUSTER_SIZE):
      mpr.start_single_process('worker', worker_id, cluster_spec)
    logging.info('workers restarted')

    stdout = mpr.join().stdout
    all_start_point = []
    for msg in stdout:
      matched_group = re.search(r'.*Restored training at (\d+)', msg)

      if matched_group:
        all_start_point.append(int(matched_group.group(1)))

    # remove duplicate logs created due to presence of multiple workers
    start_points = all_start_point[::CLUSTER_SIZE]

    # assert that after restarting, we don't repeat previous training steps
    self.assertNotEqual(start_points[-1], 0)

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
    mpr.join()


if __name__ == '__main__':
  test_util.main()
