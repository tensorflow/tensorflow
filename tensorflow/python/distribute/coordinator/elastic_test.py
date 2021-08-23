# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Elastic test for parameter server training in TF2."""
import threading
import unittest
import time
import sys
import gc

from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import context
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_runner, multi_worker_test_base, parameter_server_strategy_v2
from tensorflow.python.distribute.coordinator.fault_tolerance_test import Model
from tensorflow.python.training import server_lib


_RPC_ERROR_FROM_WORKER = "GRPC error information from remote target /job:worker"
_RPC_ERROR_FROM_PS = "GRPC error information from remote target /job:ps"
_WORKER_PREEMPTION_THREAD_NAME = "WorkerPreemptionHandler"
_WORKER_THREAD_PREFIX = "WorkerClosureProcessingLoop"


class BaseElasticTest(object):  # pylint: disable=missing-docstring

  def setUp(self, num_workers, num_ps, thread_coordinator=None):
    super(BaseElasticTest, self).setUp()

    self._cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=num_workers, num_ps=num_ps, rpc_layer="grpc")
    self._cluster_def = self._cluster.cluster_resolver.cluster_spec().as_dict()
    self._cluster_def["chief"] = [
      "localhost:%d" % multi_worker_test_base.pick_unused_port()
    ]
    self.cluster_resolver = SimpleClusterResolver(
      server_lib.ClusterSpec(self._cluster_def), rpc_layer="grpc")

    # The strategy's constructor would connect to the cluster.
    self.strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
      self.cluster_resolver)
    self.cluster_coord = cluster_coordinator.ClusterCoordinator(self.strategy)

    self.thread_coord = thread_coordinator.Coordinator(
      clean_stop_exception_types=[])
    self.num_workers = num_workers
    self.num_ps = num_ps
    self.max_idx_worker = num_workers - 1
    self.max_idx_ps = num_ps - 1

  def tearDown(self):
    super(BaseElasticTest, self).tearDown()
    self._cluster.stop()
    self._cluster = None

  def _restart(self, downtime_secs, task_type, killed_idx, restart_idx):
    """Kills `job` (index: 0) and restarts it after `downtime_secs` with
       a new name if specified.

    Args:
      downtime_secs: secs before restarting the job.
      task_type: a string specifying the type of job to restart.
      killed_idx: a string specifying the index of the old job.
      restart_idx: a string specifying the index of the new job.
    """
    self._remove_participant(downtime_secs, task_type=task_type,
                             killed_idx=killed_idx)
    self._add_participant(task_type=task_type, restart_idx=restart_idx)

  def _restart_in_thread(self,
                         downtime_secs, task_type, killed_idx, restart_idx):
    def _restart_fn():
      with self.thread_coord.stop_on_exception():
        self._restart(downtime_secs, task_type, killed_idx, restart_idx)

    restart_thread = threading.Thread(target=_restart_fn)
    restart_thread.start()
    return restart_thread

  def _remove_participant(self, downtime_secs, task_type, killed_idx):
    self._cluster.kill_task(task_type=task_type, task_id=killed_idx)
    time.sleep(downtime_secs)
    self.cluster_resolver._cluster_spec.remove_task(task_type, killed_idx)

  def _add_participant(self, task_type, restart_idx):
    self.assertFalse(
      context.check_alive(f"/job:{task_type}/replica:0/task:{restart_idx}"))
    self._cluster.start_task(task_type=task_type, task_id=restart_idx)
    while not context.check_alive(
            f"/job:{task_type}/replica:0/task:{restart_idx}"):
      time.sleep(1)
    self.cluster_resolver._cluster_spec.add_task(task_type, restart_idx)
    if task_type.lower() == "worker":
      self.max_idx_worker = max(self.max_idx_worker, restart_idx)
    elif task_type.lower() == "ps" or task_type.lower() == "parameterserver":
      self.max_idx_ps = max(self.max_idx_ps, restart_idx)

  def _ensure_threads_closed(self):
    """Ensures worker and preemption threads are closed."""
    # Worker and preemption threads should exist before releasing
    # ClusterCoordinator.
    running_threads = test_util.get_running_threads()
    self.assertTrue(
      test_util.has_thread(_WORKER_THREAD_PREFIX, running_threads))
    self.assertIn(_WORKER_PREEMPTION_THREAD_NAME, running_threads)

    # Print object graph if ClusterCoordinator may leak.
    if sys.getrefcount(self.cluster_coord) > 2:
      try:
        test_util.show_backref(self.cluster_coord)
      except:  # pylint: disable=bare-except
        pass

    # Wait for threads to close.
    self.cluster_coord = None
    self.strategy = None
    gc.collect()
    time.sleep(1)

    # Verify thread names.
    running_threads = test_util.get_running_threads()
    self.assertNotIn(_WORKER_PREEMPTION_THREAD_NAME, running_threads)
    self.assertFalse(
      test_util.has_thread(_WORKER_THREAD_PREFIX, running_threads),
      "Worker thread is not stopped properly.")

  def _create_model_and_run_indefinitely(self):
    model = Model(self.cluster_coord)
    model.do_infinite_step.assign(True)
    model.schedule_training_functions(10)
    # Model does infinite training step, so at this moment, we expect to have
    # `self.num_workers` infinite closures inflight, and `10-self.num_workers`
    # closures in the queue.
    while (self.cluster_coord._cluster.closure_queue._inflight_closure_count <
           self.num_workers):
      time.sleep(0.1)
    return model
  
  def testClusterCoordinatorDestroyed(self):
    self._ensure_threads_closed()

  def testWorkerAdditionBetweenFunctions(self):
    model = Model(self.cluster_coord)
    model.schedule_training_functions(2)
    model.join_training_functions()
    self.assertEqual(model.iterations.numpy(), 2)

    self._add_participant("worker", self.max_idx_worker + 1)
    # TODO(zw0610): update coordinator with the new worker

    model.schedule_training_functions(2)
    model.join_training_functions()
    self.assertEqual(model.iterations.numpy(), 4)

  def testWorkerAdditionMidstFunction(self):
    model = Model(self.cluster_coord)
    model.do_infinite_step.assign(True)

    model.schedule_training_functions(4)
    # Model does infinite training step, so at this moment, we expect to have
    # `self.num_workers` infinite closures inflight, and `4-self.num_workers`
    # closures in the queue.
    while (self.cluster_coord._cluster.closure_queue._inflight_closure_count <
           self.num_workers):
      time.sleep(0.1)
    self.assertFalse(self.cluster_coord.done())
    self._add_participant("worker", self.max_idx_worker + 1)
    # TODO(zw0610): update coordinator with the new worker
    model.join_training_functions()
    self.assertGreaterEqual(model.iterations.numpy(), 4)

  def testWorkerRestoreWithSameName(self):
    model = Model(self.cluster_coord)
    model.schedule_training_functions(2)
    model.join_training_functions()
    self.assertEqual(model.iterations.numpy(), 2)

    self._restart(
      downtime_secs=2, task_type="worker", killed_idx=0, restart_idx=0)

    model.schedule_training_functions(2)
    model.join_training_functions()
    self.assertEqual(model.iterations.numpy(), 4)

  def testWorkerRestoreWithAnotherName(self):
    model = Model(self.cluster_coord)
    model.schedule_training_functions(2)
    model.join_training_functions()
    self.assertEqual(model.iterations.numpy(), 2)

    self._restart(downtime_secs=2,
                  task_type="worker",
                  killed_idx=0,
                  restart_idx=self.max_idx_worker + 1)

    model.schedule_training_functions(2)
    model.join_training_functions()
    self.assertEqual(model.iterations.numpy(), 4)


class MultiWorkerElasticTest(BaseElasticTest, unittest.TestCase):
  """Multi worker elastic tests.

  This covers the ordinary cases where multiple workers and PS are used.
  """

  def setUp(self):
    super(MultiWorkerElasticTest, self).setUp(2, 2)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
