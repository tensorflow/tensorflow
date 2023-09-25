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
"""Fault tolerance test for parameter server training in TF2."""

import threading
import time

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.distribute.coordinator import fault_tolerance_test_base
from tensorflow.python.eager import test
from tensorflow.python.training import coordinator as thread_coordinator
from tensorflow.python.training import server_lib


class MultiWorkerFaultToleranceTest(
    fault_tolerance_test_base.BaseFaultToleranceTest, test.TestCase):
  """Multi worker fault tolerance tests.

  This covers the ordinary cases where multiple workers and PS are used.
  """

  def setUp(self):
    super(MultiWorkerFaultToleranceTest, self).setUp(2, 2)


class SingleWorkerFaultToleranceTest(
    fault_tolerance_test_base.BaseFaultToleranceTest, test.TestCase):
  """Single worker fault tolerance tests.

  This covers the cases that ensure training can continue in a single-worker
  cluster, even if the only worker can become unavailable at some point and
  recovered (if there are multiple workers, it is possible that the training
  succeeds with the workers that did not fail). Realistically single worker
  is very rarely used, but the tests are important to ensure the correct
  behaviors.
  """

  def setUp(self):
    super(SingleWorkerFaultToleranceTest, self).setUp(1, 1)


class InitFaultToleranceTest(test.TestCase):
  """Test preemptions during strategy init."""

  def setUp(self):
    super().setUp()
    self.num_workers = 2
    self.num_ps = 2
    self._cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=self.num_workers,
        num_ps=self.num_ps,
        rpc_layer="grpc",
        stream_output=True,
    )
    self._cluster_def = self._cluster.cluster_resolver.cluster_spec().as_dict()
    self._cluster_def["chief"] = [
        "localhost:%d" % multi_worker_test_base.pick_unused_port()
    ]
    self._cluster_resolver = SimpleClusterResolver(
        server_lib.ClusterSpec(self._cluster_def), rpc_layer="grpc"
    )

    self.thread_coord = thread_coordinator.Coordinator(
        clean_stop_exception_types=[]
    )

  def tearDown(self):
    super().tearDown()
    self._cluster.stop()
    self._cluster = None

  def testWorkerPreemptionDuringInit(self):

    worker_down = threading.Condition()

    def _restart_in_thread(downtime_secs, restart_job):
      # During initial connection in SetOrUpdateServerDef, there is a step that
      # waits to receive a `GetStatus` response from each worker, in an attempt
      # to ensure workers are available before sending `CreateContext` to them.
      # In high-preemption environments, the time between these two requests can
      # be enough for a worker to be preempted, at which point the
      # `CreateContext` request will fail and training cannot start. b/298187302
      # We solve this by enabling retries on `CreateContext` calls when used
      # with PSS.

      # This test reproduces this behavior by bringing down a worker during
      # startup, then waiting long enough for strategy startup to reach the
      # point where it is just waiting for a `GetStatus` response from that
      # worker. Then, we restart the first downed worker and bring down another
      # worker at the same time, so the strategy startup progresses to the
      # `CreateContext` requests, but not before another worker is down.

      def _restart_fn():
        with self.thread_coord.stop_on_exception():
          self._cluster.kill_task(restart_job, 0)
          with worker_down:
            worker_down.notify_all()
          time.sleep(downtime_secs)
          self._cluster.start_task(restart_job, 0)
          self._cluster.kill_task(restart_job, 1)
          time.sleep(downtime_secs)
          self._cluster.start_task(restart_job, 1)

      restart_thread = threading.Thread(target=_restart_fn)
      restart_thread.start()
      return restart_thread

    _restart_in_thread(downtime_secs=2, restart_job="worker")

    with worker_down:
      worker_down.wait()

    # The strategy's constructor would connect to the cluster.
    self.strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self._cluster_resolver
    )
    self.cluster_coord = cluster_coordinator.ClusterCoordinator(self.strategy)

    # Now do a simple training as a sanity check
    model = fault_tolerance_test_base.Model(self.cluster_coord)
    model.schedule_training_functions(2)
    model.join_training_functions()
    self.assertEqual(model.iterations.numpy(), 2)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
