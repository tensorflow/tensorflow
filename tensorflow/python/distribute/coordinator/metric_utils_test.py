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
"""Tests for metrics collecting in coordinator."""

import time
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.training import coordinator as thread_coordinator
from tensorflow.python.training.server_lib import ClusterSpec


# A function that takes long enough that inflight closures can be measured
# as nonzero.
@def_function.function
def long_function():
  x = random_ops.random_uniform((1000, 1000))
  for _ in math_ops.range(100):
    a = random_ops.random_uniform((1000, 1000))
    b = random_ops.random_uniform((1000, 1000))
    x += math_ops.matmul(a, b)
  return x


class MetricUtilsTest(test.TestCase):

  def get_rpc_layer(self):
    return 'grpc'

  def setUp(self):
    super().setUp()

    self._cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=1,
        num_ps=1,
        rpc_layer=self.get_rpc_layer(),
        stream_output=True,
    )
    self._cluster_def = self._cluster.cluster_resolver.cluster_spec().as_dict()
    self._cluster_def['chief'] = [
        'localhost:%d' % multi_worker_test_base.pick_unused_port()
    ]
    cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        ClusterSpec(self._cluster_def), rpc_layer=self.get_rpc_layer()
    )
    self.strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        cluster_resolver
    )
    self.coordinator = coordinator_lib.ClusterCoordinator(self.strategy)

    self.thread_coord = thread_coordinator.Coordinator(
        clean_stop_exception_types=[]
    )
    metric_utils._init()  # Ensure metrics are reset between tests.

  def tearDown(self):
    super().tearDown()
    self._cluster.stop()
    self._cluster = None

  def _restart(self, downtime_secs, job):
    """Kills `job` (index: 0) and restarts it after `downtime_secs`.

    Args:
      downtime_secs: secs before restarting the job.
      job: a string specifying the job to restart.
    """
    self._cluster.kill_task(job, 0)
    time.sleep(downtime_secs)
    self.assertFalse(context.check_alive('/job:%s/replica:0/task:0' % job))
    self._cluster.start_task(job, 0)
    while not context.check_alive('/job:%s/replica:0/task:0' % job):
      time.sleep(1)

  def testSimpleMetrics(self):
    # Add a sleep to increase tracing time
    @def_function.function
    def func():
      time.sleep(0.5)
      return 3

    self.assertEqual(metric_utils.get_metric_summary('queued_closures'), 0)
    self.assertEqual(metric_utils.get_metric_summary('inflight_closures'), 0)

    self.coordinator.schedule(func, args=None, kwargs=None)
    result = self.coordinator.schedule(func, args=None, kwargs=None)
    self.coordinator.join()

    self.assertEqual(metric_utils.get_metric_summary('queued_closures'), 0)
    self.assertEqual(metric_utils.get_metric_summary('inflight_closures'), 0)
    # Tracing, closure execution, and remote_value fetching should be executed
    # exactly once for running this function.
    metric_tracing = metric_utils.get_metric_summary('function_tracing')
    self.assertEqual(metric_tracing['num'], 1)
    # Tracing time should be longer than the sleep time in Python function.
    self.assertGreater(metric_tracing['sum'], 0.5)
    metric_closure = metric_utils.get_metric_summary('closure_execution')
    self.assertEqual(metric_closure['num'], 2)
    metric_remote_value = metric_utils.get_metric_summary('remote_value_fetch')
    self.assertEqual(metric_remote_value['num'], 2)

    self.assertEqual(result.fetch(), 3)

  def testInflightClosures(self):
    self.coordinator.schedule(long_function)
    self.coordinator.schedule(long_function)
    self.assertGreater(metric_utils.get_metric_summary('queued_closures'), 0)

    # inflight closures should be greater than 0 at some point
    max_inflight = 0
    while not self.coordinator.done():
      with self.coordinator._cluster.closure_queue._queue_lock:
        inflight_metric = metric_utils.get_metric_summary('inflight_closures')
      max_inflight = max(max_inflight, inflight_metric)
      time.sleep(0.01)
    self.assertGreater(max_inflight, 0)

  def testWorkerFailureCount(self):
    self.coordinator.schedule(long_function)
    self._restart(downtime_secs=2, job='worker')
    self.coordinator.schedule(long_function)

    self.coordinator.join()
    metric_closure = metric_utils.get_metric_summary('closure_execution')
    self.assertEqual(metric_closure['num'], 2)
    num_failures = metric_utils.get_metric_summary('worker_failures')
    self.assertEqual(num_failures, 1)
    recovery_times = metric_utils.get_metric_summary('server_def_update')
    self.assertEqual(recovery_times['num'], 1)


if __name__ == '__main__':
  multi_process_runner.test_main()
