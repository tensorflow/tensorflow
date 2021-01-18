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
"""Tests for metrics collecting in coordinator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.training.server_lib import ClusterSpec


class MetricUtilsTest(test.TestCase):

  def get_rpc_layer(self):
    return 'grpc'

  def testClusterCoordinatorMetrics(self):

    metric_utils.enable_metrics = True

    cluster_def = multi_worker_test_base.create_in_process_cluster(
        num_workers=1, num_ps=1, rpc_layer=self.get_rpc_layer())
    cluster_def['chief'] = [
        'localhost:%d' % multi_worker_test_base.pick_unused_port()
    ]
    cluster_resolver = SimpleClusterResolver(
        ClusterSpec(cluster_def), rpc_layer=self.get_rpc_layer())
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        cluster_resolver)
    cluster = coordinator_lib.Cluster(strategy)

    @def_function.function
    def func():
      time.sleep(0.5)
      return 3

    result = cluster.schedule(func, args=None, kwargs=None)
    result = cluster.schedule(func, args=None, kwargs=None)
    cluster.join()
    self.assertEqual(result.fetch(), 3)

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


if __name__ == '__main__':
  test.main()
