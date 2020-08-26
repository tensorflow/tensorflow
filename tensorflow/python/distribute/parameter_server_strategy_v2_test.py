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
"""Tests for parameter_server_strategy_v2.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform
import sys

from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.ops import variables
from tensorflow.python.training.server_lib import ClusterSpec


class ParameterServerStrategyV2Test(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ParameterServerStrategyV2Test, cls).setUpClass()
    cluster_def = multi_worker_test_base.create_in_process_cluster(
        num_workers=2, num_ps=3, rpc_layer="grpc")
    cls.cluster_resolver = SimpleClusterResolver(
        ClusterSpec(cluster_def), rpc_layer="grpc")
    remote.connect_to_cluster(
        cls.cluster_resolver.cluster_spec(),
        job_name="chief",
        protocol=cls.cluster_resolver.rpc_layer)

  def testVariablePlacement(self):

    if sys.version_info >= (3, 8) and platform.system() == "Windows":
      # TODO(b/165013260): Fix this
      self.skipTest("Test is currently broken on Windows with Python 3.8")

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    v1 = variables.Variable(initial_value=0.0)
    with strategy.scope():
      v2 = variables.Variable(initial_value=1.0)
      v3 = variables.Variable(initial_value=2.0)
      v4 = variables.Variable(initial_value=3.0)
      v5 = variables.Variable(initial_value=4.0)
    # v1 was created outside scope so should be on client.
    self.assertEqual(v1.device, "/job:chief/replica:0/task:0/device:CPU:0")
    # v2 through v5 are created in scope and in a round-robin manner.
    self.assertEqual(v2.device, "/job:ps/replica:0/task:0/device:CPU:0")
    self.assertEqual(v3.device, "/job:ps/replica:0/task:1/device:CPU:0")
    self.assertEqual(v4.device, "/job:ps/replica:0/task:2/device:CPU:0")
    self.assertEqual(v5.device, "/job:ps/replica:0/task:0/device:CPU:0")


if __name__ == "__main__":
  test.main()
