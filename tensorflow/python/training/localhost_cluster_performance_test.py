# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests and benchmarks for creating RPC clusters on localhost."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import portpicker

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter
from tensorflow.python.training import server_lib


def create_local_cluster(num_workers, num_ps, protocol="grpc"):
  """Create local GRPC servers and return their servers."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]
  cluster_dict = {
      "worker": ["localhost:%s" % port for port in worker_ports],
      "ps": ["localhost:%s" % port for port in ps_ports]
  }
  cs = server_lib.ClusterSpec(cluster_dict)

  workers = [
      server_lib.Server(
          cs, job_name="worker", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_workers)
  ]
  ps_servers = [
      server_lib.Server(
          cs, job_name="ps", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_ps)
  ]

  return workers, ps_servers


class CreateLocalClusterTest(test.TestCase):

  def testCreateLocalCluster(self):
    workers, _ = create_local_cluster(num_workers=2, num_ps=2)
    worker_sessions = [session_lib.Session(w.target) for w in workers]
    with ops.device("/job:ps/task:0"):
      var0 = variables.Variable(0.0)
    with ops.device("/job:ps/task:1"):
      var1 = variables.Variable(1.0)
    worker_sessions[0].run([var0.initializer, var1.initializer])
    with ops.device("/job:ps/task:0"):
      var2 = variables.Variable(2.0)
    with ops.device("/job:ps/task:1"):
      var3 = variables.Variable(3.0)
    worker_sessions[1].run([var2.initializer, var3.initializer])

    # Read values back in the opposite session
    self.assertAllEqual(0.0, var0.eval(session=worker_sessions[1]))
    self.assertAllEqual(1.0, var1.eval(session=worker_sessions[1]))
    self.assertAllEqual(2.0, var2.eval(session=worker_sessions[0]))
    self.assertAllEqual(3.0, var3.eval(session=worker_sessions[0]))


class CreateLocalClusterBenchmark(test.Benchmark):

  def benchmarkCreateLocalCluster(self):
    deltas = []
    iters = 5
    for _ in range(iters):
      start_time = time.time()
      create_local_cluster(num_workers=1, num_ps=10)
      end_time = time.time()
      deltas.append(end_time - start_time)

    median_deltas = np.median(deltas)
    print("\n\nbenchmark_create_local_cluster_1_worker_10_ps.  "
          "iterations: %d, median wall time: %g\n\n" % (iters, median_deltas))
    self.report_benchmark(
        iters=iters,
        wall_time=median_deltas,
        name="benchmark_create_local_cluster_1_worker_10_ps")


class PartitionedVariablesBenchmark(test.Benchmark):

  def benchmark_create_1000_partitions_with_100_parameter_servers(self):
    workers, _ = create_local_cluster(num_workers=1, num_ps=100)
    worker_sessions = [session_lib.Session(w.target) for w in workers]
    worker = worker_sessions[0]
    partition_sizes = (1, 512, 1024 * 32, 1024 * 128)

    partitioned = []

    for partition_size in partition_sizes:
      # max_shard_bytes is 4, shape is 1000*partition_size float32s which should
      # partition into 1000 shards, each containing partition_size float32s.
      print("Building partitioned variable with %d floats per partition" %
            partition_size)
      with ops.device(device_setter.replica_device_setter(ps_tasks=100)):
        partitioned_ix = variable_scope.get_variable(
            "partitioned_%d" % partition_size,
            shape=[1000 * partition_size],
            dtype=dtypes.float32,
            # Each partition to have exactly N float32s
            partitioner=partitioned_variables.variable_axis_size_partitioner(
                max_shard_bytes=4 * partition_size))
        # Concatenates along axis 0
        partitioned.append(ops.convert_to_tensor(partitioned_ix))

    variables.global_variables_initializer().run(session=worker)

    for ix, partition_size in enumerate(partition_sizes):
      print("Running benchmark having partitions with %d floats" %
            partition_size)
      self.run_op_benchmark(
          worker,
          partitioned[ix],
          name=("read_concat_1000_partitions_from_"
                "100_parameter_servers_partsize_%d_floats" % partition_size))


if __name__ == "__main__":
  test.main()
