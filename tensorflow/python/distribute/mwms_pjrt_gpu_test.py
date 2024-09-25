# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

import copy

from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops


# Based on collective_ops_multi_worker_test.py
def enable_collective_ops(cluster_resolver):
  context.context().configure_collective_ops(
      collective_leader="/job:worker/replica:0/task:0")
  config_proto = copy.deepcopy(context.context().config)
  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_resolver.cluster_spec().as_cluster_def(),
      default_session_config=config_proto,
      job_name=cluster_resolver.task_type,
      task_index=cluster_resolver.task_id,
      protocol=cluster_resolver.rpc_layer or "grpc")
  context.context().enable_collective_ops(server_def)


def configure_coordination_service():
  context.context().configure_coordination_service(
      service_type="standalone",
      service_leader="/job:worker/replica:0/task:0",
      enable_health_check=False,
  )


class MultiWorkerMirroredStrategyPjRtRemoteGpuTest(
    multi_worker_test_base.MultiWorkerTestBase, test.TestCase
):
  def testRemoteGpusFound(self):

    def worker_fn():

      configure_coordination_service()
      cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
      enable_collective_ops(cluster_resolver=cluster_resolver)
      context.context().ensure_initialized()

      group_size = 2
      group_key = 1
      instance_key1 = 1
      instance_key2 = 2
      tensor_size = 10

      # cluster_resolver.task_id is int 0 or 1.
      tensor_val = [cluster_resolver.task_id + 1.] * tensor_size
      constant = constant_op.constant(tensor_val)

      @def_function.function(jit_compile=True)
      def g():

        def f(x):
          return 2 * x + 1

        input_tensor1 = array_ops.identity(f(constant))
        input_tensor2 = array_ops.identity(f(constant))

        reduced_tensor1 = collective_ops.all_reduce_v2(
            input_tensor1, group_size, group_key, instance_key1, "Add", "Id")
        reduced_tensor2 = collective_ops.all_reduce_v2(
            input_tensor2, group_size, group_key, instance_key2, "Add", "Id")
        return reduced_tensor1, reduced_tensor2

      return g()

    num_gpus = len(tf_config.list_physical_devices("GPU"))
    skip_flag = (num_gpus < 2) or not test_util.is_xla_enabled()
    if skip_flag:
      self.skipTest(
          "This test is intended to test the 2 GPU (1 per worker with 2"
          " workers) with XLA case (%d GPUs found, using XLA = %s)."
          % (num_gpus, test_util.is_xla_enabled())
      )

    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    mpr = multi_process_runner.MultiProcessRunner(
        worker_fn, cluster_spec, share_gpu=False
    )
    mpr.start()
    mpr_result = mpr.join()
    self.assertLen(mpr_result.return_value, 2)
    for rval in mpr_result.return_value:
      for t in rval:
        # for IDs 0 and 1: (2*(0+1)+1) + (2*(1+1)+1) = 8
        self.assertAllClose(t.numpy(), [8., 8., 8., 8., 8., 8., 8., 8., 8., 8.])


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
