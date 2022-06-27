# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utility wrappers for working with multi-client setups."""

from typing import List, Optional

from absl import logging

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.eager import context
from tensorflow.python.platform import remote_utils


def initialize_multi_client_cluster(job_name: str,
                                    dtensor_jobs: List[str],
                                    client_id: int,
                                    collective_leader: str,
                                    port: Optional[int] = None,
                                    enable_coordination_service: bool = False):
  """Initialize GRPC servers and collectives for multi-client DTensor setup.

  While single clients (e.g. Forge) can use local mode of collectives, GRPC
  servers are necessary in mutli-client setup. This function can be used to
  initialize a cluster and enable collective ops.

  NOTE: this function must be called in an eager context.

  Args:
    job_name: The job name used by all clients in the DTensor cluster.
    dtensor_jobs: A list of the DTensor client jobs participating in the
      cluster. Must be strings of the form "hostname:port".
    client_id: The ID of the DTensor client this function is being called in.
    collective_leader: The job/task that will be used to run collectives.
    port: The port this client's GRPC server will run on.
    enable_coordination_service: If true, enable distributed coordination
      service to make sure that workers know the devices on each other, a
      prerequisite for data transfer through cross-worker rendezvous.

  Raises:
    RuntimeError: If running inside a tf.function.
  """
  assert context.executing_eagerly()

  if not collective_leader.startswith("/job:"):
    collective_leader = "/job:" + collective_leader

  context.context().configure_collective_ops(
      collective_leader=collective_leader)
  if enable_coordination_service:
    context.context().configure_coordination_service(
        service_type="standalone", service_leader=collective_leader)

  config_proto = context.get_config()
  config_proto.experimental.collective_group_leader = collective_leader
  # Construct server def from the host directly instead of relying on
  # TF_CONFIG.
  cluster_def = cluster_pb2.ClusterDef()
  # Note that we will currently rely on the sorted string of job name as the
  # order of assigning task ids. This might be brittle once we have jobs
  # across multiple cells.
  cluster_def.job.add(name=job_name, tasks=dict(enumerate(dtensor_jobs)))
  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_def,
      default_session_config=config_proto,
      job_name=job_name,
      task_index=client_id,
      protocol=remote_utils.get_default_communication_protocol(),
      port=port)
  server_def.default_session_config.rpc_options.num_channels_per_target = 4
  server_def.default_session_config.experimental.recv_buf_max_chunk = -1

  logging.info("Enabling collectives with server_def: %s", server_def)
  context.context().enable_collective_ops(server_def)
  context.ensure_initialized()
