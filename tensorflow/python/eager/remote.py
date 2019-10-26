# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Helpers to connect to remote servers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from absl import logging

from tensorflow.core.protobuf.tensorflow_server_pb2 import ServerDef
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.cluster_resolver import cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import remote_utils
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


_GRPC_PREFIX = "grpc://"


@tf_export("config.experimental_connect_to_host")
def connect_to_remote_host(remote_host=None, job_name="worker"):
  """Connects to a single machine to enable remote execution on it.

  Will make devices on the remote host available to use. Note that calling this
  more than once will work, but will invalidate any tensor handles on the old
  remote devices.

  Using the default job_name of worker, you can schedule ops to run remotely as
  follows:
  ```python
  # Enable eager execution, and connect to the remote host.
  tf.compat.v1.enable_eager_execution()
  tf.contrib.eager.connect_to_remote_host("exampleaddr.com:9876")

  with ops.device("job:worker/replica:0/task:1/device:CPU:0"):
    # The following tensors should be resident on the remote device, and the op
    # will also execute remotely.
    x1 = array_ops.ones([2, 2])
    x2 = array_ops.ones([2, 2])
    y = math_ops.matmul(x1, x2)
  ```

  Args:
    remote_host: a single or a list the remote server addr in host-port format.
    job_name: The job name under which the new server will be accessible.

  Raises:
    ValueError: if remote_host is None.
  """
  if not remote_host:
    raise ValueError("Must provide at least one remote_host")

  remote_hosts = nest.flatten(remote_host)
  cluster_spec = server_lib.ClusterSpec(
      {job_name: [_strip_prefix(host, _GRPC_PREFIX) for host in remote_hosts]})

  connect_to_cluster(cluster_spec)


@tf_export("config.experimental_connect_to_cluster")
def connect_to_cluster(cluster_spec_or_resolver,
                       job_name="localhost",
                       task_index=0,
                       protocol=None,
                       make_master_device_default=True):
  """Connects to the given cluster.

  Will make devices on the cluster available to use. Note that calling this more
  than once will work, but will invalidate any tensor handles on the old remote
  devices.

  If the given local job name is not present in the cluster specification, it
  will be automatically added, using an unused port on the localhost.

  Args:
    cluster_spec_or_resolver: A `ClusterSpec` or `ClusterResolver` describing
      the cluster.
    job_name: The name of the local job.
    task_index: The local task index.
    protocol: The communication protocol, such as `"grpc"`. If unspecified, will
      use the default from `python/platform/remote_utils.py`.
    make_master_device_default: If True and a cluster resolver is passed, will
      automatically enter the master task device scope, which indicates the
      master becomes the default device to run ops. It won't do anything if
      a cluster spec is passed. Will throw an error if the caller is currently
      already in some device scope.
  """
  protocol = protocol or remote_utils.get_default_communication_protocol()
  if isinstance(cluster_spec_or_resolver, server_lib.ClusterSpec):
    cluster_spec = cluster_spec_or_resolver
  elif isinstance(cluster_spec_or_resolver, cluster_resolver.ClusterResolver):
    cluster_spec = cluster_spec_or_resolver.cluster_spec()
  else:
    raise ValueError(
        "`cluster_spec_or_resolver` must be a `ClusterSpec` or a "
        "`ClusterResolver`.")

  cluster_def = copy.deepcopy(cluster_spec.as_cluster_def())

  # Automatically add local job, if not part of the cluster spec.
  if job_name not in cluster_spec.jobs:
    local_port = pywrap_tensorflow.TF_PickUnusedPortOrDie()
    job_def = cluster_def.job.add()
    job_def.name = job_name
    # TODO(fishx): Update this to make sure remote worker has valid ip address
    # to connect with local.
    job_def.tasks[0] = "localhost:{}".format(local_port)

  server_def = ServerDef(
      cluster=cluster_def,
      job_name=job_name,
      task_index=task_index,
      protocol=protocol,
      default_session_config=context.context().config)

  # TODO(b/142500669): Don't leak existing grpc server if calling multiple
  # times.
  context.set_server_def(server_def)

  if make_master_device_default and isinstance(
      cluster_spec_or_resolver,
      cluster_resolver.ClusterResolver) and cluster_spec_or_resolver.master():
    master = cluster_spec_or_resolver.master()
    master_job_name = None
    master_task_id = None
    for job_name in cluster_spec.jobs:
      for task_id in cluster_spec.task_indices(job_name):
        task_address = cluster_spec.task_address(job_name, task_id)
        if master in task_address or task_address in master:
          master_job_name = job_name
          master_task_id = task_id
          break

    if not master_job_name:
      raise ValueError(
          "`make_master_device_default` is set to True but cannot find "
          "master %s in the cluster" % master)

    master_device = "/job:{}/replica:0/task:{}".format(master_job_name,
                                                       master_task_id)
    master_device = device_util.canonicalize(master_device)
    current_device = device_util.current()
    if current_device:
      current_device = device_util.canonicalize(current_device)
    if current_device and current_device != master_device:
      raise ValueError("`connect_to_cluster` is called inside existing device "
                       "scope %s, which is different from the master device "
                       "scope %s to enter. This is not allowed." %
                       (current_device, master_device))
    # TODO(b/138389076): Think of the entering device scope behavior in the
    # failure recovery case when dealing with preemptions.
    if not current_device:
      logging.info("Entering into master device scope: %s", master_device)
      ops.device(master_device).__enter__()


def _strip_prefix(s, prefix):
  return s[len(prefix):] if s.startswith(prefix) else s
