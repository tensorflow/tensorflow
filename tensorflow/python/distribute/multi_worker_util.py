# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for multi-worker distribution strategies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.training import server_lib


def normalize_cluster_spec(cluster_spec):
  """Makes `cluster_spec` into a `ClusterSpec` object.

  Args:
    cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the
      cluster configurations.

  Returns:
    a `ClusterSpec` object.

  Raises:
    ValueError: if `cluster_spec` is not a dict or a `ClusterSpec` or a
      `ClusterDef`.
  """
  if isinstance(cluster_spec, (dict, cluster_pb2.ClusterDef)):
    return server_lib.ClusterSpec(cluster_spec)
  elif not isinstance(cluster_spec, server_lib.ClusterSpec):
    raise ValueError(
        "`cluster_spec' should be dict or a `tf.train.ClusterSpec` or a "
        "`tf.train.ClusterDef` object")
  return cluster_spec


def is_chief(cluster_spec, task_type, task_id):
  """Returns whether the given task is chief in the cluster.

  Args:
    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object specifying the
      cluster configurations.
    task_type: the task type in the cluster.
    task_id: the task id in the cluster.

  Returns:
    a boolean indicating whether the given task is chief.

  Raises:
    ValueError: if `task_type` is not in the `cluster_spec` or `task_id` exceeds
      the maximum id of the `task_type`.
  """
  cluster_spec = normalize_cluster_spec(cluster_spec)
  if task_type not in cluster_spec.jobs:
    raise ValueError(
        "The task_type \"%s\" is not in the `cluster_spec`." % task_type)
  if task_id >= cluster_spec.num_tasks(task_type):
    raise ValueError("The `task_id` %d exceeds the maximum id of %s." % (
        task_id, task_type))

  if task_type == "chief":
    return True

  # If chief not in the cluster_spec, use the first worker as chief. This is
  # common in CollectiveAllReduceStrategy.
  if ("chief" not in cluster_spec.jobs and task_type == "worker" and
      task_id == 0):
    return True
  return False
