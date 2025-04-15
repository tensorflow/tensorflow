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
"""Implementation of Cluster Resolvers for SageMaker Environment."""

import json
import os

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec

# List of envs
# https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
# Only support Multi-Worker Mirrored Strategy

_SESSION_MASTER_KEY = 'session_master'
_RPC_LAYER_KEY = 'rpc_layer'
_TASK_KEY = 'task'
_CLUSTER_KEY = 'cluster'
_WORKER_KEY = 'worker'
_INDEX_KEY = 'index'
_TYPE_KEY = 'type'

_SM_CURRENT_HOST = 'SM_CURRENT_HOST'
_SM_HOSTS = 'SM_HOSTS'


def format_master_url(master, rpc_layer=None):
  if rpc_layer:
    return '%s://%s' % (rpc_layer, master)
  else:
    return master


def _load_tf_config(port):
  # Create a tf_config from SM Variables
  assert all([x in os.environ for x in [_SM_CURRENT_HOST, _SM_HOSTS]
             ]), 'Not a SageMaker Environment'
  hosts = sorted(json.loads(
      os.environ[_SM_HOSTS])) if os.environ[_SM_HOSTS] != '' else []
  current_host = os.environ[_SM_CURRENT_HOST]

  if current_host not in hosts:
    return {}

  host_index = hosts.index(current_host)
  # Assign ports
  hosts = ['%s:%s' % (host, port) for host in hosts]

  tf_config = {
      _CLUSTER_KEY: {
          _WORKER_KEY: hosts
      },
      _TASK_KEY: {
          _TYPE_KEY: _WORKER_KEY,
          _INDEX_KEY: host_index
      }
  }
  return tf_config


def _get_value_in_tfconfig(key, port, default=None):
  tf_config = _load_tf_config(port)
  return tf_config[key] if key in tf_config else default


class SageMakerClusterResolver(ClusterResolver):
  """Implementation of a ClusterResolver which reads the Sagemaker EnvVars. This is an implementation of cluster resolvers when running in a SageMaker environment to set information about the cluster.

  The cluster spec returned will be initialized from the SageMaker
  environment variables.
  Currently this Cluster Resolver only supports Multi-Worker Mirrored Strategy.
  It assumes all nodes in a SageMaker Cluster are workers.
  """

  def __init__(self,
               port=2223,
               task_type=None,
               task_id=None,
               rpc_layer=None,
               environment=None):
    """Creates a new SageMakerClusterResolver.

    Args:
      port: (integer, optional) Override default port usage of 2223
      task_type: (String, optional) Overrides the task type.
      task_id: (Integer, optional) Overrides the task index.
      rpc_layer: (String, optional) Overrides the rpc layer TensorFlow uses.
      environment: (String, optional) Overrides the environment TensorFlow
        operates in.
    """
    self._task_type = task_type
    self._task_id = task_id
    self._rpc_layer = rpc_layer
    self._environment = environment
    self._port = str(port)

  @property
  def task_type(self):
    if self._task_type is None:
      task_info = _get_value_in_tfconfig(_TASK_KEY, self._port, {})
      return str(task_info['type']) if 'type' in task_info else None
    else:
      return str(self._task_type)

  @property
  def task_id(self):
    if self._task_id is None:
      task_info = _get_value_in_tfconfig(_TASK_KEY, self._port, {})
      return int(task_info['index']) if 'index' in task_info else None
    else:
      return int(self._task_id)

  @task_type.setter
  def task_type(self, task_type):
    self._task_type = task_type

  @task_id.setter
  def task_id(self, task_id):
    self._task_id = task_id

  @property
  def environment(self):
    return self._environment

  @property
  def rpc_layer(self):
    if self._rpc_layer is None:
      return _get_value_in_tfconfig(_RPC_LAYER_KEY, self._port)
    else:
      return self._rpc_layer

  @rpc_layer.setter
  def rpc_layer(self, rpc_layer):
    self._rpc_layer = rpc_layer

  def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
    task_type = self.task_type if task_type is None else task_type
    task_id = self.task_id if task_id is None else task_id
    return super(SageMakerClusterResolver,
                 self).num_accelerators(task_type, task_id, config_proto)

  def cluster_spec(self):
    """Returns a ClusterSpec based on the SageMaker environment variables.

    Returns:
      A ClusterSpec with information from the SageMaker environment variables.
    """
    tf_config = _load_tf_config(self._port)
    if 'cluster' not in tf_config:
      return ClusterSpec({})
    return ClusterSpec(tf_config['cluster'])

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    """Returns the master address to use when creating a TensorFlow session.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (String, optional) Overrides and sets the task_type of the
        master.
      task_id: (Integer, optional) Overrides and sets the task id of the master.
      rpc_layer: (String, optional) Overrides and sets the protocol over which
        TensorFlow nodes communicate with each other.

    Returns:
      The address of the master.

    Raises:
      RuntimeError: If the task_type or task_id is not specified and the
        SageMaker environment variables does not contain a task section.
    """

    # If `session_master` is set, just use that.
    session_master = _get_value_in_tfconfig(_SESSION_MASTER_KEY, self._port)
    if session_master is not None:
      return session_master

    # Return an empty string if we are the only job in the ClusterSpec.
    cluster_spec = self.cluster_spec()
    if (not cluster_spec.jobs or
        (len(cluster_spec.jobs) == 1 and
         len(cluster_spec.job_tasks(cluster_spec.jobs[0])) == 1)):
      return ''

    # We try to auto-detect the task type and id, but uses the user-supplied one
    # where available
    task_type = task_type if task_type is not None else self.task_type
    task_id = task_id if task_id is not None else self.task_id
    rpc_layer = rpc_layer if rpc_layer is not None else self.rpc_layer

    return format_master_url(
        cluster_spec.task_address(task_type, task_id), rpc_layer)
