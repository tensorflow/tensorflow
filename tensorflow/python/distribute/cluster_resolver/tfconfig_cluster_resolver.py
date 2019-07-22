# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of Cluster Resolvers for TF_CONFIG Environment Variables."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export

_TF_CONFIG_ENV = 'TF_CONFIG'
_SESSION_MASTER_KEY = 'session_master'
_RPC_LAYER_KEY = 'rpc_layer'
_TASK_KEY = 'task'


def format_master_url(master, rpc_layer=None):
  if rpc_layer:
    return '%s://%s' % (rpc_layer, master)
  else:
    return master


def _load_tf_config():
  return json.loads(os.environ.get(_TF_CONFIG_ENV, '{}'))


def _get_value_in_tfconfig(key, default=None):
  tf_config = _load_tf_config()
  return tf_config[key] if key in tf_config else default


@tf_export('distribute.cluster_resolver.TFConfigClusterResolver')
class TFConfigClusterResolver(ClusterResolver):
  """Implementation of a ClusterResolver which reads the TF_CONFIG EnvVar.

  This is an implementation of cluster resolvers when using TF_CONFIG to set
  information about the cluster. The cluster spec returned will be
  initialized from the TF_CONFIG environment variable.
  """

  def __init__(self,
               task_type=None,
               task_id=None,
               rpc_layer=None,
               environment=None):
    """Creates a new TFConfigClusterResolver.

    Args:
      task_type: (String, optional) Overrides the task type specified in the
        TF_CONFIG environment variable.
      task_id: (Integer, optional) Overrides the task index specified in the
        TF_CONFIG environment variable.
      rpc_layer: (String, optional) Overrides the rpc layer TensorFlow uses.
      environment: (String, optional) Overrides the environment TensorFlow
        operates in.
    """
    self._task_type = task_type
    self._task_id = task_id
    self._rpc_layer = rpc_layer
    self._environment = environment

  @property
  def task_type(self):
    if self._task_type is None:
      task_info = _get_value_in_tfconfig(_TASK_KEY, {})
      return str(task_info['type']) if 'type' in task_info else None
    else:
      return str(self._task_type)

  @property
  def task_id(self):
    if self._task_type is None:
      task_info = _get_value_in_tfconfig(_TASK_KEY, {})
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
      return _get_value_in_tfconfig(_RPC_LAYER_KEY)
    else:
      return self._rpc_layer

  @rpc_layer.setter
  def rpc_layer(self, rpc_layer):
    self._rpc_layer = rpc_layer

  def num_accelerators(self,
                       task_type=None,
                       task_id=None,
                       config_proto=None):
    task_type = self.task_type if task_type is None else task_type
    task_id = self.task_id if task_id is None else task_id
    return super(TFConfigClusterResolver, self).num_accelerators(
        task_type, task_id, config_proto)

  def cluster_spec(self):
    """Returns a ClusterSpec based on the TF_CONFIG environment variable.

    Returns:
      A ClusterSpec with information from the TF_CONFIG environment variable.
    """
    tf_config = _load_tf_config()
    if 'cluster' not in tf_config:
      return ClusterSpec({})
    return ClusterSpec(tf_config['cluster'])

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    """Returns the master address to use when creating a TensorFlow session.

    Args:
      task_type: (String, optional) Overrides and sets the task_type of the
        master.
      task_id: (Integer, optional) Overrides and sets the task id of the
        master.
      rpc_layer: (String, optional) Overrides and sets the protocol over which
        TensorFlow nodes communicate with each other.

    Returns:
      The address of the master.

    Raises:
      RuntimeError: If the task_type or task_id is not specified and the
        `TF_CONFIG` environment variable does not contain a task section.
    """

    # If `session_master` is set, just use that.
    session_master = _get_value_in_tfconfig(_SESSION_MASTER_KEY)
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

    return format_master_url(cluster_spec.task_address(task_type, task_id),
                             self.rpc_layer)
