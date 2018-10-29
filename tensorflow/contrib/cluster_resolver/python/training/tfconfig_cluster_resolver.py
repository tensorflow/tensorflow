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

from tensorflow.contrib.cluster_resolver.python.training.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec

_TF_CONFIG_ENV = 'TF_CONFIG'
_SESSION_MASTER_KEY = 'session_master'


class TFConfigClusterResolver(ClusterResolver):
  """Implementation of a ClusterResolver which reads the TF_CONFIG EnvVar."""

  def _load_tf_config(self):
    return json.loads(os.environ.get(_TF_CONFIG_ENV, '{}'))

  def cluster_spec(self):
    """Returns a ClusterSpec based on the TF_CONFIG environment variable.

    Returns:
      A ClusterSpec with information from the TF_CONFIG environment variable.
    """
    tf_config = self._load_tf_config()
    if 'cluster' not in tf_config:
      return ClusterSpec({})
    return ClusterSpec(tf_config['cluster'])

  def master(self, task_type=None, task_index=0):
    """Returns the master address to use when creating a TensorFlow session.

    Args:
      task_type: (String, optional) Overrides and sets the task_type of the
        master.
      task_index: (Integer, optional) Overrides and sets the task id of the
        master.

    Returns:
      The address of the master.

    Raises:
      RuntimeError: If the task_type or task_id is not specified and the
        `TF_CONFIG` environment variable does not contain a task section.
    """

    # If `session_master` is set, just use that.
    tf_config = self._load_tf_config()
    if _SESSION_MASTER_KEY in tf_config:
      return tf_config[_SESSION_MASTER_KEY]

    if 'rpc_layer' in tf_config:
      rpclayer = '%s://' % tf_config['rpc_layer']
    else:
      rpclayer = ''

    # Return an empty string if we are the only job in the ClusterSpec.
    cluster_spec = self.cluster_spec()
    if (not cluster_spec.jobs or
        (len(cluster_spec.jobs) == 1 and
         len(cluster_spec.job_tasks(cluster_spec.jobs[0])) == 1)):
      return ''

    # We try to auto-detect the task type and id, but uses the user-supplied one
    # where available
    if not task_type:
      if 'task' not in tf_config:
        raise RuntimeError('You must either specify a `task_type`, or your '
                           'TF_CONFIG must contain a `task` section.')
      task_type = tf_config['task']['type']
      task_index = tf_config['task']['index']

    return rpclayer + cluster_spec.task_address(task_type, task_index)
