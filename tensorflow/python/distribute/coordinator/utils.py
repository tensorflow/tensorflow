# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""TF2 parameter server training utilities.

Parameter server training in TF2 is currently under development.
"""
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib


def start_server(cluster_resolver, protocol):
  """Start a server and block the process from exiting."""
  # This function is for multi-processing test or users who would like to have
  # every job run the same binary for simplicity.
  if not (cluster_resolver.task_type == 'worker' or
          cluster_resolver.task_type == 'ps'):
    raise ValueError('Unexpected task_type to start a server: {}'.format(
        cluster_resolver.task_type))

  server = server_lib.Server(
      cluster_resolver.cluster_spec().as_cluster_def(),
      job_name=cluster_resolver.task_type,
      task_index=cluster_resolver.task_id,
      protocol=protocol)

  logging.info('TensorFlow server started for job %s, task %d.',
               cluster_resolver.task_type, cluster_resolver.task_id)

  # Blocking the process that starts a server from exiting.
  server.join()
