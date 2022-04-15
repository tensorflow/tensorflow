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
#==============================================================================
"""Contains utility functions used by summary ops in distribution strategy."""


from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util


def skip_summary():
  """Determines if summary should be skipped.

  If using multiple replicas in distributed strategy, skip summaries on all
  replicas except the first one (replica_id=0).

  Returns:
    True if the summary is skipped; False otherwise.
  """

  # TODO(priyag): Add a new optional argument that will provide multiple
  # alternatives to override default behavior. (e.g. run on last replica,
  # compute sum or mean across replicas).
  replica_context = distribution_strategy_context.get_replica_context()
  if not replica_context:
    return False
  # TODO(b/118385803): when replica_id of _TPUReplicaContext is properly
  # initialized, remember to change here as well.
  replica_id = replica_context.replica_id_in_sync_group
  if isinstance(replica_id, ops.Tensor):
    replica_id = tensor_util.constant_value(replica_id)
  return replica_id and replica_id > 0
