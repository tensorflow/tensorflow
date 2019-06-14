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
"""Class CollectiveAllReduceStrategy implementing DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver


# TODO(yuefengz): support in-graph replication.
class CollectiveAllReduceStrategy(distribute_lib.StrategyV1):
  """Distribution strategy that uses collective ops for all-reduce.

  *** contrib version ***

  It is similar to the MirroredStrategy but it uses collective ops for
  reduction.

  When `cluster_spec` is given by the `configure` method, it turns into the
  mulit-worker version that works on multiple workers with between-graph
  replication.

  Note: `configure` will be called by higher-level APIs if running in
  distributed environment.
  """

  def __init__(self,
               num_gpus_per_worker=0,
               communication=cross_device_ops_lib.CollectiveCommunication.AUTO):
    """Initializes the object.

    Args:
      num_gpus_per_worker: number of local GPUs or GPUs per worker, the default
        is 0 meaning CPU only.
      communication: optional Enum of type
        `distribute.experimental.CollectiveCommunication`.  This provides a way
        for the user to override the choice of collective op communication.
        Possible values include `AUTO`, `RING`, and `NCCL`.
    """
    super(CollectiveAllReduceStrategy, self).__init__(
        CollectiveAllReduceExtended(
            self,
            num_gpus_per_worker=num_gpus_per_worker,
            communication=communication))


class CollectiveAllReduceExtended(
    collective_all_reduce_strategy.CollectiveAllReduceExtended):
  """Implementation of CollectiveAllReduceStrategy."""

  def __init__(self,
               container_strategy,
               num_gpus_per_worker,
               communication):
    # Use TFConfigClusterResolver to parse TF_CONFIG. We don't want to change
    # the constructor's interface to allow customized cluster resolver. Use
    # SimpleClusterResolver to override num_accelerators.
    tfconfig = TFConfigClusterResolver()
    cluster_resolver = SimpleClusterResolver(
        cluster_spec=tfconfig.cluster_spec(),
        task_type=tfconfig.task_type,
        task_id=tfconfig.task_id,
        num_accelerators={"GPU": num_gpus_per_worker},
        rpc_layer=tfconfig.rpc_layer)
    super(CollectiveAllReduceExtended, self).__init__(
        container_strategy,
        communication=communication,
        cluster_resolver=cluster_resolver)
