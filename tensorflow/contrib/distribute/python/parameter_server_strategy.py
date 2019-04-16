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
"""Classes implementing a multi-worker ps DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver

# pylint: disable=protected-access,invalid-name,line-too-long
CoreParameterServerStrategy = parameter_server_strategy.ParameterServerStrategy
CoreParameterServerExtended = parameter_server_strategy.ParameterServerStrategyExtended

# pylint: enable=protected-access,invalid-name,line-too-long


class ParameterServerStrategy(distribute_lib.StrategyV1):
  """A parameter server DistributionStrategy.

  *** contrib version ***

  This strategy class works for both local training and between-graph replicated
  training for multiple workers. If `cluster_spec` is specified, either passed
  in to __init__() method or parsed from the
  ["TF_CONFIG" environment
  variable](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig),
  variables and updates to those variables are assigned to parameter servers and
  other operations are assigned to workers. If `cluster_spec` is not set, it
  becomes local training where variables are assigned to local CPU or the only
  GPU. When each worker has more than one GPU, operations will be replicated on
  these GPUs. In both cases, operations are replicated but variables are not and
  these workers share a common view for which paramater server a variable is
  assigned to.

  This class assumes between-graph replication will be used and works on a graph
  for a particular worker. Note that each graph and worker is independent.
  This means that while each worker will synchronously compute a single gradient
  update across all GPUs, updates between workers proceed asynchronously.
  Operations that occur only on the first replica (such as incrementing the
  global step), will occur on the first replica *of every worker*.

  It is expected to call `call_for_each_replica(fn, ...)` for any
  operations which potentially can be replicated across replicas (i.e. multiple
  GPUs) even if there is only CPU or one GPU. When defining the `fn`, extra
  caution needs to be taken:

  1) Always use `tf.get_variable` instead of `tf.Variable` which is not able
  to refer to the same variable on different replicas.

  2) It is generally not recommended to open a device scope under the strategy's
  scope. A device scope (i.e. calling `tf.device`) will be merged with or
  override the device for operations but will not change the device for
  variables.

  3) It is also not recommended to open a colocation scope (i.e. calling
  `tf.colocate_with`) under the strategy's scope. For colocating variables, use
  `strategy.extended.colocate_vars_with` instead. Colocation of ops will
  possibly create conflicts of device assignment.
  """

  def __init__(self, num_gpus_per_worker=0):
    """Initializes this strategy.

    Args:
      num_gpus_per_worker: number of local GPUs or GPUs per worker, the default
        is 0 meaning CPU only.

    Raises:
      ValueError: if `cluster_spec` is given but `task_type` or `task_id` is
        not.
    """
    super(ParameterServerStrategy, self).__init__(
        ParameterServerExtended(self, num_gpus_per_worker))

  # Override to change the documentation to reflect the different handling of
  # global vs. local batch size between core and contrib.
  def make_dataset_iterator(self, dataset):  # pylint: disable=useless-super-delegation
    """Makes an iterator for input provided via `dataset`.

    NOTE: The batch size of the `dataset` argument is treated differently for
    this contrib version of `ParameterServerStrategy`.

    Data from the given dataset will be distributed evenly across all the
    compute replicas. We will assume that the input dataset is batched by the
    per-replica batch size.

    The user could also use `make_input_fn_iterator` if they want to
    customize which input is fed to which replica/worker etc.

    Args:
      dataset: `tf.data.Dataset` that will be distributed evenly across all
        replicas.

    Returns:
      An `tf.distribute.InputIterator` which returns inputs for each step of the
      computation.  User should call `initialize` on the returned iterator.
    """
    return super(ParameterServerStrategy, self).make_dataset_iterator(dataset)


class ParameterServerExtended(CoreParameterServerExtended):
  """Implementation of ParameterServerStrategy."""

  def __init__(self, container_strategy, num_gpus_per_worker):
    # Use TFConfigClusterResolver to parse TF_CONFIG. We don't want to change
    # the constructor's interface to allow customized cluster resolver. Use
    # SimpleClusterResolver to override num_accelerators.
    tfconfig = TFConfigClusterResolver()
    cluster_resolver = SimpleClusterResolver(
        cluster_spec=tfconfig.cluster_spec(),
        task_type=tfconfig.task_type,
        task_id=tfconfig.task_id,
        num_accelerators={'GPU': num_gpus_per_worker})
    super(ParameterServerExtended, self).__init__(
        container_strategy, cluster_resolver=cluster_resolver)

  def _make_dataset_iterator(self, dataset):
    return input_lib.DatasetIterator(dataset, self._input_workers)

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    """The contrib version of PS strategy uses per-replica batch size."""
    return False
