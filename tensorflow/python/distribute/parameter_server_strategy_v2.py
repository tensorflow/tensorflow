# Lint as: python3
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
"""Parameter server strategy V2 class.

This is currently under development and the API is subject to change.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_contextlib


# pylint: disable=protected-access
class ParameterServerStrategyV2(distribute_lib.Strategy):
  """An asynchronous multi-worker parameter server tf.distribute strategy.

  Currently, `ParameterServerStrategyV2` is not supported to be used as a
  standalone tf.distribute strategy. It must be used in conjunction with
  `Client`. The recommended way of using the combination is through a
  `ParameterServerClient` object. Please see `Client` and
  `ParameterServerClient` for more information.

  This is currently under development, and the API as well as implementation
  is subject to changes.
  """

  def __init__(self, cluster_resolver):
    """Initializes the V2 parameter server strategy.

    Args:
      cluster_resolver: a `tf.distribute.cluster_resolver.ClusterResolver`
        object.
    """
    self._extended = ParameterServerStrategyV2Extended(self, cluster_resolver)
    self._cluster_resolver = cluster_resolver
    self._verify_args_and_config(cluster_resolver)
    logging.info(
        "ParameterServerStrategyV2 is initialized with cluster_spec: "
        "%s", cluster_resolver.cluster_spec())
    super(ParameterServerStrategyV2, self).__init__(self._extended)

  @tf_contextlib.contextmanager
  def experimental_variable_partitioning_scope(self):
    """A context manager for creating `ShardedVariable`.

    Variables created inside a `with experimental_variable_partitioning_scope()`
    code block will be of type `ShardedVariable` and their values are
    partitioned among parameter servers along the first / outermost axis. The
    number of shards are equal to the number of parameter servers.

    Variables created within this scope must be initialized using a callable as
    `initial_value` and a known shape.

    Div partition strategy is used to partition the variable. Assuming we
    assign consective integer ids along the first axis of the variable, then ids
    are assigned to shards in a contiguous manner, while attempting to keep each
    shard size identical. If the ids do not evenly divide the number of shards,
    each of the first several shards will be assigned one more id. For instance,
    a variable whose first dimension is 13 has 13 ids, and they are split across
    5 shards as: `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

    Yields:
      A context manager for creating `ShardedVariable`.
    """
    with variable_scope.variable_creator_scope(
        self._extended._make_sharded_variable_creator()):
      yield

  def _verify_args_and_config(self, cluster_resolver):
    if not cluster_resolver.cluster_spec():
      raise ValueError("Cluster spec must be non-empty in `cluster_resolver`.")
    if self.extended._num_gpus_per_worker > 1:
      raise NotImplementedError("Multi-gpu is not supported yet.")


class ParameterServerStrategyV2Extended(
    parameter_server_strategy.ParameterServerStrategyExtended):
  """Extended class for ParameterServerStrategyV2.

  Please see `tf.distribute.StrategyExtended` doc for more information.
  """

  def __init__(self, container_strategy, cluster_resolver):
    """Initialization of ParameterServerStrategyV2Extended."""
    super(ParameterServerStrategyV2Extended, self).__init__(container_strategy)
    self._num_ps = len(cluster_resolver.cluster_spec().as_dict().get("ps", []))
    self._variable_count = 0

  def _create_variable(self, next_creator, **kwargs):

    if "colocate_with" in kwargs:
      colocate_with = kwargs["colocate_with"]
      # Clear the variable scope to avoid possible conflicts between device
      # scope and colocation scope.
      with ops.device(None):
        with ops.colocate_with(colocate_with):
          var = next_creator(**kwargs)
          logging.debug(
              "Creating variable (name:%s, shape:%r) that colocates with %s",
              var.name, var.shape, kwargs["colocate_with"].name)
          return var

    # Clear the colocation scope to avoid possible conflicts between device
    # scope and colocation scope.
    with ops.colocate_with(None, ignore_existing=True):
      with ops.device("/job:ps/task:%d" %
                      (self._variable_count % self._num_ps)):
        var = next_creator(**kwargs)
        logging.debug(
            "Creating variable (name:%s, shape:%r) on /job:ps/task:%d",
            var.name, var.shape, (self._variable_count % self._num_ps))
        self._variable_count += 1
        return var

  def _make_sharded_variable_creator(self):
    """Returns a function conforming to the `variable_creator` signature.

    The returned function creates `ShardedVariable` when called.
    """

    def sharded_variable_creator(next_creator, **kwargs):
      if "shape" not in kwargs or kwargs["shape"] is None:
        raise ValueError("shape must be explicitly specified when creating "
                         "sharded variables")
      init_fn = kwargs.get("initial_value", None)
      # We intentionally don't allow non-callable initial_value to ensure the
      # value is created on PS but not client. If the value is created on
      # client, it will needed to be sent to PS for variable initialization,
      # which is inefficient and can potentially hit the 2GB limit on protobuf
      # serialization.
      if init_fn is None or not callable(init_fn):
        raise ValueError("initial_value must be specified as a callable when "
                         "creating sharded variables")

      # Use "div" partition strategy to partition the variable.
      full_shape = kwargs["shape"]
      if self._num_ps < full_shape[0]:
        num_shards = self._num_ps
      else:
        num_shards = full_shape[0]
      offsets = []
      base = full_shape[0] // num_shards
      extra = full_shape[0] % num_shards
      for i in range(num_shards):
        if i == 0:
          offsets.append(0)
        else:
          prev_shard_size = base + (1 if i - 1 < extra else 0)
          offsets.append(offsets[i - 1] + prev_shard_size)

      # Note: The way we initialize sharded variables is suboptimal, as it
      # needs to create the full value tensor separately on each PS which the
      # variable is going to be placed on. The full value could be very large
      # and consume a lot of memory. The ideal way is to only create what's
      # needed on the shard, however that's not practical because:
      #  1. Initializers don't have sharded behavior support, even though some
      #     initializers (e.g, uniform) can be used directly.
      #  2. tf.Variable signature requires "initial_value" to be either a value
      #     or a callable without arguments, meaning it is not straightforward
      #     to make the sharded component from it.
      def init_shard_fn(shard_index):
        full_value = init_fn()
        if shard_index < num_shards - 1:
          return full_value[offsets[shard_index]:offsets[shard_index + 1]]
        else:
          return full_value[offsets[shard_index]:]

      var_list = []
      for i in range(num_shards):
        kwargs["shape"] = None
        kwargs["initial_value"] = lambda: init_shard_fn(i)
        var_list.append(next_creator(**kwargs))

      result = sharded_variable.ShardedVariable(var_list)
      return result

    return sharded_variable_creator

  def _call_for_each_replica(self, fn, args, kwargs):
    # TODO(rchao): Consider implementing sync PS training.
    raise NotImplementedError("Sync PS training is not implemented yet.")
