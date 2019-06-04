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
# =============================================================================

"""Optimizer that implements cross-shard gradient reduction for TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["tpu.CrossShardOptimizer"])
class CrossShardOptimizer(optimizer.Optimizer):
  """An optimizer that averages gradients across TPU shards."""

  def __init__(self,
               opt,
               reduction=losses.Reduction.MEAN,
               name="CrossShardOptimizer",
               group_assignment=None):
    """Construct a new cross-shard optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      reduction: The reduction to apply to the shard losses.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "CrossShardOptimizer".
      group_assignment: Optional 2d int32 lists with shape
        [num_groups, num_replicas_per_group] which describles how to apply
        optimizer to subgroups.

    Raises:
      ValueError: If reduction is not a valid cross-shard reduction.
    """
    if reduction not in (losses.Reduction.SUM, losses.Reduction.MEAN):
      raise ValueError("Unsupported reduction: %s." % reduction)

    super(CrossShardOptimizer, self).__init__(False, name)
    self._opt = opt
    self._reduction = reduction
    self._group_assignment = group_assignment

  def _verify_and_get_subgroup_size(self, group_assignment, num_shards):
    """Verify group_assignment and get the subgroup size".

    Args:
      group_assignment: list of group ids for applying the optimizer
        to subgroups.
      num_shards: The number of TPU shards.

    Returns:
      The size of one subgroup in group_assignment.

    Raises:
      ValueError: If group_assignment is invalid.
    """
    if not group_assignment:
      return None
    if not (isinstance(group_assignment, list) and
            all(isinstance(i, list) for i in group_assignment)):
      raise ValueError("group_assignment must be a list of list. Got {}".format(
          group_assignment))

    replica_ids = set()
    for g in group_assignment:
      for i in g:
        replica_ids.add(i)

    if set(range(num_shards)) != replica_ids:
      raise ValueError("group_assignment must be a permutation of range({0})."
                       " Got group_assignment={1}".format(
                           num_shards, group_assignment))

    subgroup_size_list = [len(group) for group in group_assignment]
    if all(subgroup_size_list[0] == size for size in subgroup_size_list):
      return subgroup_size_list[0]
    else:
      raise ValueError("The size of each subgroup in group_assignment must "
                       "be equal. Got group_assignment={}".format(
                           self._group_assignment))

  def compute_gradients(self, loss, var_list=None, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping with per replica global norm if needed.
    The global norm with aggregated gradients can be bad as one replica's huge
    gradients can hurt the gradients from other replicas.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.

    Raises:
      ValueError: If not within a tpu_shard_context or group_assignment is
        invalid.
    """
    num_shards = tpu_function.get_tpu_context().number_of_shards
    if num_shards is None:
      logging.warning(
          "CrossShardOptimizer should be used within a tpu_shard_context, but "
          "got unset number_of_shards. Assuming 1.")
      num_shards = 1

    subgroup_size = self._verify_and_get_subgroup_size(self._group_assignment,
                                                       num_shards)

    if num_shards > 1 and self._reduction == losses.Reduction.MEAN:
      if self._group_assignment:
        scale = 1.0 / subgroup_size
      else:
        scale = 1.0 / num_shards
      loss *= scale

    return self._opt.compute_gradients(loss, var_list=var_list, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    Calls tpu_ops.cross_replica_sum() to sum gradient contributions across
    replicas, and then applies the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    Raises:
      ValueError: If the grads_and_vars is malformed.
    """
    summed_grads_and_vars = []
    for (grad, var) in grads_and_vars:
      if grad is None:
        summed_grads_and_vars.append((grad, var))
      else:
        with ops.colocate_with(grad):
          summed_grads_and_vars.append((tpu_ops.cross_replica_sum(
              grad, self._group_assignment), var))
    return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)

  def get_slot(self, *args, **kwargs):
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)

  def variables(self):
    """Forwarding the variables from the underlying optimizer."""
    return self._opt.variables()
