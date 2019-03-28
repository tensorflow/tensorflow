# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Optimizer that implements cross-replica gradient reduction for IPU.
Based on the TPU CrossShardOptimizer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.contrib.ipu.python import popops_cross_replica_sum


class CrossReplicaOptimizer(optimizer.Optimizer):
  """An optimizer that averages gradients across IPU replicas."""

  def __init__(self,
               opt,
               reduction=losses.Reduction.SUM,
               name="CrossReplicaOptimizer"):
    """Construct a new cross-replica optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      reduction: The reduction to apply to the shard losses.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "CrossReplicaOptimizer".

    Raises:
      ValueError: If reduction is not a valid cross-replica reduction.
    """
    if reduction not in (losses.Reduction.SUM):
      raise ValueError("Unsupported reduction: %s." % reduction)

    super(CrossReplicaOptimizer, self).__init__(False, name)
    self._opt = opt
    self._reduction = reduction

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
    """

    return self._opt.compute_gradients(loss, var_list=var_list, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    Calls popops_cross_replica_sum.cross_replica_sum() to sum gradient
    contributions across replicas, and then applies the real optimizer.

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
          summed_grads_and_vars.append(
              (popops_cross_replica_sum.cross_replica_sum(grad), var))
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
