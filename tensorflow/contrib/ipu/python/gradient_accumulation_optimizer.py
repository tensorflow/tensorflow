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
"""
Optimizer wrapper which performs local gradient accumulation.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import cast
from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.contrib.ipu.python import ipu_optimizer


class GradientAccumulationOptimizer(optimizer.Optimizer):
  """An optimizer where instead of back-propagating for every batch feedforward,
  gradients across multiple batches are accumulated. After multiple
  feedforwards, the accumulated gradients are back-propagated through the
  network.

  This feature of neural networks allows us to simulate bigger batch sizes. For
  example if we have a model of batch size 16 and we accumulate the gradients
  of 4 batches, this simulates an input batch of size 64.
  """

  def __init__(self,
               opt,
               num_mini_batches,
               name="GradientAccumulationOptimizer"):
    """Construct a Gradient Accumulation Optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      num_mini_batches: Number of mini-batches the gradients will be accumulated
        for.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientAccumulationOptimizer".
    """

    super(GradientAccumulationOptimizer, self).__init__(False, name)
    self._opt = opt

    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    self._num_mini_batches = num_mini_batches

  def compute_gradients(self, loss, var_list=None, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping.

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
              (gen_poputil_ops.ipu_stateful_gradient_accumulate(
                  grad, num_mini_batches=self._num_mini_batches), var))
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


class CrossReplicaGradientAccumulationOptimizer(optimizer.Optimizer):
  """An optimizer for data parallel models, where instead of back-propagating
  for every batch feedforward, gradients across multiple batches are
  accumulated. After multiple feedforwards, the accumulated gradients are
  reduced across the replicas and are back-propagated through the network.

  This feature of neural networks allows us to simulate bigger batch sizes. For
  example if we have a model of batch size 16 and we accumulate the gradients
  of 4 batches, this simulates an input batch of size 64.

  This optimizer is similar GradientAccumulationOptimizer, however using this
  optimizer guarantees that the accumulated gradients will only be exchanged
  between IPUs when the accumulated gradients are back-propagated through the
  network.
  """

  def __init__(self,
               opt,
               num_mini_batches,
               name="CrossReplicaGradientAccumulationOptimizer"):
    """Construct a Cross Replica Gradient Accumulation Optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      num_mini_batches: Number of mini-batches the gradients will be accumulated
        for.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "CrossReplicaGradientAccumulationOptimizer".
    """

    super(CrossReplicaGradientAccumulationOptimizer, self).__init__(
        False, name)

    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    # Internally we just wrap the optimizer in a GradientAccumulationOptimizer and CrossReplicaOptimizer.
    self._opt = ipu_optimizer.CrossReplicaOptimizer(
        GradientAccumulationOptimizer(opt, num_mini_batches))

  def compute_gradients(self, loss, var_list=None, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping.

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

    """

    return self._opt.apply_gradients(grads_and_vars, global_step, name)

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
