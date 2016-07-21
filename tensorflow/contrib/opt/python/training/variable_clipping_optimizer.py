# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Delegating optimizer to clip norm for specified variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer

__all__ = ["VariableClippingOptimizer"]


class VariableClippingOptimizer(optimizer.Optimizer):
  """Wrapper optimizer that clips the norm of specified variables after update.

  This optimizer delegates all aspects of gradient calculation and application
  to an underlying optimizer.  After applying gradients, this optimizer then
  clips the variable to have a maximum L2 norm along specified dimensions.
  NB: this is quite different from clipping the norm of the gradients.

  Multiple instances of `VariableClippingOptimizer` may be chained to specify
  different max norms for different subsets of variables.

  @@__init__
  """

  def __init__(self,
               opt,
               vars_to_clip_dims,
               max_norm,
               use_locking=False,
               colocate_clip_ops_with_vars=False,
               name="VariableClipping"):
    """Construct a new clip-norm optimizer.

    Args:
      opt: The actual optimizer that will be used to compute and apply the
        gradients. Must be one of the Optimizer classes.
      vars_to_clip_dims: A dict with keys as Variables and values as lists
        of dimensions along which to compute the L2-norm.  See
        `tf.clip_by_norm` for more details.
      max_norm: The L2-norm to clip to, for all variables specified.
      use_locking: If `True` use locks for clip update operations.
      colocate_clip_ops_with_vars: If `True`, try colocating the clip norm
        ops with the corresponding variable.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "VariableClipping".
    """
    super(VariableClippingOptimizer, self).__init__(use_locking, name)
    self._opt = opt
    # Defensive copy of input dict
    self._vars_to_clip_dims = {
        var: clip_dims[:] for var, clip_dims in vars_to_clip_dims.items()}
    self._max_norm = max_norm
    self._colocate_clip_ops_with_vars = colocate_clip_ops_with_vars

  def compute_gradients(self, *args, **kwargs):
    return self._opt.compute_gradients(*args, **kwargs)

  def get_slot(self, *args, **kwargs):
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    return self._opt.get_slot_names(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    with ops.op_scope([], name, self._name) as name:
      update_op = self._opt.apply_gradients(
          grads_and_vars, global_step=global_step)
      clip_update_ops = []
      with ops.control_dependencies([update_op]):
        for grad, var in grads_and_vars:
          if grad is None or var not in self._vars_to_clip_dims:
            continue
          with ops.name_scope("clip_" + var.op.name):
            if isinstance(grad, ops.Tensor):
              clip_update_ops.append(self._clip_dense(var))
            else:
              clip_update_ops.append(self._clip_sparse(grad, var))

      # In case no var was clipped, still need to run the update_op.
      return control_flow_ops.group(*([update_op] + clip_update_ops), name=name)

  def _clip_dense(self, var):
    with self._maybe_colocate_with(var):
      updated_var_value = array_ops.identity(var.ref())
      normalized_var = clip_ops.clip_by_norm(
          updated_var_value, self._max_norm, self._vars_to_clip_dims[var])
      delta = updated_var_value - normalized_var
    with ops.colocate_with(var):
      return var.assign_sub(delta, use_locking=self._use_locking)

  def _clip_sparse(self, grad, var):
    assert isinstance(grad, ops.IndexedSlices)
    clip_dims = self._vars_to_clip_dims[var]
    if 0 in clip_dims:
      logging.warning("Clipping norm across dims %s for %s is inefficient "
                      "when including sparse dimension 0.", clip_dims,
                      var.op.name)
      return self._clip_dense(var)

    with ops.colocate_with(var):
      var_subset = array_ops.gather(var.ref(), grad.indices)
    with self._maybe_colocate_with(var):
      normalized_var_subset = clip_ops.clip_by_norm(
          var_subset, self._max_norm, clip_dims)
      delta = ops.IndexedSlices(
          var_subset - normalized_var_subset, grad.indices, grad.dense_shape)
    with ops.colocate_with(var):
      return var.scatter_sub(delta, use_locking=self._use_locking)

  @contextlib.contextmanager
  def _maybe_colocate_with(self, var):
    """Context to colocate with `var` if `colocate_clip_ops_with_vars`."""
    if self._colocate_clip_ops_with_vars:
      with ops.colocate_with(var):
        yield
    else:
      yield
