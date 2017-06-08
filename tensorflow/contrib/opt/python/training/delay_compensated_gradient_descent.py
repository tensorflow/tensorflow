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
# ==============================================================================

"""DelayCompensatedGradientDescentOptimizer for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class _RefVariableAsynchronousProcessor(optimizer._RefVariableProcessor):
  """Processor for Variable."""
  def update_op_asynchronous(self, optimizer, g, index):
    if isinstance(g, ops.Tensor):
      return optimizer._apply_dense(g, self._v, index)
    else:
      assert isinstance(g, ops.IndexedSlices), ("Gradient ", g, " is neither a "
                                                "tensor nor IndexedSlices.")
      # pylint: disable=protected-access
      return optimizer._apply_sparse_duplicate_indices(g, self._v, index)


class _DenseResourceVariableAsynchronousProcessor(optimizer._DenseResourceVariableProcessor):
  """Processor for dense ResourceVariables."""
  def update_op_asynchronous(self, optimizer, g, index):
    # pylint: disable=protected-access
    if isinstance(g, ops.IndexedSlices):
      return optimizer._resource_apply_sparse_duplicate_indices(
        g.values, self._v, g.indices, index)
    return optimizer._resource_apply_dense(g, self._v, index)


def _get_processor(v):
  """The processor of v."""
  if v.op.type == "VarHandleOp":
    return _DenseResourceVariableAsynchronousProcessor(v)
  if isinstance(v, variables.Variable):
    return _RefVariableAsynchronousProcessor(v)
  raise NotImplementedError("Trying to optimize unsupported type ", v)


class DelayCompensatedGradientDescentOptimizer(optimizer.Optimizer):
  """Optimizer that implements gradient descent with delay compensation.

  See [Zheng, Shuxin, et al., 2016](https://arxiv.org/abs/1609.08326)
  ([pdf](https://arxiv.org/pdf/1609.08326.pdf)).
  """

  def __init__(self, learning_rate, variance_parameter, num_workers=1,
               use_locking=False, name="DelayCompensatedGradientDescent"):
    """Construct a new gradient descent optimizer with delay compensation.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      variance_parameter: A Tensor or a floating point value. The lambda
        value to use.
      num_workers: A value to indicate number of workers computing gradients
        asynchronously.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "DelayCompensatedGradientDescent".
      """
    if num_workers <= 0:
      raise ValueError("num_workers must be positive: %s" % num_workers)
    super(DelayCompensatedGradientDescentOptimizer, self).__init__(
          use_locking, name)
    self._learning_rate = learning_rate
    self._lambda = variance_parameter
    self._num_workers = num_workers

  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=optimizer.Optimizer.GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None, worker_index=None):
    """Add operations to minimize `loss` by updating `var_list`.

    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying
    them call `compute_gradients()` and `apply_gradients()` explicitly instead
    of using this function.

    Args:
      loss: A `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.  Defaults to the list of variables collected in
        the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      worker_index: Optional. A value to indicate the instance of worker
        minimizing if computing asynchronously.

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    Raises:
      ValueError: If some of the variables are not `Variable` objects.
    """
    if (worker_index < 0 and worker_index is not None) or worker_index >= self._num_workers:
      raise ValueError("worker index must be in the range [0, num_workers): %s" %
                        worker_index)
    grads_and_vars = self.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))

    return self.apply_gradients(grads_and_vars, global_step=global_step,
                                name=name, worker_index=worker_index)

  def apply_gradients(self,
                      grads_and_vars,
                      global_step=None,
                      name=None,
                      worker_index=None):
    """Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the `Optimizer` constructor.
      worker_index: Optional value to indicate the instance of worker
        minimizing if computing asynchronously.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
    """
    # This is a default implementation of apply_gradients() that can be shared
    # by most optimizers.  It relies on the subclass implementing the following
    # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().

    grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
    if not grads_and_vars:
      raise ValueError("No variables provided.")
    converted_grads_and_vars = []
    for g, v in grads_and_vars:
      if g is not None:
        try:
          # Convert the grad to Tensor or IndexedSlices if necessary.
          g = ops.convert_to_tensor_or_indexed_slices(g)
        except TypeError:
          raise TypeError(
              "Gradient must be convertible to a Tensor"
              " or IndexedSlices, or None: %s" % g)
        if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
          raise TypeError(
              "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
      p = _get_processor(v)
      converted_grads_and_vars.append((g, v, p))

    converted_grads_and_vars = tuple(converted_grads_and_vars)
    var_list = [v for g, v, _ in converted_grads_and_vars if g is not None]
    if not var_list:
      raise ValueError("No gradients provided for any variable: %s." %
                       ([str(v) for _, _, v in converted_grads_and_vars],))
    with ops.control_dependencies(None):
      self._create_slots([optimizer._get_variable_for(v) for v in var_list])
    update_ops = []
    with ops.name_scope(name, self._name) as name:
      self._prepare()
      for grad, var, processor in converted_grads_and_vars:
        if grad is None:
          continue
        # We colocate all ops created in _apply_dense or _apply_sparse
        # on the same device as the variable.
        with ops.name_scope("update_" + var.op.name), ops.colocate_with(var):
          if worker_index is None:
            update_ops.append(processor.update_op(self, grad))
          else:
            update_ops.append(processor.update_op_asynchronous(self, grad,
                                                               worker_index))
      if global_step is None:
        apply_updates = self._finish(update_ops, name)
      else:
        with ops.control_dependencies([self._finish(update_ops, "update")]):
          with ops.colocate_with(global_step):
            apply_updates = state_ops.assign_add(global_step, 1, name=name).op

      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      if apply_updates not in train_op:
        train_op.append(apply_updates)

      return apply_updates

  def _create_slots(self, var_list):
    """Initialize slots for all the vars of each worker to store
        the previous values of it
    """
    for index in range(self._num_workers):
      for v in var_list:
        var2 = array_ops.identity(v.initialized_value())
        self._get_or_make_slot(v, var2, "shadow_{0}".format(index),
                               self._name)

  def _resource_apply_dense(self, grad, var, worker_index=0):
    # Get previous value of the variable from the slot
    shadow = self.get_slot(var, "shadow_{0}".format(worker_index))
    return training_ops.apply_delay_compensated_gradient_descent(
        var.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        grad,
        math_ops.cast(self._lambda_tensor, grad.dtype.base_dtype),
        shadow.handle,
        use_locking=self._use_locking)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._lambda_tensor = ops.convert_to_tensor(self._lambda,
                                                name="lambda")
