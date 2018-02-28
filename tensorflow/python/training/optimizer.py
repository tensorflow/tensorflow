# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Base class for optimizers."""
# pylint: disable=g-bad-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import checkpointable
from tensorflow.python.training import slot_creator
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


def _get_variable_for(v):
  """Returns the ResourceVariable responsible for v, or v if not necessary."""
  if context.in_eager_mode():
    return v
  if v.op.type == "VarHandleOp":
    for var in variables.trainable_variables():
      if (isinstance(var, resource_variable_ops.ResourceVariable)
          and var.handle.op is v.op):
        return var
    raise ValueError("Got %s but could not locate source variable." % (str(v)))
  return v


def _deduplicate_indexed_slices(values, indices):
  """Sums `values` associated with any non-unique `indices`.

  Args:
    values: A `Tensor` with rank >= 1.
    indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
  Returns:
    A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
    de-duplicated version of `indices` and `summed_values` contains the sum of
    `values` slices associated with each unique index.
  """
  unique_indices, new_index_positions = array_ops.unique(indices)
  summed_values = math_ops.unsorted_segment_sum(
      values, new_index_positions,
      array_ops.shape(unique_indices)[0])
  return (summed_values, unique_indices)


def _var_key(var):
  if context.in_eager_mode():
    return var._shared_name  # pylint: disable=protected-access
  return (var.op.graph, var.op.name)


class _OptimizableVariable(object):
  """Interface for abstracting over variables in the optimizers."""

  @abc.abstractmethod
  def target(self):
    """Returns the optimization target for this variable."""
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def update_op(self, optimizer, g):
    """Returns the update ops for updating the variable."""
    raise NotImplementedError("Calling an abstract method.")


class _RefVariableProcessor(_OptimizableVariable):
  """Processor for Variable."""

  def __init__(self, v):
    self._v = v

  def __str__(self):
    return "<_RefVariableProcessor(%s)>" % self._v

  def target(self):
    return self._v._ref()  # pylint: disable=protected-access

  def update_op(self, optimizer, g):
    if isinstance(g, ops.Tensor):
      update_op = optimizer._apply_dense(g, self._v)  # pylint: disable=protected-access
      if self._v.constraint is not None:
        with ops.control_dependencies([update_op]):
          return self._v.assign(self._v.constraint(self._v))
      else:
        return update_op
    else:
      assert isinstance(g, ops.IndexedSlices), ("Gradient ", g, " is neither a "
                                                "tensor nor IndexedSlices.")
      if self._v.constraint is not None:
        raise RuntimeError(
            "Cannot use a constraint function on a sparse variable.")
      # pylint: disable=protected-access
      return optimizer._apply_sparse_duplicate_indices(g, self._v)


class _DenseReadResourceVariableProcessor(_OptimizableVariable):
  """Processor for dense ResourceVariables."""

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

  def update_op(self, optimizer, g):
    # pylint: disable=protected-access
    update_op = optimizer._resource_apply_dense(g, self._v.op.inputs[0])
    if self._v.constraint is not None:
      with ops.control_dependencies([update_op]):
        return self._v.assign(self._v.constraint(self._v))
    else:
      return update_op


class _DenseResourceVariableProcessor(_OptimizableVariable):
  """Processor for dense ResourceVariables."""

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

  def update_op(self, optimizer, g):
    # pylint: disable=protected-access
    if isinstance(g, ops.IndexedSlices):
      if self._v.constraint is not None:
        raise RuntimeError(
            "Cannot use a constraint function on a sparse variable.")
      return optimizer._resource_apply_sparse_duplicate_indices(
          g.values, self._v, g.indices)
    update_op = optimizer._resource_apply_dense(g, self._v)
    if self._v.constraint is not None:
      with ops.control_dependencies([update_op]):
        return self._v.assign(self._v.constraint(self._v))
    else:
      return update_op


class _StreamingModelPortProcessor(_OptimizableVariable):
  """Processor for streaming ModelPorts."""

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

  def update_op(self, optimizer, g):
    return g


class _TensorProcessor(_OptimizableVariable):
  """Processor for ordinary Tensors.

  Even though a Tensor can't really be updated, sometimes it is useful to
  compute the gradients with respect to a Tensor using the optimizer. Updating
  the Tensor is, of course, unsupported.
  """

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

  def update_op(self, optimizer, g):
    raise NotImplementedError("Trying to update a Tensor ", self._v)


def _get_processor(v):
  """The processor of v."""
  if context.in_eager_mode():
    if isinstance(v, ops.Tensor):
      return _TensorProcessor(v)
    else:
      return _DenseResourceVariableProcessor(v)
  if v.op.type == "VarHandleOp":
    return _DenseResourceVariableProcessor(v)
  if isinstance(v, variables.Variable):
    return _RefVariableProcessor(v)
  if v.op.type == "SubmodelPort":
    return _StreamingModelPortProcessor(v)
  if isinstance(v, ops.Tensor):
    return _TensorProcessor(v)
  raise NotImplementedError("Trying to optimize unsupported type ", v)


@tf_export("train.Optimizer")
class Optimizer(checkpointable.Checkpointable):
  """Base class for optimizers.

  This class defines the API to add Ops to train a model.  You never use this
  class directly, but instead instantiate one of its subclasses such as
  `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.

  ### Usage

  ```python
  # Create an optimizer with the desired parameters.
  opt = GradientDescentOptimizer(learning_rate=0.1)
  # Add Ops to the graph to minimize a cost by updating a list of variables.
  # "cost" is a Tensor, and the list of variables contains tf.Variable
  # objects.
  opt_op = opt.minimize(cost, var_list=<list of variables>)
  ```

  In the training program you will just have to run the returned Op.

  ```python
  # Execute opt_op to do one step of training:
  opt_op.run()
  ```

  ### Processing gradients before applying them.

  Calling `minimize()` takes care of both computing the gradients and
  applying them to the variables.  If you want to process the gradients
  before applying them you can instead use the optimizer in three steps:

  1.  Compute the gradients with `compute_gradients()`.
  2.  Process the gradients as you wish.
  3.  Apply the processed gradients with `apply_gradients()`.

  Example:

  ```python
  # Create an optimizer.
  opt = GradientDescentOptimizer(learning_rate=0.1)

  # Compute the gradients for a list of variables.
  grads_and_vars = opt.compute_gradients(loss, <list of variables>)

  # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
  # need to the 'gradient' part, for example cap them, etc.
  capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

  # Ask the optimizer to apply the capped gradients.
  opt.apply_gradients(capped_grads_and_vars)
  ```

  ### Gating Gradients

  Both `minimize()` and `compute_gradients()` accept a `gate_gradients`
  argument that controls the degree of parallelism during the application of
  the gradients.

  The possible values are: `GATE_NONE`, `GATE_OP`, and `GATE_GRAPH`.

  <b>`GATE_NONE`</b>: Compute and apply gradients in parallel.  This provides
  the maximum parallelism in execution, at the cost of some non-reproducibility
  in the results.  For example the two gradients of `matmul` depend on the input
  values: With `GATE_NONE` one of the gradients could be applied to one of the
  inputs _before_ the other gradient is computed resulting in non-reproducible
  results.

  <b>`GATE_OP`</b>: For each Op, make sure all gradients are computed before
  they are used.  This prevents race conditions for Ops that generate gradients
  for multiple inputs where the gradients depend on the inputs.

  <b>`GATE_GRAPH`</b>: Make sure all gradients for all variables are computed
  before any one of them is used.  This provides the least parallelism but can
  be useful if you want to process all gradients before applying any of them.

  ### Slots

  Some optimizer subclasses, such as `MomentumOptimizer` and `AdagradOptimizer`
  allocate and manage additional variables associated with the variables to
  train.  These are called <i>Slots</i>.  Slots have names and you can ask the
  optimizer for the names of the slots that it uses.  Once you have a slot name
  you can ask the optimizer for the variable it created to hold the slot value.

  This can be useful if you want to log debug a training algorithm, report stats
  about the slots, etc.
  """

  # Values for gate_gradients.
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2

  def __init__(self, use_locking, name):
    """Create a new Optimizer.

    This must be called by the constructors of subclasses.

    Args:
      use_locking: Bool. If True apply use locks to prevent concurrent updates
        to variables.
      name: A non-empty string.  The name to use for accumulators created
        for the optimizer.

    Raises:
      ValueError: If name is malformed.
    """
    if not name:
      raise ValueError("Must specify the optimizer name")
    self._use_locking = use_locking
    self._name = name
    # Dictionary of slots.
    #  {slot_name :
    #      {_var_key(variable_to_train): slot_for_the_variable, ... },
    #   ... }
    self._slots = {}
    self._non_slot_dict = {}
    # For implementing Checkpointable. Stores information about how to restore
    # slot variables which have not yet been created
    # (checkpointable._CheckpointPosition objects).
    #  {slot_name :
    #      {_var_key(variable_to_train): [checkpoint_position, ... ], ... },
    #   ... }
    self._deferred_slot_restorations = {}

  def get_name(self):
    return self._name

  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
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

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    Raises:
      ValueError: If some of the variables are not `Variable` objects.

    @compatibility(eager)
    When eager execution is enabled, `loss` should be a Python function that
    takes elements of `var_list` as arguments and computes the value to be
    minimized. If `var_list` is None, `loss` should take no arguments.
    Minimization (and gradient computation) is done with respect to the
    elements of `var_list` if not None, else with respect to any trainable
    variables created during the execution of the `loss` function.
    `gate_gradients`, `aggregation_method`, `colocate_gradients_with_ops` and
    `grad_loss` are ignored when eager execution is enabled.
    @end_compatibility
    """
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
                                name=name)

  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Compute gradients of `loss` for the variables in `var_list`.

    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: A Tensor containing the value to minimize or a callable taking
        no arguments which returns the value to minimize. When eager execution
        is enabled it must be a callable.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.

    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid.
      RuntimeError: If called with eager execution enabled and `loss` is
        not callable.

    @compatibility(eager)
    When eager execution is enabled, `gate_gradients`, `aggregation_method`,
    and `colocate_gradients_with_ops` are ignored.
    @end_compatibility
    """
    if callable(loss):
      with backprop.GradientTape() as tape:
        if var_list is not None:
          tape.watch(var_list)
        loss_value = loss()
      if var_list is None:
        var_list = tape.watched_variables()
      grads = tape.gradient(loss_value, var_list, grad_loss)
      return list(zip(grads, var_list))
    if context.in_eager_mode():
      raise RuntimeError(
          "`loss` passed to Optimizer.compute_gradients should "
          "be a function when eager execution is enabled.")
    if gate_gradients not in [Optimizer.GATE_NONE, Optimizer.GATE_OP,
                              Optimizer.GATE_GRAPH]:
      raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, "
                       "Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not %s" %
                       gate_gradients)
    self._assert_valid_dtypes([loss])
    if grad_loss is not None:
      self._assert_valid_dtypes([grad_loss])
    if var_list is None:
      var_list = (
          variables.trainable_variables() +
          ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    else:
      var_list = nest.flatten(var_list)
    # pylint: disable=protected-access
    var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)
    # pylint: enable=protected-access
    processors = [_get_processor(v) for v in var_list]
    if not var_list:
      raise ValueError("No variables to optimize.")
    var_refs = [p.target() for p in processors]
    grads = gradients.gradients(
        loss, var_refs, grad_ys=grad_loss,
        gate_gradients=(gate_gradients == Optimizer.GATE_OP),
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops)
    if gate_gradients == Optimizer.GATE_GRAPH:
      grads = control_flow_ops.tuple(grads)
    grads_and_vars = list(zip(grads, var_list))
    self._assert_valid_dtypes(
        [v for g, v in grads_and_vars
         if g is not None and v.dtype != dtypes.resource])
    return grads_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
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
    with ops.init_scope():
      self._create_slots([_get_variable_for(v) for v in var_list])
    update_ops = []
    with ops.name_scope(name, self._name) as name:
      self._prepare()
      for grad, var, processor in converted_grads_and_vars:
        if grad is None:
          continue
        # We colocate all ops created in _apply_dense or _apply_sparse
        # on the same device as the variable.
        # TODO(apassos): figure out how to get the variable name here.
        scope_name = var.op.name if context.in_graph_mode() else ""
        with ops.name_scope("update_" + scope_name), ops.colocate_with(var):
          update_ops.append(processor.update_op(self, grad))
      if global_step is None:
        apply_updates = self._finish(update_ops, name)
      else:
        with ops.control_dependencies([self._finish(update_ops, "update")]):
          with ops.colocate_with(global_step):
            if isinstance(global_step, resource_variable_ops.ResourceVariable):
              # TODO(apassos): the implicit read in assign_add is slow; consider
              # making it less so.
              apply_updates = resource_variable_ops.assign_add_variable_op(
                  global_step.handle,
                  ops.convert_to_tensor(1, dtype=global_step.dtype),
                  name=name)
            else:
              apply_updates = state_ops.assign_add(global_step, 1, name=name)

      if context.in_graph_mode():
        if isinstance(apply_updates, ops.Tensor):
          apply_updates = apply_updates.op
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        if apply_updates not in train_op:
          train_op.append(apply_updates)

      return apply_updates

  def get_slot(self, var, name):
    """Return a slot named `name` created for `var` by the Optimizer.

    Some `Optimizer` subclasses use additional variables.  For example
    `Momentum` and `Adagrad` use variables to accumulate updates.  This method
    gives access to these `Variable` objects if for some reason you need them.

    Use `get_slot_names()` to get the list of slot names created by the
    `Optimizer`.

    Args:
      var: A variable passed to `minimize()` or `apply_gradients()`.
      name: A string.

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    named_slots = self._slots.get(name, None)
    if not named_slots:
      return None
    return named_slots.get(_var_key(var), None)

  def get_slot_names(self):
    """Return a list of the names of slots created by the `Optimizer`.

    See `get_slot()`.

    Returns:
      A list of strings.
    """
    return sorted(self._slots.keys())

  def variables(self):
    """A list of variables which encode the current state of `Optimizer`.

    Includes slot variables and additional global variables created by the
    optimizer in the current default graph.

    Returns:
      A list of variables.
    """
    executing_eagerly = context.in_eager_mode()
    current_graph = ops.get_default_graph()

    def _from_current_graph(variable):
      if executing_eagerly:
        # No variable.op in eager mode. We don't expect lots of eager graphs,
        # but behavior should be consistent with graph mode.
        return variable._graph_key == current_graph._graph_key  # pylint: disable=protected-access
      else:
        return variable.op.graph is current_graph

    optimizer_variables = [v for v in self._non_slot_variables()
                           if _from_current_graph(v)]
    for _, variable_dict in self._slots.items():
      for _, slot_for_variable in variable_dict.items():
        if _from_current_graph(slot_for_variable):
          optimizer_variables.append(slot_for_variable)
    # Sort variables by name so that the return is deterministic.
    return sorted(optimizer_variables, key=lambda v: v.name)

  def _create_non_slot_variable(self, initial_value, name, colocate_with):
    """Add an extra variable, not associated with a slot."""
    if context.in_graph_mode():
      graph = colocate_with.graph
    else:
      graph = None

    key = (name, graph)
    v = self._non_slot_dict.get(key, None)
    if v is None:
      with ops.colocate_with(colocate_with):
        v = variable_scope.variable(initial_value, name=name, trainable=False)
      self._non_slot_dict[key] = v

    return v

  def _get_non_slot_variable(self, name, graph=None):
    return self._non_slot_dict.get((name, graph), None)

  def _non_slot_variables(self):
    """Additional variables created by the `Optimizer`.

    Returns:
      A list or tuple of variables.
    """
    return self._non_slot_dict.values()

  def _assert_valid_dtypes(self, tensors):
    """Asserts tensors are all valid types (see `_valid_dtypes`).

    Args:
      tensors: Tensors to check.

    Raises:
      ValueError: If any tensor is not a valid type.
    """
    valid_dtypes = self._valid_dtypes()
    for t in tensors:
      dtype = t.dtype.base_dtype
      if dtype not in valid_dtypes:
        raise ValueError(
            "Invalid type %r for %s, expected: %s." % (
                dtype, t.name, [v for v in valid_dtypes]))

  # --------------
  # Methods to be implemented by subclasses if they want to use the
  # inherited implementation of apply_gradients() or compute_gradients().
  # --------------
  def _valid_dtypes(self):
    """Valid types for loss, variables and gradients.

    Subclasses should override to allow other float types.

    Returns:
      Valid types for loss, variables and gradients.
    """
    return set(
        [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64])

  def _create_slots(self, var_list):
    """Create all slots needed by the variables.

    Args:
      var_list: A list of `Variable` objects.
    """
    # No slots needed by default
    pass

  def _prepare(self):
    """Create all needed tensors before applying gradients.

    This is called with the name_scope using the "name" that
    users have chosen for the application of gradients.
    """
    pass

  def _apply_dense(self, grad, var):
    """Add ops to apply dense gradients to `var`.

    Args:
      grad: A `Tensor`.
      var: A `Variable` object.

    Returns:
      An `Operation`.
    """
    raise NotImplementedError()

  def _resource_apply_dense(self, grad, handle):
    """Add ops to apply dense gradients to the variable `handle`.

    Args:
      grad: a `Tensor` representing the gradient.
      handle: a `Tensor` of dtype `resource` which points to the variable
       to be updated.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    raise NotImplementedError()

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    """Add ops to apply sparse gradients to `handle`, with repeated indices.

    Optimizers which override this method must deal with repeated indices. See
    the docstring of `_apply_sparse_duplicate_indices` for details. By default
    the correct behavior, to sum non-unique indices and their associated
    gradients, is enforced by first pre-processing `grad` and `indices` and
    passing them on to `_resource_apply_sparse`. Optimizers which deal correctly
    with duplicate indices may instead override this method to avoid the
    overhead of summing.

    Args:
      grad: a `Tensor` representing the gradient for the affected indices.
      handle: a `Tensor` of dtype `resource` which points to the variable
       to be updated.
      indices: a `Tensor` of integral type representing the indices for
       which the gradient is nonzero. Indices may be repeated.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    summed_grad, unique_indices = _deduplicate_indexed_slices(
        values=grad, indices=indices)
    return self._resource_apply_sparse(summed_grad, handle, unique_indices)

  def _resource_apply_sparse(self, grad, handle, indices):
    """Add ops to apply sparse gradients to the variable `handle`.

    Similar to `_apply_sparse`, the `indices` argument to this method has been
    de-duplicated. Optimizers which deal correctly with non-unique indices may
    instead override `_resource_apply_sparse_duplicate_indices` to avoid this
    overhead.

    Args:
      grad: a `Tensor` representing the gradient for the affected indices.
      handle: a `Tensor` of dtype `resource` which points to the variable
       to be updated.
      indices: a `Tensor` of integral type representing the indices for
       which the gradient is nonzero. Indices are unique.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    raise NotImplementedError()

  def _apply_sparse_duplicate_indices(self, grad, var):
    """Add ops to apply sparse gradients to `var`, with repeated sparse indices.

    Optimizers which override this method must deal with IndexedSlices objects
    such as the following:

      IndexedSlicesValue(values=[1, 1], indices=[0, 0], dense_shape=[1])

    The correct interpretation is:

      IndexedSlicesValue(values=[2], indices=[0], dense_shape=[1])

    Many optimizers deal incorrectly with repeated indices when updating based
    on sparse gradients (e.g. summing squares rather than squaring the sum, or
    applying momentum terms multiple times). Adding first is always the correct
    behavior, so this is enforced here by reconstructing the IndexedSlices to
    have only unique indices, then calling _apply_sparse.

    Optimizers which deal correctly with repeated indices may instead override
    this method to avoid the overhead of summing indices.

    Args:
      grad: `IndexedSlices`.
      var: A `Variable` object.

    Returns:
      An `Operation`.
    """
    summed_values, unique_indices = _deduplicate_indexed_slices(
        values=grad.values, indices=grad.indices)
    gradient_no_duplicate_indices = ops.IndexedSlices(
        indices=unique_indices,
        values=summed_values,
        dense_shape=grad.dense_shape)
    return self._apply_sparse(gradient_no_duplicate_indices, var)

  def _apply_sparse(self, grad, var):
    """Add ops to apply sparse gradients to `var`.

    The IndexedSlices object passed to `grad` in this function is by default
    pre-processed in `_apply_sparse_duplicate_indices` to remove duplicate
    indices (see its docstring for details). Optimizers which can tolerate or
    have correct special cases for duplicate sparse indices may override
    `_apply_sparse_duplicate_indices` instead of this function, avoiding that
    overhead.

    Args:
      grad: `IndexedSlices`, with no repeated indices.
      var: A `Variable` object.

    Returns:
      An `Operation`.
    """
    raise NotImplementedError()

  def _finish(self, update_ops, name_scope):
    """Do what is needed to finish the update.

    This is called with the `name_scope` using the "name" that
    users have chosen for the application of gradients.

    Args:
      update_ops: List of `Operation` objects to update variables.  This list
        contains the values returned by the `_apply_dense()` and
        `_apply_sparse()` calls.
      name_scope: String.  Name to use for the returned operation.

    Returns:
      The operation to apply updates.
    """
    return control_flow_ops.group(*update_ops, name=name_scope)

  # --------------
  # Utility methods for subclasses.
  # --------------

  def _slot_dict(self, slot_name):
    """Returns a dict for caching slots created under the given name.

    Args:
      slot_name: Name for the slot.

    Returns:
      A dict that maps primary `Variable` objects to the slot created
      for that variable, under the given slot name.
    """
    named_slots = self._slots.get(slot_name, None)
    if named_slots is None:
      named_slots = {}
      self._slots[slot_name] = named_slots
    return named_slots

  def _get_or_make_slot(self, var, val, slot_name, op_name):
    """Find or create a slot for a variable.

    Args:
      var: A `Variable` object.
      val: A `Tensor`.  The initial value of the slot.
      slot_name: Name for the slot.
      op_name: Name to use when scoping the Variable that
        needs to be created for the slot.

    Returns:
      A `Variable` object.
    """
    named_slots = self._slot_dict(slot_name)
    if _var_key(var) not in named_slots:
      new_slot_variable = slot_creator.create_slot(var, val, op_name)
      self._restore_slot_variable(
          slot_name=slot_name, variable=var,
          slot_variable=new_slot_variable)
      named_slots[_var_key(var)] = new_slot_variable
    return named_slots[_var_key(var)]

  def _get_or_make_slot_with_initializer(self, var, initializer, shape, dtype,
                                         slot_name, op_name):
    """Find or create a slot for a variable, using an Initializer.

    Args:
      var: A `Variable` object.
      initializer: An `Initializer`.  The initial value of the slot.
      shape: Shape of the initial value of the slot.
      dtype: Type of the value of the slot.
      slot_name: Name for the slot.
      op_name: Name to use when scoping the Variable that
        needs to be created for the slot.

    Returns:
      A `Variable` object.
    """
    named_slots = self._slot_dict(slot_name)
    if _var_key(var) not in named_slots:
      new_slot_variable = slot_creator.create_slot_with_initializer(
          var, initializer, shape, dtype, op_name)
      self._restore_slot_variable(
          slot_name=slot_name, variable=var,
          slot_variable=new_slot_variable)
      named_slots[_var_key(var)] = new_slot_variable
    return named_slots[_var_key(var)]

  def _zeros_slot(self, var, slot_name, op_name):
    """Find or create a slot initialized with 0.0.

    Args:
      var: A `Variable` object.
      slot_name: Name for the slot.
      op_name: Name to use when scoping the Variable that
        needs to be created for the slot.

    Returns:
      A `Variable` object.
    """
    named_slots = self._slot_dict(slot_name)
    if _var_key(var) not in named_slots:
      new_slot_variable = slot_creator.create_zeros_slot(var, op_name)
      self._restore_slot_variable(
          slot_name=slot_name, variable=var,
          slot_variable=new_slot_variable)
      named_slots[_var_key(var)] = new_slot_variable
    return named_slots[_var_key(var)]

  # --------------
  # For implementing the Checkpointable interface.
  # --------------

  def _restore_slot_variable(self, slot_name, variable, slot_variable):
    """Restore a newly created slot variable's value."""
    variable_key = _var_key(variable)
    deferred_restorations = self._deferred_slot_restorations.get(
        slot_name, {}).pop(variable_key, [])
    # Iterate over restores, highest restore UID first to minimize the number
    # of assignments.
    deferred_restorations.sort(key=lambda position: position.restore_uid,
                               reverse=True)
    for checkpoint_position in deferred_restorations:
      checkpoint_position.restore(slot_variable)

  def _create_or_restore_slot_variable(
      self, slot_variable_position, slot_name, variable):
    """Restore a slot variable's value, possibly creating it.

    Called when a variable which has an associated slot variable is created or
    restored. When executing eagerly, we create the slot variable with a
    restoring initializer.

    No new variables are created when graph building. Instead,
    _restore_slot_variable catches these after normal creation and adds restore
    ops to the graph. This method is nonetheless important when graph building
    for the case when a slot variable has already been created but `variable`
    has just been added to a dependency graph (causing us to realize that the
    slot variable needs to be restored).

    Args:
      slot_variable_position: A `checkpointable._CheckpointPosition` object
        indicating the slot variable `Checkpointable` object to be restored.
      slot_name: The name of this `Optimizer`'s slot to restore into.
      variable: The variable object this slot is being created for.
    """
    named_slots = self._slot_dict(slot_name)
    variable_key = _var_key(variable)
    slot_variable = named_slots.get(variable_key, None)
    if (slot_variable is None
        and context.in_eager_mode()
        and slot_variable_position.is_simple_variable()):
      initializer = checkpointable.CheckpointInitialValue(
          checkpoint_position=slot_variable_position)
      slot_variable = self._get_or_make_slot(
          var=variable,
          val=initializer,
          slot_name=slot_name,
          op_name=self._name)
      # Slot variables are not owned by any one object (because we don't want to
      # save the slot variable if the optimizer is saved without the non-slot
      # variable, or if the non-slot variable is saved without the optimizer;
      # it's a dependency hypergraph with edges of the form (optimizer, non-slot
      # variable, variable)). So we don't _track_ slot variables anywhere, and
      # instead special-case this dependency and otherwise pretend it's a normal
      # graph.
    if slot_variable is not None:
      # If we've either made this slot variable, or if we've pulled out an
      # existing slot variable, we should restore it.
      slot_variable_position.restore(slot_variable)
    else:
      # We didn't make the slot variable. Defer restoring until it gets created
      # normally. We keep a list rather than the one with the highest restore
      # UID in case slot variables have their own dependencies, in which case
      # those could differ between restores.
      self._deferred_slot_restorations.setdefault(
          slot_name, {}).setdefault(variable_key, []).append(
              slot_variable_position)
