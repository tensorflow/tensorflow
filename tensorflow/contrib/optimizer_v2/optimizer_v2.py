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

"""Version 2 of class Optimizer."""
# pylint: disable=g-bad-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import distribution_strategy_context as distribute_ctx
from tensorflow.python.training import optimizer as optimizer_v1
from tensorflow.python.training import slot_creator
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import nest


@six.add_metaclass(abc.ABCMeta)
class _OptimizableVariable(object):
  """Interface for abstracting over variables in the optimizers."""

  @abc.abstractmethod
  def target(self):
    """Returns the optimization target for this variable."""
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def update_op(self, optimizer, g, *args):
    """Returns the update ops for updating the variable."""
    raise NotImplementedError("Calling an abstract method.")


class _RefVariableProcessor(_OptimizableVariable):
  """Processor for Variable."""

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v._ref()  # pylint: disable=protected-access

  def update_op(self, optimizer, g, *args):
    if isinstance(g, ops.Tensor):
      update_op = optimizer._apply_dense(g, self._v, *args)  # pylint: disable=protected-access
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
      return optimizer._apply_sparse_duplicate_indices(g, self._v, *args)


class _DenseReadResourceVariableProcessor(_OptimizableVariable):
  """Processor for dense ResourceVariables."""

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

  def update_op(self, optimizer, g, *args):
    # pylint: disable=protected-access
    update_op = optimizer._resource_apply_dense(g, self._v.op.inputs[0], *args)
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

  def update_op(self, optimizer, g, *args):
    # pylint: disable=protected-access
    if isinstance(g, ops.IndexedSlices):
      if self._v.constraint is not None:
        raise RuntimeError(
            "Cannot use a constraint function on a sparse variable.")
      return optimizer._resource_apply_sparse_duplicate_indices(
          g.values, self._v, g.indices, *args)
    update_op = optimizer._resource_apply_dense(g, self._v, *args)
    if self._v.constraint is not None:
      with ops.control_dependencies([update_op]):
        return self._v.assign(self._v.constraint(self._v))
    else:
      return update_op


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

  def update_op(self, optimizer, g, *args):
    raise NotImplementedError("Trying to update a Tensor ", self._v)


def _get_processor(v):
  """The processor of v."""
  if context.executing_eagerly():
    if isinstance(v, ops.Tensor):
      return _TensorProcessor(v)
    else:
      return _DenseResourceVariableProcessor(v)
  if v.op.type == "VarHandleOp":
    return _DenseResourceVariableProcessor(v)
  if isinstance(v, variables.Variable):
    return _RefVariableProcessor(v)
  if isinstance(v, ops.Tensor):
    return _TensorProcessor(v)
  raise NotImplementedError("Trying to optimize unsupported type ", v)


def _var_key_v2(var):
  """Key for representing a primary variable, for looking up slots."""
  # pylint: disable=protected-access
  if hasattr(var, "_distributed_container"):
    distributed_container = var._distributed_container()
    assert distributed_container is not None
    if context.executing_eagerly():
      return distributed_container._unique_id
    return distributed_container._shared_name
  if context.executing_eagerly():
    return var._unique_id
  return var.op.name


def _resolve(value, name):
  if callable(value):
    value = value()
  return ops.convert_to_tensor(value, name=name)


def _is_dynamic(value):
  """Returns true if __init__ arg `value` should be re-evaluated each step."""
  if callable(value):
    return True
  # Don't need to do anything special in graph mode, since dynamic values
  # will propagate correctly automatically.
  # TODO(josh11b): Add per-device caching across steps using variables for
  # truly static values once we add distributed support.
  if context.executing_eagerly() and isinstance(
      value, resource_variable_ops.ResourceVariable):
    return True
  return False


class _OptimizerV2State(object):
  """Holds per-graph and per-step optimizer state.

  Use _init_with_static_hyper() to create the state for a graph, and then
  _copy_with_dynamic_hyper() to convert that to state for a particular step.
  The difference between the two is that the former only has hyper
  parameter values that are static and the latter also has values that
  can change every step (according to _is_dynamic()).
  """

  def __init__(self, op_name):
    self._op_name = op_name

  def _init_with_static_hyper(self, hyper):
    """Initialize a fresh state object from hyper dict."""
    # self._hyper contains a dict from name to a dict with the Tensor values.
    # This dict starts with a single item with key "None" with the hyper
    # parameter value converted to a Tensor. Other items have dtype keys
    # with that Tensor cast to that dtype.
    with ops.init_scope():
      self._hyper = {
          name: {
              None: ops.convert_to_tensor(value, name=name)
          } for name, (dynamic, value) in sorted(hyper.items()) if not dynamic
      }
    self._slots = {}
    self._non_slot_dict = {}
    # Extra state to help Optimizers implement Checkpointable. Holds information
    # about variables which will be restored as soon as they're created.
    self._deferred_dependencies = {}  # Non-slot variables
    self._deferred_slot_restorations = {}  # Slot variables

  def _copy_with_dynamic_hyper(self, hyper, distribution, non_slot_devices):
    """Create a new state object for a particular step."""
    ret = _OptimizerV2State(self._op_name)
    # pylint: disable=protected-access
    ret._slots = self._slots
    ret._non_slot_dict = self._non_slot_dict
    ret._deferred_dependencies = self._deferred_dependencies
    ret._deferred_slot_restorations = self._deferred_slot_restorations
    ret._hyper = {
        name: {
            None: _resolve(value, name)
        } for name, (dynamic, value) in sorted(hyper.items()) if dynamic
    }
    ret._hyper.update(self._hyper)
    ret._non_slot_devices = non_slot_devices
    ret._distribution = distribution
    return ret

  def _variables(self):
    """Returns a list of all variables held by self."""
    optimizer_variables = list(self._non_slot_dict.values())
    for variable_dict in self._slots.values():
      for slot_for_variable in variable_dict.values():
        optimizer_variables.append(slot_for_variable)
    # Sort variables by name so that the return is deterministic.
    return sorted(optimizer_variables, key=lambda v: v.name)

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

  def create_slot(self, var, val, slot_name, optional_op_name=None):
    """Find or create a slot for a variable.

    Args:
      var: A `Variable` object.
      val: A `Tensor`.  The initial value of the slot.
      slot_name: Name for the slot.
      optional_op_name: Name to use when scoping the Variable that needs to be
        created for the slot.

    Returns:
      A `Variable` object.
    """
    named_slots = self._slot_dict(slot_name)
    var_key = _var_key_v2(var)
    if var_key not in named_slots:
      new_slot_variable = slot_creator.create_slot(
          var, val, optional_op_name or self._op_name)
      self._restore_slot_variable(
          slot_name=slot_name, variable=var, slot_variable=new_slot_variable)
      named_slots[var_key] = new_slot_variable
    return named_slots[var_key]

  def create_slot_with_initializer(self,
                                   var,
                                   initializer,
                                   shape,
                                   dtype,
                                   slot_name,
                                   optional_op_name=None):
    """Find or create a slot for a variable, using an Initializer.

    Args:
      var: A `Variable` object.
      initializer: An `Initializer`.  The initial value of the slot.
      shape: Shape of the initial value of the slot.
      dtype: Type of the value of the slot.
      slot_name: Name for the slot.
      optional_op_name: Name to use when scoping the Variable that needs to be
        created for the slot.

    Returns:
      A `Variable` object.
    """
    named_slots = self._slot_dict(slot_name)
    var_key = _var_key_v2(var)
    if var_key not in named_slots:
      new_slot_variable = slot_creator.create_slot_with_initializer(
          var, initializer, shape, dtype, optional_op_name or self._op_name)
      self._restore_slot_variable(
          slot_name=slot_name, variable=var, slot_variable=new_slot_variable)
      named_slots[var_key] = new_slot_variable
    return named_slots[var_key]

  def zeros_slot(self, var, slot_name, optional_op_name=None):
    """Find or create a slot initialized with 0.0.

    Args:
      var: A `Variable` object.
      slot_name: Name for the slot.
      optional_op_name: Name to use when scoping the Variable that needs to be
        created for the slot.

    Returns:
      A `Variable` object.
    """
    named_slots = self._slot_dict(slot_name)
    var_key = _var_key_v2(var)
    if var_key not in named_slots:
      new_slot_variable = slot_creator.create_zeros_slot(
          var, optional_op_name or self._op_name)
      self._restore_slot_variable(
          slot_name=slot_name, variable=var, slot_variable=new_slot_variable)
      named_slots[var_key] = new_slot_variable
    return named_slots[var_key]

  def _create_or_restore_slot_variable(self,
                                       slot_variable_position,
                                       slot_name,
                                       variable,
                                       optional_op_name=None):
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
      optional_op_name: Name to use when scoping the Variable that needs to be
        created for the slot.
    """
    slot_variable = self.get_slot(var=variable, name=slot_name)
    if (slot_variable is None and context.executing_eagerly() and
        slot_variable_position.is_simple_variable()
        # Defer slot variable creation if there is an active variable creator
        # scope. Generally we'd like to eagerly create/restore slot variables
        # when possible, but this may mean that scopes intended to catch
        # `variable` also catch its eagerly created slot variable
        # unintentionally (specifically make_template would add a dependency on
        # a slot variable if not for this case). Deferring is mostly harmless
        # (aside from double initialization), and makes variable creator scopes
        # behave the same way they do when graph building.
        and not ops.get_default_graph()._variable_creator_stack):  # pylint: disable=protected-access
      initializer = checkpointable.CheckpointInitialValue(
          checkpoint_position=slot_variable_position)
      slot_variable = self.create_slot(
          var=variable,
          val=initializer,
          slot_name=slot_name,
          optional_op_name=optional_op_name)
      # Optimizers do not have unconditional dependencies on their slot
      # variables (nor do any other objects). They are only saved if the
      # variables they were created for are also saved.
    if slot_variable is not None:
      # If we've either made this slot variable, or if we've pulled out an
      # existing slot variable, we should restore it.
      slot_variable_position.restore(slot_variable)
    else:
      # We didn't make the slot variable. Defer restoring until it gets created
      # normally. We keep a list rather than the one with the highest restore
      # UID in case slot variables have their own dependencies, in which case
      # those could differ between restores.
      variable_key = _var_key_v2(variable)
      self._deferred_slot_restorations.setdefault(slot_name, {}).setdefault(
          variable_key, []).append(slot_variable_position)

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
    return named_slots.get(_var_key_v2(var), None)

  def get_slot_names(self):
    """Return a list of the names of slots created by the `Optimizer`.

    See `get_slot()`.

    Returns:
      A list of strings.
    """
    return sorted(self._slots.keys())

  def create_non_slot(self, initial_value, name, colocate_with=None):
    """Add an extra variable, not associated with a slot."""
    v = self._non_slot_dict.get(name, None)
    if v is None:
      if colocate_with is None:
        colocate_with = self._non_slot_devices
      with self._distribution.colocate_vars_with(colocate_with):
        # TODO(josh11b): Use get_variable() except for the legacy Adam use case.
        v = variable_scope.variable(initial_value, name=name, trainable=False)
      self._non_slot_dict[name] = v
      deferred_dependencies_list = self._deferred_dependencies.pop(name, ())
      for checkpoint_position in sorted(
          deferred_dependencies_list,
          key=lambda restore: restore.checkpoint.restore_uid,
          reverse=True):
        checkpoint_position.restore(v)
    return v

  def _restore_slot_variable(self, slot_name, variable, slot_variable):
    """Restore a newly created slot variable's value."""
    variable_key = _var_key_v2(variable)
    deferred_restorations = self._deferred_slot_restorations.get(
        slot_name, {}).pop(variable_key, [])
    # Iterate over restores, highest restore UID first to minimize the number
    # of assignments.
    deferred_restorations.sort(
        key=lambda position: position.restore_uid, reverse=True)
    for checkpoint_position in deferred_restorations:
      checkpoint_position.restore(slot_variable)

  def get_non_slot(self, name):
    """Returns the non-slot variable identified by `name`."""
    return self._non_slot_dict.get(name, None)

  def get_hyper(self, name, dtype=None):
    """Returns the `name` hyper parameter, optionally cast to `dtype`."""
    dtype_dict = self._hyper[name]
    # Do we have the value cast to dtype already cached? This should always
    # succeed when dtype is None.
    if dtype in dtype_dict:
      return dtype_dict[dtype]
    # Not cached, cast to dtype and save the result in the cache.
    result = math_ops.cast(dtype_dict[None], dtype)
    dtype_dict[dtype] = result
    return result


class OptimizerV2(optimizer_v1.Optimizer):
  """Updated base class for optimizers.

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

  ### Non-slot variables

  Some optimizer subclasses, such as `AdamOptimizer` have variables that
  are not associated with the variables to train, just the step itself.

  ### Hyper parameters

  These are arguments passed to the optimizer subclass constructor
  (the `__init__` method), and then passed to `self._set_hyper()`.
  They can be either regular Python values (like 1.0), tensors, or
  callables. If they are callable, the callable will be called during
  `apply_gradients()` to get the value for the hyper parameter.

  ### State

  Internal methods are passed a `state` argument with the correct
  values to use for the slot and non-slot variables, and the hyper
  parameters.
  """

  # Values for gate_gradients.
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2

  def __init__(self, use_locking, name):
    """Create a new Optimizer.

    This must be called by the constructors of subclasses.
    Note that Optimizer instances should not bind to a single graph,
    and so shouldn't keep Tensors as member variables. Generally
    you should be able to use the _set_hyper()/state.get_hyper()
    facility instead.

    Args:
      use_locking: Bool. If True apply use locks to prevent concurrent updates
        to variables.
      name: A non-empty string.  The name to use for accumulators created
        for the optimizer.

    Raises:
      ValueError: If name is malformed.
      RuntimeError: If _create_slots has been overridden instead of
          _create_vars.
    """
    # Note: We intentionally don't call parent __init__.

    # Optimizer._create_slots was replaced by _create_vars in OptimizerV2.
    if (self.__class__._create_slots.__code__ is not  # pylint: disable=protected-access
        OptimizerV2._create_slots.__code__):
      raise RuntimeError(
          "Override _create_vars instead of _create_slots when "
          "descending from OptimizerV2 (class %s)" % self.__class__.__name__)
    if not name:
      raise ValueError("Must specify the optimizer name")

    self._use_locking = use_locking
    self._name = name
    # Map from graph_key to state for that graph. We use the graph_key
    # since it works in both eager and graph mode, and gives the outer
    # graph inside functions.
    replica_context = distribute_ctx.get_replica_context()
    if replica_context is None:
      # In a cross-replica context for a DistributionStrategy, which means
      # only one Optimizer will be created, not one per replica.
      self._per_graph_state = {}
    else:
      # We use get_replica_context().merge_call() to get a single dict
      # shared across all model replicas when running with a
      # DistributionStrategy.
      self._per_graph_state = replica_context.merge_call(lambda _: {})

    # Hyper parameters, and whether they should be re-evaluated every step.
    self._hyper = {}

  def _set_hyper(self, name, value):
    self._hyper[name] = (_is_dynamic(value), value)

  def minimize(self,
               loss,
               global_step=None,
               var_list=None,
               gate_gradients=GATE_OP,
               aggregation_method=None,
               colocate_gradients_with_ops=False,
               name=None,
               grad_loss=None,
               stop_gradients=None,
               scale_loss_by_num_replicas=None):
    """Add operations to minimize `loss` by updating `var_list`.

    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying
    them call `compute_gradients()` and `apply_gradients()` explicitly instead
    of using this function.

    Args:
      loss: A `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.  Defaults to the list of variables collected in the
        graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with the
        corresponding op.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      stop_gradients: Optional. A Tensor or list of tensors not to differentiate
        through.
      scale_loss_by_num_replicas: Optional boolean. If true, scale the loss down
        by the number of replicas. By default, auto-detects whether this is
        needed.

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
        loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss,
        stop_gradients=stop_gradients,
        scale_loss_by_num_replicas=scale_loss_by_num_replicas)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))

    return self.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None,
                        stop_gradients=None,
                        scale_loss_by_num_replicas=None):
    """Compute gradients of `loss` for the variables in `var_list`.

    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: A Tensor containing the value to minimize or a callable taking no
        arguments which returns the value to minimize. When eager execution is
        enabled it must be a callable.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph under
        the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with the
        corresponding op.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      stop_gradients: Optional. A Tensor or list of tensors not to differentiate
        through.
      scale_loss_by_num_replicas: Optional boolean. If true, scale the loss down
        by the number of replicas. By default, auto-detects whether this is
        needed.

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
    # TODO(josh11b): Test that we handle weight decay in a reasonable way.
    if callable(loss):
      with backprop.GradientTape() as tape:
        if var_list is not None:
          tape.watch(var_list)
        loss_value = loss()

        # Scale loss for number of replicas (callable-loss case). In this case,
        # we have to be careful to call distribute_lib.get_loss_reduction()
        # *after* loss() is evaluated, so we know what loss reduction it uses.
        loss_value = self._scale_loss(loss_value, scale_loss_by_num_replicas)

      if var_list is None:
        var_list = tape.watched_variables()
      grads = tape.gradient(loss_value, var_list, grad_loss)
      return list(zip(grads, var_list))
    if context.executing_eagerly():
      raise RuntimeError("`loss` passed to Optimizer.compute_gradients should "
                         "be a function when eager execution is enabled.")

    # Scale loss for number of replicas (non-callable-loss case).
    loss = self._scale_loss(loss, scale_loss_by_num_replicas)

    if gate_gradients not in [
        optimizer_v1.Optimizer.GATE_NONE, optimizer_v1.Optimizer.GATE_OP,
        optimizer_v1.Optimizer.GATE_GRAPH
    ]:
      raise ValueError(
          "gate_gradients must be one of: Optimizer.GATE_NONE, "
          "Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not %s" % gate_gradients)
    self._assert_valid_dtypes([loss])
    if grad_loss is not None:
      self._assert_valid_dtypes([grad_loss])
    if var_list is None:
      var_list = (
          variables.trainable_variables() + ops.get_collection(
              ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
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
        loss,
        var_refs,
        grad_ys=grad_loss,
        gate_gradients=(gate_gradients == optimizer_v1.Optimizer.GATE_OP),
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        stop_gradients=stop_gradients)
    if gate_gradients == optimizer_v1.Optimizer.GATE_GRAPH:
      grads = control_flow_ops.tuple(grads)
    grads_and_vars = list(zip(grads, var_list))
    self._assert_valid_dtypes([
        v for g, v in grads_and_vars
        if g is not None and v.dtype != dtypes.resource
    ])
    return grads_and_vars

  @staticmethod
  def _scale_loss(loss_value, scale_loss_by_num_replicas):
    """Scale loss for the number of replicas."""
    if scale_loss_by_num_replicas is None:
      scale_loss_by_num_replicas = (
          distribute_lib.get_loss_reduction() == ds_reduce_util.ReduceOp.MEAN)
    if scale_loss_by_num_replicas:
      num_replicas = \
        distribute_ctx.get_distribution_strategy().num_replicas_in_sync
      if num_replicas > 1:
        loss_value *= 1. / num_replicas
    return loss_value

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
    """
    # This is a default implementation of apply_gradients() that can be shared
    # by most optimizers.  It relies on the subclass implementing the following
    # methods: _create_vars(), _prepare(), _apply_dense(), and _apply_sparse().

    # Filter out variables with gradients of `None`.
    grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
    if not grads_and_vars:
      raise ValueError("No variables provided.")
    filtered = tuple((g, v) for (g, v) in grads_and_vars if g is not None)
    if not filtered:
      raise ValueError("No gradients provided for any variable: %s." %
                       ([str(v) for _, v in grads_and_vars],))
    return distribute_ctx.get_replica_context().merge_call(
        self._distributed_apply, args=(filtered,),
        kwargs={"global_step": global_step, "name": name})

  def _get_or_create_state(self, var_list=None):
    """Either looks up or creates `_OptimizerV2State`.

    If any variables are available, they should be passed via the `var_list`
    argument, and these will be used to determine the graph to create/retrieve
    state for. Otherwise the returned state is for the current default graph.

    Args:
      var_list: A list of variables to extract a graph from.

    Returns:
      An `_OptimizerV2State` object.
    """
    # Determine the graph_key from the current graph.
    eager_execution = context.executing_eagerly()
    if eager_execution or var_list is None:
      graph = ops.get_default_graph()
    else:
      graph = ops._get_graph_from_inputs(var_list)  # pylint: disable=protected-access
    assert graph is not None
    graph_key = graph._graph_key  # pylint: disable=protected-access

    # Get the per graph state by looking up the graph_key.
    if graph_key in self._per_graph_state:
      per_graph_state = self._per_graph_state[graph_key]
    else:
      per_graph_state = _OptimizerV2State(self._name)
      per_graph_state._init_with_static_hyper(self._hyper)  # pylint: disable=protected-access
      self._per_graph_state[graph_key] = per_graph_state
    return per_graph_state

  def _distributed_apply(self, distribution, grads_and_vars, global_step, name):
    """`apply_gradients` for use with a `DistributionStrategy`."""
    reduced_grads = distribution.batch_reduce(
        ds_reduce_util.ReduceOp.SUM, grads_and_vars)
    var_list = [v for _, v in grads_and_vars]
    grads_and_vars = zip(reduced_grads, var_list)

    unwrapped_var_list = [x for v in var_list for x in distribution.unwrap(v)]
    eager_execution = context.executing_eagerly()
    if eager_execution:
      # Give a clear error in this case instead of "name not supported
      # for Eager Tensors" when we compute non_slot_devices.
      for v in unwrapped_var_list:
        if isinstance(v, ops.Tensor):
          raise NotImplementedError("Trying to update a Tensor ", v)

    with ops.name_scope(name, self._name) as name:
      per_graph_state = self._get_or_create_state(var_list=unwrapped_var_list)
      # Include the current value of any dynamic hyper parameters in `state`.
      non_slot_devices = distribution.non_slot_devices(var_list)
      state = per_graph_state._copy_with_dynamic_hyper(  # pylint: disable=protected-access
          self._hyper, distribution, non_slot_devices)

    # Create any slot and non-slot variables we need in `state`.
    with ops.init_scope():
      self._create_vars(var_list, state)

    with ops.name_scope(name):  # Re-enter name_scope created above
      # Give the child class a chance to do something before we start
      # applying gradients.
      self._prepare(state)

      def update(v, g):
        """Update variable `v` using gradient `g`."""
        assert v is not None

        # Convert the grad to Tensor or IndexedSlices if necessary, and
        # look up a processor for each variable's type.
        try:
          g = ops.convert_to_tensor_or_indexed_slices(g)
        except TypeError:
          raise TypeError("Gradient must be convertible to a Tensor"
                          " or IndexedSlices, or None: %s" % g)
        if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
          raise TypeError(
              "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
        processor = _get_processor(v)

        # We colocate all ops created in _apply_dense or _apply_sparse
        # on the same device as the variable.
        # TODO(apassos): figure out how to get the variable name here.
        scope_name = "" if eager_execution else v.op.name
        # device_policy is set because non-mirrored tensors will be read in
        # `update_op`.
        # TODO(josh11b): Make different state objects for each device to
        # avoid needing to set the device_policy.
        device_policy = context.context().device_policy(
            context.DEVICE_PLACEMENT_SILENT)
        with ops.name_scope("update_" + scope_name), device_policy:
          return processor.update_op(self, g, state)

      # Use the processors to update the variables.
      update_ops = []
      for grad, var in grads_and_vars:
        update_ops.extend(distribution.update(var, update, grad, grouped=False))

      # Give the child class a chance to do something after applying
      # gradients
      def finish():
        # TODO(josh11b): Make different state objects for each device to
        # avoid needing to set the device_policy.
        with context.context().device_policy(context.DEVICE_PLACEMENT_SILENT):
          return self._finish(state)

      update_ops = control_flow_ops.group(update_ops)
      with ops.control_dependencies([update_ops]):
        finish_updates = distribution.update_non_slot(
            non_slot_devices, finish, grouped=False)
      # We said grouped=False, which means finish_updates is always a list.
      # It will be [None] when finish() returns None.
      if finish_updates == [None]:
        finish_updates = [update_ops]

      # Update `global_step` (if any).
      if global_step is None:
        apply_updates = distribution.group(finish_updates, name=name)
      else:
        with ops.control_dependencies(finish_updates):

          def update_global_step(global_step, name):
            return global_step.assign_add(1, read_value=False, name=name)

          apply_updates = distribution.update(global_step, update_global_step,
                                              name)

      # Add the training op to the TRAIN_OP graph collection in graph mode.
      if not eager_execution:
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
    state = self._get_state_for_var(var)
    return state.get_slot(var, name) if state is not None else None

  def get_slot_names(self):
    """Return a list of the names of slots created by the `Optimizer`.

    See `get_slot()`.

    Returns:
      A list of strings.
    """
    state = self._get_per_graph_state()
    return state.get_slot_names() if state is not None else []

  def variables(self):
    """A list of variables which encode the current state of `Optimizer`.

    Includes slot variables and additional global variables created by the
    optimizer in the current default graph.

    Returns:
      A list of variables.
    """
    state = self._get_per_graph_state()
    return state._variables() if state is not None else []  # pylint: disable=protected-access

  # --------------
  # Methods to be implemented by subclasses if they want to use the
  # inherited implementation of apply_gradients() or compute_gradients().
  # --------------
  def _create_vars(self, var_list, state):
    """Create all slots needed by the variables and any non-slot variables.

    Args:
      var_list: A list of `Variable` objects.
      state: An object with these methods: `create_slot(var, val, slot_name,
        optional_op_name)`, `create_slot_with_initializer(` `var, initializer,
        shape, dtype, slot_name, optional_op_name)`, `zeros_slot(var, slot_name,
        optional_op_name)`, `create_non_slot_variable(initial_value, name,
        colocate_with)`, `get_hyper(name)`
    """
    # No slots needed by default
    pass

  def _prepare(self, state):
    """Code to execute before applying gradients.

    Note that most uses of _prepare() in Optimizer have been subsumed
    by explicit support for hyper parameters in OptimizerV2

    Args:
      state: An object with a `get_hyper(name)` method.

    Returns:
      Return value will be ignored.
    """
    pass

  def _apply_dense(self, grad, var, state):
    """Add ops to apply dense gradients to `var`.

    Args:
      grad: A `Tensor`.
      var: A `Variable` object.
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      An `Operation`.
    """
    raise NotImplementedError()

  def _resource_apply_dense(self, grad, handle, state):
    """Add ops to apply dense gradients to the variable `handle`.

    Args:
      grad: a `Tensor` representing the gradient.
      handle: a `Tensor` of dtype `resource` which points to the variable to be
        updated.
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    raise NotImplementedError()

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices,
                                               state):
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
      handle: a `Tensor` of dtype `resource` which points to the variable to be
        updated.
      indices: a `Tensor` of integral type representing the indices for which
        the gradient is nonzero. Indices may be repeated.
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    # pylint: disable=protected-access
    summed_grad, unique_indices = optimizer_v1._deduplicate_indexed_slices(
        values=grad, indices=indices)
    # pylint: enable=protected-access
    return self._resource_apply_sparse(summed_grad, handle, unique_indices,
                                       state)

  def _resource_apply_sparse(self, grad, handle, indices, state):
    """Add ops to apply sparse gradients to the variable `handle`.

    Similar to `_apply_sparse`, the `indices` argument to this method has been
    de-duplicated. Optimizers which deal correctly with non-unique indices may
    instead override `_resource_apply_sparse_duplicate_indices` to avoid this
    overhead.

    Args:
      grad: a `Tensor` representing the gradient for the affected indices.
      handle: a `Tensor` of dtype `resource` which points to the variable to be
        updated.
      indices: a `Tensor` of integral type representing the indices for which
        the gradient is nonzero. Indices are unique.
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    raise NotImplementedError()

  def _apply_sparse_duplicate_indices(self, grad, var, state):
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
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      An `Operation`.
    """
    # pylint: disable=protected-access
    summed_values, unique_indices = optimizer_v1._deduplicate_indexed_slices(
        values=grad.values, indices=grad.indices)
    # pylint: enable=protected-access
    gradient_no_duplicate_indices = ops.IndexedSlices(
        indices=unique_indices,
        values=summed_values,
        dense_shape=grad.dense_shape)
    return self._apply_sparse(gradient_no_duplicate_indices, var, state)

  def _apply_sparse(self, grad, var, state):
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
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      An `Operation`.
    """
    raise NotImplementedError()

  def _finish(self, state):
    """Do what is needed to finish the update.

    This is called inside a scope colocated with any non-slot variables.

    Args:
      state: An object with `get_slot(var, name)`, `get_non_slot(self, name)`,
        and `get_hyper(name)` methods.

    Returns:
      The operation to apply updates, or None if no updates.
    """
    return None

  # --------------
  # Utility methods for subclasses.
  # --------------
  def _get_per_graph_state(self):
    # pylint: disable=protected-access
    return self._per_graph_state.get(ops.get_default_graph()._graph_key, None)

  def _get_state_for_var(self, var):
    # pylint: disable=protected-access
    return self._per_graph_state.get(var._graph_key, None)

  # --------------
  # Overridden methods from Checkpointable.
  # --------------

  def _track_checkpointable(self, *args, **kwargs):
    """Optimizers may not track dependencies. Raises an error."""
    raise NotImplementedError(
        "Optimizers may not have dependencies. File a feature request if this "
        "limitation bothers you.")

  @property
  def _checkpoint_dependencies(self):
    """From Checkpointable. Gather graph-specific non-slot variables to save."""
    current_graph_non_slot_variables = []
    state = self._get_per_graph_state()
    if state is not None:
      for name, variable_object in sorted(
          state._non_slot_dict.items(),  # pylint: disable=protected-access
          # Avoid comparing variables
          key=lambda item: item[0]):
        current_graph_non_slot_variables.append(
            checkpointable.CheckpointableReference(
                name=name, ref=variable_object))
    # Note: ignores super(); Optimizers may not have any dependencies outside of
    # state objects.
    return current_graph_non_slot_variables

  def _lookup_dependency(self, name):
    """From Checkpointable. Find a non-slot variable in the current graph."""
    state = self._get_per_graph_state()
    if state is None:
      return None
    else:
      return state.get_non_slot(name)

  @property
  def _deferred_dependencies(self):
    """Lets Checkpointable know where non-slot variables are created.

    If necessary, creates a new state object for the current default graph.
    Checkpointable will then add entries to that state's deferred dependency
    dictionary. The state object will check that dictionary when creating
    non-slot variables, restoring their value if an entry is found.

    Returns:
      A dictionary which holds deferred dependencies for the current default
      graph.
    """
    state = self._get_or_create_state()
    return state._deferred_dependencies  # pylint: disable=protected-access

  def _create_or_restore_slot_variable(self, slot_variable_position, slot_name,
                                       variable):
    """Checkpointable: Restore a slot variable's value, possibly creating it.

    Called when a variable which has an associated slot variable is created or
    restored.

    Args:
      slot_variable_position: A `checkpointable._CheckpointPosition` object
        indicating the slot variable `Checkpointable` object to be restored.
      slot_name: The name of this `Optimizer`'s slot to restore into.
      variable: The variable object this slot is being created for.
    """
    state = self._get_or_create_state(var_list=[variable])
    state._create_or_restore_slot_variable(  # pylint: disable=protected-access
        slot_variable_position=slot_variable_position,
        slot_name=slot_name,
        variable=variable,
        optional_op_name=self._name)

  # --------------
  # Unsupported parent methods
  # --------------
  def _slot_dict(self, slot_name):
    raise NotImplementedError("_slot_dict() method unsupported in OptimizerV2")

  def _get_or_make_slot(self, var, val, slot_name, op_name):
    raise NotImplementedError(
        "_get_or_make_slot() method unsupported in OptimizerV2")

  def _get_or_make_slot_with_initializer(self, var, initializer, shape, dtype,
                                         slot_name, op_name):
    raise NotImplementedError(
        "_get_or_make_slot_with_initializer() method unsupported in "
        "OptimizerV2")

  def _create_non_slot_variable(self, initial_value, name, colocate_with):
    raise NotImplementedError(
        "_create_non_slot_variable() method unsupported in OptimizerV2")

  def _get_non_slot_variable(self, name, graph=None):
    raise NotImplementedError(
        "_get_non_slot_variable() method unsupported in OptimizerV2")

  def _non_slot_variables(self):
    raise NotImplementedError(
        "_non_slot_variables() method unsupported in OptimizerV2")
