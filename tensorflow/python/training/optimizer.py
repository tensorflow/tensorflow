# Copyright 2015 Google Inc. All Rights Reserved.
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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator


class Optimizer(object):
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

  @@__init__

  @@minimize
  @@compute_gradients
  @@apply_gradients

  ### Gating Gradients

  Both `minimize()` and `compute_gradients()` accept a `gate_gradient` argument
  that controls the degree of parallelism during the application of the
  gradients.

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

  @@get_slot_names
  @@get_slot
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
    #  {slot_name : { variable_to_train: slot_for_the_variable, ...}, ... }
    self._slots = {}

  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None):
    """Add operations to minimize `loss` by updating `var_list`.

    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying
    them call `compute_gradients()` and `apply_gradients()` explicitly instead
    of using this function.

    Args:
      loss: A `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      var_list: Optional list of `Variable` objects to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      name: Optional name for the returned operation.

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    Raises:
      ValueError: If some of the variables are not `Variable` objects.
    """
    grads_and_vars = self.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops)
    return self.apply_gradients(grads_and_vars, global_step=global_step,
                                name=name)

  def compute_gradients(self, loss, var_list=None, gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False):
    """Compute gradients of `loss` for the variables in `var_list`.

    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list of tf.Variable to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.

    Returns:
      A list of (gradient, variable) pairs.

    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid.
    """
    if gate_gradients not in [Optimizer.GATE_NONE, Optimizer.GATE_OP,
                              Optimizer.GATE_GRAPH]:
      raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, "
                       "Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not %s" %
                       gate_gradients)
    self._assert_valid_dtypes([loss])
    if var_list is None:
      var_list = variables.trainable_variables()
    for var in var_list:
      if not isinstance(var, variables.Variable):
        raise TypeError("Argument is not a tf.Variable: %s" % var)
    if not var_list:
      raise ValueError("No variables to optimize")
    var_refs = [v.ref() for v in var_list]
    grads = gradients.gradients(
        loss, var_refs, gate_gradients=(gate_gradients == Optimizer.GATE_OP),
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops)
    if gate_gradients == Optimizer.GATE_GRAPH:
      grads = control_flow_ops.tuple(grads)
    grads_and_vars = list(zip(grads, var_list))
    self._assert_valid_dtypes([v for g, v in grads_and_vars if g is not None])
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
    grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works
    for g, v in grads_and_vars:
      if not isinstance(g, (ops.Tensor, ops.IndexedSlices, type(None))):
        raise TypeError(
            "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
      if not isinstance(v, variables.Variable):
        raise TypeError(
            "Variable must be a tf.Variable: %s" % v)
      if g is not None:
        self._assert_valid_dtypes([g, v])
    var_list = [v for g, v in grads_and_vars if g is not None]
    if not var_list:
      raise ValueError("No gradients provided for any variable: %s" %
                       (grads_and_vars,))
    with ops.control_dependencies(None):
      self._create_slots(var_list)
    update_ops = []
    with ops.op_scope([], name, self._name) as name:
      self._prepare()
      for grad, var in grads_and_vars:
        if grad is None:
          continue
        # We colocate all ops created in _apply_dense or _apply_sparse
        # on the same device as the variable.
        with ops.name_scope("update_" + var.op.name), ops.colocate_with(var):
          if isinstance(grad, ops.Tensor):
            update_ops.append(self._apply_dense(grad, var))
          else:
            update_ops.append(self._apply_sparse(grad, var))
      if global_step is None:
        return self._finish(update_ops, name)
      else:
        with ops.control_dependencies([self._finish(update_ops, "update")]):
          with ops.colocate_with(global_step):
            return state_ops.assign_add(global_step, 1, name=name).op

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
    return named_slots.get(var, None)

  def get_slot_names(self):
    """Return a list of the names of slots created by the `Optimizer`.

    See `get_slot()`.

    Returns:
      A list of strings.
    """
    return sorted(self._slots.keys())

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

    Defaults to `float32`. Subclasses should override to allow other types.

    Returns:
      Valid types for loss, variables and gradients.
    """
    return set([dtypes.float32])

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

    Return:
      An `Operation`.
    """
    raise NotImplementedError()

  def _apply_sparse(self, grad, var):
    """Add ops to apply sparse gradients to `var`.

    Args:
      grad: `IndexedSlices`.
      var: A `Variable` object.

    Return:
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
        needs to be created for  the slot.

    Returns:
      A `Variable` object.
    """
    named_slots = self._slot_dict(slot_name)
    if var not in named_slots:
      named_slots[var] = slot_creator.create_slot(var, val, op_name)
    return named_slots[var]

  def _zeros_slot(self, var, slot_name, op_name):
    """Find or create a slot initialized with 0.0.

    Args:
      var: A `Variable` object.
      slot_name: Name for the slot.
      op_name: Name to use when scoping the Variable that
        needs to be created for  the slot.

    Returns:
      A `Variable` object.
    """
    named_slots = self._slot_dict(slot_name)
    if var not in named_slots:
      named_slots[var] = slot_creator.create_zeros_slot(var, op_name)
    return named_slots[var]
