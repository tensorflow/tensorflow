"""Base class for optimizers."""
# pylint: disable=g-bad-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import types as tf_types
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables


class Optimizer(object):
  """Base class for optimizers.

  This class defines the API to add Ops to train a model.  You never use this
  class directly, but instead instantiate one of its subclasses such as
  `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.

  ### Usage

  ```
  # Create an optimizer with the desired parameters.
  opt = GradientDescentOptimizer(learning_rate=0.1)
  # Add Ops to the graph to minimize a cost by updating a list of variables.
  # "cost" is a Tensor, and the list of variables contains variables.Variable
  # objects.
  opt_op = opt.minimize(cost, <list of variables>)
  ```

  In the training program you will just have to run the returned Op.

  ```
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

  ```
  # Create an optimizer.
  opt = GradientDescentOptimizer(learning_rate=0.1)

  # Compute the gradients for a list of variables.
  grads_and_vars = opt.compute_gradients(loss, <list of variables>)

  # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
  # need to the 'gradient' part, for example cap them, etc.
  capped_grads_and_vars = [(MyCapper(gv[0]), gv[1])) for gv in grads_and_vars]

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

  <b>GATE_NONE</b>: Compute and apply gradients in parallel.  This provides the
  maximum parallelism in execution, at the cost of some non-reproducibility in
  the results.  For example the two gradients of MatMul depend on the input
  values: With `GATE_NONE` one of the gradients could be applied to one of the
  inputs _before_ the other gradient is computed resulting in non-reproducible
  results.

  <b>GATE_OP</b>: For each Op, make sure all gradients are computed before they
  are used.  This prevents race conditions for Ops that generate gradients for
  multiple inputs where the gradients depend on the inputs.

  <b>GATE_GRAPH</b>: Make sure all gradients for all variables are computed
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
      ValueError: if name is malformed.
    """
    if not name:
      raise ValueError("Must specify the optimizer name")
    self._use_locking = use_locking
    self._name = name
    # Dictionary of slots.
    #  {slot_name : { variable_to_train: slot_for_the_variable, ...}, ... }
    self._slots = {}

  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None, name=None):
    """Add operations to minimize 'loss' by updating 'var_list'.

    This method simply combines calls compute_gradients() and
    apply_gradients(). If you want to process the gradient before applying them
    call compute_gradients() and apply_gradients() explicitly instead of using
    this function.

    Args:
      loss: A Tensor containing the value to minimize.
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      var_list: Optional list of variables.Variable to update to minimize
        'loss'.  Defaults to the list of variables collected in the graph
        under the key GraphKeys.TRAINABLE_VARIABLES.
      gate_gradients: How to gate the computation of gradients.  Can be
        GATE_NONE, GATE_OP, or  GATE_GRAPH.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      name: Optional name for the returned operation.

    Returns:
      An Operation that updates the variables in 'var_list'.  If 'global_step'
      was not None, that operation also increments global_step.

    Raises:
      ValueError: if some of the variables are not variables.Variable objects.
    """
    grads_and_vars = self.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method)
    return self.apply_gradients(grads_and_vars, global_step=global_step,
                                name=name)

  def compute_gradients(self, loss, var_list=None, gate_gradients=GATE_OP,
                        aggregation_method=None):
    """Compute gradients of "loss" for the variables in "var_list".

    This is the first part of minimize().  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a Tensor, a
    IndexedSlices, or None if there is no gradient for the
    given variable.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list of variables.Variable to update to minimize
        "loss".  Defaults to the list of variables collected in the graph
        under the key GraphKey.TRAINABLE_VARIABLES.
      gate_gradients: How to gate the computation of gradients.  Can be
        GATE_NONE, GATE_OP, or  GATE_GRAPH.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.

    Returns:
      A list of (gradient, variable) pairs.

    Raises:
      TypeError: If var_list contains anything else than variables.Variable.
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
        raise TypeError("Argument is not a variables.Variable: %s" % var)
    grads = gradients.gradients(
        loss, var_list, gate_gradients=(gate_gradients == Optimizer.GATE_OP),
        aggregation_method=aggregation_method)
    if gate_gradients == Optimizer.GATE_GRAPH:
      grads = control_flow_ops.tuple(grads)
    grads_and_vars = list(zip(grads, var_list))
    self._assert_valid_dtypes([v for g, v in grads_and_vars if g is not None])
    return grads_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This is the second part of minimize(). It returns an Operation that
    applies gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An Operation that applies the specified gradients. If 'global_step'
      was not None, that operation also increments global_step.

    Raises:
      TypeError: if grads_and_vars is malformed.
    """
    # This is a default implementation of apply_gradients() that can be shared
    # by most optimizers.  It relies on the subclass implementing the following
    # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().
    for g, v in grads_and_vars:
      if not isinstance(g, (ops.Tensor, ops.IndexedSlices, type(None))):
        raise TypeError(
            "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
      if not isinstance(v, variables.Variable):
        raise TypeError(
            "Variable must be a variables.Variable: %s" % v)
      if g is not None:
        self._assert_valid_dtypes([g, v])
    self._create_slots([v for g, v in grads_and_vars if g is not None])
    update_ops = []
    with ops.op_scope([], name, self._name) as name:
      self._prepare()
      for grad, var in grads_and_vars:
        if not grad:
          continue
        with ops.name_scope("update_" + var.op.name), ops.device(var.device):
          if isinstance(grad, ops.Tensor):
            update_ops.append(self._apply_dense(grad, var))
          else:
            update_ops.append(self._apply_sparse(grad, var))
      if global_step is None:
        return self._finish(update_ops, name)
      else:
        with ops.control_dependencies([self._finish(update_ops, "update")]):
          with ops.device(global_step.device):
            return state_ops.assign_add(global_step, 1, name=name).op

  def get_slot(self, var, name):
    """Return a slot named "name" created for "var" by the Optimizer.

    Some Optimizer subclasses use additional variables.  For example
    Momentum and Adagrad use variables to accumulate updates.  This method
    gives access to these Variables if for some reason you need them.

    Use get_slot_names() to get the list of slot names created by the Optimizer.

    Args:
      var: A variable passed to minimize() or apply_gradients().
      name: A string.

    Returns:
      The Variable for the slot if it was created, None otherwise.
    """
    named_slots = self._slots.get(name, None)
    if not named_slots:
      return None
    return named_slots.get(var, None)

  def get_slot_names(self):
    """Return a list of the names of slots created by the Optimizer.

    See get_slot().

    Returns:
      A list of strings.
    """
    return sorted(self._slots.keys())

  def _assert_valid_dtypes(self, tensors):
    """Asserts tensors are all valid types (see _valid_dtypes).

    Args:
      tensors: tensors to check.
    Raises:
      ValueError: if any tensor is not a valid type.
    """
    valid_dtypes = self._valid_dtypes()
    for t in tensors:
      dtype = t.dtype.base_dtype
      if dtype not in valid_dtypes:
        raise ValueError(
            "Invalid type %s for %s, expected: %s." % (
                dtype, t.name, [v for v in valid_dtypes]))

  # --------------
  # Methods to be implemented by subclasses if they want to use the
  # inherited implementation of apply_gradients() or compute_gradients().
  # --------------
  def _valid_dtypes(self):
    """Valid types for loss, variables and gradients.

    Defaults to float32. Subclasses should override to allow other types.

    Returns:
      Valid types for loss, variables and gradients.
    """
    return set([tf_types.float32])

  def _create_slots(self, var_list):
    """Create all slots needed by the variables.

    Args:
      var_list: A list of variables.Variable.
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
    """Add ops to apply dense gradients to "var".

    Args:
      grad: A Tensor.
      var: A variables.Variable.

    Return:
      An Operation.
    """
    raise NotImplementedError()

  def _apply_sparse(self, grad, var):
    """Add ops to apply sparse gradients to "var".

    Args:
      grad: IndexedSlices.
      var: A variables.Variable.

    Return:
      An Operation.
    """
    raise NotImplementedError()

  def _finish(self, update_ops, name_scope):
    """Do what is needed to finish the update.

    This is called with the name_scope using the "name" that
    users have chosen for the application of gradients.

    Args:
      update_ops: List of Operations to update variables.  This list contains
        the values returned by the _apply_dense() and _apply_sparse() calls.
      name_scope: string.  Name to use for the returned operation.

    Returns:
      The operation to apply updates.
    """
    return control_flow_ops.group(*update_ops, name=name_scope)

  # --------------
  # Utility methods for subclasses.
  # --------------

  def _get_or_make_slot(self, var, val, slot_name, op_name):
    """Find or create a slot for a variable.

    Args:
      var: A variables.Variable.
      val: A Tensor.  The initial value of the slot.
      slot_name: Name for the slot.
      op_name: Name to use when scoping the Variable that
        needs to be created for  the slot.

    Returns:
      A variables.Variable.
    """
    named_slots = self._slots.get(slot_name, None)
    if named_slots is None:
      named_slots = {}
      self._slots[slot_name] = named_slots
    slot = named_slots.get(var, None)
    if slot is None:
      # Scope the slot name in the namespace of the Variable and
      # create the slot on the same device as the variable.
      with ops.name_scope(var.op.name + "/" + op_name) as scope:
        with ops.device(var.device):
          slot = variables.Variable(val, name=scope, trainable=False)
      named_slots[var] = slot
    return slot

  def _zeros_slot(self, var, slot_name, op_name):
    """Find or create a slot initialized with 0.0.

    Args:
      var: A variables.Variable.
      slot_name: Name for the slot.
      op_name: Name to use when scoping the Variable that
        needs to be created for  the slot.

    Returns:
      A variables.Variable.
    """
    val = array_ops.zeros(var.get_shape().as_list(), dtype=var.dtype)
    return self._get_or_make_slot(var, val, slot_name, op_name)
