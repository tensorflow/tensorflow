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

import abc

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import slot_creator
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


def get_filtered_grad_fn(grad_fn):
  # `distributed_context.join()` requires that its arguments are parallel
  # across threads, and in particular that `grads_and_vars` has the same
  # variables in the same order.

  # When computing gradients in eager mode with multiple threads, you
  # can get extra variables with a gradient of `None`. This happens when
  # those variables are accessed in another thread during the gradient
  # computation. To get a consistent set of variables, we filter out
  # those with `None` gradients.
  def filtered_grad_fn(*args, **kwargs):
    return [(g, v) for g, v in grad_fn(*args, **kwargs) if g is not None]

  return filtered_grad_fn


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
  """Returns slot key for `var`."""
  # pylint: disable=protected-access
  var = distribute_utils.value_container(var)
  if (distribute_utils.is_distributed_variable(var) and
      not ops.executing_eagerly_outside_functions()):
    return (var.graph, var._shared_name)
  if hasattr(var, "op"):
    return (var.op.graph, var.op.name)
  return var._unique_id
  # pylint: enable=protected-access


class _OptimizableVariable(metaclass=abc.ABCMeta):
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
    if isinstance(g, tensor.Tensor):
      update_op = optimizer._apply_dense(g, self._v)  # pylint: disable=protected-access
      if self._v.constraint is not None:
        with ops.control_dependencies([update_op]):
          return self._v.assign(self._v.constraint(self._v))
      else:
        return update_op
    else:
      assert isinstance(g, indexed_slices.IndexedSlices), (
          "Gradient ", g, " is neither a tensor nor IndexedSlices.")
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
    if isinstance(g, indexed_slices.IndexedSlices):
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
  if context.executing_eagerly():
    if isinstance(v, tensor.Tensor):
      return _TensorProcessor(v)
    else:
      return _DenseResourceVariableProcessor(v)
  if resource_variable_ops.is_resource_variable(v) and not v._in_graph_mode:  # pylint: disable=protected-access
    # True if and only if `v` was initialized eagerly.
    return _DenseResourceVariableProcessor(v)
  if v.op.type == "VarHandleOp":
    return _DenseResourceVariableProcessor(v)
  if isinstance(v, variables.Variable):
    return _RefVariableProcessor(v)
  if isinstance(v, tensor.Tensor):
    return _TensorProcessor(v)
  raise NotImplementedError("Trying to optimize unsupported type ", v)


@tf_export(v1=["train.Optimizer"])
class Optimizer(
    # Optimizers inherit from Trackable rather than AutoTrackable
    # since they do most of their dependency management themselves (slot
    # variables are special-cased, and non-slot variables are keyed to graphs).
    trackable.Trackable):
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

  @compatibility(TF2)
  `tf.compat.v1.train.Optimizer` can be used in eager mode and `tf.function`,
  but it is not recommended. Please use the subclasses of
  `tf.keras.optimizers.Optimizer` instead in TF2. Please see [Basic training
  loops](https://www.tensorflow.org/guide/basic_training_loops) or
  [Writing a training loop from scratch]
  (https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
  for examples.

  If your TF1 code contains a `tf.compat.v1.train.Optimizer` symbol, whether it
  is used with or without a `tf.estimator.Estimator`, you cannot simply replace
  that with the corresponding `tf.keras.optimizers.Optimizer`s. To migrate to
  TF2, it is advised the whole training program used with `Estimator` to be
  migrated to Keras `Model.fit` based or TF2 custom training loops.

  #### Structural Mapping to Native TF2

  Before:

  ```python
  sgd_op = tf.compat.v1.train.GradientDescentOptimizer(3.0)
  opt_op = sgd_op.minimize(cost, global_step, [var0, var1])
  opt_op.run(session=session)
  ```

  After:

  ```python
  sgd = tf.keras.optimizers.SGD(3.0)
  sgd.minimize(cost_fn, [var0, var1])
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name    | Note                       |
  | :-------------------- | :-------------- | :------------------------- |
  | `use_locking`         | Not supported   | -                          |
  | `name`                | `name. `        | -                          |

  #### Before & After Usage Example

  Before:

  >>> g = tf.compat.v1.Graph()
  >>> with g.as_default():
  ...   var0 = tf.compat.v1.Variable([1.0, 2.0])
  ...   var1 = tf.compat.v1.Variable([3.0, 4.0])
  ...   cost = 5 * var0 + 3 * var1
  ...   global_step = tf.compat.v1.Variable(
  ...       tf.compat.v1.zeros([], tf.compat.v1.int64), name='global_step')
  ...   init_op = tf.compat.v1.initialize_all_variables()
  ...   sgd_op = tf.compat.v1.train.GradientDescentOptimizer(3.0)
  ...   opt_op = sgd_op.minimize(cost, global_step, [var0, var1])
  >>> session = tf.compat.v1.Session(graph=g)
  >>> session.run(init_op)
  >>> opt_op.run(session=session)
  >>> print(session.run(var0))
  [-14. -13.]


  After:
  >>> var0 = tf.Variable([1.0, 2.0])
  >>> var1 = tf.Variable([3.0, 4.0])
  >>> cost_fn = lambda: 5 * var0 + 3 * var1
  >>> sgd = tf.keras.optimizers.SGD(3.0)
  >>> sgd.minimize(cost_fn, [var0, var1])
  >>> print(var0.numpy())
  [-14. -13.]

  @end_compatibility


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
    # For implementing Trackable. Stores information about how to restore
    # slot variables which have not yet been created
    # (trackable._CheckpointPosition objects).
    #  {slot_name :
    #      {_var_key(variable_to_train): [checkpoint_position, ... ], ... },
    #   ... }
    self._deferred_slot_restorations = {}

    # TODO(isaprykin): When using a DistributionStrategy, and when an
    # optimizer is created in each replica, it might be dangerous to
    # rely on some Optimizer methods.  When such methods are called on a
    # per-replica optimizer, an exception needs to be thrown.  We do
    # allow creation per-replica optimizers however, because the
    # compute_gradients()->apply_gradients() sequence is safe.

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
    takes no arguments and computes the value to be minimized. Minimization (and
    gradient computation) is done with respect to the elements of `var_list` if
    not None, else with respect to any trainable variables created during the
    execution of the `loss` function. `gate_gradients`, `aggregation_method`,
    `colocate_gradients_with_ops` and `grad_loss` are ignored when eager
    execution is enabled.
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

    @compatibility(TF2)
    `tf.keras.optimizers.Optimizer` in TF2 does not provide a
    `compute_gradients` method, and you should use a `tf.GradientTape` to
    obtain the gradients:

    ```python
    @tf.function
    def train step(inputs):
      batch_data, labels = inputs
      with tf.GradientTape() as tape:
        predictions = model(batch_data, training=True)
        loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)(labels, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    ```

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

        # Scale loss if using a "mean" loss reduction and multiple replicas.
        # Have to be careful to call distribute_utils.get_loss_reduction()
        # *after* loss() is evaluated, so we know what loss reduction it uses.
        # TODO(josh11b): Test that we handle weight decay in a reasonable way.
        loss_value = self._scale_loss(loss_value)

      if var_list is None:
        var_list = tape.watched_variables()
      # TODO(jhseu): Figure out why GradientTape's gradients don't require loss
      # to be executed.
      with ops.control_dependencies([loss_value]):
        grads = tape.gradient(loss_value, var_list, grad_loss)
      return list(zip(grads, var_list))

    # Non-callable/Tensor loss case
    if context.executing_eagerly():
      raise RuntimeError(
          "`loss` passed to Optimizer.compute_gradients should "
          "be a function when eager execution is enabled.")

    # Scale loss if using a "mean" loss reduction and multiple replicas.
    loss = self._scale_loss(loss)

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

  @staticmethod
  def _scale_loss(loss_value):
    ops.get_default_graph()._is_loss_scaled_by_optimizer = False  # pylint: disable=protected-access
    if distribute_utils.get_loss_reduction() == ds_reduce_util.ReduceOp.MEAN:
      num_replicas = distribute_lib.get_strategy().num_replicas_in_sync
      if num_replicas > 1:
        loss_value *= (1. / num_replicas)
        ops.get_default_graph()._is_loss_scaled_by_optimizer = True  # pylint: disable=protected-access
    return loss_value

  def apply_gradients(
      self,
      grads_and_vars,
      global_step=None,
      name=None,
      skip_gradients_aggregation=False,
  ):
    """Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    @compatibility(TF2)
    #### How to Map Arguments

    | TF1 Arg Name          | TF2 Arg Name    | Note                       |
    | :-------------------- | :-------------- | :------------------------- |
    | `grads_and_vars`      | `grads_and_vars`| -                          |
    | `global_step`         | Not supported.  | Use `optimizer.iterations` |
    | `name`                | `name. `        | -                          |

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.
      skip_gradients_aggregation: If true, gradients aggregation will not be
        performed inside optimizer. Usually this arg is set to True when you
        write custom code aggregating gradients outside the optimizer.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
      RuntimeError: If you should use `_distributed_apply()` instead.
    """
    # This is a default implementation of apply_gradients() that can be shared
    # by most optimizers.  It relies on the subclass implementing the following
    # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().

    # TODO(isaprykin): Get rid of `has_strategy()` check by
    # always calling _distributed_apply(), using the default distribution
    # as needed.
    if distribute_lib.has_strategy() and not skip_gradients_aggregation:
      # Handle DistributionStrategy case.
      if distribute_lib.in_cross_replica_context():
        raise RuntimeError("Use `_distributed_apply()` instead of "
                           "`apply_gradients()` in a cross-replica context.")

      grads_and_vars = get_filtered_grad_fn(lambda: grads_and_vars)()
      return distribute_lib.get_replica_context().merge_call(
          self._distributed_apply, args=(grads_and_vars, global_step, name))

    # No DistributionStrategy case.
    grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
    if not grads_and_vars:
      raise ValueError("No variables provided.")
    converted_grads_and_vars = []
    for g, v in grads_and_vars:
      if g is not None:
        try:
          # Convert the grad to Tensor or IndexedSlices if necessary.
          g = indexed_slices.convert_to_tensor_or_indexed_slices(g)
        except TypeError:
          raise TypeError(
              "Gradient must be convertible to a Tensor"
              " or IndexedSlices, or None: %s" % g)
        if not isinstance(g, (tensor.Tensor, indexed_slices.IndexedSlices)):
          raise TypeError(
              "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
      p = _get_processor(v)
      converted_grads_and_vars.append((g, v, p))

    converted_grads_and_vars = tuple(converted_grads_and_vars)
    var_list = [v for g, v, _ in converted_grads_and_vars if g is not None]
    if not var_list:
      raise ValueError("No gradients provided for any variable: %s." %
                       ([str(v) for _, v, _ in converted_grads_and_vars],))
    with ops.init_scope():
      self._create_slots(var_list)
    update_ops = []
    with ops.name_scope(name, self._name, skip_on_eager=False) as name:
      self._prepare()
      for grad, var, processor in converted_grads_and_vars:
        if grad is None:
          continue
        # We colocate all ops created in _apply_dense or _apply_sparse
        # on the same device as the variable.
        # TODO(apassos): figure out how to get the variable name here.
        if (context.executing_eagerly() or
            resource_variable_ops.is_resource_variable(var)
            and not var._in_graph_mode):  # pylint: disable=protected-access
          scope_name = ""
        else:
          scope_name = var.op.name
        with ops.name_scope(
            "update_" + scope_name,
            skip_on_eager=False), ops.colocate_with(var):
          update_ops.append(processor.update_op(self, grad))
      if global_step is None:
        apply_updates = self._finish(update_ops, name)
      else:
        with ops.control_dependencies([self._finish(update_ops, "update")]):
          with ops.colocate_with(global_step):
            if isinstance(
                global_step, resource_variable_ops.BaseResourceVariable):
              # TODO(apassos): the implicit read in assign_add is slow; consider
              # making it less so.
              apply_updates = resource_variable_ops.assign_add_variable_op(
                  global_step.handle,
                  ops.convert_to_tensor(1, dtype=global_step.dtype),
                  name=name)
            else:
              apply_updates = state_ops.assign_add(global_step, 1, name=name)

      if not context.executing_eagerly():
        if isinstance(apply_updates, tensor.Tensor):
          apply_updates = apply_updates.op
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        if apply_updates not in train_op:
          train_op.append(apply_updates)

      return apply_updates

  def _distributed_apply(self,
                         distribution,
                         grads_and_vars,
                         global_step=None,
                         name=None):
    """A version of `apply_gradients` for cross-replica context.

    This is a version of `apply_gradients()` for when you are using a
    `DistributionStrategy` and are in a cross-replica context. If in a
    replica context, use `apply_gradients()` as normal.

    Args:
      distribution: A `DistributionStrategy` object.
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`, and then aggregated across replicas.
      global_step: Optional (mirrored) `Variable` to increment by one
        after the variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients across all
      replicas. If `global_step` was not None, that operation also
      increments `global_step`
    """
    reduced_grads = distribution.extended.batch_reduce_to(
        ds_reduce_util.ReduceOp.SUM, grads_and_vars)
    var_list = [v for _, v in grads_and_vars]
    grads_and_vars = zip(reduced_grads, var_list)

    # Note that this is called in a cross-replica context.
    with ops.init_scope():
      self._create_slots(var_list)

    def update(v, g):
      """Apply gradients to a replica variable."""
      assert v is not None

      try:
        # Convert the grad to Tensor or IndexedSlices if necessary.
        g = indexed_slices.convert_to_tensor_or_indexed_slices(g)
      except TypeError:
        raise TypeError("Gradient must be convertible to a Tensor"
                        " or IndexedSlices, or None: %s" % g)
      if not isinstance(g, (tensor.Tensor, indexed_slices.IndexedSlices)):
        raise TypeError(
            "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
      p = _get_processor(v)

      if context.executing_eagerly() or (
          resource_variable_ops.is_resource_variable(v) and
          not v._in_graph_mode):  # pylint: disable=protected-access
        scope_name = v.name.split(":")[0]
      else:
        scope_name = v.op.name

      # device_policy is set because non-mirrored tensors will be read in
      # `update_op`. `_resource_apply_dense`, `lr_t`, `beta1_t` and `beta2_t`
      # is an example.
      with ops.name_scope("update_" + scope_name):
        return p.update_op(self, g)

    with ops.name_scope(name, self._name) as name:
      self._prepare()

      update_ops = [
          op
          for grad, var in grads_and_vars
          for op in distribution.extended.update(
              var, update, args=(grad,), group=False)
      ]

      def finish(self, update_ops):
        return self._finish(update_ops, "update")

      non_slot_devices = distribution.extended.non_slot_devices(var_list)
      finish_updates = distribution.extended.update_non_slot(
          non_slot_devices, finish, args=(self, update_ops), group=False)
      if global_step is None:
        apply_updates = distribution.group(finish_updates, name=name)
      else:
        with ops.control_dependencies(finish_updates):
          apply_updates = distribution.extended.update(
              global_step, state_ops.assign_add, args=(1,),
              kwargs={"name": name})

      if not context.executing_eagerly():
        if isinstance(apply_updates, tensor.Tensor):
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
    slot = named_slots.get(_var_key(var), None)
    if (distribute_utils.is_distributed_variable(slot) and
        not distribute_utils.is_distributed_variable(var)):
      # Make sure var and slot are either both DistributedVariable, or both
      # per replica variables.
      slot = slot._get_on_device_or_primary()  # pylint: disable=protected-access
    return slot

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
    current_graph = ops.get_default_graph()

    def _from_current_graph(variable):
      if variable._in_graph_mode:  # pylint: disable=protected-access
        return variable.op.graph is current_graph
      else:
        # No variable.op in eager mode. We don't expect lots of eager graphs,
        # but behavior should be consistent with graph mode.
        return variable._graph_key == current_graph._graph_key  # pylint: disable=protected-access

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
    # Recommendation: Use OptimizerV2 if your optimizer uses non-slot variables.
    eager = ops.executing_eagerly_outside_functions()
    graph = None if eager else colocate_with.graph

    key = (name, graph)
    v = self._non_slot_dict.get(key, None)
    if v is None:
      self._maybe_initialize_trackable()
      distribution_strategy = distribute_lib.get_strategy()
      with distribution_strategy.extended.colocate_vars_with(colocate_with):
        if eager:
          restored_initial_value = self._preload_simple_restoration(
              name=name)
          if restored_initial_value is not None:
            initial_value = restored_initial_value
        v = variable_v1.VariableV1(
            initial_value, name=name, trainable=False,
            use_resource=resource_variable_ops.is_resource_variable(
                colocate_with))
      # Restore this variable by name if necessary, but don't add a
      # Trackable dependency. Optimizers return the current graph's
      # non-slot variables from _checkpoint_dependencies explicitly rather
      # than unconditionally adding dependencies (since there may be multiple
      # non-slot variables with the same name in different graphs, trying to
      # save all of them would result in errors).
      self._handle_deferred_dependencies(name=name, trackable=v)
      self._non_slot_dict[key] = v

    return v

  def _trackable_children(self,
                          save_type=trackable.SaveType.CHECKPOINT,
                          **kwargs):
    """From Trackable. Gather graph-specific non-slot variables to save."""
    current_graph_non_slot_variables = {}
    current_graph_key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    for (name, _), variable_object in sorted(self._non_slot_dict.items(),
                                             # Avoid comparing graphs
                                             key=lambda item: item[0][0]):
      # Skip checking for graph key for eager mode since there's only one graph.
      # This is necessary because there are cases where _trackable_children() is
      # called in a differenr thread from the main thread (e.g., async
      # checkpoint) and hence the default graph key would be different.
      if (context.executing_eagerly()
          or variable_object._graph_key == current_graph_key):  # pylint: disable=protected-access
        current_graph_non_slot_variables[name] = variable_object
    current_graph_non_slot_variables.update(
        super()._trackable_children(save_type, **kwargs)
    )
    return current_graph_non_slot_variables

  def _lookup_dependency(self, name, cached_dependencies=None):
    """From Trackable. Find a non-slot variable in the current graph."""
    unconditional = super()._lookup_dependency(name, cached_dependencies)
    if unconditional is not None:
      return unconditional
    graph = None if context.executing_eagerly() else ops.get_default_graph()
    return self._get_non_slot_variable(name, graph=graph)

  def _get_non_slot_variable(self, name, graph=None):
    non_slot = self._non_slot_dict.get((name, graph), None)
    if distribute_utils.value_container(non_slot) is not non_slot:
      # This is a mirrored non-slot.  In order to enable code like `_finish`
      # to assign to a non-slot, return the current context replica.
      return non_slot.get()
    else:
      return non_slot

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
    gradient_no_duplicate_indices = indexed_slices.IndexedSlices(
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
      new_slot_variable = slot_creator.create_slot(
          var, val, op_name, copy_xla_sharding=True)
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
          var, initializer, shape, dtype, op_name, copy_xla_sharding=True)
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
      new_slot_variable = slot_creator.create_zeros_slot(
          var, op_name, copy_xla_sharding=True)
      self._restore_slot_variable(
          slot_name=slot_name, variable=var,
          slot_variable=new_slot_variable)
      named_slots[_var_key(var)] = new_slot_variable
    return named_slots[_var_key(var)]

  # --------------
  # For implementing the Trackable interface.
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
      slot_variable_position: A `trackable._CheckpointPosition` object
        indicating the slot variable `Trackable` object to be restored.
      slot_name: The name of this `Optimizer`'s slot to restore into.
      variable: The variable object this slot is being created for.
    """
    named_slots = self._slot_dict(slot_name)
    variable_key = _var_key(variable)
    slot_variable = named_slots.get(variable_key, None)
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
      initializer = trackable.CheckpointInitialValueCallable(
          checkpoint_position=slot_variable_position)
      # CheckpointInitialValueCallable will ignore the shape and dtype
      # parameters but they must be passed.
      slot_variable = self._get_or_make_slot_with_initializer(
          var=variable,
          initializer=initializer,
          shape=variable.shape,
          dtype=variable.dtype,
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

  def _call_if_callable(self, param):
    """Call the function if param is callable."""
    return param() if callable(param) else param
