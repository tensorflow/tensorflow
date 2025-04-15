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
"""Variable class."""

import abc
import enum
import functools
import itertools
import os
from typing_extensions import Self

from tensorflow.core.framework import variable_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_getitem_override
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_should_use
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export


def default_variable_creator_v2(next_creator=None, **kwds):
  from tensorflow.python.ops import resource_variable_ops  # pylint: disable=g-import-not-at-top

  return resource_variable_ops.default_variable_creator_v2(
      next_creator=next_creator, **kwds)


def _make_getter(captured_getter, captured_previous):
  """To avoid capturing loop variables."""

  def getter(**kwargs):
    return captured_getter(captured_previous, **kwargs)

  return getter


@tf_export("VariableSynchronization")
class VariableSynchronization(enum.Enum):
  """Indicates when a distributed variable will be synced.

  * `AUTO`: Indicates that the synchronization will be determined by the current
    `DistributionStrategy` (eg. With `MirroredStrategy` this would be
    `ON_WRITE`).
  * `NONE`: Indicates that there will only be one copy of the variable, so
    there is no need to sync.
  * `ON_WRITE`: Indicates that the variable will be updated across devices
    every time it is written.
  * `ON_READ`: Indicates that the variable will be aggregated across devices
    when it is read (eg. when checkpointing or when evaluating an op that uses
    the variable).

    Example:
  >>> temp_grad=[tf.Variable([0.], trainable=False,
  ...                      synchronization=tf.VariableSynchronization.ON_READ,
  ...                      aggregation=tf.VariableAggregation.MEAN
  ...                      )]
  """
  AUTO = 0
  NONE = 1
  ON_WRITE = 2
  ON_READ = 3


# LINT.IfChange
@tf_export("VariableAggregation", v1=[])
class VariableAggregationV2(enum.Enum):
  """Indicates how a distributed variable will be aggregated.

  `tf.distribute.Strategy` distributes a model by making multiple copies
  (called "replicas") acting on different elements of the input batch in a
  data parallel model. When performing some variable-update operation,
  for example `var.assign_add(x)`, in a model, we need to resolve how to combine
  the different values for `x` computed in the different replicas.

  * `NONE`: This is the default, giving an error if you use a
    variable-update operation with multiple replicas.
  * `SUM`: Add the updates across replicas.
  * `MEAN`: Take the arithmetic mean ("average") of the updates across replicas.
  * `ONLY_FIRST_REPLICA`: This is for when every replica is performing the same
    update, but we only want to perform the update once. Used, e.g., for the
    global step counter.

  For example:

  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> with strategy.scope():
  ...   v = tf.Variable(5.0, aggregation=tf.VariableAggregation.MEAN)
  >>> @tf.function
  ... def update_fn():
  ...   return v.assign_add(1.0)
  >>> strategy.run(update_fn)
  PerReplica:{
    0: <tf.Tensor: shape=(), dtype=float32, numpy=6.0>,
    1: <tf.Tensor: shape=(), dtype=float32, numpy=6.0>
  }

  """
  NONE = 0
  SUM = 1
  MEAN = 2
  ONLY_FIRST_REPLICA = 3

  def __hash__(self):
    return hash(self.value)

  def __eq__(self, other):
    if self is other:
      return True
    elif isinstance(other, VariableAggregation):
      return int(self.value) == int(other.value)
    else:
      return False


@tf_export(v1=["VariableAggregation"])
class VariableAggregation(enum.Enum):
  NONE = 0
  SUM = 1
  MEAN = 2
  ONLY_FIRST_REPLICA = 3
  ONLY_FIRST_TOWER = 3  # DEPRECATED

  def __hash__(self):
    return hash(self.value)


# LINT.ThenChange(//tensorflow/core/framework/variable.proto)
#
# Note that we are currently relying on the integer values of the Python enums
# matching the integer values of the proto enums.

VariableAggregation.__doc__ = (
    VariableAggregationV2.__doc__ +
    "* `ONLY_FIRST_TOWER`: Deprecated alias for `ONLY_FIRST_REPLICA`.\n  ")


def validate_synchronization_aggregation_trainable(synchronization, aggregation,
                                                   trainable, name):
  """Given user-provided variable properties, sets defaults and validates."""
  if aggregation is None:
    aggregation = VariableAggregation.NONE
  else:
    if not isinstance(aggregation,
                      (VariableAggregation, VariableAggregationV2)):
      try:
        aggregation = VariableAggregationV2(aggregation)
      except ValueError:
        raise ValueError(
            "Invalid variable aggregation mode: {} for variable: {}".format(
                aggregation, name))
  if synchronization is None:
    synchronization = VariableSynchronization.AUTO
  else:
    try:
      synchronization = VariableSynchronization(synchronization)
    except ValueError:
      raise ValueError(
          "Invalid variable synchronization mode: {} for variable: {}".format(
              synchronization, name))
  if trainable is None:
    trainable = synchronization != VariableSynchronization.ON_READ
  return synchronization, aggregation, trainable


class VariableMetaclass(abc.ABCMeta):
  """Metaclass to allow construction of tf.Variable to be overridden."""

  @traceback_utils.filter_traceback
  def __call__(cls, *args, **kwargs):
    if hasattr(cls, "_variable_call") and callable(cls._variable_call):
      variable_call = cls._variable_call(*args, **kwargs)
      if variable_call is not None:
        return variable_call
    return super(VariableMetaclass, cls).__call__(*args, **kwargs)


@tf_export("Variable", v1=[])
# TODO(mdan): This should subclass core.Tensor, and not all its subclasses?
class Variable(trackable.Trackable, metaclass=VariableMetaclass):
  """See the [variable guide](https://tensorflow.org/guide/variable).

  A variable maintains shared, persistent state manipulated by a program.

  The `Variable()` constructor requires an initial value for the variable, which
  can be a `Tensor` of any type and shape. This initial value defines the type
  and shape of the variable. After construction, the type and shape of the
  variable are fixed. The value can be changed using one of the assign methods.

  >>> v = tf.Variable(1.)
  >>> v.assign(2.)
  <tf.Variable ... shape=() dtype=float32, numpy=2.0>
  >>> v.assign_add(0.5)
  <tf.Variable ... shape=() dtype=float32, numpy=2.5>

  The `shape` argument to `Variable`'s constructor allows you to construct a
  variable with a less defined shape than its `initial_value`:

  >>> v = tf.Variable(1., shape=tf.TensorShape(None))
  >>> v.assign([[1.]])
  <tf.Variable ... shape=<unknown> dtype=float32, numpy=array([[1.]], ...)>

  Just like any `Tensor`, variables created with `Variable()` can be used as
  inputs to operations. Additionally, all the operators overloaded for the
  `Tensor` class are carried over to variables.

  >>> w = tf.Variable([[1.], [2.]])
  >>> x = tf.constant([[3., 4.]])
  >>> tf.matmul(w, x)
  <tf.Tensor:... shape=(2, 2), ... numpy=
    array([[3., 4.],
           [6., 8.]], dtype=float32)>
  >>> tf.sigmoid(w + x)
  <tf.Tensor:... shape=(2, 2), ...>

  When building a machine learning model it is often convenient to distinguish
  between variables holding trainable model parameters and other variables such
  as a `step` variable used to count training steps. To make this easier, the
  variable constructor supports a `trainable=<bool>`
  parameter. `tf.GradientTape` watches trainable variables by default:

  >>> with tf.GradientTape(persistent=True) as tape:
  ...   trainable = tf.Variable(1.)
  ...   non_trainable = tf.Variable(2., trainable=False)
  ...   x1 = trainable * 2.
  ...   x2 = non_trainable * 3.
  >>> tape.gradient(x1, trainable)
  <tf.Tensor:... shape=(), dtype=float32, numpy=2.0>
  >>> assert tape.gradient(x2, non_trainable) is None  # Unwatched

  Variables are automatically tracked when assigned to attributes of types
  inheriting from `tf.Module`.

  >>> m = tf.Module()
  >>> m.v = tf.Variable([1.])
  >>> m.trainable_variables
  (<tf.Variable ... shape=(1,) ... numpy=array([1.], dtype=float32)>,)

  This tracking then allows saving variable values to
  [training checkpoints](https://www.tensorflow.org/guide/checkpoint), or to
  [SavedModels](https://www.tensorflow.org/guide/saved_model) which include
  serialized TensorFlow graphs.

  Variables are often captured and manipulated by `tf.function`s. This works the
  same way the un-decorated function would have:

  >>> v = tf.Variable(0.)
  >>> read_and_decrement = tf.function(lambda: v.assign_sub(0.1))
  >>> read_and_decrement()
  <tf.Tensor: shape=(), dtype=float32, numpy=-0.1>
  >>> read_and_decrement()
  <tf.Tensor: shape=(), dtype=float32, numpy=-0.2>

  Variables created inside a `tf.function` must be owned outside the function
  and be created only once:

  >>> class M(tf.Module):
  ...   @tf.function
  ...   def __call__(self, x):
  ...     if not hasattr(self, "v"):  # Or set self.v to None in __init__
  ...       self.v = tf.Variable(x)
  ...     return self.v * x
  >>> m = M()
  >>> m(2.)
  <tf.Tensor: shape=(), dtype=float32, numpy=4.0>
  >>> m(3.)
  <tf.Tensor: shape=(), dtype=float32, numpy=6.0>
  >>> m.v
  <tf.Variable ... shape=() dtype=float32, numpy=2.0>

  See the `tf.function` documentation for details.
  """

  @deprecated_args(
      None, "A variable's value can be manually cached by calling "
      "tf.Variable.read_value() under a tf.device scope. The caching_device "
      "argument does not work properly.", "caching_device")
  def __init__(self,
               initial_value=None,
               trainable=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               variable_def=None,
               dtype=None,
               import_scope=None,
               constraint=None,
               synchronization=VariableSynchronization.AUTO,
               aggregation=VariableAggregation.NONE,
               shape=None,
               experimental_enable_variable_lifting=True,
               ):
    """Creates a new variable with value `initial_value`.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
      trainable: If `True`, GradientTapes automatically watch uses of this
        variable. Defaults to `True`, unless `synchronization` is set to
        `ON_READ`, in which case it defaults to `False`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Note: This argument is only valid when using a v1-style
        `Session`. Optional device string describing where the Variable should
        be cached for reading. Defaults to the Variable's device. If not `None`,
        caches on another device. Typical use is to cache on the device where
        the Ops using the Variable reside, to deduplicate copying through
        `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      variable_def: `VariableDef` protocol buffer. If not `None`, recreates the
        Variable object with its contents, referencing the variable's nodes in
        the graph, which must already exist. The graph is not changed.
        `variable_def` and the other arguments are mutually exclusive.
      dtype: If set, initial_value will be converted to the given type. If
        `None`, either the datatype will be kept (if `initial_value` is a
        Tensor), or `convert_to_tensor` will decide.
      import_scope: Optional `string`. Name scope to add to the `Variable.` Only
        used when initializing from protocol buffer.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.
      experimental_enable_variable_lifting: Whether to lift the variable out if
        it's in a `tf.function`. Default is `True`. When this argument
        is `True`, variable creation will follow the behavior and
        restrictions described
        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).
        If this argument is `False`, that description doesn't apply,
        and you can freely create and use the variable in the
        `tf.function`, as if it's a "mutable `tf.Tensor`". You can't
        return the variable though.

    Raises:
      ValueError: If both `variable_def` and initial_value are specified.
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
    """
    raise NotImplementedError

  def __repr__(self):
    raise NotImplementedError

  def value(self):
    """Returns the last snapshot of this variable.

    You usually do not need to call this method as all ops that need the value
    of the variable call it automatically through a `convert_to_tensor()` call.

    Returns a `Tensor` which holds the value of the variable.  You can not
    assign a new value to this tensor as it is not a reference to the variable.

    To avoid copies, if the consumer of the returned value is on the same device
    as the variable, this actually returns the live value of the variable, not
    a copy.  Updates to the variable are seen by the consumer.  If the consumer
    is on a different device it will get a copy of the variable.

    Returns:
      A `Tensor` containing the value of the variable.
    """
    raise NotImplementedError

  def read_value(self):
    """Returns the value of this variable, read in the current context.

    Can be different from value() if it's on another device, with control
    dependencies, etc.

    Returns:
      A `Tensor` containing the value of the variable.
    """
    raise NotImplementedError

  def set_shape(self, shape):
    """Overrides the shape for this variable.

    Args:
      shape: the `TensorShape` representing the overridden shape.
    """
    raise NotImplementedError

  @property
  def trainable(self):
    raise NotImplementedError

  @property
  def synchronization(self):
    raise NotImplementedError

  @property
  def aggregation(self):
    raise NotImplementedError

  def eval(self, session=None):
    """In a session, computes and returns the value of this variable.

    This is not a graph construction method, it does not add ops to the graph.

    This convenience method requires a session where the graph
    containing this variable has been launched. If no session is
    passed, the default session is used.  See `tf.compat.v1.Session` for more
    information on launching a graph and on sessions.

    ```python
    v = tf.Variable([1, 2])
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # Usage passing the session explicitly.
        print(v.eval(sess))
        # Usage with the default session.  The 'with' block
        # above makes 'sess' the default session.
        print(v.eval())
    ```

    Args:
      session: The session to use to evaluate this variable. If none, the
        default session is used.

    Returns:
      A numpy `ndarray` with a copy of the value of this variable.
    """
    raise NotImplementedError

  @deprecated(
      None, "Use Variable.read_value. Variables in 2.X are initialized "
      "automatically both in eager and graph (inside tf.defun) contexts.")
  def initialized_value(self):
    """Returns the value of the initialized variable.

    You should use this instead of the variable itself to initialize another
    variable with a value that depends on the value of this variable.

    ```python
    # Initialize 'v' with a random tensor.
    v = tf.Variable(tf.random.truncated_normal([10, 40]))
    # Use `initialized_value` to guarantee that `v` has been
    # initialized before its value is used to initialize `w`.
    # The random values are picked only once.
    w = tf.Variable(v.initialized_value() * 2.0)
    ```

    Returns:
      A `Tensor` holding the value of this variable after its initializer
      has run.
    """
    raise NotImplementedError

  @property
  def initial_value(self):
    """Returns the Tensor used as the initial value for the variable.

    Note that this is different from `initialized_value()` which runs
    the op that initializes the variable before returning its value.
    This method returns the tensor that is used by the op that initializes
    the variable.

    Returns:
      A `Tensor`.
    """
    raise NotImplementedError

  @property
  def constraint(self):
    """Returns the constraint function associated with this variable.

    Returns:
      The constraint function that was passed to the variable constructor.
      Can be `None` if no constraint was passed.
    """
    raise NotImplementedError

  def assign(self, value, use_locking=False, name=None, read_value=True):
    """Assigns a new value to the variable.

    This is essentially a shortcut for `assign(self, value)`.

    Args:
      value: A `Tensor`. The new value for this variable.
      use_locking: If `True`, use locking during the assignment.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the new
        value of the variable; if False will return the assign op.

    Returns:
      The updated variable. If `read_value` is false, instead returns None in
      Eager mode and the assign op in graph mode.
    """
    raise NotImplementedError

  def assign_add(self, delta, use_locking=False, name=None, read_value=True):
    """Adds a value to this variable.

     This is essentially a shortcut for `assign_add(self, delta)`.

    Args:
      delta: A `Tensor`. The value to add to this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the new
        value of the variable; if False will return the assign op.

    Returns:
      The updated variable. If `read_value` is false, instead returns None in
      Eager mode and the assign op in graph mode.
    """
    raise NotImplementedError

  def assign_sub(self, delta, use_locking=False, name=None, read_value=True):
    """Subtracts a value from this variable.

    This is essentially a shortcut for `assign_sub(self, delta)`.

    Args:
      delta: A `Tensor`. The value to subtract from this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the new
        value of the variable; if False will return the assign op.

    Returns:
      The updated variable. If `read_value` is false, instead returns None in
      Eager mode and the assign op in graph mode.
    """
    raise NotImplementedError

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    """Subtracts `tf.IndexedSlices` from this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be subtracted from this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    """Adds `tf.IndexedSlices` to this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be added to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    """Updates this variable with the max of `tf.IndexedSlices` and itself.

    Args:
      sparse_delta: `tf.IndexedSlices` to use as an argument of max with this
        variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    """Updates this variable with the min of `tf.IndexedSlices` and itself.

    Args:
      sparse_delta: `tf.IndexedSlices` to use as an argument of min with this
        variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    """Multiply this variable by `tf.IndexedSlices`.

    Args:
      sparse_delta: `tf.IndexedSlices` to multiply this variable by.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    """Divide this variable by `tf.IndexedSlices`.

    Args:
      sparse_delta: `tf.IndexedSlices` to divide this variable by.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    """Assigns `tf.IndexedSlices` to this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError

  def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
    """Assigns `tf.IndexedSlices` to this variable batch-wise.

    Analogous to `batch_gather`. This assumes that this variable and the
    sparse_delta IndexedSlices have a series of leading dimensions that are the
    same for all of them, and the updates are performed on the last dimension of
    indices. In other words, the dimensions should be the following:

    `num_prefix_dims = sparse_delta.indices.ndims - 1`
    `batch_dim = num_prefix_dims + 1`
    `sparse_delta.updates.shape = sparse_delta.indices.shape + var.shape[
         batch_dim:]`

    where

    `sparse_delta.updates.shape[:num_prefix_dims]`
    `== sparse_delta.indices.shape[:num_prefix_dims]`
    `== var.shape[:num_prefix_dims]`

    And the operation performed can be expressed as:

    `var[i_1, ..., i_n,
         sparse_delta.indices[i_1, ..., i_n, j]] = sparse_delta.updates[
            i_1, ..., i_n, j]`

    When sparse_delta.indices is a 1D tensor, this operation is equivalent to
    `scatter_update`.

    To avoid this operation one can looping over the first `ndims` of the
    variable and using `scatter_update` on the subtensors that result of slicing
    the first dimension. This is a valid option for `ndims = 1`, but less
    efficient than this implementation.

    Args:
      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      The updated variable.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    """
    raise NotImplementedError

  def scatter_nd_sub(self, indices, updates, name=None):
    """Applies sparse subtraction to individual values or slices in a Variable.

    Assuming the variable has rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into self.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of self.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, self.shape[K], ..., self.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        v.scatter_nd_sub(indices, updates)
        print(v)
    ```

    After the update `v` would look like this:

        [1, -9, 3, -6, -4, 6, 7, -4]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
    raise NotImplementedError

  def scatter_nd_add(self, indices, updates, name=None):
    """Applies sparse addition to individual values or slices in a Variable.

    The Variable has rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into self.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of self.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, self.shape[K], ..., self.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        v.scatter_nd_add(indices, updates)
        print(v)
    ```

    The resulting update to v would look like this:

        [1, 13, 3, 14, 14, 6, 7, 20]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
    raise NotImplementedError

  def scatter_nd_update(self, indices, updates, name=None):
    """Applies sparse assignment to individual values or slices in a Variable.

    The Variable has rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into self.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of self.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, self.shape[K], ..., self.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        v.scatter_nd_update(indices, updates)
        print(v)
    ```

    The resulting update to v would look like this:

        [1, 11, 3, 10, 9, 6, 7, 12]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      The updated variable.
    """
    raise NotImplementedError

  def sparse_read(self, indices, name=None):
    r"""Gather slices from params axis axis according to indices.

    This function supports a subset of tf.gather, see tf.gather for details on
    usage.

    Args:
      indices: The index `Tensor`.  Must be one of the following types: `int32`,
        `int64`. Must be in range `[0, params.shape[axis])`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `params`.
    """
    raise AttributeError

  def gather_nd(self, indices, name=None):
    r"""Gather slices from `params` into a Tensor with shape specified by `indices`.

    See tf.gather_nd for details.

    Args:
      indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
        Index tensor.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `params`.
    """
    raise AttributeError

  @deprecated(None, "Prefer Dataset.range instead.")
  def count_up_to(self, limit):
    """Increments this variable until it reaches `limit`.

    When that Op is run it tries to increment the variable by `1`. If
    incrementing the variable would bring it above `limit` then the Op raises
    the exception `OutOfRangeError`.

    If no error is raised, the Op outputs the value of the variable before
    the increment.

    This is essentially a shortcut for `count_up_to(self, limit)`.

    Args:
      limit: value at which incrementing the variable raises an error.

    Returns:
      A `Tensor` that will hold the variable value before the increment. If no
      other Op modifies this variable, the values produced will all be
      distinct.
    """
    raise NotImplementedError

  @deprecated(None,
              "Prefer Variable.assign which has equivalent behavior in 2.X.")
  def load(self, value, session=None):
    """Load new value into this variable.

    Writes new value to variable's memory. Doesn't add ops to the graph.

    This convenience method requires a session where the graph
    containing this variable has been launched. If no session is
    passed, the default session is used.  See `tf.compat.v1.Session` for more
    information on launching a graph and on sessions.

    ```python
    v = tf.Variable([1, 2])
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # Usage passing the session explicitly.
        v.load([2, 3], sess)
        print(v.eval(sess)) # prints [2 3]
        # Usage with the default session.  The 'with' block
        # above makes 'sess' the default session.
        v.load([3, 4], sess)
        print(v.eval()) # prints [3 4]
    ```

    Args:
        value: New variable value
        session: The session to use to evaluate this variable. If none, the
          default session is used.

    Raises:
        ValueError: Session is not passed and no default session
    """
    if context.executing_eagerly():
      self.assign(value)
    else:
      session = session or ops.get_default_session()
      if session is None:
        raise ValueError(
            "Either session argument should be provided or default session "
            "should be established")
      session.run(self.initializer, {self.initializer.inputs[1]: value})

  # Conversion to tensor.
  @staticmethod
  def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):  # pylint: disable=invalid-name
    """Utility function for converting a Variable to a Tensor."""
    _ = name
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          f"Incompatible type conversion requested to type '{dtype.name}' for "
          f"variable of type '{v.dtype.name}' (Variable: {v}).")
    if as_ref:
      return v._ref()  # pylint: disable=protected-access
    else:
      return v.value()

  @classmethod
  def _OverloadAllOperators(cls):  # pylint: disable=invalid-name
    """Register overloads for all operators."""
    for operator in tensor_lib.Tensor.OVERLOADABLE_OPERATORS:
      cls._OverloadOperator(operator)
    # For slicing, bind getitem differently than a tensor (use _slice_helper_var
    # instead)
    # pylint: disable=protected-access
    setattr(cls, "__getitem__", tensor_getitem_override._slice_helper_var)

  @classmethod
  def _OverloadOperator(cls, operator):  # pylint: disable=invalid-name
    """Defer an operator overload to `tensor_lib.Tensor`.

    We pull the operator out of tensor_lib.Tensor dynamically to avoid ordering
    issues.

    Args:
      operator: string. The operator name.
    """
    # We can't use the overload mechanism on __eq__ & __ne__ since __eq__ is
    # called when adding a variable to sets. As a result we call a.value() which
    # causes infinite recursion when operating within a GradientTape
    # TODO(gjn): Consider removing this
    if operator == "__eq__" or operator == "__ne__":
      return

    tensor_oper = getattr(tensor_lib.Tensor, operator)

    def _run_op(a, *args, **kwargs):
      # pylint: disable=protected-access
      return tensor_oper(a.value(), *args, **kwargs)

    functools.update_wrapper(_run_op, tensor_oper)
    setattr(cls, operator, _run_op)

  def __hash__(self):
    if (
        tensor_lib.Tensor._USE_EQUALITY
        and ops.executing_eagerly_outside_functions()
    ):  # pylint: disable=protected-access
      raise TypeError(
          "Variable is unhashable. "
          f"Instead, use variable.ref() as the key. (Variable: {self})"
      )
    else:
      return id(self)

  # TODO(gjn): duplicate of math_ops.tensor_equals, consider removing
  def __eq__(self, other):
    """Compares two variables element-wise for equality."""
    if (
        tensor_lib.Tensor._USE_EQUALITY
        and ops.executing_eagerly_outside_functions()
    ):  # pylint: disable=protected-access
      return gen_math_ops.equal(self, other, incompatible_shape_error=False)
    else:
      # In legacy graph mode, tensor equality is object equality
      return self is other

  # TODO(gjn): duplicate of math_ops.tensor_not_equals, consider removing
  def __ne__(self, other):
    """Compares two variables element-wise for equality."""
    if (
        tensor_lib.Tensor._USE_EQUALITY
        and ops.executing_eagerly_outside_functions()
    ):  # pylint: disable=protected-access
      return gen_math_ops.not_equal(self, other, incompatible_shape_error=False)
    else:
      # In legacy graph mode, tensor equality is object equality
      return self is not other

  def __iter__(self):
    """When executing eagerly, iterates over the value of the variable."""
    return iter(self.read_value())

  # NOTE(mrry): This enables the Variable's overloaded "right" binary
  # operators to run when the left operand is an ndarray, because it
  # accords the Variable class higher priority than an ndarray, or a
  # numpy matrix.
  # TODO(mrry): Convert this to using numpy's __numpy_ufunc__
  # mechanism, which allows more control over how Variables interact
  # with ndarrays.
  __array_priority__ = 100

  @property
  def name(self):
    """The name of this variable."""
    raise NotImplementedError

  @property
  def _shared_name(self):
    """The shared name of the variable.

      Unlike name(), shared_name doesn't have ":0" suffix. It is user-specified
      name with name scope prefix.

    Returns:
      variable name.
    """
    return self.name[:self.name.index(":")]

  @property
  def initializer(self):
    """The initializer operation for this variable."""
    raise NotImplementedError

  @property
  def device(self):
    """The device of this variable."""
    raise NotImplementedError

  @property
  def dtype(self):
    """The `DType` of this variable."""
    raise NotImplementedError

  @property
  def op(self):
    """The `Operation` of this variable."""
    raise NotImplementedError

  @property
  def graph(self):
    """The `Graph` of this variable."""
    raise NotImplementedError

  @property
  def shape(self):
    """The `TensorShape` of this variable.

    Returns:
      A `TensorShape`.
    """
    raise NotImplementedError

  def get_shape(self) -> tensor_shape.TensorShape:
    """Alias of `Variable.shape`."""
    return self.shape

  def _gather_saveables_for_checkpoint(self):
    """For implementing `Trackable`. This object is saveable on its own."""
    return {trackable.VARIABLE_VALUE_KEY: self}

  def to_proto(self, export_scope=None):
    """Converts a `Variable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    """
    raise NotImplementedError

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    """Returns a `Variable` object created from `variable_def`."""
    raise NotImplementedError

  def _set_save_slice_info(self, save_slice_info):
    """Sets the slice info for this `Variable`.

    Args:
      save_slice_info: A `Variable.SaveSliceInfo` object.
    """
    self._save_slice_info = save_slice_info

  def _get_save_slice_info(self):
    return self._save_slice_info

  @deprecated(None, "Use ref() instead.")
  def experimental_ref(self):
    return self.ref()

  def ref(self):
    # tf.Tensor also has the same ref() API.  If you update the
    # documentation here, please update tf.Tensor.ref() as well.
    """Returns a hashable reference object to this Variable.

    The primary use case for this API is to put variables in a set/dictionary.
    We can't put variables in a set/dictionary as `variable.__hash__()` is no
    longer available starting Tensorflow 2.0.

    The following will raise an exception starting 2.0

    >>> x = tf.Variable(5)
    >>> y = tf.Variable(10)
    >>> z = tf.Variable(10)
    >>> variable_set = {x, y, z}
    Traceback (most recent call last):
      ...
    TypeError: Variable is unhashable. Instead, use tensor.ref() as the key.
    >>> variable_dict = {x: 'five', y: 'ten'}
    Traceback (most recent call last):
      ...
    TypeError: Variable is unhashable. Instead, use tensor.ref() as the key.

    Instead, we can use `variable.ref()`.

    >>> variable_set = {x.ref(), y.ref(), z.ref()}
    >>> x.ref() in variable_set
    True
    >>> variable_dict = {x.ref(): 'five', y.ref(): 'ten', z.ref(): 'ten'}
    >>> variable_dict[y.ref()]
    'ten'

    Also, the reference object provides `.deref()` function that returns the
    original Variable.

    >>> x = tf.Variable(5)
    >>> x.ref().deref()
    <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=5>
    """
    return object_identity.Reference(self)

  @classmethod
  def _variable_call(
      cls,
      initial_value=None,
      trainable=None,
      validate_shape=True,
      caching_device=None,
      name=None,
      variable_def=None,
      dtype=None,
      import_scope=None,
      constraint=None,
      synchronization=VariableSynchronization.AUTO,
      aggregation=VariableAggregation.NONE,
      shape=None,
      experimental_enable_variable_lifting=None,
      **kwargs,
    ):
    """Variable class getter. Useful to force the signature."""
    if cls is not Variable:
      return None
    previous_getter = lambda **kws: default_variable_creator_v2(None, **kws)
    for _, getter in ops.get_default_graph()._variable_creator_stack:  # pylint: disable=protected-access
      previous_getter = _make_getter(getter, previous_getter)

    # Reset `aggregation` that is explicitly set as `None` to the enum NONE.
    if aggregation is None:
      aggregation = VariableAggregation.NONE
    return previous_getter(
        initial_value=initial_value,
        trainable=trainable,
        validate_shape=validate_shape,
        caching_device=caching_device,
        name=name,
        variable_def=variable_def,
        dtype=dtype,
        import_scope=import_scope,
        constraint=constraint,
        synchronization=synchronization,
        aggregation=aggregation,
        shape=shape,
        experimental_enable_variable_lifting=experimental_enable_variable_lifting,
        **kwargs
    )

  class SaveSliceInfo:
    """Information on how to save this Variable as a slice.

    Provides internal support for saving variables as slices of a larger
    variable.  This API is not public and is subject to change.

    Available properties:

    * full_name
    * full_shape
    * var_offset
    * var_shape
    """

    def __init__(self,
                 full_name=None,
                 full_shape=None,
                 var_offset=None,
                 var_shape=None,
                 save_slice_info_def=None,
                 import_scope=None):
      """Create a `SaveSliceInfo`.

      Args:
        full_name: Name of the full variable of which this `Variable` is a
          slice.
        full_shape: Shape of the full variable, as a list of int.
        var_offset: Offset of this `Variable` into the full variable, as a list
          of int.
        var_shape: Shape of this `Variable`, as a list of int.
        save_slice_info_def: `SaveSliceInfoDef` protocol buffer. If not `None`,
          recreates the SaveSliceInfo object its contents. `save_slice_info_def`
          and other arguments are mutually exclusive.
        import_scope: Optional `string`. Name scope to add. Only used when
          initializing from protocol buffer.
      """
      if save_slice_info_def:
        assert isinstance(save_slice_info_def, variable_pb2.SaveSliceInfoDef)
        self.full_name = ops.prepend_name_scope(
            save_slice_info_def.full_name, import_scope=import_scope)
        self.full_shape = list(save_slice_info_def.full_shape)
        self.var_offset = list(save_slice_info_def.var_offset)
        self.var_shape = list(save_slice_info_def.var_shape)
      else:
        self.full_name = full_name
        self.full_shape = full_shape
        self.var_offset = var_offset
        self.var_shape = var_shape

    @property
    def spec(self):
      """Computes the spec string used for saving."""
      full_shape_str = " ".join("%d" % d for d in self.full_shape) + " "
      sl_spec = ":".join(
          "%d,%d" % (o, s) for o, s in zip(self.var_offset, self.var_shape))
      return full_shape_str + sl_spec

    @classmethod
    def from_spec(cls, spec: str) -> Self:
      """Parses a SaveSliceInfo spec string and returns a SaveSliceInfo object.

      Args:
        spec: The tensor slice spec string according to the SaveSliceInfo.spec
          property. The spec contains the space-separated shape of the full
          variable, followed by colon-separated pairs of the variable's offset
          and shape, where each pair is comma-separated. For example, consider a
          variable whose full shape is [4 3 5], offset is [0 1 3], and shape is
          [4 1 2]. This variable's SaveSliceInfo.spec would be
          "4 3 5 0,4:1,1:3,2".

      Returns:
        A SaveSliceInfo object containing the extracted information.

      Raises:
        ValueError: If the input string is not in the expected format.
      """
      if not spec:
        return cls()

      try:
        full_shape_str, slice_str = spec.rsplit(" ", 1)
      except ValueError as e:
        raise ValueError(
            "Spec string must contain space-separated full_shape info.") from e

      # Parse the full shape.
      full_shape = []
      for dim in full_shape_str.split():
        try:
          full_shape.append(int(dim))
        except ValueError as e:
          raise ValueError(
              "Spec string full_shape must be a sequence of integers. "
              f"Found '{dim}', which is not an integer.") from e

      # Parse the slice specification.
      var_offset = []
      var_shape = []
      for dim_spec in slice_str.split(":"):
        try:
          offset, shape = dim_spec.split(",")
        except ValueError as e:
          raise ValueError(
              "Spec string must contain comma-separated pairs of offsets and "
              "shapes.") from e

        try:
          var_offset.append(int(offset))
        except ValueError as e:
          raise ValueError(
              "Spec string var_offset must be an integer. "
              f"Found '{offset}', which is not an integer.") from e
        try:
          var_shape.append(int(shape))
        except ValueError as e:
          raise ValueError(
              "Spec string var_shape must be an integer. "
              f"Found '{shape}', which is not an integer.") from e

      return cls(
          full_shape=full_shape,
          var_offset=var_offset,
          var_shape=var_shape
      )

    def to_proto(self, export_scope=None):
      """Returns a SaveSliceInfoDef() proto.

      Args:
        export_scope: Optional `string`. Name scope to remove.

      Returns:
        A `SaveSliceInfoDef` protocol buffer, or None if the `Variable` is not
        in the specified name scope.
      """
      if (export_scope is None or self.full_name.startswith(export_scope)):
        save_slice_info_def = variable_pb2.SaveSliceInfoDef()
        save_slice_info_def.full_name = ops.strip_name_scope(
            self.full_name, export_scope)
        for i in self.full_shape:
          save_slice_info_def.full_shape.append(i)
        for i in self.var_offset:
          save_slice_info_def.var_offset.append(i)
        for i in self.var_shape:
          save_slice_info_def.var_shape.append(i)
        return save_slice_info_def
      else:
        return None


Variable._OverloadAllOperators()  # pylint: disable=protected-access


def _try_guard_against_uninitialized_dependencies(name, initial_value):
  """Attempt to guard against dependencies on uninitialized variables.

  Replace references to variables in `initial_value` with references to the
  variable's initialized values. The initialized values are essentially
  conditional TensorFlow graphs that return a variable's value if it is
  initialized or its `initial_value` if it hasn't been initialized. This
  replacement is done on a best effort basis:

  - If the `initial_value` graph contains cycles, we don't do any
    replacements for that graph.
  - If the variables that `initial_value` depends on are not present in the
    `GLOBAL_VARIABLES` or `LOCAL_VARIABLES` we don't replace them.

  In these cases, it is up to the caller to ensure that the `initial_value`
  graph uses initialized variables or that they guard access to variables
  using their `initialized_value` method.

  Args:
    name: Variable name.
    initial_value: `Tensor`. The initial value.

  Returns:
    A `Tensor` suitable to initialize a variable.
  Raises:
    TypeError: If `initial_value` is not a `Tensor`.
  """
  if not isinstance(initial_value, tensor_lib.Tensor):
    raise TypeError("initial_value needs to be a Tensor: %s" % initial_value)

  # Don't modify initial_value if it contains any cyclic dependencies.
  if _has_cycle(initial_value.op, state={}):
    return initial_value
  return _safe_initial_value_from_tensor(name, initial_value, op_cache={})


_UNKNOWN, _STARTED, _FINISHED = range(3)


def _has_cycle(op, state):
  """Detect cycles in the dependencies of `initial_value`."""
  op_state = state.get(op.name, _UNKNOWN)
  if op_state == _STARTED:
    return True
  elif op_state == _FINISHED:
    return False

  state[op.name] = _STARTED
  for i in itertools.chain((i.op for i in op.inputs), op.control_inputs):
    if _has_cycle(i, state):
      return True
  state[op.name] = _FINISHED
  return False


def _safe_initial_value_from_tensor(name, tensor, op_cache):
  """Replace dependencies on variables with their initialized values.

  Args:
    name: Variable name.
    tensor: A `Tensor`. The tensor to replace.
    op_cache: A dict mapping operation names to `Operation`s. Used to memoize
      the results so as to avoid creating redundant operations.

  Returns:
    A `Tensor` compatible with `tensor`. Any inputs that lead to variable
    values will be replaced with a corresponding graph that uses the
    variable's initialized values. This is done on a best-effort basis. If no
    modifications need to be made then `tensor` will be returned unchanged.
  """
  op = tensor.op
  new_op = op_cache.get(op.name)
  if new_op is None:
    new_op = _safe_initial_value_from_op(name, op, op_cache)
    op_cache[op.name] = new_op
  return new_op.outputs[tensor.value_index]


def _safe_initial_value_from_op(name, op, op_cache):
  """Replace dependencies on variables with their initialized values.

  Args:
    name: Variable name.
    op: An `Operation`. The operation to replace.
    op_cache: A dict mapping operation names to `Operation`s. Used to memoize
      the results so as to avoid creating redundant operations.

  Returns:
    An `Operation` compatible with `op`. Any inputs that lead to variable
    values will be replaced with a corresponding graph that uses the
    variable's initialized values. This is done on a best-effort basis. If no
    modifications need to be made then `op` will be returned unchanged.
  """
  op_type = op.node_def.op
  if op_type in ("IsVariableInitialized", "VarIsInitializedOp",
                 "ReadVariableOp", "If"):
    return op

  # Attempt to find the initialized_value of any variable reference / handles.
  # TODO(b/70206927): Fix handling of ResourceVariables.
  if op_type in ("Variable", "VariableV2", "VarHandleOp"):
    initialized_value = _find_initialized_value_for_variable(op)
    return op if initialized_value is None else initialized_value.op

  # Recursively build initializer expressions for inputs.
  modified = False
  new_op_inputs = []
  for op_input in op.inputs:
    new_op_input = _safe_initial_value_from_tensor(name, op_input, op_cache)
    new_op_inputs.append(new_op_input)
    modified = modified or (new_op_input != op_input)

  # If at least one input was modified, replace the op.
  if modified:
    new_op_type = op_type
    if new_op_type == "RefSwitch":
      new_op_type = "Switch"
    new_op_name = op.node_def.name + "_" + name
    new_op_name = new_op_name.replace(":", "_")
    return op.graph.create_op(
        new_op_type,
        new_op_inputs,
        op._output_types,  # pylint: disable=protected-access
        name=new_op_name,
        attrs=op.node_def.attr)

  return op


def _find_initialized_value_for_variable(variable_op):
  """Find the initialized value for a variable op.

  To do so, lookup the variable op in the variables collection.

  Args:
    variable_op: A variable `Operation`.

  Returns:
    A `Tensor` representing the initialized value for the variable or `None`
    if the initialized value could not be found.
  """
  try:
    var_names = [variable_op.node_def.name, variable_op.node_def.name + ":0"]
    for collection_name in (ops.GraphKeys.GLOBAL_VARIABLES,
                            ops.GraphKeys.LOCAL_VARIABLES):
      for var in variable_op.graph.get_collection(collection_name):
        if var.name in var_names:
          return var.initialized_value()
  except AttributeError:
    # Return None when an incomplete user-defined variable type was put in
    # the collection.
    return None
  return None


class PartitionedVariable:
  """A container for partitioned `Variable` objects.

  @compatibility(eager) `tf.PartitionedVariable` is not compatible with
  eager execution.  Use `tf.Variable` instead which is compatible
  with both eager execution and graph construction.  See [the
  TensorFlow Eager Execution
  guide](https://www.tensorflow.org/guide/eager#variables_and_optimizers)
  for details on how variables work in eager execution.
  @end_compatibility
  """

  def __init__(self, name, shape, dtype, variable_list, partitions):
    """Creates a new partitioned variable wrapper.

    Variables passed via the variable_list must contain a save_slice_info
    field.  Concatenation and iteration is in lexicographic order according
    to the var_offset property of the save_slice_info.

    Args:
      name: String. Overall name of the variables.
      shape: List of integers.  Overall shape of the variables.
      dtype: Type of the variables.
      variable_list: List of `Variable` that comprise this partitioned variable.
      partitions: List of integers.  Number of partitions for each dimension.

    Raises:
      TypeError: If `variable_list` is not a list of `Variable` objects, or
        `partitions` is not a list.
      ValueError: If `variable_list` is empty, or the `Variable` shape
        information does not match `shape`, or `partitions` has invalid values.
    """
    if not isinstance(variable_list, (list, tuple)):
      raise TypeError("variable_list is not a list or tuple: %s" %
                      variable_list)
    if not isinstance(partitions, (list, tuple)):
      raise TypeError("partitions is not a list or tuple: %s" % partitions)
    if not all(p >= 1 for p in partitions):
      raise ValueError("partition values must be positive: %s" % partitions)
    if not variable_list:
      raise ValueError("variable_list may not be empty")
    # pylint: disable=protected-access
    for v in variable_list:
      # Sort the variable_list lexicographically according to var offset value.
      if not all(v._get_save_slice_info() is not None for v in variable_list):
        raise ValueError(
            "All variables must have a save_slice_info available: %s" %
            [v.name for v in variable_list])
      if len(shape) != len(partitions):
        raise ValueError("len(shape) != len(partitions): %s vs. %s" %
                         (shape, partitions))
      if v._get_save_slice_info().full_shape != shape:
        raise ValueError("All variables' full shapes must match shape: %s; "
                         "but full shapes were: %s" %
                         (shape, str([v._get_save_slice_info().full_shape])))
    self._variable_list = sorted(
        variable_list, key=lambda v: v._get_save_slice_info().var_offset)
    # pylint: enable=protected-access

    self._name = name
    self._shape = shape
    self._dtype = dtype
    self._partitions = partitions
    self._as_tensor = None

  def __iter__(self):
    """Return an iterable for accessing the underlying partition Variables."""
    return iter(self._variable_list)

  def __len__(self):
    num_partition_axes = len(self._partition_axes())
    if num_partition_axes > 1:
      raise ValueError("Cannot get a length for %d > 1 partition axes" %
                       num_partition_axes)
    return len(self._variable_list)

  def _partition_axes(self):
    if all(p == 1 for p in self._partitions):
      return [0]
    else:
      return [i for i, p in enumerate(self._partitions) if p > 1]

  def _concat(self):
    """Returns the overall concatenated value as a `Tensor`.

    This is different from using the partitioned variable directly as a tensor
    (through tensor conversion and `as_tensor`) in that it creates a new set of
    operations that keeps the control dependencies from its scope.

    Returns:
      `Tensor` containing the concatenated value.
    """
    if len(self._variable_list) == 1:
      with ops.name_scope(None):
        return array_ops.identity(self._variable_list[0], name=self._name)

    partition_axes = self._partition_axes()

    if len(partition_axes) > 1:
      raise NotImplementedError(
          "Cannot concatenate along more than one dimension: %s.  "
          "Multi-axis partition concat is not supported" % str(partition_axes))
    partition_ix = partition_axes[0]

    with ops.name_scope(self._name + "/ConcatPartitions/"):
      concatenated = array_ops.concat(self._variable_list, partition_ix)

    with ops.name_scope(None):
      return array_ops.identity(concatenated, name=self._name)

  def as_tensor(self):
    """Returns the overall concatenated value as a `Tensor`.

    The returned tensor will not inherit the control dependencies from the scope
    where the value is used, which is similar to getting the value of
    `Variable`.

    Returns:
      `Tensor` containing the concatenated value.
    """
    with ops.control_dependencies(None):
      return self._concat()

  @staticmethod
  def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):
    # pylint: disable=invalid-name
    _ = name
    if dtype is not None and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          "Incompatible type conversion requested to type '%s' for variable "
          "of type '%s'" % (dtype.name, v.dtype.name))
    if as_ref:
      raise NotImplementedError(
          "PartitionedVariable doesn't support being used as a reference.")
    else:
      return v.as_tensor()

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._dtype

  @property
  def shape(self):
    return self.get_shape()

  @property
  def _distribute_strategy(self):
    """The `tf.distribute.Strategy` that this variable was created under."""
    # NOTE(yuefengz): Today, no partitioned variables in a distribute strategy.
    return None

  def get_shape(self) -> tensor_shape.TensorShape:
    return self._shape

  def _get_variable_list(self):
    return self._variable_list

  def _get_partitions(self):
    return self._partitions

  def _apply_assign_fn(self, assign_fn, value):
    partition_axes = self._partition_axes()
    if len(partition_axes) > 1:
      raise NotImplementedError(
          "Cannot do assign action along more than one dimension: %s.  "
          "Multi-axis partition assign action is not supported " %
          str(partition_axes))
    if isinstance(value, list):
      assert len(value) == len(self._variable_list)
      value_list = value
    elif isinstance(value, PartitionedVariable):
      value_list = list(value)
    else:
      partition_ix = partition_axes[0]
      size_splits_list = [
          tensor_shape.dimension_value(var.shape[partition_ix])
          for var in self._variable_list
      ]
      value_list = array_ops.split(value, size_splits_list, axis=partition_ix)

    op_list = [
        assign_fn(var, value_list[idx])
        for idx, var in enumerate(self._variable_list)
    ]
    return op_list

  def assign(self, value, use_locking=False, name=None, read_value=True):
    assign_fn = lambda var, r_value: var.assign(
        r_value, use_locking=use_locking, name=name, read_value=read_value)
    assign_list = self._apply_assign_fn(assign_fn, value)
    if read_value:
      return assign_list
    return [assign.op for assign in assign_list]

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    assign_fn = lambda var, r_value: var.assign_add(
        r_value, use_locking=use_locking, name=name, read_value=read_value)
    assign_list = self._apply_assign_fn(assign_fn, value)
    if read_value:
      return assign_list
    return [assign.op for assign in assign_list]

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    assign_fn = lambda var, r_value: var.assign_sub(
        r_value, use_locking=use_locking, name=name, read_value=read_value)
    assign_list = self._apply_assign_fn(assign_fn, value)
    if read_value:
      return assign_list
    return [assign.op for assign in assign_list]


@tf_export(v1=["global_variables"])
def global_variables(scope=None):
  """Returns global variables.

  Global variables are variables that are shared across machines in a
  distributed environment. The `Variable()` constructor or `get_variable()`
  automatically adds new variables to the graph collection
  `GraphKeys.GLOBAL_VARIABLES`.
  This convenience function returns the contents of that collection.

  An alternative to global variables are local variables. See
  `tf.compat.v1.local_variables`

  @compatibility(TF2)
  Not compatible with eager execution and `tf.function`. In particular, Graph
  collections are deprecated in TF2. Instead please create a
  [tf.Module](https://www.tensorflow.org/guide/intro_to_modules)
  container for all your model state, including variables.
  You can then list all the variables in your `tf.Module` through the
  `variables` attribute.
  @end_compatibility

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered to
      include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a scope
      is supplied. The choice of `re.match` means that a `scope` without special
      tokens filters by prefix.

  Returns:
    A list of `Variable` objects.
  """
  return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope)


@tf_export(v1=["all_variables"])
@deprecated("2017-03-02", "Please use tf.global_variables instead.")
def all_variables():
  """Use `tf.compat.v1.global_variables` instead."""
  return global_variables()


def _all_saveable_objects(scope=None):
  """Returns all variables and `SaveableObject`s that must be checkpointed.

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered to
      include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a scope
      is supplied. The choice of `re.match` means that a `scope` without special
      tokens filters by prefix.

  Returns:
    A list of `Variable` and `SaveableObject` to be checkpointed
  """
  # TODO(andreasst): make this function public once things are settled.
  return (ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope) +
          ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS, scope))


@tf_export(v1=["local_variables"])
def local_variables(scope=None):
  """Returns local variables.

  Local variables - per process variables, usually not saved/restored to
  checkpoint and used for temporary or intermediate values.
  For example, they can be used as counters for metrics computation or
  number of epochs this machine has read data.
  The `tf.contrib.framework.local_variable()` function automatically adds the
  new variable to `GraphKeys.LOCAL_VARIABLES`.
  This convenience function returns the contents of that collection.

  An alternative to local variables are global variables. See
  `tf.compat.v1.global_variables`

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered to
      include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a scope
      is supplied. The choice of `re.match` means that a `scope` without special
      tokens filters by prefix.

  Returns:
    A list of local `Variable` objects.
  """
  return ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES, scope)


@tf_export(v1=["model_variables"])
def model_variables(scope=None):
  """Returns all variables in the MODEL_VARIABLES collection.

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered to
      include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a scope
      is supplied. The choice of `re.match` means that a `scope` without special
      tokens filters by prefix.

  Returns:
    A list of local Variable objects.
  """
  return ops.get_collection(ops.GraphKeys.MODEL_VARIABLES, scope)


@tf_export(v1=["trainable_variables"])
def trainable_variables(scope=None):
  """Returns all variables created with `trainable=True`.

  When passed `trainable=True`, the `Variable()` constructor automatically
  adds new variables to the graph collection
  `GraphKeys.TRAINABLE_VARIABLES`. This convenience function returns the
  contents of that collection.

  @compatibility(TF2)
  Not compatible with eager execution and `tf.function`. In particular, Graph
  collections are deprecated in TF2. Instead please create a `tf.Module`
  container for all your model state, including variables.
  You can then list all the trainable variables in your `tf.Module` through the
  `trainable_variables` attribute.
  @end_compatibility

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered to
      include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a scope
      is supplied. The choice of `re.match` means that a `scope` without special
      tokens filters by prefix.

  Returns:
    A list of Variable objects.
  """
  return ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope)


@tf_export(v1=["moving_average_variables"])
def moving_average_variables(scope=None):
  """Returns all variables that maintain their moving averages.

  If an `ExponentialMovingAverage` object is created and the `apply()`
  method is called on a list of variables, these variables will
  be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection.
  This convenience function returns the contents of that collection.

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered to
      include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a scope
      is supplied. The choice of `re.match` means that a `scope` without special
      tokens filters by prefix.

  Returns:
    A list of Variable objects.
  """
  return ops.get_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, scope)


@tf_export(v1=["initializers.variables", "variables_initializer"])
def variables_initializer(var_list, name="init"):
  """Returns an Op that initializes a list of variables.

  After you launch the graph in a session, you can run the returned Op to
  initialize all the variables in `var_list`. This Op runs all the
  initializers of the variables in `var_list` in parallel.

  Calling `initialize_variables()` is equivalent to passing the list of
  initializers to `Group()`.

  If `var_list` is empty, however, the function still returns an Op that can
  be run. That Op just has no effect.

  @compatibility(TF2)
  In TF2, variables are initialized immediately when they are created. There is
  no longer a need to run variable initializers before using them.
  @end_compatibility

  Args:
    var_list: List of `Variable` objects to initialize.
    name: Optional name for the returned operation.

  Returns:
    An Op that run the initializers of all the specified variables.
  """
  if var_list and not context.executing_eagerly():
    return control_flow_ops.group(*[v.initializer for v in var_list], name=name)
  return control_flow_ops.no_op(name=name)


@tf_export(v1=["initialize_variables"])
@tf_should_use.should_use_result
@deprecated("2017-03-02", "Use `tf.variables_initializer` instead.")
def initialize_variables(var_list, name="init"):
  """See `tf.compat.v1.variables_initializer`."""
  return variables_initializer(var_list, name=name)


@tf_export(v1=["initializers.global_variables", "global_variables_initializer"])
def global_variables_initializer():
  """Returns an Op that initializes global variables.

  This is just a shortcut for `variables_initializer(global_variables())`

  @compatibility(TF2)
  In TF2, variables are initialized immediately when they are created. There is
  no longer a need to run variable initializers before using them.
  @end_compatibility

  Returns:
    An Op that initializes global variables in the graph.
  """
  if context.executing_eagerly():
    return control_flow_ops.no_op(name="global_variables_initializer")
  return variables_initializer(global_variables())


@tf_export(v1=["initialize_all_variables"])
@tf_should_use.should_use_result
@deprecated("2017-03-02", "Use `tf.global_variables_initializer` instead.")
def initialize_all_variables():
  """See `tf.compat.v1.global_variables_initializer`."""
  return global_variables_initializer()


@tf_export(v1=["initializers.local_variables", "local_variables_initializer"])
def local_variables_initializer():
  """Returns an Op that initializes all local variables.

  This is just a shortcut for `variables_initializer(local_variables())`

  @compatibility(TF2)
  In TF2, variables are initialized immediately when they are created. There is
  no longer a need to run variable initializers before using them.
  @end_compatibility

  Returns:
    An Op that initializes all local variables in the graph.
  """
  if context.executing_eagerly():
    return control_flow_ops.no_op(name="local_variables_initializer")
  return variables_initializer(local_variables())


@tf_export(v1=["initialize_local_variables"])
@tf_should_use.should_use_result
@deprecated("2017-03-02", "Use `tf.local_variables_initializer` instead.")
def initialize_local_variables():
  """See `tf.compat.v1.local_variables_initializer`."""
  return local_variables_initializer()


@tf_export(v1=["assert_variables_initialized"])
@tf_should_use.should_use_result
def assert_variables_initialized(var_list=None):
  """Returns an Op to check if variables are initialized.

  NOTE: This function is obsolete and will be removed in 6 months.  Please
  change your implementation to use `report_uninitialized_variables()`.

  When run, the returned Op will raise the exception `FailedPreconditionError`
  if any of the variables has not yet been initialized.

  Note: This function is implemented by trying to fetch the values of the
  variables. If one of the variables is not initialized a message may be
  logged by the C++ runtime. This is expected.

  Args:
    var_list: List of `Variable` objects to check. Defaults to the value of
      `global_variables().`

  Returns:
    An Op, or None if there are no variables.
  """
  if var_list is None:
    var_list = global_variables() + local_variables()
  # Backwards compatibility for old-style variables. TODO(touts): remove.
  if not var_list:
    var_list = []
    for op in ops.get_default_graph().get_operations():
      if op.type in ["Variable", "VariableV2", "AutoReloadVariable"]:
        var_list.append(op.outputs[0])
  if not var_list:
    return None
  else:
    ranks = []
    for var in var_list:
      with ops.colocate_with(var.op):
        ranks.append(array_ops.rank_internal(var, optimize=False))
    if len(ranks) == 1:
      return ranks[0]
    else:
      return array_ops_stack.stack(ranks)


@tf_export(v1=["report_uninitialized_variables"])
@tf_should_use.should_use_result
def report_uninitialized_variables(var_list=None,
                                   name="report_uninitialized_variables"):
  """Adds ops to list the names of uninitialized variables.

  When run, it returns a 1-D tensor containing the names of uninitialized
  variables if there are any, or an empty array if there are none.

  Args:
    var_list: List of `Variable` objects to check. Defaults to the value of
      `global_variables() + local_variables()`
    name: Optional name of the `Operation`.

  Returns:
    A 1-D tensor containing names of the uninitialized variables, or an empty
    1-D tensor if there are no variables or no uninitialized variables.
  """
  if var_list is None:
    var_list = global_variables() + local_variables()
    # Backwards compatibility for old-style variables. TODO(touts): remove.
    if not var_list:
      var_list = []
      for op in ops.get_default_graph().get_operations():
        if op.type in ["Variable", "VariableV2", "AutoReloadVariable"]:
          var_list.append(op.outputs[0])
  with ops.name_scope(name):
    # Run all operations on CPU
    if var_list:
      init_vars = [state_ops.is_variable_initialized(v) for v in var_list]
    local_device = os.environ.get(
        "TF_DEVICE_FOR_UNINITIALIZED_VARIABLE_REPORTING", "/cpu:0")
    with ops.device(local_device):
      if not var_list:
        # Return an empty tensor so we only need to check for returned tensor
        # size being 0 as an indication of model ready.
        return array_ops.constant([], dtype=dtypes.string)
      else:
        # Get a 1-D boolean tensor listing whether each variable is initialized.
        variables_mask = math_ops.logical_not(array_ops_stack.stack(init_vars))
        # Get a 1-D string tensor containing all the variable names.
        variable_names_tensor = array_ops.constant(
            [s.op.name for s in var_list])
        # Return a 1-D tensor containing all the names of
        # uninitialized variables.
        return array_ops.boolean_mask(variable_names_tensor, variables_mask)


tensor_conversion_registry.register_tensor_conversion_function(
    PartitionedVariable, PartitionedVariable._TensorConversionFunction)  # pylint: disable=protected-access
