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

"""Variable class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import variable_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops


class Variable(object):
  """See the [Variables How To](../../how_tos/variables/index.md) for a high
  level overview.

  A variable maintains state in the graph across calls to `run()`. You add a
  variable to the graph by constructing an instance of the class `Variable`.

  The `Variable()` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  If you want to change the shape of a variable later you have to use an
  `assign` Op with `validate_shape=False`.

  Just like any `Tensor`, variables created with `Variable()` can be used as
  inputs for other Ops in the graph. Additionally, all the operators
  overloaded for the `Tensor` class are carried over to variables, so you can
  also add nodes to the graph by just doing arithmetic on variables.

  ```python
  import tensorflow as tf

  # Create a variable.
  w = tf.Variable(<initial-value>, name=<optional-name>)

  # Use the variable in the graph like any Tensor.
  y = tf.matmul(w, ...another variable or tensor...)

  # The overloaded operators are available too.
  z = tf.sigmoid(w + y)

  # Assign a new value to the variable with `assign()` or a related method.
  w.assign(w + 1.0)
  w.assign_add(1.0)
  ```

  When you launch the graph, variables have to be explicitly initialized before
  you can run Ops that use their value. You can initialize a variable by
  running its *initializer op*, restoring the variable from a save file, or
  simply running an `assign` Op that assigns a value to the variable. In fact,
  the variable *initializer op* is just an `assign` Op that assigns the
  variable's initial value to the variable itself.

  ```python
  # Launch the graph in a session.
  with tf.Session() as sess:
      # Run the variable initializer.
      sess.run(w.initializer)
      # ...you now can run ops that use the value of 'w'...
  ```

  The most common initialization pattern is to use the convenience function
  `initialize_all_variables()` to add an Op to the graph that initializes
  all the variables. You then run that Op after launching the graph.

  ```python
  # Add an Op to initialize all variables.
  init_op = tf.initialize_all_variables()

  # Launch the graph in a session.
  with tf.Session() as sess:
      # Run the Op that initializes all variables.
      sess.run(init_op)
      # ...you can now run any Op that uses variable values...
  ```

  If you need to create a variable with an initial value dependent on another
  variable, use the other variable's `initialized_value()`. This ensures that
  variables are initialized in the right order.

  All variables are automatically collected in the graph where they are
  created. By default, the constructor adds the new variable to the graph
  collection `GraphKeys.VARIABLES`. The convenience function
  `all_variables()` returns the contents of that collection.

  When building a machine learning model it is often convenient to distinguish
  betwen variables holding the trainable model parameters and other variables
  such as a `global step` variable used to count training steps. To make this
  easier, the variable constructor supports a `trainable=<bool>` parameter. If
  `True`, the new variable is also added to the graph collection
  `GraphKeys.TRAINABLE_VARIABLES`. The convenience function
  `trainable_variables()` returns the contents of this collection. The
  various `Optimizer` classes use this collection as the default list of
  variables to optimize.


  Creating a variable.

  @@__init__
  @@initialized_value

  Changing a variable value.

  @@assign
  @@assign_add
  @@assign_sub
  @@scatter_sub
  @@count_up_to

  @@eval

  Properties.

  @@name
  @@dtype
  @@get_shape
  @@device
  @@initializer
  @@graph
  @@op
  """
  # TODO(touts): Add @@value and @@ref in the docstring above once they are
  # ready for consumption.

  def __init__(self, initial_value=None, trainable=True, collections=None,
               validate_shape=True, caching_device=None, name=None,
               variable_def=None, dtype=None):
    """Creates a new variable with value `initial_value`.

    The new variable is added to the graph collections listed in `collections`,
    which defaults to `[GraphKeys.VARIABLES]`.

    If `trainable` is `True` the variable is also added to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES`.

    This constructor creates both a `variable` Op and an `assign` Op to set the
    variable to its initial value.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.VARIABLES]`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Optional device string describing where the Variable
        should be cached for reading.  Defaults to the Variable's device.
        If not `None`, caches on another device.  Typical use is to cache
        on the device where the Ops using the Variable reside, to deduplicate
        copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      variable_def: `VariableDef` protocol buffer. If not `None`, recreates
        the Variable object with its contents. `variable_def` and the other
        arguments are mutually exclusive.
      dtype: If set, initial_value will be converted to the given type.
        If `None`, either the datatype will be kept (if `initial_value` is
        a Tensor), or `convert_to_tensor` will decide.

    Returns:
      A Variable.

    Raises:
      ValueError: If both `variable_def` and initial_value are specified.
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
    """
    if variable_def:
      # If variable_def is provided, recreates the variable from its fields.
      if initial_value:
        raise ValueError("variable_def and initial_value are mutually "
                         "exclusive.")
      self._init_from_proto(variable_def)
    else:
      # Create from initial_value.
      self._init_from_args(initial_value=initial_value,
                           trainable=trainable,
                           collections=collections,
                           validate_shape=validate_shape,
                           caching_device=caching_device,
                           name=name,
                           dtype=dtype)

  def _init_from_args(self, initial_value=None, trainable=True,
                      collections=None, validate_shape=True,
                      caching_device=None, name=None, dtype=None):
    """Creates a new variable from arguments.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.VARIABLES]`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type.
        If None, either the datatype will be kept (if initial_value is
       a Tensor) or float32 will be used (if it is a Python object convertible
       to a Tensor).

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
    """
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)
    if init_from_fn and dtype is None:
      raise ValueError(
          "dtype must also be specified when initial_value is callable.")

    if collections is None:
      collections = [ops.GraphKeys.VARIABLES]
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    with ops.control_dependencies(None):
      with ops.op_scope(
          [] if init_from_fn else [initial_value], name, "Variable") as name:

        # Get the initial value from a callable function. The real shape of the
        # variable will be set later, since under the init_from_fn case, the
        # shape won't be known until after the function is invoked.
        if init_from_fn:
          self._variable = state_ops.variable_op(
              [],
              dtype.base_dtype,
              set_shape=False,
              name=name)
          with ops.colocate_with(self._variable.op):
            with ops.name_scope("Initializer"):
              # Colocate the tensors created by the initial_value() function
              # with the variable itself.
              self._initial_value = ops.convert_to_tensor(initial_value(),
                                                          name="initial_value",
                                                          dtype=dtype)

        # Or get the initial value from a Tensor or Python object.
        else:
          self._initial_value = ops.convert_to_tensor(initial_value,
                                                      name="initial_value",
                                                      dtype=dtype)
          # In this case, the variable op can't be created until after the
          # initial_value has been converted to a Tensor with a known type.
          self._variable = state_ops.variable_op(
              [],
              self._initial_value.dtype.base_dtype,
              set_shape=False,
              name=name)

        # Manually overrides the variable's shape with the initial value's.
        if validate_shape:
          initial_value_shape = self._initial_value.get_shape()
          if not initial_value_shape.is_fully_defined():
            raise ValueError("initial_value must have a shape specified: %s"
                             % self._initial_value)
          self._variable.set_shape(initial_value_shape)
          # TODO(b/28152992): Remove the below hack modifying the node_def shape
          # directly once set_shape() handles it.
          self._variable.op.node_def.attr["shape"].shape.CopyFrom(
              initial_value_shape.as_proto())

        # Assigns initial value.
        with ops.colocate_with(self._variable.op):
          self._initializer_op = state_ops.assign(
              self._variable, self._initial_value,
              validate_shape=validate_shape).op

        # TODO(vrv): Change this class to not take caching_device, but
        # to take the op to colocate the snapshot with, so we can use
        # colocation rather than devices.
        if caching_device is not None:
          with ops.device(caching_device):
            self._snapshot = array_ops.identity(self._variable, name="read")
        else:
          with ops.colocate_with(self._variable.op):
            self._snapshot = array_ops.identity(self._variable, name="read")

    ops.add_to_collections(collections, self)
    self._caching_device = caching_device
    self._save_slice_info = None

  def _init_from_proto(self, variable_def):
    """Creates a new variable from `VariableDef` protocol buffer.

    Args:
      variable_def: `VariableDef` protocol buffer.
    """
    assert isinstance(variable_def, variable_pb2.VariableDef)
    # Create from variable_def.
    g = ops.get_default_graph()
    self._variable = g.as_graph_element(variable_def.variable_name)
    self._initializer_op = g.as_graph_element(variable_def.initializer_name)
    self._snapshot = g.as_graph_element(variable_def.snapshot_name)
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def)
    else:
      self._save_slice_info = None
    self._caching_device = None

  def _as_graph_element(self):
    """Conversion function for Graph.as_graph_element()."""
    return self._variable

  def _AsTensor(self):
    """Converts this variable to a Tensor.

    See [`value()`](#Variable.value).

    Returns:
      A `Tensor` containing the value of the variable.
    """
    return self._snapshot

  def __iter__(self):
    """Dummy method to prevent iteration. Do not call.

    NOTE(mrry): If we register __getitem__ as an overloaded operator,
    Python will valiantly attempt to iterate over the variable's Tensor from 0
    to infinity.  Declaring this method prevents this unintended behavior.

    Raises:
      TypeError: when invoked.
    """
    raise TypeError("'Variable' object is not iterable.")

  def value(self):
    """Returns the last snapshot of this variable.

    You usually do not need to call this method as all ops that need the value
    of the variable call it automatically through a `convert_to_tensor()` call.

    Returns a `Tensor` which holds the value of the variable.  You can not
    assign a new value to this tensor as it is not a reference to the variable.
    See [`ref()`](#Variable.ref) if you want to get a reference to the
    variable.

    To avoid copies, if the consumer of the returned value is on the same device
    as the variable, this actually returns the live value of the variable, not
    a copy.  Updates to the variable are seen by the consumer.  If the consumer
    is on a different device it will get a copy of the variable.

    Returns:
      A `Tensor` containing the value of the variable.
    """
    return self._snapshot

  def ref(self):
    """Returns a reference to this variable.

    You usually do not need to call this method as all ops that need a reference
    to the variable call it automatically.

    Returns is a `Tensor` which holds a reference to the variable.  You can
    assign a new value to the variable by passing the tensor to an assign op.
    See [`value()`](#Variable.value) if you want to get the value of the
    variable.

    Returns:
      A `Tensor` that is a reference to the variable.
    """
    return self._variable

  def eval(self, session=None):
    """In a session, computes and returns the value of this variable.

    This is not a graph construction method, it does not add ops to the graph.

    This convenience method requires a session where the graph containing this
    variable has been launched. If no session is passed, the default session is
    used.  See the [Session class](../../api_docs/python/client.md#Session) for
    more information on launching a graph and on sessions.

    ```python
    v = tf.Variable([1, 2])
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        # Usage passing the session explicitly.
        print(v.eval(sess))
        # Usage with the default session.  The 'with' block
        # above makes 'sess' the default session.
        print(v.eval())
    ```

    Args:
      session: The session to use to evaluate this variable. If
        none, the default session is used.

    Returns:
      A numpy `ndarray` with a copy of the value of this variable.
    """
    return self._variable.eval(session=session)

  def initialized_value(self):
    """Returns the value of the initialized variable.

    You should use this instead of the variable itself to initialize another
    variable with a value that depends on the value of this variable.

    ```python
    # Initialize 'v' with a random tensor.
    v = tf.Variable(tf.truncated_normal([10, 40]))
    # Use `initialized_value` to guarantee that `v` has been
    # initialized before its value is used to initialize `w`.
    # The random values are picked only once.
    w = tf.Variable(v.initialized_value() * 2.0)
    ```

    Returns:
      A `Tensor` holding the value of this variable after its initializer
      has run.
    """
    with ops.control_dependencies(None):
      with ops.control_dependencies([self._initializer_op]):
        # TODO(vrv): Change this class to not take caching_device, but
        # to take the op to colocate the snapshot with, so we can use
        # colocation rather than devices.
        if self._caching_device is not None:
          with ops.device(self._caching_device):
            return array_ops.identity(self._variable)
        else:
          with ops.colocate_with(self._variable.op):
            return array_ops.identity(self._variable)

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
    return self._initial_value

  def assign(self, value, use_locking=False):
    """Assigns a new value to the variable.

    This is essentially a shortcut for `assign(self, value)`.

    Args:
      value: A `Tensor`. The new value for this variable.
      use_locking: If `True`, use locking during the assignment.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the assignment has completed.
    """
    return state_ops.assign(self._variable, value, use_locking=use_locking)

  def assign_add(self, delta, use_locking=False):
    """Adds a value to this variable.

     This is essentially a shortcut for `assign_add(self, delta)`.

    Args:
      delta: A `Tensor`. The value to add to this variable.
      use_locking: If `True`, use locking during the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the addition has completed.
    """
    return state_ops.assign_add(self._variable, delta, use_locking=use_locking)

  def assign_sub(self, delta, use_locking=False):
    """Subtracts a value from this variable.

    This is essentially a shortcut for `assign_sub(self, delta)`.

    Args:
      delta: A `Tensor`. The value to subtract from this variable.
      use_locking: If `True`, use locking during the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the subtraction has completed.
    """
    return state_ops.assign_sub(self._variable, delta, use_locking=use_locking)

  def scatter_sub(self, sparse_delta, use_locking=False):
    """Subtracts `IndexedSlices` from this variable.

    This is essentially a shortcut for `scatter_sub(self, sparse_delta.indices,
    sparse_delta.values)`.

    Args:
      sparse_delta: `IndexedSlices` to be subtracted from this variable.
      use_locking: If `True`, use locking during the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.

    Raises:
      ValueError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return state_ops.scatter_sub(self._variable,
                                 sparse_delta.indices,
                                 sparse_delta.values,
                                 use_locking=use_locking)

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
    return state_ops.count_up_to(self._variable, limit=limit)

  # Conversion to tensor.
  @staticmethod
  def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):
    """Utility function for converting a Variable to a Tensor."""
    _ = name
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          "Incompatible type conversion requested to type '%s' for variable "
          "of type '%s'" % (dtype.name, v.dtype.name))
    if as_ref:
      return v.ref()
    else:
      return v.value()

  # Operator overloading.
  #
  # To carry over all overloaded operators from ops.Tensor to Variable, we
  # register the _RunOp() static method as the implementation of all operators.
  # That function dynamically discovers the overloaded operator in ops.Tensor
  # and invokes it after converting the Variable to a tensor.
  @staticmethod
  def _OverloadAllOperators():
    """Register overloads for all operators."""
    for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
      Variable._OverloadOperator(operator)

  @staticmethod
  def _OverloadOperator(operator):
    """Register _RunOp as the implementation of 'operator'.

    Args:
      operator: string. The operator name.
    """
    if operator in ["__invert__", "__neg__", "__abs__"]:
      setattr(Variable, operator, lambda a: Variable._RunOp(operator, a, None))
    else:
      setattr(Variable, operator, lambda a, b: Variable._RunOp(operator, a, b))

  @staticmethod
  def _RunOp(operator, a, b):
    """Run the operator 'op' for 'a'.

    Args:
      operator: string. The operator name.
      a: A Variable.
      b: Second argument to the operator. None if unary.
    Returns:
      The result of the operator.
    """
    # pylint: disable=protected-access
    if b is not None:
      return getattr(ops.Tensor, operator)(a._AsTensor(), b)
    else:
      return getattr(ops.Tensor, operator)(a._AsTensor())
    # pylint: enable=protected-access

  @property
  def name(self):
    """The name of this variable."""
    return self._variable.name

  @property
  def initializer(self):
    """The initializer operation for this variable."""
    return self._initializer_op

  @property
  def device(self):
    """The device of this variable."""
    return self._variable.device

  @property
  def dtype(self):
    """The `DType` of this variable."""
    return self._variable.dtype

  @property
  def op(self):
    """The `Operation` of this variable."""
    return self._variable.op

  @property
  def graph(self):
    """The `Graph` of this variable."""
    return self._variable.graph

  def get_shape(self):
    """The `TensorShape` of this variable.

    Returns:
      A `TensorShape`.
    """
    return self._variable.get_shape()

  def to_proto(self):
    """Converts a `Variable` to a `VariableDef` protocol buffer.

    Returns:
      A `VariableDef` protocol buffer.
    """
    var_def = variable_pb2.VariableDef()
    var_def.variable_name = self._variable.name
    var_def.initializer_name = self.initializer.name
    var_def.snapshot_name = self._snapshot.name
    if self._save_slice_info:
      var_def.save_slice_info_def.MergeFrom(self._save_slice_info.to_proto())
    return var_def

  @staticmethod
  def from_proto(variable_def):
    """Returns a `Variable` object created from `variable_def`."""
    return Variable(variable_def=variable_def)

  # Experimental support for saving variables as slices of a larger variable.
  class SaveSliceInfo(object):
    """Information on how to save this Variable as a slice."""

    def __init__(self, full_name=None, full_shape=None, var_offset=None,
                 var_shape=None, save_slice_info_def=None):
      """Create a `SaveSliceInfo`.

      Args:
        full_name: Name of the full variable of which this `Variable` is a
            slice.
        full_shape: Shape of the full variable, as a list of int.
        var_offset: Offset of this `Variable` into the full variable, as a
            list of int.
        var_shape: Shape of this `Variable`, as a list of int.
        save_slice_info_def: `SaveSliceInfoDef` protocol buffer. If not `None`,
          recreates the SaveSliceInfo object its contents.
          `save_slice_info_def` and other arguments are mutually
          exclusive.
      """
      if save_slice_info_def:
        assert isinstance(save_slice_info_def, variable_pb2.SaveSliceInfoDef)
        self.full_name = save_slice_info_def.full_name
        self.full_shape = [i for i in save_slice_info_def.full_shape]
        self.var_offset = [i for i in save_slice_info_def.var_offset]
        self.var_shape = [i for i in save_slice_info_def.var_shape]
      else:
        self.full_name = full_name
        self.full_shape = full_shape
        self.var_offset = var_offset
        self.var_shape = var_shape

    @property
    def spec(self):
      """Computes the spec string used for saving."""
      full_shape_str = " ".join(["%d" % d for d in self.full_shape]) + " "
      sl_spec = ":".join([
          "%d,%d" % (o, s) for o, s in zip(self.var_offset, self.var_shape)])
      return full_shape_str + sl_spec

    def to_proto(self):
      """Returns a SaveSliceInfoDef() proto."""
      save_slice_info_def = variable_pb2.SaveSliceInfoDef()
      save_slice_info_def.full_name = self.full_name
      for i in self.full_shape:
        save_slice_info_def.full_shape.append(i)
      for i in self.var_offset:
        save_slice_info_def.var_offset.append(i)
      for i in self.var_shape:
        save_slice_info_def.var_shape.append(i)
      return save_slice_info_def

  def _set_save_slice_info(self, save_slice_info):
    """Sets the slice info for this `Variable`.

    Args:
      save_slice_info: A `Variable.SaveSliceInfo` object.
    """
    self._save_slice_info = save_slice_info


def all_variables():
  """Returns all variables that must be saved/restored.

  The `Variable()` constructor automatically adds new variables to the graph
  collection `GraphKeys.VARIABLES`. This convenience function returns the
  contents of that collection.

  Returns:
    A list of `Variable` objects.
  """
  return ops.get_collection(ops.GraphKeys.VARIABLES)


def trainable_variables():
  """Returns all variables created with `trainable=True`.

  When passed `trainable=True`, the `Variable()` constructor automatically
  adds new variables to the graph collection
  `GraphKeys.TRAINABLE_VARIABLES`. This convenience function returns the
  contents of that collection.

  Returns:
    A list of Variable objects.
  """
  return ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)

def local_variables():
  """Returns all variables created with collection=[LOCAL_VARIABLES].

  Returns:
    A list of local Variable objects.
  """
  return ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES)

def moving_average_variables():
  """Returns all variables that maintain their moving averages.

  If an `ExponentialMovingAverage` object is created and the `apply()`
  method is called on a list of variables, these variables will
  be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection.
  This convenience function returns the contents of that collection.

  Returns:
    A list of Variable objects.
  """
  return ops.get_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES)


def initialize_variables(var_list, name="init"):
  """Returns an Op that initializes a list of variables.

  After you launch the graph in a session, you can run the returned Op to
  initialize all the variables in `var_list`. This Op runs all the
  initializers of the variables in `var_list` in parallel.

  Calling `initialize_variables()` is equivalent to passing the list of
  initializers to `Group()`.

  If `var_list` is empty, however, the function still returns an Op that can
  be run. That Op just has no effect.

  Args:
    var_list: List of `Variable` objects to initialize.
    name: Optional name for the returned operation.

  Returns:
    An Op that run the initializers of all the specified variables.
  """
  if var_list:
    return control_flow_ops.group(
        *[v.initializer for v in var_list], name=name)
  return control_flow_ops.no_op(name=name)


def initialize_all_variables():
  """Returns an Op that initializes all variables.

  This is just a shortcut for `initialize_variables(all_variables())`

  Returns:
    An Op that initializes all variables in the graph.
  """
  return initialize_variables(all_variables())


def initialize_local_variables():
  """Returns an Op that initializes all local variables.

  This is just a shortcut for `initialize_variables(local_variables())`

  Returns:
    An Op that initializes all local variables in the graph.
  """
  return initialize_variables(local_variables())


def is_variable_initialized(variable):
  """Returns an Op to check if a variable has been initialized.

  Args:
    variable: A `Variable`.

  Returns:
    An operation to check whether a variable has been initialized.
  """
  return state_ops.is_variable_initialized(variable)


def assert_variables_initialized(var_list=None):
  """Returns an Op to check if variables are initialized.

  When run, the returned Op will raise the exception `FailedPreconditionError`
  if any of the variables has not yet been initialized.

  Note: This function is implemented by trying to fetch the values of the
  variables. If one of the variables is not initialized a message may be
  logged by the C++ runtime. This is expected.

  Args:
    var_list: List of `Variable` objects to check. Defaults to the
      value of `all_variables().`

  Returns:
    An Op, or None if there are no variables.
  """
  if var_list is None:
    var_list = all_variables() + local_variables()
  # Backwards compatibility for old-style variables. TODO(touts): remove.
  if not var_list:
    var_list = []
    for op in ops.get_default_graph().get_operations():
      if op.type in ["Variable", "AutoReloadVariable"]:
        var_list.append(op.outputs[0])
  if not var_list:
    return None
  else:
    ranks = []
    for var in var_list:
      with ops.colocate_with(var.op):
        ranks.append(array_ops.rank(var))
    if len(ranks) == 1:
      return ranks[0]
    else:
      return array_ops.pack(ranks)


# pylint: disable=protected-access
ops.register_tensor_conversion_function(Variable,
                                        Variable._TensorConversionFunction)
Variable._OverloadAllOperators()
# pylint: enable=protected-access

ops.register_proto_function(ops.GraphKeys.VARIABLES,
                            proto_type=variable_pb2.VariableDef,
                            to_proto=Variable.to_proto,
                            from_proto=Variable.from_proto)
ops.register_proto_function(ops.GraphKeys.TRAINABLE_VARIABLES,
                            proto_type=variable_pb2.VariableDef,
                            to_proto=Variable.to_proto,
                            from_proto=Variable.from_proto)
ops.register_proto_function(ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
                            proto_type=variable_pb2.VariableDef,
                            to_proto=Variable.to_proto,
                            from_proto=Variable.from_proto)
