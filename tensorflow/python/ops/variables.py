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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.util import compat
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.deprecation import deprecated


class Variable(object):
  """See the @{$variables$Variables How To} for a high level overview.

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
  `global_variables_initializer()` to add an Op to the graph that initializes
  all the variables. You then run that Op after launching the graph.

  ```python
  # Add an Op to initialize global variables.
  init_op = tf.global_variables_initializer()

  # Launch the graph in a session.
  with tf.Session() as sess:
      # Run the Op that initializes global variables.
      sess.run(init_op)
      # ...you can now run any Op that uses variable values...
  ```

  If you need to create a variable with an initial value dependent on another
  variable, use the other variable's `initialized_value()`. This ensures that
  variables are initialized in the right order.

  All variables are automatically collected in the graph where they are
  created. By default, the constructor adds the new variable to the graph
  collection `GraphKeys.GLOBAL_VARIABLES`. The convenience function
  `global_variables()` returns the contents of that collection.

  When building a machine learning model it is often convenient to distinguish
  between variables holding the trainable model parameters and other variables
  such as a `global step` variable used to count training steps. To make this
  easier, the variable constructor supports a `trainable=<bool>` parameter. If
  `True`, the new variable is also added to the graph collection
  `GraphKeys.TRAINABLE_VARIABLES`. The convenience function
  `trainable_variables()` returns the contents of this collection. The
  various `Optimizer` classes use this collection as the default list of
  variables to optimize.

  @compatibility(eager)
  `tf.Variable` is not compatible with eager execution.  Use
  `tfe.Variable` instead which is compatible with both eager execution
  and graph construction.  See [the TensorFlow Eager Execution
  guide](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/g3doc/guide.md#variables-and-optimizers)
  for details on how variables work in eager execution.
  @end_compatibility
  """

  def __init__(self,
               initial_value=None,
               trainable=True,
               collections=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               variable_def=None,
               dtype=None,
               expected_shape=None,
               import_scope=None,
               constraint=None):
    """Creates a new variable with value `initial_value`.

    The new variable is added to the graph collections listed in `collections`,
    which defaults to `[GraphKeys.GLOBAL_VARIABLES]`.

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
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
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
        the Variable object with its contents, referencing the variable's nodes
        in the graph, which must already exist. The graph is not changed.
        `variable_def` and the other arguments are mutually exclusive.
      dtype: If set, initial_value will be converted to the given type.
        If `None`, either the datatype will be kept (if `initial_value` is
        a Tensor), or `convert_to_tensor` will decide.
      expected_shape: A TensorShape. If set, initial_value is expected
        to have this shape.
      import_scope: Optional `string`. Name scope to add to the
        `Variable.` Only used when initializing from protocol buffer.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value
        (which must have the same shape). Constraints are not safe to
        use when doing asynchronous distributed training.

    Raises:
      ValueError: If both `variable_def` and initial_value are specified.
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
      RuntimeError: If eager execution is enabled.

    @compatibility(eager)
    `tf.Variable` is not compatible with eager execution.  Use
    `tfe.Variable` instead which is compatible with both eager execution
    and graph construction.  See [the TensorFlow Eager Execution
    guide](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/g3doc/guide.md#variables-and-optimizers)
    for details on how variables work in eager execution.
    @end_compatibility
    """
    if not context.in_graph_mode():
      raise RuntimeError("tf.Variable not supported in Eager mode. "
                         "Please use tfe.Variable instead")
    if variable_def:
      # If variable_def is provided, recreates the variable from its fields.
      if initial_value:
        raise ValueError("variable_def and initial_value are mutually "
                         "exclusive.")
      self._init_from_proto(variable_def, import_scope=import_scope)
    else:
      # Create from initial_value.
      self._init_from_args(
          initial_value=initial_value,
          trainable=trainable,
          collections=collections,
          validate_shape=validate_shape,
          caching_device=caching_device,
          name=name,
          dtype=dtype,
          expected_shape=expected_shape,
          constraint=constraint)

  def __repr__(self):
    if context.in_eager_mode():
      return "<tf.Variable '%s' shape=%s dtype=%s, numpy=%s>" % (
          self.name, self.get_shape(), self.dtype.name,
          ops.numpy_text(self.read_value(), is_repr=True))
    else:
      return "<tf.Variable '%s' shape=%s dtype=%s>" % (
          self.name, self.get_shape(), self.dtype.name)

  def _init_from_args(self,
                      initial_value=None,
                      trainable=True,
                      collections=None,
                      validate_shape=True,
                      caching_device=None,
                      name=None,
                      dtype=None,
                      expected_shape=None,
                      constraint=None):
    """Creates a new variable from arguments.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound
         to a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
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
      expected_shape: Deprecated. Ignored.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value
        (which must have the same shape). Constraints are not safe to
        use when doing asynchronous distributed training.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
    """
    _ = expected_shape
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
          "collections argument to Variable constructor must be a list, tuple, "
          "or set. Got %s of type %s" % (collections, type(collections)))
    if constraint is not None and not callable(constraint):
      raise ValueError("The `constraint` argument must be a callable.")

    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    with ops.control_dependencies(None):
      with ops.name_scope(name, "Variable", [] if init_from_fn else
                          [initial_value]) as name:

        if init_from_fn:
          # Use attr_scope and device(None) to simulate the behavior of
          # colocate_with when the variable we want to colocate with doesn't
          # yet exist.
          true_name = ops._name_from_scope_name(name)  # pylint: disable=protected-access
          attr = attr_value_pb2.AttrValue(
              list=attr_value_pb2.AttrValue.ListValue(
                  s=[compat.as_bytes("loc:@%s" % true_name)]))
          # pylint: disable=protected-access
          with ops.get_default_graph()._attr_scope({"_class": attr}):
            with ops.name_scope("Initializer"), ops.device(None):
              self._initial_value = ops.convert_to_tensor(
                  initial_value(), name="initial_value", dtype=dtype)
              shape = (self._initial_value.get_shape()
                       if validate_shape else tensor_shape.unknown_shape())
            self._variable = state_ops.variable_op_v2(
                shape,
                self._initial_value.dtype.base_dtype,
                name=name)
          # pylint: enable=protected-access

        # Or get the initial value from a Tensor or Python object.
        else:
          self._initial_value = ops.convert_to_tensor(
              initial_value, name="initial_value", dtype=dtype)
          # pylint: disable=protected-access
          if self._initial_value.op._get_control_flow_context() is not None:
            raise ValueError(
                "Initializer for variable %s is from inside a control-flow "
                "construct, such as a loop or conditional. When creating a "
                "variable inside a loop or conditional, use a lambda as the "
                "initializer." % name)
          # pylint: enable=protected-access
          shape = (self._initial_value.get_shape()
                   if validate_shape else tensor_shape.unknown_shape())
          # In this case, the variable op can't be created until after the
          # initial_value has been converted to a Tensor with a known type.
          self._variable = state_ops.variable_op_v2(
              shape,
              self._initial_value.dtype.base_dtype,
              name=name)

        # Manually overrides the variable's shape with the initial value's.
        if validate_shape:
          initial_value_shape = self._initial_value.get_shape()
          if not initial_value_shape.is_fully_defined():
            raise ValueError("initial_value must have a shape specified: %s" %
                             self._initial_value)

        # If 'initial_value' makes use of other variables, make sure we don't
        # have an issue if these other variables aren't initialized first by
        # using their initialized_value() method.
        self._initializer_op = state_ops.assign(
            self._variable,
            self._build_initializer_expr(self._initial_value),
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
    self._constraint = constraint

  def _init_from_proto(self, variable_def, import_scope=None):
    """Recreates the Variable object from a `VariableDef` protocol buffer.

    Args:
      variable_def: `VariableDef` protocol buffer, describing a variable
          whose nodes already exists in the graph.
      import_scope: Optional `string`. Name scope to add.
    """
    assert isinstance(variable_def, variable_pb2.VariableDef)
    # Create from variable_def.
    g = ops.get_default_graph()
    self._variable = g.as_graph_element(
        ops.prepend_name_scope(variable_def.variable_name,
                               import_scope=import_scope))
    self._initializer_op = g.as_graph_element(
        ops.prepend_name_scope(variable_def.initializer_name,
                               import_scope=import_scope))
    # Tests whether initial_value_name exists first for backwards compatibility.
    if (hasattr(variable_def, "initial_value_name") and
        variable_def.initial_value_name):
      self._initial_value = g.as_graph_element(
          ops.prepend_name_scope(variable_def.initial_value_name,
                                 import_scope=import_scope))
    else:
      self._initial_value = None
    self._snapshot = g.as_graph_element(
        ops.prepend_name_scope(variable_def.snapshot_name,
                               import_scope=import_scope))
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def,
          import_scope=import_scope)
    else:
      self._save_slice_info = None
    self._caching_device = None
    self._constraint = None

  def _as_graph_element(self):
    """Conversion function for Graph.as_graph_element()."""
    return self._variable

  def _AsTensor(self):  # pylint: disable=invalid-name
    """Converts this variable to a Tensor.

    See @{tf.Variable.value}.

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

    To avoid copies, if the consumer of the returned value is on the same device
    as the variable, this actually returns the live value of the variable, not
    a copy.  Updates to the variable are seen by the consumer.  If the consumer
    is on a different device it will get a copy of the variable.

    Returns:
      A `Tensor` containing the value of the variable.
    """
    return self._snapshot

  def read_value(self):
    """Returns the value of this variable, read in the current context.

    Can be different from value() if it's on another device, with control
    dependencies, etc.

    Returns:
      A `Tensor` containing the value of the variable.
    """
    return array_ops.identity(self._variable, name="read")

  def _ref(self):
    """Returns a reference to this variable.

    You usually do not need to call this method as all ops that need a reference
    to the variable call it automatically.

    Returns is a `Tensor` which holds a reference to the variable.  You can
    assign a new value to the variable by passing the tensor to an assign op.
    See @{tf.Variable.value} if you want to get the value of the
    variable.

    Returns:
      A `Tensor` that is a reference to the variable.
    """
    return self._variable

  def set_shape(self, shape):
    """Overrides the shape for this variable.

    Args:
      shape: the `TensorShape` representing the overridden shape.
    """
    self._ref().set_shape(shape)
    self.value().set_shape(shape)

  def eval(self, session=None):
    """In a session, computes and returns the value of this variable.

    This is not a graph construction method, it does not add ops to the graph.

    This convenience method requires a session where the graph
    containing this variable has been launched. If no session is
    passed, the default session is used.  See @{tf.Session} for more
    information on launching a graph and on sessions.

    ```python
    v = tf.Variable([1, 2])
    init = tf.global_variables_initializer()

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
      return control_flow_ops.cond(is_variable_initialized(self),
                                   self.read_value,
                                   lambda: self.initial_value)

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

  @property
  def constraint(self):
    """Returns the constraint function associated with this variable.

    Returns:
      The constraint function that was passed to the variable constructor.
      Can be `None` if no constraint was passed.
    """
    return self._constraint

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
    return state_ops.scatter_sub(
        self._variable,
        sparse_delta.indices,
        sparse_delta.values,
        use_locking=use_locking)

  def _strided_slice_assign(self,
                            begin,
                            end,
                            strides,
                            value,
                            name,
                            begin_mask,
                            end_mask,
                            ellipsis_mask,
                            new_axis_mask,
                            shrink_axis_mask):
    return gen_array_ops.strided_slice_assign(ref=self._ref(),
                                              begin=begin,
                                              end=end,
                                              strides=strides,
                                              value=value,
                                              name=name,
                                              begin_mask=begin_mask,
                                              end_mask=end_mask,
                                              ellipsis_mask=ellipsis_mask,
                                              new_axis_mask=new_axis_mask,
                                              shrink_axis_mask=shrink_axis_mask)

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

  def load(self, value, session=None):
    """Load new value into this variable.

    Writes new value to variable's memory. Doesn't add ops to the graph.

    This convenience method requires a session where the graph
    containing this variable has been launched. If no session is
    passed, the default session is used.  See @{tf.Session} for more
    information on launching a graph and on sessions.

    ```python
    v = tf.Variable([1, 2])
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
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
        session: The session to use to evaluate this variable. If
          none, the default session is used.

    Raises:
        ValueError: Session is not passed and no default session
    """
    if context.in_graph_mode():
      session = session or ops.get_default_session()
      if session is None:
        raise ValueError(
            "Either session argument should be provided or default session "
            "should be established")
      session.run(self._initializer_op, {self._initializer_op.inputs[1]: value})
    else:
      self.assign(value)

  # Conversion to tensor.
  @staticmethod
  def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):  # pylint: disable=invalid-name
    """Utility function for converting a Variable to a Tensor."""
    _ = name
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          "Incompatible type conversion requested to type '%s' for variable "
          "of type '%s'" % (dtype.name, v.dtype.name))
    if as_ref:
      return v._ref()  # pylint: disable=protected-access
    else:
      return v.value()

  @staticmethod
  def _OverloadAllOperators():  # pylint: disable=invalid-name
    """Register overloads for all operators."""
    for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
      Variable._OverloadOperator(operator)
    # For slicing, bind getitem differently than a tensor (use SliceHelperVar
    # instead)
    # pylint: disable=protected-access
    setattr(Variable, "__getitem__", array_ops._SliceHelperVar)

  @staticmethod
  def _OverloadOperator(operator):  # pylint: disable=invalid-name
    """Defer an operator overload to `ops.Tensor`.

    We pull the operator out of ops.Tensor dynamically to avoid ordering issues.

    Args:
      operator: string. The operator name.
    """

    def _run_op(a, *args):
      # pylint: disable=protected-access
      return getattr(ops.Tensor, operator)(a._AsTensor(), *args)
    # Propagate __doc__ to wrapper
    try:
      _run_op.__doc__ = getattr(ops.Tensor, operator).__doc__
    except AttributeError:
      pass

    setattr(Variable, operator, _run_op)

  def _build_initializer_expr(self, initial_value):
    """Build an expression suitable to initialize a variable.

    Replace references to variables in initial_value with references to the
    variable initial values instead.

    Args:
      initial_value: original expression
    Returns:
      A tensorflow expression suitable to initialize a variable.
    """
    if isinstance(initial_value, Variable):
      return initial_value.initialized_value()
    elif isinstance(initial_value, ops.Tensor):
      new_op = self._build_initializer_expr(initial_value.op)
      if new_op != initial_value.op:
        if isinstance(new_op, ops.Tensor):
          return new_op
        else:
          return ops.Tensor(new_op, initial_value.value_index,
                            initial_value.dtype)
      else:
        return initial_value
    elif isinstance(initial_value, ops.Operation):
      if initial_value.node_def.op in [
          "IsVariableInitialized", "VarIsInitializedOp", "ReadVariableOp"
      ]:
        return initial_value
      if initial_value.node_def.op in ["Variable", "VariableV2", "VarHandleOp"]:
        return self._find_initialized_value_for_variable(initial_value)
      modified = False
      new_inputs = []
      for tensor in initial_value.inputs:
        new_tensor = self._build_initializer_expr(tensor)
        new_inputs.append(new_tensor)
        if new_tensor != tensor:
          modified = True

      if modified:
        new_name = initial_value.node_def.name + "_" + self.name
        new_name = new_name.replace(":", "_")
        new_op = initial_value.node_def.op
        new_op = new_op.replace("RefSwitch", "Switch")
        new_value = self.graph.create_op(
            new_op,
            new_inputs,
            # pylint: disable=protected-access
            initial_value._output_types,
            # pylint: enable=protected-access
            name=new_name,
            attrs=initial_value.node_def.attr)
        return new_value
      else:
        return initial_value
    else:
      return initial_value

  def _find_initialized_value_for_variable(self, variable_op):
    """Find the initial value for a variable op.

    To do so, lookup the variable op in the variables collection.

    Args:
      variable_op: a TensorFlow variable Operation
    Returns:
      The initial value for the variable.
    """
    try:
      var_names = [variable_op.node_def.name, variable_op.node_def.name + ":0"]
      global_vars = self.graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      for var in global_vars:
        if var.name in var_names:
          return var.initialized_value()
      local_vars = self.graph.get_collection(ops.GraphKeys.LOCAL_VARIABLES)
      for var in local_vars:
        if var.name == var_names:
          return var.initialized_value()
    except AttributeError:
      # Return the variable itself when an incomplete user defined variable type
      # was put in the collection.
      return variable_op
    return variable_op

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
    return self._variable.name

  @property
  def _shared_name(self):
    """The shared name of the variable.

      Unlike name(), shared_name doesn't have ":0" suffix. It is user-specified
      name with name scope prefix.

    Returns:
      variable name.
    """
    return self.name[:-2]

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

  @property
  def shape(self):
    """The `TensorShape` of this variable.

    Returns:
      A `TensorShape`.
    """
    return self._variable.get_shape()

  def get_shape(self):
    """Alias of Variable.shape."""
    return self.shape

  def to_proto(self, export_scope=None):
    """Converts a `Variable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    """
    if (export_scope is None or
        self._variable.name.startswith(export_scope)):
      var_def = variable_pb2.VariableDef()
      var_def.variable_name = ops.strip_name_scope(
          self._variable.name, export_scope)
      if self._initial_value is not None:
        # For backwards compatibility.
        var_def.initial_value_name = ops.strip_name_scope(
            self._initial_value.name, export_scope)
      var_def.initializer_name = ops.strip_name_scope(
          self.initializer.name, export_scope)
      var_def.snapshot_name = ops.strip_name_scope(
          self._snapshot.name, export_scope)
      if self._save_slice_info:
        var_def.save_slice_info_def.MergeFrom(self._save_slice_info.to_proto(
            export_scope=export_scope))
      return var_def
    else:
      return None

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    """Returns a `Variable` object created from `variable_def`."""
    return Variable(variable_def=variable_def,
                    import_scope=import_scope)

  class SaveSliceInfo(object):
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
        var_offset: Offset of this `Variable` into the full variable, as a
            list of int.
        var_shape: Shape of this `Variable`, as a list of int.
        save_slice_info_def: `SaveSliceInfoDef` protocol buffer. If not `None`,
          recreates the SaveSliceInfo object its contents.
          `save_slice_info_def` and other arguments are mutually
          exclusive.
        import_scope: Optional `string`. Name scope to add. Only used
          when initializing from protocol buffer.
      """
      if save_slice_info_def:
        assert isinstance(save_slice_info_def, variable_pb2.SaveSliceInfoDef)
        self.full_name = ops.prepend_name_scope(
            save_slice_info_def.full_name, import_scope=import_scope)
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
          "%d,%d" % (o, s) for o, s in zip(self.var_offset, self.var_shape)
      ])
      return full_shape_str + sl_spec

    def to_proto(self, export_scope=None):
      """Returns a SaveSliceInfoDef() proto.

      Args:
        export_scope: Optional `string`. Name scope to remove.

      Returns:
        A `SaveSliceInfoDef` protocol buffer, or None if the `Variable` is not
        in the specified name scope.
      """
      if (export_scope is None or
          self.full_name.startswith(export_scope)):
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

  def _set_save_slice_info(self, save_slice_info):
    """Sets the slice info for this `Variable`.

    Args:
      save_slice_info: A `Variable.SaveSliceInfo` object.
    """
    self._save_slice_info = save_slice_info

  def _get_save_slice_info(self):
    return self._save_slice_info


class PartitionedVariable(object):
  """A container for partitioned `Variable` objects.

  @compatibility(eager) `tf.PartitionedVariable` is not compatible with
  eager execution.  Use `tfe.Variable` instead which is compatible
  with both eager execution and graph construction.  See [the
  TensorFlow Eager Execution
  guide](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/g3doc/guide.md#variables-and-optimizers)
  for details on how variables work in eager execution.
  @end_compatibility
  """

  class PartitionedVariableIterator(object):
    """An iterator that allows accessing the underlying `Variable` objects.

    This iterator is necessary to control order of access when Variables
    are not partitioned in a standard way along a single axis.

    Allows e.g. `list(partitioned_variable)` to return a proper list.
    """

    def __init__(self, partitioned_variable):
      self._ix = 0
      self._partitioned_variable = partitioned_variable

    def __iter__(self):
      return self

    def __next__(self):  # For python3 compatibility.
      return self.next()

    def next(self):
      # pylint: disable=protected-access
      if self._ix >= len(self._partitioned_variable._variable_list):
        raise StopIteration()
      variable = self._partitioned_variable._variable_list[self._ix]
      # pylint: enable=protected-access
      self._ix += 1
      return variable

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
      RuntimeError: If eager execution is enabled
    """
    if not context.in_graph_mode():
      raise RuntimeError("tf.PartitionedVariable not supported in "
                         "eager mode. Please use tfe.Variable instead")
    if not isinstance(variable_list, (list, tuple)):
      raise TypeError(
          "variable_list is not a list or tuple: %s" % variable_list)
    if not isinstance(partitions, (list, tuple)):
      raise TypeError("partitions is not a list or tuple: %s" % partitions)
    if not all([p >= 1 for p in partitions]):
      raise ValueError("partition values must be positive: %s" % partitions)
    if not variable_list:
      raise ValueError("variable_list may not be empty")
    # pylint: disable=protected-access
    for v in variable_list:
      # Sort the variable_list lexicographically according to var offset value.
      if not all([v._get_save_slice_info() is not None for v in variable_list]):
        raise ValueError(
            "All variables must have a save_slice_info available: %s"
            % [v.name for v in variable_list])
      if len(shape) != len(partitions):
        raise ValueError("len(shape) != len(partitions): %s vs. %s"
                         % (shape, partitions))
      if not all([v._get_save_slice_info().full_shape == shape]):
        raise ValueError(
            "All variables' full shapes must match shape: %s; "
            "but full shapes were: %s"
            % (shape, str([v._get_save_slice_info().full_shape])))
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
    return self.PartitionedVariableIterator(self)

  def __len__(self):
    num_partition_axes = len(self._partition_axes())
    if num_partition_axes > 1:
      raise ValueError("Cannot get a length for %d > 1 partition axes"
                       % num_partition_axes)
    return len(self._variable_list)

  def _partition_axes(self):
    if all([p == 1 for p in self._partitions]):
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

  def get_shape(self):
    return self._shape

  def _get_variable_list(self):
    return self._variable_list

  def _get_partitions(self):
    return self._partitions

  def assign(self, value, use_locking=False):
    _ = value, use_locking
    raise NotImplementedError(
        "assign() has not been implemented for PartitionedVariable.")


def global_variables(scope=None):
  """Returns global variables.

  Global variables are variables that are shared across machines in a
  distributed environment. The `Variable()` constructor or `get_variable()`
  automatically adds new variables to the graph collection
  `GraphKeys.GLOBAL_VARIABLES`.
  This convenience function returns the contents of that collection.

  An alternative to global variables are local variables. See
  @{tf.local_variables}

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered
      to include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a
      scope is supplied. The choice of `re.match` means that a `scope` without
      special tokens filters by prefix.

  Returns:
    A list of `Variable` objects.
  """
  return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope)


@deprecated("2017-03-02", "Please use tf.global_variables instead.")
def all_variables():
  """See `tf.global_variables`."""
  return global_variables()


def _all_saveable_objects(scope=None):
  """Returns all variables and `SaveableObject`s that must be checkpointed.

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered
      to include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a
      scope is supplied. The choice of `re.match` means that a `scope` without
      special tokens filters by prefix.

  Returns:
    A list of `Variable` and `SaveableObject` to be checkpointed
  """
  # TODO(andreasst): make this function public once things are settled.
  return (ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope) +
          ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS, scope))


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
  @{tf.global_variables}

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered
      to include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a
      scope is supplied. The choice of `re.match` means that a `scope` without
      special tokens filters by prefix.

  Returns:
    A list of local `Variable` objects.
  """
  return ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES, scope)


def model_variables(scope=None):
  """Returns all variables in the MODEL_VARIABLES collection.

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered
      to include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a
      scope is supplied. The choice of `re.match` means that a `scope` without
      special tokens filters by prefix.

  Returns:
    A list of local Variable objects.
  """
  return ops.get_collection(ops.GraphKeys.MODEL_VARIABLES, scope)


def trainable_variables(scope=None):
  """Returns all variables created with `trainable=True`.

  When passed `trainable=True`, the `Variable()` constructor automatically
  adds new variables to the graph collection
  `GraphKeys.TRAINABLE_VARIABLES`. This convenience function returns the
  contents of that collection.

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered
      to include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a
      scope is supplied. The choice of `re.match` means that a `scope` without
      special tokens filters by prefix.

  Returns:
    A list of Variable objects.
  """
  return ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope)


def moving_average_variables(scope=None):
  """Returns all variables that maintain their moving averages.

  If an `ExponentialMovingAverage` object is created and the `apply()`
  method is called on a list of variables, these variables will
  be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection.
  This convenience function returns the contents of that collection.

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered
      to include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a
      scope is supplied. The choice of `re.match` means that a `scope` without
      special tokens filters by prefix.

  Returns:
    A list of Variable objects.
  """
  return ops.get_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, scope)


def variables_initializer(var_list, name="init"):
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
  if var_list and context.in_graph_mode():
    return control_flow_ops.group(*[v.initializer for v in var_list], name=name)
  return control_flow_ops.no_op(name=name)


@tf_should_use.should_use_result
@deprecated("2017-03-02", "Use `tf.variables_initializer` instead.")
def initialize_variables(var_list, name="init"):
  """See `tf.variables_initializer`."""
  return variables_initializer(var_list, name=name)


def global_variables_initializer():
  """Returns an Op that initializes global variables.

  This is just a shortcut for `variables_initializer(global_variables())`

  Returns:
    An Op that initializes global variables in the graph.
  """
  if context.in_eager_mode():
    return control_flow_ops.no_op(name="global_variables_initializer")
  return variables_initializer(global_variables())


@tf_should_use.should_use_result
@deprecated("2017-03-02", "Use `tf.global_variables_initializer` instead.")
def initialize_all_variables():
  """See `tf.global_variables_initializer`."""
  return global_variables_initializer()


def local_variables_initializer():
  """Returns an Op that initializes all local variables.

  This is just a shortcut for `variables_initializer(local_variables())`

  Returns:
    An Op that initializes all local variables in the graph.
  """
  if context.in_eager_mode():
    return control_flow_ops.no_op(name="local_variables_initializer")
  return variables_initializer(local_variables())


@tf_should_use.should_use_result
@deprecated("2017-03-02", "Use `tf.local_variables_initializer` instead.")
def initialize_local_variables():
  """See `tf.local_variables_initializer`."""
  return local_variables_initializer()


@tf_should_use.should_use_result
def is_variable_initialized(variable):
  """Tests if a variable has been initialized.

  Args:
    variable: A `Variable`.

  Returns:
    Returns a scalar boolean Tensor, `True` if the variable has been
    initialized, `False` otherwise.
  """
  return state_ops.is_variable_initialized(variable)


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
    var_list: List of `Variable` objects to check. Defaults to the
      value of `global_variables().`

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
      return array_ops.stack(ranks)


@tf_should_use.should_use_result
def report_uninitialized_variables(var_list=None,
                                   name="report_uninitialized_variables"):
  """Adds ops to list the names of uninitialized variables.

  When run, it returns a 1-D tensor containing the names of uninitialized
  variables if there are any, or an empty array if there are none.

  Args:
    var_list: List of `Variable` objects to check. Defaults to the
      value of `global_variables() + local_variables()`
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
    with ops.device("/cpu:0"):
      if not var_list:
        # Return an empty tensor so we only need to check for returned tensor
        # size being 0 as an indication of model ready.
        return array_ops.constant([], dtype=dtypes.string)
      else:
        # Get a 1-D boolean tensor listing whether each variable is initialized.
        variables_mask = math_ops.logical_not(
            array_ops.stack(
                [state_ops.is_variable_initialized(v) for v in var_list]))
        # Get a 1-D string tensor containing all the variable names.
        variable_names_tensor = array_ops.constant(
            [s.op.name for s in var_list])
        # Return a 1-D tensor containing all the names of
        # uninitialized variables.
        return array_ops.boolean_mask(variable_names_tensor, variables_mask)

# pylint: disable=protected-access
Variable._OverloadAllOperators()

ops.register_tensor_conversion_function(
    PartitionedVariable, PartitionedVariable._TensorConversionFunction)
# pylint: enable=protected-access


ops.register_dense_tensor_like_type(Variable)
