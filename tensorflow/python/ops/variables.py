"""Variable class."""
import tensorflow.python.platform

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
  z = tf.sigmoid(w + b)

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

  def __init__(self, initial_value, trainable=True, collections=None,
               validate_shape=True, name=None):
    """Creates a new variable with value `initial_value`.

    The new variable is added to the graph collections listed in `collections`,
    which defaults to `[GraphKeys.VARIABLES]`.

    If `trainable` is `True` the variable is also added to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES`.

    This constructor creates both a `variable` Op and an `assign` Op to set the
    variable to its initial value.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`.
        The initial value for the Variable. Must have a shape specified unless
        `validate_shape` is set to False.
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.VARIABLES]`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.

    Returns:
      A Variable.

    Raises:
      ValueError: If the initial value does not have a shape and
        `validate_shape` is `True`.
    """
    if collections is None:
      collections = [ops.GraphKeys.VARIABLES]
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      # pylint: disable=g-no-augmented-assignment
      #
      # Pylint wants us to write collections += [...TRAINABLE_VARIABLES] which
      # is not the same (it modifies the list in place.)  Here, we only want to
      # modify the value of the variable, not the list.
      collections = collections + [ops.GraphKeys.TRAINABLE_VARIABLES]
      # pylint: enable=g-no-augmented-assignment
    with ops.op_scope([initial_value], name, "Variable") as name:
      self._initial_value = ops.convert_to_tensor(initial_value,
                                                  name="initial_value")
      if not self._initial_value.get_shape().is_fully_defined():
        if validate_shape:
          raise ValueError(
              "initial_value must have a shape specified: %s"
              % self._initial_value)
        self._variable = state_ops.variable_op(
            [], self._initial_value.dtype.base_dtype, set_shape=False,
            name=name)
        with ops.device(self._variable.device):
          self._initializer_op = state_ops.assign(
              self._variable, self._initial_value, validate_shape=False).op
      else:
        self._variable = state_ops.variable_op(
            self._initial_value.get_shape(),
            self._initial_value.dtype.base_dtype,
            name=name)
        with ops.device(self._variable.device):
          self._initializer_op = state_ops.assign(
              self._variable, self._initial_value).op
    for key in collections:
      ops.add_to_collection(key, self)
    self._save_slice_info = None

  def _as_graph_element(self):
    """Conversion function for Graph.as_graph_element()."""
    return self._variable

  def _AsTensor(self):
    """Conversion function for ops.convert_to_tensor()."""
    return self._variable

  def eval(self, session=None):
    """In a session, computes and returns the value of this variable.

    This is not a graph construction method, it does not add ops to the graph.

    This convenience method requires a session where the graph containing this
    variable has been launched. If no session is passed, the default session is
    used.  See the [Session class](../../api_docs/python/client.md#Session) for more information on
    launching a graph and on sessions.

    ```python
    v = tf.Variable([1, 2])
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        # Usage passing the session explicitly.
        print v.eval(sess)
        # Usage with the default session.  The 'with' block
        # above makes 'sess' the default session.
        print v.eval()
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
    return control_flow_ops.with_dependencies(
        [self._initializer_op], self._variable)

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
  def _TensorConversionFunction(v, dtype=None, name=None):
    """Utility function for converting a Variable to a Tensor."""
    _ = name
    ret = v._AsTensor()  # pylint: disable=protected-access
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          "Incompatible type conversion requested to type '%s' for variable "
          "of type '%s'" % (dtype.name, v.dtype.name))
    return ret

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

  # Experimental support for saving variables as slices of a larger variable.
  class SaveSliceInfo(object):
    """Information on how to save this Variable as a slice."""

    def  __init__(self, name, spec):
      """Create a SliceInfo.

      Args:
        name: Name of the larger Tensor that this variable is a slice of.
        spec: Slice specification for the saver.
      """
      self.name = name
      self.spec = spec

  def _set_save_slice_info(self, save_slice_info):
    """Sets the slice info for this Variable.

    Args:
      save_slice_info: A Variable.SliceInfo object.
    """
    self._save_slice_info = save_slice_info


def all_variables():
  """Returns all variables collected in the graph.

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
    var_list = all_variables()
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
      with ops.device(var.device):
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
