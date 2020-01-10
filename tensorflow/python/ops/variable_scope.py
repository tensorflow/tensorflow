"""A class to store named variables and a scope operator to manage sharing."""

import contextlib

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import types
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import logging


class _VariableStore(object):
  """Variable store that carries a number of named Variables.

  New variable names and new variables can be created; all stored
  variables are initialized with the initializer passed to __init__.

  Attributes:
    vars: a dictionary with string names (same as passed in GetVar) as keys
          and the corresponding TensorFlow Variables as values.
  """

  def __init__(self):
    """Create a variable store."""
    self._vars = {}  # A dictionary of the stored TensorFlow variables.

  def get_variable(self, name, shape=None, dtype=types.float32,
                   initializer=None, reuse=None, trainable=True,
                   collections=None):
    """Gets an existing variable with these parameters or create a new one.

    If a variable with the given name is already stored, we return the stored
    variable. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    If `reuse` is `None` (the default), both new and existing variables are
    returned.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `UniformUnitScalingInitializer`.

    Args:
      name: the name of the new or existing variable.
      shape: shape of the new or existing variable.
      dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
      initializer: initializer for the variable.
      reuse: a Boolean or `None`. Controls reuse or creation of variables.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see variables.Variable).
      collections: List of graph collections keys to add the Variable to.
        Defaults to `[GraphKeys.VARIABLES]` (see variables.Variable).

    Returns:
      The created or existing variable.

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        or when violating reuse during variable creation.
    """
    should_check = reuse is not None
    dtype = types.as_dtype(dtype)
    shape = tensor_shape.as_shape(shape)
    if name in self._vars:
      # Here we handle the case when returning an existing variable.
      if should_check and not reuse:
        raise ValueError("Over-sharing: Variable %s already exists, disallowed."
                         " Did you mean to set reuse=True in VarScope?" % name)
      found_var = self._vars[name]
      if not shape.is_compatible_with(found_var.get_shape()):
        raise ValueError("Trying to share variable %s, but specified shape %s"
                         " and found shape %s." % (name, str(shape),
                                                   str(found_var.get_shape())))
      if not dtype.is_compatible_with(found_var.dtype):
        dtype_str = dtype.name
        found_type_str = found_var.dtype.name
        raise ValueError("Trying to share variable %s, but specified dtype %s"
                         " and found dtype %s." % (name, str(dtype_str),
                                                   str(found_type_str)))
      return found_var

    # The code below handles only the case of creating a new variable.
    if should_check and reuse:
      raise ValueError("Under-sharing: Variable %s does not exist, disallowed."
                       " Did you mean to set reuse=None in VarScope?" % name)
    if not shape.is_fully_defined():
      raise ValueError("Shape of a new variable (%s) must be fully defined, "
                       "but instead was %s." % (name, shape))
    if initializer is None:
      initializer = init_ops.uniform_unit_scaling_initializer()
    with ops.name_scope(name + "/Initializer/"):
      init_val = initializer(shape.as_list(), dtype=dtype)
    v = variables.Variable(init_val, name=name, trainable=trainable,
                           collections=collections)
    self._vars[name] = v
    logging.info("Created variable %s with shape %s and init %s", v.name,
                 format(shape), str(initializer))
    return v


class _VariableScope(object):
  """Variable scope object to carry defaults to provide to get_variable.

  Many of the arguments we need for get_variable in a variable store are most
  easily handled with a context. This object is used for the defaults.

  Attributes:
    name: name of the current scope, used as prefix in get_variable.
    initializer: default initializer passed to get_variable.
    reuse: Boolean or None, setting the reuse in get_variable.
  """

  def __init__(self, reuse, name="", initializer=None):
    self._name = name
    self._initializer = initializer
    self._reuse = reuse

  @property
  def name(self):
    return self._name

  @property
  def reuse(self):
    return self._reuse

  @property
  def initializer(self):
    return self._initializer

  def reuse_variables(self):
    """Reuse variables in this scope."""
    self._reuse = True

  def set_initializer(self, initializer):
    """Set initializer for this scope."""
    self._initializer = initializer

  def get_variable(self, var_store, name, shape=None, dtype=types.float32,
                   initializer=None, trainable=True, collections=None):
    """Gets an existing variable with this name or create a new one."""
    if initializer is None and self._initializer:
      initializer = self._initializer
    full_name = self.name + "/" + name if self.name else name
    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None):
      return var_store.get_variable(full_name, shape, dtype, initializer,
                                    self.reuse, trainable, collections)


_VARSTORE_KEY = ("__variable_store",)
_VARSCOPE_KEY = ("__varscope",)


def get_variable_scope():
  """Returns the current variable scope."""
  scope = ops.get_collection(_VARSCOPE_KEY)
  if scope:  # This collection has at most 1 element, the default scope at [0].
    return scope[0]
  scope = _VariableScope(False)
  ops.add_to_collection(_VARSCOPE_KEY, scope)
  return scope


def _get_default_variable_store():
  store = ops.get_collection(_VARSTORE_KEY)
  if store:
    return store[0]
  store = _VariableStore()
  ops.add_to_collection(_VARSTORE_KEY, store)
  return store


def get_variable(name, shape=None, dtype=types.float32, initializer=None,
                 trainable=True, collections=None):
  """Gets an existing variable with these parameters or create a new one.

  This function prefixes the name with the current variable scope
  and performs reuse checks. See the
  [Variable Scope How To](../../how_tos/variable_scope/index.md)
  for an extensive description of how reusing works. Here is a basic example:

  ```python
  with tf.variable_scope("foo"):
      v = get_variable("v", [1])  # v.name == "foo/v:0"
      w = get_variable("w", [1])  # w.name == "foo/w:0"
  with tf.variable_scope("foo", reuse=True)
      v1 = get_variable("v")  # The same as v above.
  ```

  If initializer is `None` (the default), the default initializer passed in
  the constructor is used. If that one is `None` too, a
  `UniformUnitScalingInitializer` will be used.

  Args:
    name: the name of the new or existing variable.
    shape: shape of the new or existing variable.
    dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: initializer for the variable if one is created.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see variables.Variable).
    collections: List of graph collections keys to add the Variable to.
      Defaults to `[GraphKeys.VARIABLES]` (see variables.Variable).

  Returns:
    The created or existing variable.

  Raises:
    ValueError: when creating a new variable and shape is not declared,
      or when violating reuse during variable creation. Reuse is set inside
      `variable_scope`.
  """
  return get_variable_scope().get_variable(_get_default_variable_store(), name,
                                           shape, dtype, initializer,
                                           trainable, collections)


@contextlib.contextmanager
def variable_scope(name_or_scope, reuse=None, initializer=None):
  """Returns a context for variable scope.

  Variable scope allows to create new variables and to share already created
  ones while providing checks to not create or share by accident. For details,
  see the [Variable Scope How To](../../how_tos/variable_scope/index.md),
  here we present only a few basic examples.

  Simple example of how to create a new variable:

  ```python
  with tf.variable_scope("foo"):
      with tf.variable_scope("bar"):
          v = tf.get_variable("v", [1])
          assert v.name == "foo/bar/v:0"
  ```

  Basic example of sharing a variable:

  ```python
  with tf.variable_scope("foo"):
      v = get_variable("v", [1])
  with tf.variable_scope("foo", reuse=True):
      v1 = tf.get_variable("v", [1])
  assert v1 == v
  ```

  Sharing a variable by capturing a scope and setting reuse:

  ```python
  with tf.variable_scope("foo") as scope.
      v = get_variable("v", [1])
      scope.reuse_variables()
      v1 = tf.get_variable("v", [1])
  assert v1 == v
  ```

  To prevent accidental sharing of variables, we raise an exception when
  getting an existing variable in a non-reusing scope.

  ```python
  with tf.variable_scope("foo") as scope.
      v = get_variable("v", [1])
      v1 = tf.get_variable("v", [1])
      #  Raises ValueError("... v already exists ...").
  ```

  Similarly, we raise an exception when trying to get a variable that
  does not exist in reuse mode.

  ```python
  with tf.variable_scope("foo", reuse=True):
      v = get_variable("v", [1])
      #  Raises ValueError("... v does not exists ...").
  ```

  Note that the `reuse` flag is inherited: if we open a reusing scope,
  then all its sub-scopes become reusing as well.

  Args:
    name_or_scope: `string` or `VariableScope`: the scope to open.
    reuse: `True` or `None`; if `True`, we go into reuse mode for this scope as
      well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
    initializer: default initializer for variables within this scope.

  Yields:
    A scope that can be to captured and reused.

  Raises:
    ValueError: when trying to reuse within a create scope, or create within
      a reuse scope, or if reuse is not `None` or `True`.
    TypeError: when the types of some arguments are not appropriate.
  """
  if not isinstance(name_or_scope, (_VariableScope, basestring)):
    raise TypeError("VariableScope: name_scope must be a string or "
                    "VariableScope.")
  if reuse not in [None, True]:
    raise ValueError("VariableScope reuse parameter must be True or None.")
  if not reuse and isinstance(name_or_scope, (_VariableScope)):
    logging.info("Passing VariableScope to a non-reusing scope, intended?")
  if reuse and isinstance(name_or_scope, (basestring)):
    logging.info("Re-using string-named scope, consider capturing as object.")
  get_variable_scope()  # Ensure that a default exists, then get a pointer.
  default_varscope = ops.get_collection(_VARSCOPE_KEY)
  try:
    old = default_varscope[0]
    reuse = reuse or old.reuse  # Re-using is inherited by sub-scopes.
    if isinstance(name_or_scope, _VariableScope):
      # Handler for the case when we jump to a shared scope.
      #   In this case, we leave the current name_scope unchanged.
      #   We create a new VariableScope (default_varscope[0]) that contains
      #   a copy of the provided shared scope, possibly with changed reuse
      #   and initializer, if the user requested this.
      default_varscope[0] = _VariableScope(reuse, name_or_scope.name,
                                           name_or_scope.initializer)
      if initializer:
        default_varscope[0].set_initializer(initializer)
      yield default_varscope[0]
    else:
      # Handler for the case when we just prolong current variable scope.
      #   In this case we prolong the current name_scope and create a new
      #   VariableScope with name extended by the provided one, and inherited
      #   reuse and initializer (except if the user provided values to set).
      with ops.name_scope(name_or_scope):
        new_name = old.name + "/" + name_or_scope if old.name else name_or_scope
        default_varscope[0] = _VariableScope(reuse, name=new_name,
                                             initializer=old.initializer)
        if initializer:
          default_varscope[0].set_initializer(initializer)
        yield default_varscope[0]
  finally:
    default_varscope[0] = old
