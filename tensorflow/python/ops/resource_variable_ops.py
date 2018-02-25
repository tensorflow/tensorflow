# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Ops to use variables as resources."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import variables
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_resource_variable_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.training import checkpointable
from tensorflow.python.util import compat


def _eager_safe_variable_handle(shape, dtype, shared_name, name, graph_mode):
  """Creates a variable handle with information to do shape inference."""
  container = ops.get_default_graph()._container  # pylint: disable=protected-access
  if container is None:
    container = ""
  if not graph_mode:
    # When in eager mode use a uid for the shared_name, to prevent accidental
    # sharing.
    shared_name = str(ops.uid())
  handle = gen_resource_variable_ops.var_handle_op(shape=shape, dtype=dtype,
                                                   shared_name=shared_name,
                                                   name=name,
                                                   container=container)
  if graph_mode:
    return handle

  # We do not want two distinct ResourceVariable objects for the same
  # underlying resource in the runtime.
  # When in eager mode, explicitly ensure so here. When in graph mode, it's
  # ensured by always generating different variable names.
  exists = gen_resource_variable_ops.var_is_initialized_op(handle)
  if exists:
    raise ValueError("variable object with name '%s' already created. Use "
                     "get_variable() if reuse is desired." %
                     shared_name)
  with context.graph_mode(), ops.Graph().as_default() as graph:
    h = gen_resource_variable_ops.var_handle_op(shape=shape, dtype=dtype,
                                                shared_name=shared_name,
                                                name=name,
                                                container=container)

    # Tensor._handle_data contains information for the shape-inference code to
    # know the shape and dtype of the variable pointed to by a handle. Since
    # shape inference doesn't run in eager mode we copy this data here for when
    # the handle is captured by an eager mode function.
    handle._handle_data = h._handle_data  # pylint: disable=protected-access
  # Clean up our reference cycles to avoid making the garbage collector run.
  # pylint: disable=protected-access
  # OrderedDict, constructed on Graph creation, makes a simple reference loop
  # and hides it in an __attribute in some Python versions. We don't need to
  # throw an error if we can't find it, but if we do find it we can break the
  # loop to avoid creating work for the garbage collector.
  problematic_cycle = graph._functions.__dict__.get("_OrderedDict__root", None)
  # pylint: enable=protected-access
  if problematic_cycle:
    try:
      del problematic_cycle[0][:]
    except TypeError:
      # This is probably not one of the problematic Python versions. Continue
      # with the rest of our cleanup.
      pass
  # Now clean up our own reference cycles by clearing all of the attributes for
  # the Graph and op we created.
  h.__dict__ = {}
  graph.__dict__ = {}
  return handle


class EagerResourceDeleter(object):
  """An object which cleans up a resource handle.

  An alternative to defining a __del__ method on an object. The intended use is
  that ResourceVariables or other objects with resource handles will maintain a
  single reference to this object. When the parent object is collected, this
  object will be too. Even if the parent object is part of a reference cycle,
  the cycle will be collectable.
  """

  def __init__(self, handle, handle_device):
    if not isinstance(handle, ops.Tensor):
      raise ValueError(
          ("Passed handle=%s to EagerResourceDeleter. Was expecting a handle "
           "Tensor." % (handle,)))
    self._handle = handle
    self._handle_device = handle_device

  def __del__(self):
    # Resources follow object-identity when executing eagerly, so it is safe to
    # delete the resource we have a handle to.
    try:
      # This resource was created in eager mode. However, this destructor may be
      # running in graph mode (especially during unit tests). To clean up
      # successfully, we switch back into eager mode temporarily.
      with context.eager_mode():
        with ops.device(self._handle_device):
          gen_resource_variable_ops.destroy_resource_op(
              self._handle, ignore_lookup_error=True)
    except TypeError:
      # Suppress some exceptions, mainly for the case when we're running on
      # module deletion. Things that can go wrong include the context module
      # already being unloaded, self._handle._handle_data no longer being
      # valid, and so on. Printing warnings in these cases is silly
      # (exceptions raised from __del__ are printed as warnings to stderr).
      pass  # 'NoneType' object is not callable when the handle has been
            # partially unloaded.
    except AttributeError:
      pass  # 'NoneType' object has no attribute 'eager_mode' when context has
            # been unloaded. Will catch other module unloads as well.


def shape_safe_assign_variable_handle(handle, shape, value, name=None):
  """Helper that checks shape compatibility and assigns variable."""
  value_tensor = ops.convert_to_tensor(value)
  shape.assert_is_compatible_with(value_tensor.shape)
  return gen_resource_variable_ops.assign_variable_op(handle,
                                                      value_tensor,
                                                      name=name)


class ResourceVariable(variables.Variable):
  """Variable based on resource handles.

  See the ${variables} documentation for more details.

  A `ResourceVariable` allows you to maintain state across subsequent calls to
  session.run.

  The `ResourceVariable` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  Just like any `Tensor`, variables created with `ResourceVariable()` can be
  used as inputs for other Ops in the graph. Additionally, all the operators
  overloaded for the `Tensor` class are carried over to variables, so you can
  also add nodes to the graph by just doing arithmetic on variables.

  Unlike tf.Variable, a tf.ResourceVariable has well-defined semantics. Each
  usage of a ResourceVariable in a TensorFlow graph adds a read_value operation
  to the graph. The Tensors returned by a read_value operation are guaranteed
  to see all modifications to the value of the variable which happen in any
  operation on which the read_value depends on (either directly, indirectly, or
  via a control dependency) and guaranteed to not see any modification to the
  value of the variable on which the read_value operation does not depend on.

  For example, if there is more than one assignment to a ResourceVariable in
  a single session.run call there is a well-defined value for each operation
  which uses the variable's value if the assignments and the read are connected
  by edges in the graph. Consider the following example, in which two writes
  can cause tf.Variable and tf.ResourceVariable to behave differently:

   ```python
    a = tf.ResourceVariable(1.0)
    a.initializer.run()

    assign = a.assign(2.0)
    with tf.control_dependencies([assign]):
      b = a.read_value()
    with tf.control_dependencies([b]):
      other_assign = a.assign(3.0)
    with tf.control_dependencies([other_assign]):
      # Will print 2.0 because the value was read before other_assign ran. If
      # `a` was a tf.Variable instead, 2.0 or 3.0 could be printed.
      tf.Print(b, [b]).eval()
  ```

  To enforce these consistency properties tf.ResourceVariable might make more
  copies than an equivalent tf.Variable under the hood, so tf.Variable is still
  not deprecated.
  """

  def __init__(self,
               initial_value=None,
               trainable=True,
               collections=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               dtype=None,
               variable_def=None,
               import_scope=None,
               constraint=None):
    """Creates a variable.

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
      validate_shape: Ignored. Provided for compatibility with tf.Variable.
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
      variable_def: `VariableDef` protocol buffer. If not None, recreates the
        `ResourceVariable` object with its contents. `variable_def` and other
        arguments (except for import_scope) are mutually exclusive.
      import_scope: Optional `string`. Name scope to add to the
        ResourceVariable. Only used when `variable_def` is provided.
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

    @compatibility(eager)
    When Eager Execution is enabled, the default for the `collections` argument
    is `None`, which signifies that this `Variable` will not be added to any
    collections.
    @end_compatibility
    """
    if variable_def:
      if initial_value is not None:
        raise ValueError("variable_def and initial_value are mutually "
                         "exclusive.")
      if not context.in_graph_mode():
        raise ValueError("Creating ResourceVariable from variable_def"
                         " only supported in GRAPH mode.")
      self._init_from_proto(variable_def, import_scope=import_scope)
    else:
      self._init_from_args(
          initial_value=initial_value,
          trainable=trainable,
          collections=collections,
          validate_shape=validate_shape,
          caching_device=caching_device,
          name=name,
          dtype=dtype,
          constraint=constraint)

  # pylint: disable=unused-argument
  def _init_from_args(self,
                      initial_value=None,
                      trainable=True,
                      collections=None,
                      validate_shape=True,
                      caching_device=None,
                      name=None,
                      dtype=None,
                      constraint=None):
    """Creates a variable.

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
      validate_shape: Ignored. Provided for compatibility with tf.Variable.
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

    @compatibility(eager)
    When Eager Execution is enabled, variables are never added to collections.
    It is not implicitly added to the `GLOBAL_VARIABLES` or
    `TRAINABLE_VARIABLES` collections, and the `collections` argument is
    ignored.
    @end_compatibility
    """
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

    if isinstance(initial_value, checkpointable.CheckpointInitialValue):
      self._maybe_initialize_checkpointable()
      self._update_uid = initial_value.checkpoint_position.restore_uid
      initial_value = initial_value.wrapped_value

    self._trainable = trainable
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    self._save_slice_info = None
    # Store the graph key so optimizers know how to only retrieve variables from
    # this graph.
    self._graph_key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    with ops.init_scope():
      self._in_graph_mode = context.in_graph_mode()
      with ops.name_scope(name, "Variable", []
                          if init_from_fn else [initial_value]) as name:
        # pylint: disable=protected-access
        handle_name = ops._name_from_scope_name(name)
        if init_from_fn:
          # Use attr_scope and device(None) to simulate the behavior of
          # colocate_with when the variable we want to colocate with doesn't
          # yet exist.
          if self._in_graph_mode:
            attr = attr_value_pb2.AttrValue(
                list=attr_value_pb2.AttrValue.ListValue(
                    s=[compat.as_bytes("loc:@%s" % handle_name)]))
            with ops.get_default_graph()._attr_scope({"_class": attr}):
              with ops.name_scope("Initializer"), ops.device(None):
                initial_value = ops.convert_to_tensor(
                    initial_value(), name="initial_value", dtype=dtype)
              self._handle = _eager_safe_variable_handle(
                  shape=initial_value.get_shape(),
                  dtype=initial_value.dtype.base_dtype,
                  shared_name=handle_name,
                  name=name,
                  graph_mode=self._in_graph_mode)
              self._handle_device = (
                  self._handle.device if self._in_graph_mode else
                  context.get_default_context().device_name)
              self._shape = initial_value.get_shape()
          else:
            initial_value = initial_value()
            with ops.name_scope("Initializer"):
              initial_value = ops.convert_to_tensor(
                  initial_value, name="initial_value", dtype=dtype)
            self._handle = _eager_safe_variable_handle(
                shape=initial_value.get_shape(),
                dtype=initial_value.dtype.base_dtype,
                shared_name=handle_name,
                name=name,
                graph_mode=False)
            self._handle_device = (
                self._handle.device if self._in_graph_mode else
                context.get_default_context().device_name)
            self._shape = initial_value.get_shape()
        # pylint: enable=protected-access

        # Or get the initial value from a Tensor or Python object.
        else:
          with ops.name_scope("Initializer"):
            initial_value = ops.convert_to_tensor(
                initial_value, name="initial_value", dtype=dtype)
          # pylint: disable=protected-access
          if (self._in_graph_mode and initial_value is not None and
              initial_value.op._get_control_flow_context() is not None):
            raise ValueError(
                "Initializer for variable %s is from inside a control-flow "
                "construct, such as a loop or conditional. When creating a "
                "variable inside a loop or conditional, use a lambda as the "
                "initializer." % name)
          # pylint: enable=protected-access
          self._handle = _eager_safe_variable_handle(
              shape=initial_value.get_shape(),
              dtype=initial_value.dtype.base_dtype,
              shared_name=handle_name,
              name=name,
              graph_mode=self._in_graph_mode)
          self._handle_device = (self._handle.device if self._in_graph_mode else
                                 context.get_default_context().device_name)
          self._shape = initial_value.get_shape()

        self._initial_value = initial_value if self._in_graph_mode else None
        self._handle_name = handle_name + ":0"
        self._dtype = initial_value.dtype.base_dtype
        self._constraint = constraint

        if self._in_graph_mode:
          with ops.name_scope("IsInitialized"):
            self._is_initialized_op = (
                gen_resource_variable_ops.var_is_initialized_op(self._handle))
          if initial_value is not None:
            with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
              self._initializer_op = (
                  gen_resource_variable_ops.assign_variable_op(
                      self._handle,
                      self._try_guard_against_uninitialized_dependencies(
                          initial_value),
                      name=n))
          with ops.name_scope("Read"), ops.colocate_with(self._handle):
            # Manually assign reads to the handle's device to avoid log
            # messages.
            with ops.device(self._handle_device):
              value = self._read_variable_op()
            self._graph_element = value
            if caching_device is not None:
              # Variables may be created in a tf.device() or ops.colocate_with()
              # context. At the same time, users would expect caching device to
              # be independent of this context, and/or would not expect the
              # current device context to be merged with the caching device
              # spec.  Therefore we reset the colocation stack before creating
              # the cached value. Note that resetting the colocation stack will
              # also reset the device stack.
              with ops.colocate_with(None, ignore_existing=True):
                with ops.device(caching_device):
                  self._cached_value = array_ops.identity(value)
            else:
              self._cached_value = None
        else:
          gen_resource_variable_ops.assign_variable_op(self._handle,
                                                       initial_value)
          self._is_initialized_op = None
          self._initializer_op = None
          self._graph_element = None
          if caching_device:
            with ops.device(caching_device):
              self._cached_value = self._read_variable_op()
          else:
            self._cached_value = None
        if context.in_graph_mode():
          ops.add_to_collections(collections, self)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)

    if not self._in_graph_mode:
      # After the handle has been created, set up a way to clean it up when
      # executing eagerly. We'll hold the only reference to the deleter, so that
      # when this object is garbage collected the deleter will be too. This
      # means ResourceVariables can be part of reference cycles without those
      # cycles being uncollectable, and means that no __del__ will be defined at
      # all in graph mode.
      self._handle_deleter = EagerResourceDeleter(
          handle=self._handle, handle_device=self._handle_device)

  def _init_from_proto(self, variable_def, import_scope=None):
    """Initializes from `VariableDef` proto."""
    # Note that init_from_proto is currently not supported in Eager mode.
    assert context.in_graph_mode()
    self._in_graph_mode = True
    assert isinstance(variable_def, variable_pb2.VariableDef)
    if not variable_def.is_resource:
      raise ValueError("Trying to restore Variable as ResourceVariable.")

    # Create from variable_def.
    g = ops.get_default_graph()
    self._handle = g.as_graph_element(
        ops.prepend_name_scope(
            variable_def.variable_name, import_scope=import_scope))
    self._shape = tensor_shape.TensorShape(
        self._handle.op.get_attr("shape"))
    self._handle_device = self._handle.device
    self._handle_name = self._handle.name
    self._initializer_op = g.as_graph_element(
        ops.prepend_name_scope(
            variable_def.initializer_name, import_scope=import_scope))
    # Check whether initial_value_name exists for backwards compatibility.
    if (hasattr(variable_def, "initial_value_name") and
        variable_def.initial_value_name):
      self._initial_value = g.as_graph_element(
          ops.prepend_name_scope(variable_def.initial_value_name,
                                 import_scope=import_scope))
    else:
      self._initial_value = None
    if variable_def.snapshot_name:
      self._cached_value = g.as_graph_element(
          ops.prepend_name_scope(
              variable_def.snapshot_name, import_scope=import_scope))
    else:
      self._cached_value = None
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = variables.Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def,
          import_scope=import_scope)
    else:
      self._save_slice_info = None
    self._caching_device = None
    self._dtype = dtypes.as_dtype(self._handle.op.get_attr("dtype"))
    self._graph_element = self.value()
    self._constraint = None

  def __nonzero__(self):
    return self.__bool__()

  def __bool__(self):
    return bool(self.read_value())

  @property
  def dtype(self):
    """The dtype of this variable."""
    return self._dtype

  @property
  def device(self):
    """The device this variable is on."""
    return self._handle_device

  @property
  def graph(self):
    """The `Graph` of this variable."""
    return self._handle.graph

  @property
  def name(self):
    """The name of the handle for this variable."""
    return self._handle_name

  @property
  def shape(self):
    """The shape of this variable."""
    return self._shape

  @property
  def create(self):
    """The op responsible for initializing this variable."""
    if not self._in_graph_mode:
      raise RuntimeError("Calling create in EAGER mode not supported.")
    return self._initializer_op

  @property
  def handle(self):
    """The handle by which this variable can be accessed."""
    return self._handle

  def value(self):
    """A cached operation which reads the value of this variable."""
    if self._cached_value is not None:
      return self._cached_value
    with ops.colocate_with(None, ignore_existing=True):
      with ops.device(self._handle_device):
        return self._read_variable_op()

  def _as_graph_element(self):
    """Conversion function for Graph.as_graph_element()."""
    return self._graph_element

  @property
  def initializer(self):
    """The op responsible for initializing this variable."""
    return self._initializer_op

  @property
  def initial_value(self):
    """Returns the Tensor used as the initial value for the variable."""
    if context.in_eager_mode():
      raise RuntimeError("initial_value not supported in EAGER mode.")
    return self._initial_value

  @property
  def constraint(self):
    """Returns the constraint function associated with this variable.

    Returns:
      The constraint function that was passed to the variable constructor.
      Can be `None` if no constraint was passed.
    """
    return self._constraint

  @property
  def op(self):
    """The op for this variable."""
    return self._handle.op

  def eval(self, session=None):
    """Evaluates and returns the value of this variable."""
    if context.in_eager_mode():
      raise RuntimeError("Trying to eval in EAGER mode")
    return self._graph_element.eval(session=session)

  def numpy(self):
    if context.in_graph_mode():
      raise NotImplementedError(
          "numpy() is only available when eager execution is enabled.")
    return self.read_value().numpy()

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
    return gen_state_ops.resource_count_up_to(self.handle, limit=limit,
                                              T=self.dtype)

  def _set_save_slice_info(self, save_slice_info):
    """Sets the slice info for this `ResourceVariable`.

    Args:
      save_slice_info: A `Variable.SaveSliceInfo` object.
    """
    self._save_slice_info = save_slice_info

  def _get_save_slice_info(self):
    return self._save_slice_info

  def _read_variable_op(self):
    if hasattr(self, "_trainable") and self._trainable:
      tape.watch_variable(self)
    return gen_resource_variable_ops.read_variable_op(self._handle,
                                                      self._dtype)

  def read_value(self):
    """Constructs an op which reads the value of this variable.

    Should be used when there are multiple reads, or when it is desirable to
    read the value only after some condition is true.

    Returns:
     the read operation.
    """
    with ops.name_scope("Read"):
      # Ensure we read the variable in the same device as the handle.
      with ops.device(self._handle_device):
        value = self._read_variable_op()
    # Return an identity so it can get placed on whatever device the context
    # specifies instead of the device where the variable is.
    return array_ops.identity(value)

  def sparse_read(self, indices, name=None):
    """Reads the value of this variable sparsely, using `gather`."""
    with ops.name_scope("Gather" if name is None else name) as name:
      if self._trainable:
        tape.watch_variable(self)
      value = gen_resource_variable_ops.resource_gather(
          self._handle, indices, dtype=self._dtype, name=name)
    return array_ops.identity(value)

  def to_proto(self, export_scope=None):
    """Converts a `ResourceVariable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Raises:
      RuntimeError: If run in EAGER mode.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    """
    if context.in_eager_mode():
      raise RuntimeError("to_proto not supported in EAGER mode.")
    if export_scope is None or self.handle.name.startswith(export_scope):
      var_def = variable_pb2.VariableDef()
      var_def.variable_name = ops.strip_name_scope(self.handle.name,
                                                   export_scope)
      if self._initial_value is not None:
        # This is inside an if-statement for backwards compatibility, since
        # self._initial_value might be None for variables constructed from old
        # protos.
        var_def.initial_value_name = ops.strip_name_scope(
            self._initial_value.name, export_scope)
      var_def.initializer_name = ops.strip_name_scope(self.initializer.name,
                                                      export_scope)
      if self._cached_value is not None:
        var_def.snapshot_name = ops.strip_name_scope(self._cached_value.name,
                                                     export_scope)
      var_def.is_resource = True
      if self._save_slice_info:
        var_def.save_slice_info_def.MergeFrom(
            self._save_slice_info.to_proto(export_scope=export_scope))
      return var_def
    else:
      return None

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    if context.in_eager_mode():
      raise RuntimeError("from_proto not supported in EAGER mode.")
    return ResourceVariable(
        variable_def=variable_def, import_scope=import_scope)

  @staticmethod
  def _OverloadAllOperators():  # pylint: disable=invalid-name
    """Register overloads for all operators."""
    for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
      ResourceVariable._OverloadOperator(operator)
    # For slicing, bind getitem differently than a tensor (use SliceHelperVar
    # instead)
    # pylint: disable=protected-access
    setattr(ResourceVariable, "__getitem__", array_ops._SliceHelperVar)

  def _AsTensor(self):
    return self.value()

  def _ref(self):
    """Unsupported."""
    raise NotImplementedError("ResourceVariable does not implement _ref()")

  def set_shape(self, shape):
    """Unsupported."""
    raise NotImplementedError("ResourceVariable does not implement set_shape()")

  @staticmethod
  def _OverloadOperator(operator):  # pylint: disable=invalid-name
    """Defer an operator overload to `ops.Tensor`.

    We pull the operator out of ops.Tensor dynamically to avoid ordering issues.

    Args:
      operator: string. The operator name.
    """

    def _run_op(a, *args):
      # pylint: disable=protected-access
      value = a._AsTensor()
      return getattr(ops.Tensor, operator)(value, *args)

    # Propagate __doc__ to wrapper
    try:
      _run_op.__doc__ = getattr(ops.Tensor, operator).__doc__
    except AttributeError:
      pass

    setattr(ResourceVariable, operator, _run_op)

  __array_priority__ = 100

  def assign_sub(self, delta, use_locking=None, name=None):
    # TODO(apassos): this here and below is not atomic. Consider making it
    # atomic if there's a way to do so without a performance cost for those who
    # don't need it.
    return self._lazy_read(gen_resource_variable_ops.assign_sub_variable_op(
        self.handle,
        ops.convert_to_tensor(delta, dtype=self.dtype),
        name=name))

  def assign_add(self, delta, use_locking=None, name=None):
    return self._lazy_read(gen_resource_variable_ops.assign_add_variable_op(
        self.handle,
        ops.convert_to_tensor(delta, dtype=self.dtype),
        name=name))

  def _lazy_read(self, op):
    if hasattr(self, "_trainable") and self._trainable:
      tape.watch_variable(self)
    return _UnreadVariable(
        self._handle, self.dtype, self._handle_device, self._shape,
        self._in_graph_mode,
        self._handle_deleter if not self._in_graph_mode else None, op)

  def assign(self, value, use_locking=None, name=None):
    value_tensor = ops.convert_to_tensor(value, dtype=self.dtype)
    self._shape.assert_is_compatible_with(value_tensor.shape)
    return self._lazy_read(
        gen_resource_variable_ops.assign_variable_op(
            self.handle,
            value_tensor,
            name=name))

  def _strided_slice_assign(self, begin, end, strides, value, name, begin_mask,
                            end_mask, ellipsis_mask, new_axis_mask,
                            shrink_axis_mask):
    return self._lazy_read(
        gen_array_ops.resource_strided_slice_assign(
            ref=self.handle,
            begin=begin,
            end=end,
            strides=strides,
            value=value,
            name=name,
            begin_mask=begin_mask,
            end_mask=end_mask,
            ellipsis_mask=ellipsis_mask,
            new_axis_mask=new_axis_mask,
            shrink_axis_mask=shrink_axis_mask))

  def __int__(self):
    if self.dtype != dtypes.int32 and self.dtype != dtypes.int64:
      raise TypeError("Non-integer variable can't be converted to integer.")
    return int(self.value().numpy())

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    del name
    if dtype is not None and dtype != self.dtype:
      print("trying to switch the dtype to ", dtype, " from ", self.dtype)
      return NotImplemented
    if as_ref:
      return self.read_value().op.inputs[0]
    else:
      return self.value()

  def __iadd__(self, unused_other):
    raise RuntimeError("Variable += value not supported. Use "
                       "variable.assign_add(value) to modify the variable "
                       "value and variable = variable + value to get a new "
                       "Tensor object.")

  def __isub__(self, unused_other):
    raise RuntimeError("Variable -= value not supported. Use "
                       "variable.assign_sub(value) to modify the variable "
                       "value and variable = variable - value to get a new "
                       "Tensor object.")

  def __imul__(self, unused_other):
    raise RuntimeError("Variable *= value not supported. Use "
                       "variable.assign_mul(value) to modify the variable "
                       "value and variable = variable * value to get a new "
                       "Tensor object.")

  def __idiv__(self, unused_other):
    raise RuntimeError("Variable /= value not supported. Use "
                       "variable.assign_div(value) to modify the variable "
                       "value and variable = variable / value to get a new "
                       "Tensor object.")

  def __itruediv__(self, unused_other):
    raise RuntimeError("Variable /= value not supported. Use "
                       "variable.assign_div(value) to modify the variable "
                       "value and variable = variable / value to get a new "
                       "Tensor object.")

  def __irealdiv__(self, unused_other):
    raise RuntimeError("Variable /= value not supported. Use "
                       "variable.assign_div(value) to modify the variable "
                       "value and variable = variable / value to get a new "
                       "Tensor object.")

  def __ipow__(self, unused_other):
    raise RuntimeError("Variable **= value not supported. Use "
                       "value and variable = variable ** value to get a new "
                       "Tensor object.")


def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


class _UnreadVariable(ResourceVariable):
  """Represents a future for a read of a variable.

  Pretends to be the tensor if anyone looks.
  """

  def __init__(self, handle, dtype, handle_device,  # pylint: disable=super-init-not-called
               shape, in_graph_mode, deleter, parent_op):
    # We do not call super init on purpose.
    self._trainable = False
    self._save_slice_info = None
    self._graph_key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    self._in_graph_mode = in_graph_mode
    self._handle = handle
    self._handle_device = handle_device
    self._shape = shape
    self._initial_value = None
    if isinstance(self._handle, ops.EagerTensor):
      self._handle_name = ""
    else:
      self._handle_name = self._handle.name
    self._dtype = dtype
    self._constraint = None
    self._cached_value = None
    self._is_initialized_op = None
    self._initializer_op = None
    self._parent_op = parent_op
    if context.in_graph_mode():
      self._graph_element = self.read_value()
    else:
      self._graph_element = None
    self._handle_deleter = deleter

  def value(self):
    return self._read_variable_op()

  def read_value(self):
    return self._read_variable_op()

  def _read_variable_op(self):
    with ops.control_dependencies([self._parent_op]):
      return gen_resource_variable_ops.read_variable_op(self._handle,
                                                        self._dtype)

  def set_shape(self, shape):
    self._shape = shape

  @property
  def op(self):
    """The op for this variable."""
    return self._parent_op

ops.register_tensor_conversion_function(_UnreadVariable, _dense_var_to_tensor)
ops.register_dense_tensor_like_type(_UnreadVariable)

# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.

# Note: registering for Variable after ResourceVariable because inheritance will
# otherwise lead to the wrong behavior.
ops.register_tensor_conversion_function(ResourceVariable, _dense_var_to_tensor)
ops.register_tensor_conversion_function(
    variables.Variable, variables.Variable._TensorConversionFunction)  # pylint: disable=protected-access

# pylint: disable=protected-access
ResourceVariable._OverloadAllOperators()
ops.register_dense_tensor_like_type(ResourceVariable)


@ops.RegisterGradient("ReadVariableOp")
def _ReadGrad(_, grad):
  """Gradient for read op."""
  return grad


@ops.RegisterGradient("ResourceGather")
def _GatherGrad(op, grad):
  """Gradient for gather op."""
  # Build appropriately shaped IndexedSlices
  handle = op.inputs[0]
  indices = op.inputs[1]
  params_shape = gen_resource_variable_ops.variable_shape(handle)
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[1:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return (ops.IndexedSlices(values, indices, params_shape), None)


def _to_proto_fn(v, export_scope=None):
  """Converts Variable and ResourceVariable to VariableDef for collections."""
  return v.to_proto(export_scope=export_scope)


def _from_proto_fn(v, import_scope=None):
  """Creates Variable or ResourceVariable from VariableDef as needed."""
  if v.is_resource:
    return ResourceVariable.from_proto(v, import_scope=import_scope)
  return variables.Variable.from_proto(v, import_scope=import_scope)


ops.register_proto_function(
    ops.GraphKeys.GLOBAL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.TRAINABLE_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.LOCAL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.MODEL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)


def is_resource_variable(var):
  """"Returns True if `var` is to be considered a ResourceVariable."""
  return isinstance(var, ResourceVariable) or hasattr(
      var, "_should_act_as_resource_variable")
