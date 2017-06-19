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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import variables
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_resource_variable_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util import compat


class ResourceVariable(variables.Variable):
  """Variable based on resource handles.

  TODO(apassos): fill this out explaining the semantics and Variable
  compatibility when the API has settled more.

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
               import_scope=None):
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

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
    """
    if variable_def:
      if initial_value:
        raise ValueError("variable_def and initial_value are mutually "
                         "exclusive.")
      self._init_from_proto(variable_def, import_scope=import_scope)
    else:
      self._init_from_args(initial_value=initial_value,
                           trainable=trainable,
                           collections=collections,
                           validate_shape=validate_shape,
                           caching_device=caching_device,
                           name=name,
                           dtype=dtype)

  # pylint: disable=unused-argument
  def _init_from_args(self,
                      initial_value=None,
                      trainable=True,
                      collections=None,
                      validate_shape=True,
                      caching_device=None,
                      name=None,
                      dtype=None):

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

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
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
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    self._save_slice_info = None
    with ops.control_dependencies(None):
      with ops.name_scope(name, "Variable", [] if init_from_fn else
                          [initial_value]) as name:
        # pylint: disable=protected-access
        true_name = ops._name_from_scope_name(name)
        if init_from_fn:
          # Use attr_scope and device(None) to simulate the behavior of
          # colocate_with when the variable we want to colocate with doesn't
          # yet exist.
          attr = attr_value_pb2.AttrValue(
              list=attr_value_pb2.AttrValue.ListValue(
                  s=[compat.as_bytes("loc:@%s" % true_name)]))
          with ops.get_default_graph()._attr_scope({"_class": attr}):
            with ops.name_scope("Initializer"), ops.device(None):
              self._initial_value = ops.convert_to_tensor(
                  initial_value(), name="initial_value", dtype=dtype)
            self._handle = gen_resource_variable_ops.var_handle_op(
                shape=self._initial_value.get_shape(),
                dtype=self._initial_value.dtype.base_dtype,
                shared_name=true_name, name=name)
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
          self._handle = gen_resource_variable_ops.var_handle_op(
              shape=self._initial_value.get_shape(),
              dtype=self._initial_value.dtype.base_dtype,
              shared_name=true_name, name=name)

        self._dtype = self._initial_value.dtype.base_dtype

        with ops.name_scope("IsInitialized"):
          self._is_initialized_op = (
              gen_resource_variable_ops.var_is_initialized_op(self._handle))
        if initial_value is not None:
          with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
            self._initializer_op = gen_resource_variable_ops.assign_variable_op(
                self._handle, self._initial_value, name=n)
        with ops.name_scope("Read"), ops.colocate_with(self._handle):
          # Manually assign reads to the handle's device to avoid log messages.
          with ops.device(self._handle.device):
            value = gen_resource_variable_ops.read_variable_op(
                self._handle, dtype=self._dtype)
          self._graph_element = value
          if caching_device is not None:
            # Variables may be created in a tf.device() or ops.colocate_with()
            # context. At the same time, users would expect caching device to be
            # independent of this context, and/or would not expect the current
            # device context to be merged with the caching device spec.
            # Therefore we reset the colocation stack before creating the cached
            # value. Note that resetting the colocation stack will also reset
            # the device stack.
            with ops.colocate_with(None, ignore_existing=True):
              with ops.device(caching_device):
                self._cached_value = array_ops.identity(value)
          else:
            self._cached_value = None
          ops.add_to_collections(collections, self)

  def _init_from_proto(self, variable_def, import_scope=None):
    """Initializes from `VariableDef` proto."""
    assert isinstance(variable_def, variable_pb2.VariableDef)
    if not variable_def.is_resource:
      raise ValueError("Trying to restore Variable as ResourceVariable.")

    # Create from variable_def.
    g = ops.get_default_graph()
    self._handle = g.as_graph_element(
        ops.prepend_name_scope(variable_def.variable_name,
                               import_scope=import_scope))
    self._initializer_op = g.as_graph_element(
        ops.prepend_name_scope(variable_def.initializer_name,
                               import_scope=import_scope))
    if variable_def.snapshot_name:
      self._cached_value = g.as_graph_element(
          ops.prepend_name_scope(variable_def.snapshot_name,
                                 import_scope=import_scope))
    else:
      self._cached_value = None
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = variables.Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def)
    else:
      self._save_slice_info = None
    self._caching_device = None
    self._dtype = dtypes.as_dtype(self._handle.op.get_attr("dtype"))
    self._graph_element = self.value()

  @property
  def dtype(self):
    """The dtype of this variable."""
    return self._dtype

  @property
  def device(self):
    """The device this variable is on."""
    return self._handle.device

  @property
  def graph(self):
    """The `Graph` of this variable."""
    return self._handle.graph

  @property
  def name(self):
    """The name of the handle for this variable."""
    return self._handle.name

  def get_shape(self):
    """The shape of this variable."""
    return tensor_shape.TensorShape(self._handle.op.get_attr("shape"))

  @property
  def create(self):
    """The op responsible for initializing this variable."""
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
      with ops.device(self._handle.device):
        return gen_resource_variable_ops.read_variable_op(
            self._handle, dtype=self._dtype)

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
    return self._initial_value

  @property
  def op(self):
    """The op for this variable."""
    return self._handle.op

  def eval(self, session=None):
    """Evaluates and returns the value of this variable."""
    return self._graph_element.eval(session=session)

  def _set_save_slice_info(self, save_slice_info):
    """Sets the slice info for this `ResourceVariable`.

    Args:
      save_slice_info: A `Variable.SaveSliceInfo` object.
    """
    self._save_slice_info = save_slice_info

  def _get_save_slice_info(self):
    return self._save_slice_info

  def read_value(self):
    """Constructs an op which reads the value of this variable.

    Should be used when there are multiple reads, or when it is desirable to
    read the value only after some condition is true.

    Returns:
     the read operation.
    """
    with ops.name_scope("Read"):
      with ops.device(self._handle.device):
        value = gen_resource_variable_ops.read_variable_op(
            self._handle, dtype=self._dtype)
    # Return an identity so it can get placed on whatever device the context
    # specifies instead of the device where the variable is.
    return array_ops.identity(value)

  def sparse_read(self, indices, name=None):
    """Reads the value of this variable sparsely, using `gather`."""
    with ops.name_scope("Gather" if name is None else name) as name:
      value = gen_resource_variable_ops.resource_gather(
          self._handle, indices, dtype=self._dtype, name=name)
    return array_ops.identity(value)

  def to_proto(self, export_scope=None):
    """Converts a `ResourceVariable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    """
    if (export_scope is None or
        self.handle.name.startswith(export_scope)):
      var_def = variable_pb2.VariableDef()
      var_def.variable_name = ops.strip_name_scope(
          self.handle.name, export_scope)
      var_def.initializer_name = ops.strip_name_scope(
          self.initializer.name, export_scope)
      if self._cached_value is not None:
        var_def.snapshot_name = ops.strip_name_scope(
            self._cached_value.name, export_scope)
      var_def.is_resource = True
      if self._save_slice_info:
        var_def.save_slice_info_def.MergeFrom(self._save_slice_info.to_proto(
            export_scope=export_scope))
      return var_def
    else:
      return None

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    return ResourceVariable(variable_def=variable_def,
                            import_scope=import_scope)

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

    setattr(ResourceVariable, operator, _run_op)

  __array_priority__ = 100

  def assign_sub(self, delta, use_locking=None, name=None):
    # TODO(apassos): this here and below is not atomic. Consider making it
    # atomic if there's a way to do so without a performance cost for those who
    # don't need it.
    with ops.control_dependencies(
        [gen_resource_variable_ops.assign_sub_variable_op(
            self.handle,
            ops.convert_to_tensor(delta, dtype=self.dtype), name=name)]):
      return self.read_value()

  def assign_add(self, delta, use_locking=None, name=None):
    with ops.control_dependencies(
        [gen_resource_variable_ops.assign_add_variable_op(
            self.handle,
            ops.convert_to_tensor(delta, dtype=self.dtype), name=name)]):
      return self.read_value()

  def assign(self, value, use_locking=None, name=None):
    with ops.control_dependencies(
        [gen_resource_variable_ops.assign_variable_op(
            self.handle,
            ops.convert_to_tensor(value, dtype=self.dtype), name=name)]):
      return self.read_value()

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
    with ops.control_dependencies([gen_array_ops.resource_strided_slice_assign(
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
        shrink_axis_mask=shrink_axis_mask)]):
      return self.value()


# pylint: disable=unused-argument,protected-access
def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
  if dtype is not None and dtype != var.value().dtype:
    print("trying to switch the dtype to ", dtype, " from ", var.value().dtype)
    return NotImplemented
  if as_ref:
    return var.read_value().op.inputs[0]
  return var.value()
# pylint: enable=unused-argument,protected-access

# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.

# Note: registering for Variable after ResourceVariable because inheritance will
# otherwise lead to the wrong behavior.
ops.register_tensor_conversion_function(ResourceVariable, _dense_var_to_tensor)
ops.register_tensor_conversion_function(
    variables.Variable,
    variables.Variable._TensorConversionFunction)  # pylint: disable=protected-access

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
  # Walk graph back until the original handle is found.
  # TODO(apassos): more robust way of getting the shape.
  handle = op.inputs[0]
  while handle.op.type != "VarHandleOp":
    handle = handle.op.inputs[0]
  params_shape = ops.convert_to_tensor(
      tensor_shape.TensorShape(handle.op.get_attr("shape")))
  indices = op.inputs[1]
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[1:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return [ops.IndexedSlices(values, indices, params_shape), None]


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
