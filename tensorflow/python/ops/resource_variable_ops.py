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
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_resource_variable_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util import compat


def _register_variable_read(read, collections, trainable):
  """Helper function to put a read from a variable in the collections."""
  if collections is None:
    collections = []
  if (trainable and ops.GraphKeys.TRAINABLE_VARIABLES
       not in collections):
    collections = (list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES])
    ops.add_to_collections(collections, read)


class ResourceVariable(object):
  """Variable based on resource handles.

  TODO(apassos): fill this out explaining the semantics and Variable
  compatibility when the API has settled more.

  """

  # pylint: disable=unused-argument
  def __init__(self,
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
        if init_from_fn:
          # Use attr_scope and device(None) to simulate the behavior of
          # colocate_with when the variable we want to colocate with doesn't
          # yet exist.
          # pylint: disable=protected-access
          true_name = ops._name_from_scope_name(name)
          attr = attr_value_pb2.AttrValue(
              list=attr_value_pb2.AttrValue.ListValue(
                  s=[compat.as_bytes("loc:@%s" % true_name)]))
          # pylint: disable=protected-access
          with ops.get_default_graph()._attr_scope({"_class": attr}):
            with ops.name_scope("Initializer"), ops.device(None):
              self._initial_value = ops.convert_to_tensor(
                  initial_value(), name="initial_value", dtype=dtype)
            self._handle = gen_resource_variable_ops.var_handle_op(
                shape=self._initial_value.get_shape(),
                dtype=self._initial_value.dtype.base_dtype,
                shared_name=name, name=name)

        # Or get the initial value from a Tensor or Python object.
        else:
          self._initial_value = ops.convert_to_tensor(
              initial_value, name="initial_value", dtype=dtype)
          self._handle = gen_resource_variable_ops.var_handle_op(
              shape=self._initial_value.get_shape(),
              dtype=self._initial_value.dtype.base_dtype,
              shared_name=name, name=name)

        self._dtype = self._initial_value.dtype.base_dtype

        with ops.name_scope("IsInitialized"):
          self._is_initialized_op = (
              gen_resource_variable_ops.var_is_initialized_op(self._handle))
        if initial_value is not None:
          with ops.name_scope("Create"):
            self._initialize_op = gen_resource_variable_ops.assign_variable_op(
                self._handle, self._initial_value)

        with ops.name_scope("Read"):
          self._value = gen_resource_variable_ops.read_variable_op(
              self._handle, dtype=self._dtype)
          if caching_device is not None:
            with ops.device(caching_device):
              self._cached_value = array_ops.identity(self._value)
          else:
            with ops.colocate_with(self._handle.op):
              self._cached_value = array_ops.identity(self._value)
          # TODO(apassos) this is terrible monkey-patching required to make
          # initialize_all_variables work. Replace self._value with an explicit
          # class instead of monkey-patching.
          self._value.initializer = self._initialize_op
          ops.add_to_collections(collections, self)

  @property
  def dtype(self):
    """The dtype of this variable."""
    return self._dtype

  @property
  def device(self):
    """The device this variable is on."""
    return self._handle.device

  @property
  def name(self):
    """The name of the handle for this variable."""
    return self._handle.name

  def get_shape(self):
    """The shape of this variable."""
    return self._value.get_shape()

  @property
  def create(self):
    """The op responsible for initializing this variable."""
    return self._initialize_op

  @property
  def handle(self):
    """The handle by which this variable can be accessed."""
    return self._handle

  def value(self):
    """A cached operation which reads the value of this variable."""
    return self._cached_value

  def _as_graph_element(self):
    """Conversion function for Graph.as_graph_element()."""
    return self._value

  @property
  def initializer(self):
    """The op responsible for initializing this variable."""
    return self._initialize_op

  @property
  def op(self):
    """The op for this variable."""
    return self._handle.op

  def eval(self, session=None):
    """Evaluates and returns the value of this variable."""
    return self._value.eval(session=session)

  def _set_save_slice_info(self, save_slice_info):
    """Sets the slice info for this `ResourceVariable`.

    Args:
      save_slice_info: A `Variable.SaveSliceInfo` object.
    """
    self._save_slice_info = save_slice_info

  def _get_save_slice_info(self):
    return self._save_slice_info

  def read_value(self, collections=None, trainable=True):
    """Constructs an op which reads the value of this variable.

    Should be used when there are multiple reads, or when it is desirable to
    read the value only after some condition is true.

    Args:
     collections: any collections in which this operation should be inserted.
     trainable: whether this read is to be used for training.

    Returns:
     the read operation.
    """
    with ops.name_scope("Read"):
      value = gen_resource_variable_ops.read_variable_op(
          self._handle, dtype=self._dtype)
    _register_variable_read(value, collections=collections, trainable=trainable)
    return array_ops.identity(value)

  def sparse_read(self, indices, collections=None, trainable=True, name=None):
    """Reads the value of this variable sparsely, using `gather`."""
    with ops.name_scope("Gather" if name is None else name) as name:
      value = gen_resource_variable_ops.resource_gather(
          self._handle, indices, dtype=self._dtype, name=name)
    _register_variable_read(value, collections=collections, trainable=trainable)
    return array_ops.identity(value)

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


# pylint: disable=unused-argument,protected-access
def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
  if dtype is not None and dtype != var.value().dtype:
    print("trying to switch the dtype to ", dtype, " from ", var.value().dtype)
    return NotImplemented
  if as_ref:
    return var._value
  return var._cached_value
# pylint: enable=unused-argument,protected-access

# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
ops.register_tensor_conversion_function(ResourceVariable, _dense_var_to_tensor)

# pylint: disable=protected-access
ResourceVariable._OverloadAllOperators()
ops.register_dense_tensor_like_type(ResourceVariable)
