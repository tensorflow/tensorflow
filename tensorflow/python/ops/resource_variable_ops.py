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

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resources
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_resource_variable_ops import *
# pylint: enable=wildcard-import


def _register_variable_read(read, collections, trainable):
  """Helper function to put a read from a variable in the collections."""
  if collections is None:
    collections = [ops.GraphKeys.GLOBAL_VARIABLES]
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
               name=None,
               caching_device=None,
               trainable=True,
               collections=None,
               dtype=None,
               shape=None):

    """Creates a variable.

    Args:
      initial_value: A `Tensor` or Python object convertible to a `Tensor`
        representing the initial value of this variable.
      name: The name of this variable. Automatically uniquified.
      caching_device: device where the variable value's read by default.
      trainable: Whether the global read of this variable will be used for
        training.
      collections: Additional collections to which the `read` operation for
        this variable is to be added. Defaults to [].
      dtype: The type of this variable. Can be omitted if it can be deduced
        from the initial_value. If different from the type of the initial
        value it will be cast to this type.
      shape: The shape of this variable. Only specify if there is no initial
        value but shape inference is desired.
    """
    if initial_value is not None:
      if callable(initial_value):
        initial_value = initial_value()
      initial_value = ops.convert_to_tensor(initial_value)
    if dtype is None:
      assert initial_value is not None, ("Trying to create a resource variable "
                                         "with no dtype or initial value. At"
                                         " least one of these must be set.")
      dtype = initial_value.dtype
    elif initial_value is not None:
      initial_value = math_ops.cast(initial_value, dtype)
    if shape is None:
      if initial_value is not None:
        shape = initial_value.get_shape().as_proto()
      else:
        shape = tensor_shape.unknown_shape()
    else:
      shape = tensor_shape.as_shape(shape)

    self._dtype = dtype
    with ops.name_scope(name, "Variable", [initial_value]) as name:
      self._handle = gen_resource_variable_ops.var_handle_op(shared_name=name,
                                                             name=name,
                                                             dtype=dtype,
                                                             shape=shape)

      with ops.name_scope("IsInitialized"):
        self._is_initialized_op = (
            gen_resource_variable_ops.var_is_initialized_op(self._handle))
      if initial_value is not None:
        with ops.name_scope("Create"):
          self._initialize_op = gen_resource_variable_ops.assign_variable_op(
              self._handle, initial_value)
        resources.register_resource(self._handle,
                                    self._initialize_op,
                                    self._is_initialized_op)

      with ops.name_scope("Read"):
        if caching_device is not None:
          with ops.device(caching_device):
            self._value = gen_resource_variable_ops.read_variable_op(
                self._handle, dtype=self._dtype)
        else:
          self._value = gen_resource_variable_ops.read_variable_op(
              self._handle, dtype=self._dtype)
        # TODO(apassos) this is terrible
        self._value.initializer = self._initialize_op
      _register_variable_read(
          self._value, trainable=trainable, collections=collections)

  @property
  def dtype(self):
    """The dtype of this variable."""
    return self._dtype

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

  @property
  def value(self):
    """A cached operation which reads the value of this variable."""
    return self._value

  def _as_graph_element(self):
    """Conversion function for Graph.as_graph_element()."""
    return self._value

  @property
  def initializer(self):
    """The op responsible for initializing this variable."""
    return self._initialize_op

  @property
  def op(self):
    """The op which reads the value of this variable."""
    return self._value.op

  def eval(self, session=None):
    """Evaluates and returns the value of this variable."""
    return self._value.eval(session=session)

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
    return value

  def sparse_read(self, indices, collections=None, trainable=True, name=None):
    """Reads the value of this variable sparsely, using `gather`."""
    with ops.name_scope("Gather" if name is None else name):
      value = gen_resource_variable_ops.resource_gather(
          self._handle, indices, dtype=self._dtype)
    _register_variable_read(value, collections=collections, trainable=trainable)
    return value


# pylint: disable=unused-argument
def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
  if dtype is not None and dtype != var.value.dtype:
    print("trying to switch the dtype to ", dtype, " from ", var.value.dtype)
    return NotImplemented
  return var.value

# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
ops.register_tensor_conversion_function(ResourceVariable, _dense_var_to_tensor)
