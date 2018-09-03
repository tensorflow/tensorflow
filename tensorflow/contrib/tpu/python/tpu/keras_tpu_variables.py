# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Distributed variable implementation for TPUs.

N.B. This is an experimental feature that should only be used for Keras support.

It is unsupported and will be removed in favor of Distribution Strategy soon.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging


@contextlib.contextmanager
def _handle_graph(handle):
  with handle.graph.as_default():
    yield


def _enclosing_tpu_context():
  # pylint: disable=protected-access
  context = ops.get_default_graph()._get_control_flow_context()
  # pylint: enable=protected-access
  while context is not None and not isinstance(
      context, control_flow_ops.XLAControlFlowContext):
    context = context.outer_context
  return context


class ReplicatedVariable(object):
  """A replicated variable for use on TPUs.

  When accessed inside a tpu.replicate() context, this variable acts as if it
  is a single variable whose handle is a replicated input to the computation.

  Outside a tpu.replicate() context currently this object has pretty murky
  semantics, especially with respect to things such as
  * initialization
  * colocation.
  """

  def __init__(self, name, variables):
    self._name = name
    self._primary_var = variables[0]
    self._vars = variables
    self._cached_value = None
    self._dtype = variables[0].dtype

  @property
  def handle(self):
    tpu_context = _enclosing_tpu_context()
    if tpu_context is None:
      return self._primary_var.handle

    return tpu_context.get_replicated_var_handle(self)

  @contextlib.contextmanager
  def _assign_dependencies(self):
    """Makes assignments depend on the cached value, if any.

    This prevents undefined behavior with reads not ordered wrt writes.

    Yields:
      None.
    """
    if self._cached_value is not None:
      with ops.control_dependencies([self._cached_value]):
        yield
    else:
      yield

  @property
  def initializer(self):
    return control_flow_ops.group([v.initializer for v in self._vars])

  @property
  def graph(self):
    return self._primary_var.graph

  @property
  def _shared_name(self):
    return self._common_name

  @property
  def _unique_id(self):
    return self._primary_var._unique_id  # pylint: disable=protected-access

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._primary_var.dtype

  @property
  def shape(self):
    return self._primary_var.shape

  def get_shape(self):
    return self._primary_var.get_shape()

  def to_proto(self, export_scope=None):
    return self._primary_var.to_proto(export_scope=export_scope)

  @property
  def constraint(self):
    return None

  @property
  def op(self):
    return self.get().op

  @property
  def is_tensor_like(self):
    return True

  def _read_variable_op(self):
    if _enclosing_tpu_context() is None:
      return self._primary_var.read_value()
    v = gen_resource_variable_ops.read_variable_op(self.handle, self._dtype)
    return v

  def read_value(self):
    return self._read_variable_op()

  def is_initialized(self, name=None):
    return self._vars[0].is_initialized(name=name)

  def __getitem__(self, *args):
    return self.read_value().__getitem__(*args)

  def assign(self, value, use_locking=None, name=None, read_value=False):
    """Assign `value` to all replicas.

    Outside of the tpu.rewrite context, assign explicitly to all replicas.
    Inside of the tpu.rewrite context, assigns to the local replica.

    Arguments:
      value: Tensor to assign
      use_locking: ignored
      name: ignored
      read_value: return the value from the assignment
    Returns:
      Assignment operation, or new value of the variable if `read_value` is True
    """
    del use_locking
    if _enclosing_tpu_context() is None:
      assign_ops = []
      with self._assign_dependencies():
        for var in self._vars:
          assign_ops.append(var.assign(value, use_locking=None, name=name))

        if read_value:
          with ops.control_dependencies(assign_ops):
            return self.read_value()
        else:
          return control_flow_ops.group(assign_ops)

    with _handle_graph(self.handle), self._assign_dependencies():
      value_tensor = ops.convert_to_tensor(value, dtype=self.dtype)
      assign_op = gen_resource_variable_ops.assign_variable_op(
          self.handle, value_tensor, name=name)
    if read_value:
      return self._read_variable_op()
    return assign_op

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    del use_locking
    with _handle_graph(self.handle), self._assign_dependencies():
      assign_add_op = gen_resource_variable_ops.assign_add_variable_op(
          self.handle,
          ops.convert_to_tensor(delta, dtype=self.dtype),
          name=name)
    if read_value:
      return self._read_variable_op()
    return assign_add_op

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    del use_locking
    with _handle_graph(self.handle), self._assign_dependencies():
      assign_sub_op = gen_resource_variable_ops.assign_sub_variable_op(
          self.handle,
          ops.convert_to_tensor(delta, dtype=self.dtype),
          name=name)
    if read_value:
      return self._read_variable_op()
    return assign_sub_op

  def get(self):
    return self._primary_var

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    # pylint: disable=protected-access
    if _enclosing_tpu_context() is None:
      return self._primary_var._dense_var_to_tensor(dtype, name, as_ref)
    # pylint: enable=protected-access
    if dtype is not None and dtype != self.dtype:
      return NotImplemented
    if as_ref:
      return self.handle
    else:
      return self.read_value()


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


def replicated_fetch_function(var):
  # pylint: disable=protected-access
  return ([var._dense_var_to_tensor()], lambda v: v[0])
  # pylint: enable=protected-access


ops.register_tensor_conversion_function(ReplicatedVariable, _tensor_conversion)
ops.register_dense_tensor_like_type(ReplicatedVariable)
session_lib.register_session_run_conversion_functions(
    ReplicatedVariable, replicated_fetch_function)


def replicated_scope(num_replicas):
  """Variable scope for constructing replicated variables."""

  def _replicated_variable_getter(getter, name, *args, **kwargs):
    """Getter that constructs replicated variables."""
    collections = kwargs.pop("collections", None)
    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    kwargs["collections"] = []

    logging.info("Constructing replicated variable %s", name)
    variables = []
    index = {}
    for i in range(num_replicas):
      replica_name = "{}/{}".format(name, i)
      with ops.device("device:TPU:{}".format(i)):
        v = getter(*args, name=replica_name, **kwargs)
        variables.append(v)
      index[i] = v
    result = ReplicatedVariable(name, variables)

    g = ops.get_default_graph()
    # If "trainable" is True, next_creator() will add the member variables
    # to the TRAINABLE_VARIABLES collection, so we manually remove
    # them and replace with the MirroredVariable. We can't set
    # "trainable" to False for next_creator() since that causes functions
    # like implicit_gradients to skip those variables.
    if kwargs.get("trainable", True):
      collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
      l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
      for v in index.values():
        if v in l:
          l.remove(v)
    g.add_to_collections(collections, result)

    return result

  return variable_scope.variable_scope(
      "", custom_getter=_replicated_variable_getter)
