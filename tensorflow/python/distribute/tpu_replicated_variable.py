# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""A Variable class that is replicated to logical cores for model parallelism."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import abc
import contextlib

from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
from tensorflow.python.distribute import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_tpu_partition_ops as tpu_partition_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable


def _on_device_update(update_fn, var, value, **kwargs):
  with ops.device(var.device):
    return update_fn(var, value, **kwargs)


class TPUReplicatedVariable(variables_lib.Variable):
  """Container for replicated `Variables` that are treated as a single variable.

  This class maintains a list of replicated variables that are stored on
  separate logic TPU devices. TF2XLA bridge accesses these variables as
  if they were a single variable.
  """

  def __init__(self, variables, name='TPUReplicatedVariable'):
    """Treats `variables` as a replicated list of `tf.Variable`s.

    Example:

    ```
    variables = [
      tf.Variable(..., shape=(10, 100), dtype=tf.float32),
      tf.Variable(..., shape=(10, 100), dtype=tf.float32),
      tf.Variable(..., shape=(10, 100), dtype=tf.float32),
      tf.Variable(..., shape=(10, 100), dtype=tf.float32),
    ]
    replicated_variable = TPUReplicatedVariable(variables)
    assert replicated_variable.shape.as_list() == [10, 100]
    ```

    Args:
      variables: A list of `ResourceVariable`s that comprise this replicated
        variable. Variables should not be shared between different
        `TPUReplicatedVariable` objects.
      name: String. Name of this container. Defaults to "TPUReplicatedVariable".
    """
    if not isinstance(variables, abc.Sequence) or not variables or any(
        not isinstance(v, variables_lib.Variable) for v in variables):
      raise TypeError('Argument `variables` should be a non-empty list of '
                      f'`variables.Variable`s. Received {variables}')

    if any(v.dtype != variables[0].dtype for v in variables):
      raise ValueError(
          'All elements in argument `variables` must have the same dtype. '
          f'Received dtypes: {[v.dtype for v in variables]}')

    if any(v.shape != variables[0].shape for v in variables):
      raise ValueError(
          'All elements in argument `variables` must have the same shape. '
          f'Received shapes: {[v.shape for v in variables]}')

    self._vars = variables
    self._name = name
    self._common_name = self._name.split(':')[0]
    self._cached_value = None

  def __iter__(self):
    """Return an iterable for accessing the underlying sharded variables."""
    return iter(self._vars)

  @property
  def name(self):
    """The name of this object. Used for checkpointing."""
    return self._name

  @property
  def dtype(self):
    """The dtype of all `Variable`s in this object."""
    return self._vars[0].dtype

  @property
  def is_initialized(self):
    return self._vars[0].is_initialized

  @property
  def trainable(self):
    return self._vars[0].trainable

  @property
  def device(self):
    """The device this variable is on."""
    return self._vars[0].device

  @contextlib.contextmanager
  def _handle_graph(self):
    with self.handle.graph.as_default():
      yield

  @contextlib.contextmanager
  def _assign_dependencies(self):
    if self._cached_value is not None:
      with ops.control_dependencies([self._cached_value]):
        yield
    else:
      yield

  @property
  def constraint(self):
    return self._vars[0].constraint

  @property
  def _in_graph_mode(self):
    return self._vars[0]._in_graph_mode  # pylint: disable=protected-access

  @property
  def _unique_id(self):
    return self._vars[0]._unique_id  # pylint: disable=protected-access

  @property
  def graph(self):
    return self._vars[0].graph

  @property
  def _shared_name(self):
    return self._common_name

  @property
  def synchronization(self):
    return variable_scope.VariableSynchronization.NONE

  @property
  def aggregation(self):
    return variable_scope.VariableAggregation.NONE

  @property
  def variables(self):
    """The list of `Variables`."""
    if save_context.in_save_context():
      return [self._vars[0]]
    return self._vars

  def _map_resources(self, save_options):
    """For implementing `Trackable`."""
    first_var = self._vars[0]
    obj_map, resource_map = first_var._map_resources(save_options)  # pylint:disable=protected-access
    for v in self._vars[1:]:
      obj_map[v] = obj_map[first_var]
      resource_map[v.handle] = resource_map[first_var.handle]
    obj_map[self] = obj_map[first_var]
    resource_map[self] = resource_map[first_var.handle]
    return obj_map, resource_map

  def _gather_saveables_for_saved_model(self):
    return {trackable.VARIABLE_VALUE_KEY: self._vars[0]}

  @property
  def shape(self):
    return self._vars[0].shape

  @property
  def handle(self):
    if save_context.in_save_context() or context.executing_eagerly():
      return self._vars[0].handle

    if tpu_util.enclosing_tpu_context() is None:
      raise NotImplementedError('TPUReplicatedVariable.handle is not available '
                                'outside tpu context or save context')
    else:
      with tpu_util.outside_or_skip_tpu_context():
        return xla_sharding.replicate(
            tpu_partition_ops.tpu_partitioned_input(
                [v.handle for v in self._vars], partition_dim=-1))

  def _read_variable_op(self):
    return gen_resource_variable_ops.read_variable_op(self.handle, self.dtype)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    # pylint: disable=protected-access
    if tpu_util.enclosing_tpu_context() is None:
      return self.read_value()
    else:
      return self._read_variable_op()

  def read_value(self):
    return self._vars[0].read_value()

  def _update(self, update_fn, value, **kwargs):
    """Converts the value to tensor and updates the variable list."""
    input_tensor = ops.convert_to_tensor(
        value, name='value_in_tensor', dtype=self.dtype)

    return control_flow_ops.group(
        *tuple(
            _on_device_update(update_fn, v, input_tensor, **kwargs)
            for v in self.variables))

  def assign(self, value, use_locking=False, name=None, read_value=True):
    if tpu_util.enclosing_tpu_context() is None or context.executing_eagerly():
      assign_fn = lambda var, *a, **ka: var.assign(*a, **ka)
      return self._update(
          assign_fn,
          value=value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_variable_op)(
              self,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    if tpu_util.enclosing_tpu_context() is None or context.executing_eagerly():
      assign_sub_fn = lambda var, *a, **ka: var.assign_sub(*a, **ka)
      return self._update(
          assign_sub_fn,
          value=value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_sub_variable_op)(
              self,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    if tpu_util.enclosing_tpu_context() is None or context.executing_eagerly():
      assign_add_fn = lambda var, *a, **ka: var.assign_add(*a, **ka)
      return self._update(
          assign_add_fn,
          value=value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_add_variable_op)(
              self,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)

  def __str__(self):
    debug_str = ',\n'.join(
        '  %d: %s' % (i, v) for i, v in enumerate(self._vars))
    return '%s:{\n%s\n}' % (self.__class__.__name__, debug_str)

  def __repr__(self):
    debug_repr = ',\n'.join(
        '  %d: %r' % (i, v) for i, v in enumerate(self._vars))
    return '%s:{\n%s\n}' % (self.__class__.__name__, debug_repr)


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_tpu_replicated_var(var,
                                          dtype=None,
                                          name=None,
                                          as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(TPUReplicatedVariable,
                                        _tensor_conversion_tpu_replicated_var)
