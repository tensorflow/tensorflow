# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""A variable which packs a list of variables distributed across devices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops


class PackedDistributedVariable(resource_variable_ops.BaseResourceVariable):
  """A variable which packs multiple variables distributed across devices.

  It's only supported when eager execution is enabled.
  For op-by-op execution, use an unpacked handle on the current device; for
  function execution, use the packed handle to reduce the overhead of function
  calls.
  """

  def __init__(self, distributed_variables=None, name=None, **unused_kwargs):
    """Packs a list of variables which are distributed across devices.

    Args:
      distributed_variables: A list of distributed Variables to pack.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
    """
    if not ops.executing_eagerly_outside_functions():
      raise ValueError(
          "PackedDistributedVariable should be created in eager mode.")
    if not distributed_variables:
      raise ValueError("Expect a non-empty list of variables to pack.")
    for i, var in enumerate(distributed_variables):
      if not resource_variable_ops.is_resource_variable(var):
        raise ValueError("Expect a list of ResourceVariables to pack, "
                         "but the %d-th variable is %s" % (i, type(var)))

    self._distributed_variables = distributed_variables
    self._devices = [v.device for v in distributed_variables]
    with ops.init_scope():
      with ops.name_scope(name, "Variable", skip_on_eager=False) as name:
        handle = ops.pack_eager_tensors(
            [var.handle for var in distributed_variables])
        handle_name = ops.name_from_scope_name(name)
        unique_id = "%s_%d" % (handle_name, ops.uid())
        super(PackedDistributedVariable, self).__init__(
            trainable=distributed_variables[0].trainable,
            shape=distributed_variables[0].shape,
            dtype=distributed_variables[0].dtype,
            handle=handle,
            synchronization=distributed_variables[0].synchronization,
            constraint=distributed_variables[0].constraint,
            aggregation=distributed_variables[0].aggregation,
            distribute_strategy=distributed_variables[0]._distribute_strategy,  # pylint: disable=protected-access
            name=name,
            unique_id=unique_id,
            handle_name=handle_name,
            graph_element=None,
            initial_value=None,
            initializer_op=None,
            is_initialized_op=None,
            cached_value=None,
            caching_device=None,
            is_distributed_variables=True)

  @property
  def devices(self):
    return self._devices

  def on_device(self, device):
    return PackedVarAndDevice(self, device)

  def get_var_on_device(self, device):
    for i, d in enumerate(self._devices):
      if d == device:
        return self._distributed_variables[i]
    raise ValueError("Device %s is not found" % device)

  def get_var_on_current_device(self):
    current_device = device_util.canonicalize(device_util.current())
    return self.get_var_on_device(current_device)

  def initial_value(self, device):
    """Returns the Tensor used as the initial value for the variable."""
    return self.get_var_on_device(device).initial_value

  @property
  def handle(self):
    if context.executing_eagerly():
      return self.get_var_on_current_device().handle
    else:
      return self._handle

  @property
  def packed_handle(self):
    return self._handle

  def _read_variable_op(self):
    if context.executing_eagerly():
      return self.get_var_on_current_device().value()
    else:
      return super(PackedDistributedVariable, self)._read_variable_op()

  def value(self):
    return self._read_variable_op()

  def is_initialized(self, name=None):
    if context.executing_eagerly():
      result = self._distributed_variables[0].is_initialized()
      for v in self._distributed_variables[1:-1]:
        result = math_ops.logical_and(result, v.is_initialized())
      result = math_ops.logical_and(
          result, self._distributed_variables[-1].is_initialized(), name=name)
    else:
      with ops.device(self._devices[0]):
        result = super(PackedDistributedVariable, self).is_initialized(name)
      for d in self._devices[1:-1]:
        with ops.device(d):
          initialized = super(PackedDistributedVariable,
                              self).is_initialized(name)
        result = math_ops.logical_and(result, initialized)
      with ops.device(self._devices[-1]):
        initialized = super(PackedDistributedVariable,
                            self).is_initialized(name)
      result = math_ops.logical_and(result, initialized, name=name)
    return result

  def _update(self, update_fn, value, **kwargs):
    if context.executing_eagerly():
      return update_fn(self.get_var_on_current_device(), value, **kwargs)
    else:
      return update_fn(super(PackedDistributedVariable, self), value, **kwargs)

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
    return self._update(
        update_fn=assign_sub_fn,
        value=delta,
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    assign_add_fn = lambda var, *a, **kw: var.assign_add(*a, **kw)
    return self._update(
        update_fn=assign_add_fn,
        value=delta,
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  def assign(self, value, use_locking=None, name=None, read_value=True):
    assign_fn = lambda var, *a, **kw: var.assign(*a, **kw)
    return self._update(
        update_fn=assign_fn,
        value=value,
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    scatter_sub_fn = lambda var, *a, **kw: var.scatter_sub(*a, **kw)
    return self._update(
        update_fn=scatter_sub_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    scatter_add_fn = lambda var, *a, **kw: var.scatter_add(*a, **kw)
    return self._update(
        update_fn=scatter_add_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    scatter_mul_fn = lambda var, *a, **kw: var.scatter_mul(*a, **kw)
    return self._update(
        update_fn=scatter_mul_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    scatter_div_fn = lambda var, *a, **kw: var.scatter_div(*a, **kw)
    return self._update(
        update_fn=scatter_div_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    scatter_min_fn = lambda var, *a, **kw: var.scatter_min(*a, **kw)
    return self._update(
        update_fn=scatter_min_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    scatter_max_fn = lambda var, *a, **kw: var.scatter_max(*a, **kw)
    return self._update(
        update_fn=scatter_max_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    scatter_update_fn = lambda var, *a, **kw: var.scatter_update(*a, **kw)
    return self._update(
        update_fn=scatter_update_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    if context.executing_eagerly():
      return self.get_var_on_current_device()._dense_var_to_tensor(  # pylint: disable=protected-access
          dtype=dtype,
          name=name,
          as_ref=as_ref)
    else:
      return super(PackedDistributedVariable, self)._dense_var_to_tensor(  # pylint: disable=protected-access
          dtype=dtype,
          name=name,
          as_ref=as_ref)


class PackedVarAndDevice(object):
  """Holds a packed distributed variable and a device."""

  def __init__(self, var, device):
    self._var = var
    self._device = device

  def __getattr__(self, name):
    return getattr(self._var, name)

  def var(self):
    return self._var

  def value(self):
    with ops.device(self._device):
      return self._var.value()

  def read_value(self):
    with ops.device(self._device):
      return self._var.read_value()

  @property
  def initial_value(self):
    return self._var.initial_value(self._device)

  def initialized_value(self):
    with ops.device(self._device):
      return self._var.initialized_value()

  @property
  def device(self):
    return self._device

  @property
  def handle(self):
    with ops.device(self._device):
      return self._var.handle

  def on_device_handle(self):
    with ops.device(self._device):
      return self._var.get_var_on_current_device().handle

  @property
  def op(self):
    with ops.device(self._device):
      return self._var.op

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    with ops.device(self._device):
      return self._var.assign_sub(delta, use_locking, name, read_value)

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    with ops.device(self._device):
      return self._var.assign_add(delta, use_locking, name, read_value)

  def assign(self, value, use_locking=None, name=None, read_value=True):
    with ops.device(self._device):
      return self._var.assign(value, use_locking, name, read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    with ops.device(self._device):
      return self._var.scatter_sub(sparse_delta, use_locking, name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    with ops.device(self._device):
      return self._var.scatter_add(sparse_delta, use_locking, name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    with ops.device(self._device):
      return self._var.scatter_mul(sparse_delta, use_locking, name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    with ops.device(self._device):
      return self._var.scatter_div(sparse_delta, use_locking, name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    with ops.device(self._device):
      return self._var.scatter_min(sparse_delta, use_locking, name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    with ops.device(self._device):
      return self._var.scatter_max(sparse_delta, use_locking, name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    with ops.device(self._device):
      return self._var.scatter_update(sparse_delta, use_locking, name)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    with ops.device(self._device):
      return self._var._dense_var_to_tensor(  # pylint: disable=protected-access
          dtype=dtype,
          name=name,
          as_ref=as_ref)

  def _as_graph_element(self):
    return self._var._as_graph_element()  # pylint: disable=protected-access


def _tensor_conversion_packed_var_and_device(var,
                                             dtype=None,
                                             name=None,
                                             as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


ops.register_tensor_conversion_function(
    PackedVarAndDevice, _tensor_conversion_packed_var_and_device)
