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
"""Various classes representing TPU distributed values.

Note that the tests are in values_test.py .

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


@contextlib.contextmanager
def _maybe_enter_graph(tensor):
  # Note: might have an eager tensor but not be executing eagerly when
  # building functions.
  if (context.executing_eagerly() or isinstance(tensor, ops.EagerTensor) or
      ops.has_default_graph()):
    yield
  else:
    with tensor.graph.as_default():
      yield


@contextlib.contextmanager
def _maybe_on_device(var):
  # Add a device scope for packed variables.
  if isinstance(var, packed.PackedVarAndDevice):
    with ops.device(var.device):
      yield
  else:
    yield


def _make_raw_assign_fn(raw_assign_fn):  # pylint: disable=missing-docstring

  def assign_fn(var, value, use_locking=False, name=None, read_value=True):  # pylint: disable=missing-docstring
    del use_locking  # Unused.

    handle = var.handle
    with _maybe_enter_graph(handle), _maybe_on_device(var):
      op = raw_assign_fn(
          handle, ops.convert_to_tensor(value, dtype=var.dtype), name=name)
      with ops.control_dependencies([op]):
        return var._read_variable_op() if read_value else op  # pylint: disable=protected-access

  return assign_fn


_scatter_error_msg = ("{op_name} is only supported for distributed "
                      "variable (variable created within certain "
                      "`tf.distribute.Strategy` scope) with NONE "
                      " aggregation, got: {aggregation}.")


def _make_raw_scatter_xxx_fn(raw_scatter_xxx_fn):
  """Wrap `raw_scatter_xxx_fn` so that it can be called w/ and w/o packed handle."""

  def scatter_xxx_fn(var, sparse_delta, use_locking=False, name=None):  # pylint: disable=missing-docstring
    del use_locking  # Unused.

    handle = var.handle
    with _maybe_enter_graph(handle), _maybe_on_device(var):
      op = raw_scatter_xxx_fn(
          handle,
          sparse_delta.indices,
          ops.convert_to_tensor(sparse_delta.values, var.dtype),
          name=name)
      with ops.control_dependencies([op]):
        return var._read_variable_op()  # pylint: disable=protected-access

  return scatter_xxx_fn


class TPUVariableMixin(object):
  """Mixin for TPU variables."""

  def __init__(self, *args, **kwargs):
    super(TPUVariableMixin, self).__init__(*args, **kwargs)

    # Handle ID is needed for `get_replicated_var_handle` to cache the variables
    # correctly since in eager mode different variables can have the same name.
    if ops.executing_eagerly_outside_functions():
      self._handle_id = self._common_name + "_" + str(id(self._primary))
    else:
      self._handle_id = self._common_name

  def __getattr__(self, name):
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self).__getattr__(name)
    else:
      raise AttributeError(
          "'{}' not accessible within a TPU context.".format(name))

  def get(self):
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self).get()
    else:
      raise NotImplementedError(
          "`TPUVariableMixin.get()` is not supported within a TPU context.")

  def _get_as_operand(self):
    return self.read_value()

  def _is_mirrored(self):
    raise NotImplementedError(
        "`TPUVariableMixin._is_mirrored()` must be implemented by subclasses.")

  @property
  def handle(self):
    """The handle by which this variable can be accessed."""
    # If we're in a tpu.rewrite(), return the replicated handle.
    tpu_context = tpu_util.enclosing_tpu_context()
    if tpu_context is None or context.executing_eagerly():
      var = self._get_on_device_or_primary()
      if isinstance(var, packed.PackedVarAndDevice):
        return var.on_device_handle()
      else:
        return var.handle
    else:
      is_packed = self._packed_var is not None
      val = self._values
      if is_packed:
        val = [self._packed_var]

      return tpu_context.get_replicated_var_handle(self._handle_id, val,
                                                   self._is_mirrored(),
                                                   is_packed)

  @property
  def device(self):
    return self.handle.device

  def _read_variable_op(self):
    """Reads the value of this variable."""
    if self.trainable:
      tape.variable_accessed(self)

    handle = self.handle
    if getattr(handle, "is_packed", False):
      # Add a device scope for a packed variable handle.
      with ops.device(self._get_on_device_or_primary().device):
        return gen_resource_variable_ops.read_variable_op(handle, self.dtype)
    else:
      return gen_resource_variable_ops.read_variable_op(handle, self.dtype)

  def read_value(self):
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self).read_value()
    else:
      return self._read_variable_op()

  def value(self):
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self).value()
    else:
      return self._read_variable_op()

  def _as_graph_element(self):
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self)._as_graph_element()  # pylint: disable=protected-access
    else:
      return None

  @property
  def op(self):
    if values_util.is_saving_non_distributed():
      return self._primary.op
    return values.DistributedVarOp(self._primary.op.name,
                                   self._primary.op.graph,
                                   self._primary.op.traceback,
                                   self._primary.op.type)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    # pylint: disable=protected-access
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self)._dense_var_to_tensor(
          dtype=dtype, name=name, as_ref=as_ref)
    # pylint: enable=protected-access
    elif dtype is not None and dtype != self.dtype:
      return math_ops.cast(self.read_value(), dtype)
    else:
      return self.handle if as_ref else self.read_value()


class TPUDistributedVariable(TPUVariableMixin, values.DistributedVariable):
  """DistributedVariable subclass for TPUStrategy."""

  def _is_mirrored(self):
    return self._policy._is_mirrored()  # pylint: disable=protected-access

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    if values_util.is_saving_non_distributed():
      return self._primary.assign_sub(value, use_locking, name, read_value)
    return self._policy.assign_sub(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    if values_util.is_saving_non_distributed():
      return self._primary.assign_add(value, use_locking, name, read_value)
    return self._policy.assign_add(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign(self, value, use_locking=False, name=None, read_value=True):
    if values_util.is_saving_non_distributed():
      return self._primary.assign(value, use_locking, name, read_value)
    return self._policy.assign(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_sub(sparse_delta, use_locking, name)
    return self._policy.scatter_sub(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_add(sparse_delta, use_locking, name)
    return self._policy.scatter_add(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_mul(sparse_delta, use_locking, name)
    return self._policy.scatter_mul(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_div(sparse_delta, use_locking, name)
    return self._policy.scatter_div(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_min(sparse_delta, use_locking, name)
    return self._policy.scatter_min(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_max(sparse_delta, use_locking, name)
    return self._policy.scatter_max(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_update(sparse_delta, use_locking, name)
    return self._policy.scatter_update(
        self, sparse_delta, use_locking=use_locking, name=name)


class TPUMirroredVariable(TPUVariableMixin, values.MirroredVariable):
  """Holds a map from replica to TPU variables whose values are kept in sync."""

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    if (tpu_util.enclosing_tpu_context() and
        self.aggregation == variable_scope.VariableAggregation.NONE):
      return _make_raw_assign_fn(
          gen_resource_variable_ops.assign_sub_variable_op)(
              self,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)
    return assign_sub(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    if (tpu_util.enclosing_tpu_context() and
        self.aggregation == variable_scope.VariableAggregation.NONE):
      return _make_raw_assign_fn(
          gen_resource_variable_ops.assign_add_variable_op)(
              self,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)
    return assign_add(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign(self, value, use_locking=False, name=None, read_value=True):
    if (tpu_util.enclosing_tpu_context() and
        self.aggregation == variable_scope.VariableAggregation.NONE):
      return _make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(
          self,
          value=value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    return assign(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def scatter_sub(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_sub(*args, **kwargs)
    raise NotImplementedError

  def scatter_add(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_add(*args, **kwargs)
    raise NotImplementedError

  def scatter_max(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_max(*args, **kwargs)
    raise NotImplementedError

  def scatter_min(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_min(*args, **kwargs)
    raise NotImplementedError

  def scatter_mul(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_mul(*args, **kwargs)
    raise NotImplementedError

  def scatter_div(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_div(*args, **kwargs)
    raise NotImplementedError

  def scatter_update(self, *args, **kwargs):
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_update(*args, **kwargs)
    raise NotImplementedError

  def _is_mirrored(self):
    return True


class TPUSyncOnReadVariable(TPUVariableMixin, values.SyncOnReadVariable):
  """Holds a map from replica to variables whose values are reduced on save."""

  def assign_sub(self, *args, **kwargs):
    if tpu_util.enclosing_tpu_context() is None:
      return values.SyncOnReadVariable.assign_sub(self, *args, **kwargs)
    else:
      return _make_raw_assign_fn(
          gen_resource_variable_ops.assign_sub_variable_op)(self, *args,
                                                            **kwargs)

  def assign_add(self, *args, **kwargs):
    if tpu_util.enclosing_tpu_context() is None:
      return values.SyncOnReadVariable.assign_add(self, *args, **kwargs)
    else:
      return _make_raw_assign_fn(
          gen_resource_variable_ops.assign_add_variable_op)(self, *args,
                                                            **kwargs)

  def assign(self, *args, **kwargs):
    if tpu_util.enclosing_tpu_context() is None:
      return values.SyncOnReadVariable.assign(self, *args, **kwargs)
    else:
      return _make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(
          self, *args, **kwargs)

  def _is_mirrored(self):
    return False


# Common method between OnWrite and Mirrored variables.
def assign_sub(var, value, use_locking=False, name=None, read_value=True):
  assign_sub_fn = _make_raw_assign_fn(
      gen_resource_variable_ops.assign_sub_variable_op)
  return var._update(  # pylint: disable=protected-access
      update_fn=assign_sub_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)


def assign_add(var, value, use_locking=False, name=None, read_value=True):
  assign_add_fn = _make_raw_assign_fn(
      gen_resource_variable_ops.assign_add_variable_op)
  return var._update(  # pylint: disable=protected-access
      update_fn=assign_add_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)


def assign(var, value, use_locking=False, name=None, read_value=True):
  assign_fn = _make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)
  return var._update(  # pylint: disable=protected-access
      update_fn=assign_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)


class TPUOnWritePolicy(values.OnWritePolicy):
  """Policy defined for `tf.VariableSynchronization.ON_WRITE` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.AUTO` or `tf.VariableSynchronization.ON_WRITE`.
  """

  def assign_sub(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    if (tpu_util.enclosing_tpu_context() and
        var.aggregation == variable_scope.VariableAggregation.NONE):
      return _make_raw_assign_fn(
          gen_resource_variable_ops.assign_sub_variable_op)(
              var,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)
    return assign_sub(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    if (tpu_util.enclosing_tpu_context() and
        var.aggregation == variable_scope.VariableAggregation.NONE):
      return _make_raw_assign_fn(
          gen_resource_variable_ops.assign_add_variable_op)(
              var,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)
    return assign_add(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign(self, var, value, use_locking=False, name=None, read_value=True):
    if (tpu_util.enclosing_tpu_context() and
        var.aggregation == variable_scope.VariableAggregation.NONE):
      return _make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(
          var,
          value=value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    return assign(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def _scatter_xxx(self,
                   raw_scater_xxx_fn,
                   op_name,
                   var,
                   sparse_delta,
                   use_locking=False,
                   name=None):
    scater_xxx_fn = _make_raw_scatter_xxx_fn(raw_scater_xxx_fn)
    if tpu_util.enclosing_tpu_context():
      if self._aggregation != variable_scope.VariableAggregation.NONE:
        raise NotImplementedError(
            _scatter_error_msg.format(
                op_name=op_name, aggregation=self._aggregation))
      return scater_xxx_fn(
          var, sparse_delta=sparse_delta, use_locking=use_locking, name=name)
    else:
      return var._update(  # pylint: disable=protected-access
          update_fn=scater_xxx_fn,
          value=sparse_delta,
          use_locking=use_locking,
          name=name)

  def scatter_sub(self, var, sparse_delta, use_locking=False, name=None):
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_sub,
                             "scatter_sub", var, sparse_delta, use_locking,
                             name)

  def scatter_add(self, var, sparse_delta, use_locking=False, name=None):
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_add,
                             "scatter_add", var, sparse_delta, use_locking,
                             name)

  def scatter_max(self, var, sparse_delta, use_locking=False, name=None):
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_max,
                             "scatter_max", var, sparse_delta, use_locking,
                             name)

  def scatter_min(self, var, sparse_delta, use_locking=False, name=None):
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_min,
                             "scatter_min", var, sparse_delta, use_locking,
                             name)

  def scatter_mul(self, var, sparse_delta, use_locking=False, name=None):
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_mul,
                             "scatter_mul", var, sparse_delta, use_locking,
                             name)

  def scatter_div(self, var, sparse_delta, use_locking=False, name=None):
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_div,
                             "scatter_div", var, sparse_delta, use_locking,
                             name)

  def scatter_update(self, var, sparse_delta, use_locking=False, name=None):
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_update,
                             "scatter_update", var, sparse_delta, use_locking,
                             name)

  def _is_mirrored(self):
    return True


class TPUOnReadPolicy(values.OnReadPolicy):
  """Policy defined for `tf.VariableSynchronization.ON_READ` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.ON_READ` and `aggregation` is set to any of the
  values allowed by the `tf.VariableAggregation` enum such as `NONE`, `SUM`,
  `MEAN` or `ONLY_FIRST_REPLICA`when creating a `tf.Variable` in `tf.distribute`
  scope.
  """

  def assign_sub(self, var, *args, **kwargs):
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUOnReadPolicy, self).assign_sub(var, *args, **kwargs)
    else:
      return _make_raw_assign_fn(
          gen_resource_variable_ops.assign_sub_variable_op)(var, *args,
                                                            **kwargs)

  def assign_add(self, var, *args, **kwargs):
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUOnReadPolicy, self).assign_add(var, *args, **kwargs)
    else:
      return _make_raw_assign_fn(
          gen_resource_variable_ops.assign_add_variable_op)(var, *args,
                                                            **kwargs)

  def assign(self, var, *args, **kwargs):
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUOnReadPolicy, self).assign(var, *args, **kwargs)
    else:
      return _make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(
          var, *args, **kwargs)

  def _is_mirrored(self):
    return False

  def scatter_sub(self, *args, **kwargs):
    raise NotImplementedError

  def scatter_add(self, *args, **kwargs):
    raise NotImplementedError

  def scatter_max(self, *args, **kwargs):
    raise NotImplementedError

  def scatter_min(self, *args, **kwargs):
    raise NotImplementedError

  def scatter_mul(self, *args, **kwargs):
    raise NotImplementedError

  def scatter_div(self, *args, **kwargs):
    raise NotImplementedError

  def scatter_update(self, *args, **kwargs):
    raise NotImplementedError
