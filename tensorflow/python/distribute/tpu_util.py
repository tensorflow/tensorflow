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
"""Utility functions for TPU."""

import contextlib

from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.tpu import tpu_replication


def enclosing_tpu_context():
  """Returns the TPUReplicateContext, which exists inside a tpu.rewrite()."""
  return enclosing_tpu_context_and_graph()[0]


def enclosing_tpu_context_and_graph():
  """Returns the TPUReplicateContext which exists inside a tpu.rewrite(), and its associated graph."""
  graph = ops.get_default_graph()
  while graph is not None:
    ctx = graph._get_control_flow_context()  # pylint: disable=protected-access
    while ctx is not None:
      if isinstance(ctx, tpu_replication.TPUReplicateContext):
        return ctx, graph
      ctx = ctx.outer_context
    # This may be a FuncGraph due to defuns or v2 control flow. We need to
    # find the original graph with the XLAControlFlowContext.
    graph = getattr(graph, "outer_graph", None)
  return None, None


@contextlib.contextmanager
def outside_or_skip_tpu_context():
  """Returns a context manager that skips current enclosing context if there is any."""
  ctx, graph = enclosing_tpu_context_and_graph()
  if ctx is None:
    yield
  else:
    saved_context = graph._get_control_flow_context()  # pylint: disable=protected-access
    graph._set_control_flow_context(ctx.outer_context)  # pylint: disable=protected-access
    yield
    graph._set_control_flow_context(saved_context)  # pylint: disable=protected-access


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


def make_raw_assign_fn(raw_assign_fn, use_handle=True):
  """Wrap `raw_assign_fn` with the proper graph context and device scope.

  Args:
    raw_assign_fn: the function to be wrapped.
    use_handle: if True, the `raw_assign_fn` will be applied to the handle of a
      variable; otherwise it will be applied to the variable itself.

  Returns:
    The wrapped function.
  """

  def assign_fn(var, value, use_locking=False, name=None, read_value=True):
    del use_locking  # Unused.

    handle = var.handle if use_handle else var
    with _maybe_enter_graph(handle), _maybe_on_device(var):
      op = raw_assign_fn(
          handle, ops.convert_to_tensor(value, dtype=var.dtype), name=name)
      with ops.control_dependencies([op]):
        if read_value:
          return var._read_variable_op() if use_handle else var.read_value()  # pylint: disable=protected-access
        else:
          return op

  return assign_fn


def make_raw_scatter_xxx_fn(raw_scatter_xxx_fn):
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


class LazyVariableTracker(object):
  """Class to track uninitialized lazy variables."""

  def __init__(self):
    self._uninitialized_var_list = []

  def initialize_all(self):
    """Initialize all uninitialized lazy variables stored in scope."""

    def assign_function(uninitialized_var_list):
      for var in uninitialized_var_list:
        val = var._initial_value  # pylint: disable=protected-access
        packed_var = getattr(var, "_packed_var", None)
        handle = getattr(packed_var, "packed_handle", var.handle)

        with ops.device(handle.device):
          resource_variable_ops.AssignVariableOp(resource=handle, value=val)
      return constant_op.constant([])

    assign_tf_function = def_function.function(
        assign_function, autograph=False, jit_compile=False,)

    with ops.init_scope():
      if len(self._uninitialized_var_list) > 1:
        assign_tf_function(self._uninitialized_var_list)
      else:
        assign_function(self._uninitialized_var_list)

    self._uninitialized_var_list = []

  def add_uninitialized_var(self, var):
    self._uninitialized_var_list.append(var)


class TPUUninitializedVariable(resource_variable_ops.UninitializedVariable):
  """UninitializedVariable component for TPU.

  Sometimes user might assign (different values) to a single component of a
  mirrored TPU variable. Thus we need to initialize_all when the assign* or read
  is invoked on a single component.
  """

  def read_value(self):
    self._lazy_scope.initialize_all()
    return super().read_value()

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    self._lazy_scope.initialize_all()
    return super().assign_sub(
        delta, use_locking=use_locking, name=name, read_value=read_value
    )

  def assign(self, value, use_locking=None, name=None, read_value=True):
    self._lazy_scope.initialize_all()
    return super().assign(
        value, use_locking=use_locking, name=name, read_value=read_value
    )

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    self._lazy_scope.initialize_all()
    return super().assign_add(
        delta, use_locking=use_locking, name=name, read_value=read_value
    )
