# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Variables. See the @{$python/state_ops} guide.

@@AUTO_REUSE
@@IndexedSlices
@@Saver
@@Variable
@@VariableScope
@@all_variables
@@assert_variables_initialized
@@assign
@@assign_add
@@assign_sub
@@constant_initializer
@@export_meta_graph
@@fixed_size_partitioner
@@get_checkpoint_state
@@get_local_variable
@@get_variable
@@get_variable_scope
@@global_variables
@@global_variables_initializer
@@glorot_normal_initializer
@@glorot_uniform_initializer
@@import_meta_graph
@@initialize_all_tables
@@initialize_all_variables
@@initialize_local_variables
@@initialize_variables
@@is_variable_initialized
@@latest_checkpoint
@@local_variables
@@local_variables_initializer
@@make_template
@@min_max_variable_partitioner
@@model_variables
@@moving_average_variables
@@no_regularizer
@@ones_initializer
@@orthogonal_initializer
@@random_normal_initializer
@@random_uniform_initializer
@@report_uninitialized_variables
@@scatter_add
@@scatter_div
@@scatter_mul
@@scatter_nd_add
@@scatter_nd_sub
@@scatter_nd_update
@@scatter_sub
@@scatter_update
@@sparse_mask
@@tables_initializer
@@trainable_variables
@@truncated_normal_initializer
@@uniform_unit_scaling_initializer
@@update_checkpoint_state
@@variable_axis_size_partitioner
@@variable_op_scope
@@variable_scope
@@variables_initializer
@@variance_scaling_initializer
@@zeros_initializer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_state_ops import *
# pylint: enable=wildcard-import


# pylint: disable=protected-access,g-doc-return-or-yield,g-doc-args
def variable_op(shape, dtype, name="Variable", set_shape=True, container="",
                shared_name=""):
  """Deprecated. Used variable_op_v2 instead."""
  if not set_shape:
    shape = tensor_shape.unknown_shape()
  ret = gen_state_ops._variable(shape=shape, dtype=dtype, name=name,
                                container=container, shared_name=shared_name)
  # TODO(mrry): Move this to where it is used, so we can get rid of this op
  #   wrapper?
  if set_shape:
    ret.set_shape(shape)
  return ret


def variable_op_v2(shape, dtype, name="Variable", container="", shared_name=""):
  """Create a variable Operation.

  See also variables.Variable.

  Args:
    shape: The shape of the tensor managed by this variable
    dtype: The underlying type of the tensor values.
    name: optional name to use for the variable op.
    container: An optional string. Defaults to "".
      If non-empty, this variable is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional string. Defaults to "".
      If non-empty, this variable is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.

  Returns:
    A variable tensor.
  """
  return gen_state_ops._variable_v2(shape=shape,
                                    dtype=dtype,
                                    name=name,
                                    container=container,
                                    shared_name=shared_name)


def init_variable(v, init, name="init"):
  """Initializes variable with "init".

  This op does the following:
  if init is a Tensor, v = init
  if callable(init): v = init(VariableShape(v), v.dtype)

  Args:
    v: Variable to initialize
    init: Tensor to assign to v,
      Or an object convertible to Tensor e.g. nparray,
      Or an Initializer that generates a tensor given the shape and type of v.
      An "Initializer" is a callable that returns a tensor that "v" should be
      set to. It will be called as init(shape, dtype).
    name: Optional name for the op.

  Returns:
    The operation that initializes v.
  """
  with ops.name_scope(None, v.op.name + "/", [v, init]):
    with ops.name_scope(name) as scope:
      with ops.colocate_with(v):
        if callable(init):
          assert v.get_shape().is_fully_defined(), "Variable shape unknown."
          # TODO(mrry): Convert to v.shape when the property and
          # accessor are reconciled (and all initializers support
          # tf.TensorShape objects).
          value = init(v.get_shape().as_list(), v.dtype.base_dtype)
          value = ops.convert_to_tensor(value, name="value")
          return gen_state_ops.assign(v, value, name=scope)
        else:
          init = ops.convert_to_tensor(init, name="init")
          return gen_state_ops.assign(v, init, name=scope)


def is_variable_initialized(ref, name=None):
  """Checks whether a tensor has been initialized.

  Outputs boolean scalar indicating whether the tensor has been initialized.

  Args:
    ref: A mutable `Tensor`.
      Should be from a `Variable` node. May be uninitialized.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  if ref.dtype._is_ref_dtype:
    return gen_state_ops.is_variable_initialized(ref=ref, name=name)
  # Handle resource variables.
  if context.in_eager_mode() or ref.op.type == "VarHandleOp":
    return gen_resource_variable_ops.var_is_initialized_op(ref.handle,
                                                           name=name)


def assign_sub(ref, value, use_locking=None, name=None):
  """Update 'ref' by subtracting 'value' from it.

  This operation outputs "ref" after the update is done.
  This makes it easier to chain operations that need to use the reset value.

  Args:
    ref: A mutable `Tensor`. Must be one of the following types:
      `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`,
      `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a `Variable` node.
    value: A `Tensor`. Must have the same type as `ref`.
      The value to be subtracted to the variable.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    Same as "ref".  Returned as a convenience for operations that want
    to use the new value after the variable has been updated.
  """
  if ref.dtype._is_ref_dtype:
    return gen_state_ops.assign_sub(
        ref, value, use_locking=use_locking, name=name)
  return ref.assign_sub(value)


def assign_add(ref, value, use_locking=None, name=None):
  """Update 'ref' by adding 'value' to it.

  This operation outputs "ref" after the update is done.
  This makes it easier to chain operations that need to use the reset value.

  Args:
    ref: A mutable `Tensor`. Must be one of the following types:
      `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`,
      `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a `Variable` node.
    value: A `Tensor`. Must have the same type as `ref`.
      The value to be added to the variable.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the addition will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    Same as "ref".  Returned as a convenience for operations that want
    to use the new value after the variable has been updated.
  """
  if ref.dtype._is_ref_dtype:
    return gen_state_ops.assign_add(
        ref, value, use_locking=use_locking, name=name)
  return ref.assign_add(value)


def assign(ref, value, validate_shape=None, use_locking=None, name=None):
  """Update 'ref' by assigning 'value' to it.

  This operation outputs a Tensor that holds the new value of 'ref' after
    the value has been assigned. This makes it easier to chain operations
    that need to use the reset value.

  Args:
    ref: A mutable `Tensor`.
      Should be from a `Variable` node. May be uninitialized.
    value: A `Tensor`. Must have the same type as `ref`.
      The value to be assigned to the variable.
    validate_shape: An optional `bool`. Defaults to `True`.
      If true, the operation will validate that the shape
      of 'value' matches the shape of the Tensor being assigned to.  If false,
      'ref' will take on the shape of 'value'.
    use_locking: An optional `bool`. Defaults to `True`.
      If True, the assignment will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` that will hold the new value of 'ref' after
      the assignment has completed.
  """
  if ref.dtype._is_ref_dtype:
    return gen_state_ops.assign(
        ref, value, use_locking=use_locking, name=name,
        validate_shape=validate_shape)
  return ref.assign(value)


def count_up_to(ref, limit, name=None):
  r"""Increments 'ref' until it reaches 'limit'.

  Args:
    ref: A Variable. Must be one of the following types: `int32`, `int64`.
      Should be from a scalar `Variable` node.
    limit: An `int`.
      If incrementing ref would bring it above limit, instead generates an
      'OutOfRange' error.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `ref`.
    A copy of the input before increment. If nothing else modifies the
    input, the values produced will all be distinct.
  """
  if ref.dtype._is_ref_dtype:
    return gen_state_ops.count_up_to(ref, limit=limit, name=name)
  return gen_state_ops.resource_count_up_to(
      ref.handle, limit, T=ref.dtype, name=name)


def scatter_update(ref, indices, updates, use_locking=True, name=None):
  # pylint: disable=line-too-long
  r"""Applies sparse updates to a variable reference.

  This operation computes

  ```python
      # Scalar indices
      ref[indices, ...] = updates[...]

      # Vector indices (for each i)
      ref[indices[i], ...] = updates[i, ...]

      # High rank indices (for each i, ..., j)
      ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
  ```

  This operation outputs `ref` after the update is done.
  This makes it easier to chain operations that need to use the reset value.

  If values in `ref` is to be updated more than once, because there are
  duplicate entries in `indices`, the order at which the updates happen
  for each value is undefined.

  Requires `updates.shape = indices.shape + ref.shape[1:]`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ScatterUpdate.png" alt>
  </div>

  Args:
    ref: A `Variable`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor of indices into the first dimension of `ref`.
    updates: A `Tensor`. Must have the same type as `ref`.
      A tensor of updated values to store in `ref`.
    use_locking: An optional `bool`. Defaults to `True`.
      If True, the assignment will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    Same as `ref`.  Returned as a convenience for operations that want
    to use the updated values after the update is done.
  """
  if ref.dtype._is_ref_dtype:
    return gen_state_ops.scatter_update(ref, indices, updates,
                                        use_locking=use_locking, name=name)
  return gen_resource_variable_ops.resource_scatter_update(
      ref.handle, indices, ops.convert_to_tensor(updates, ref.dtype), name=name)
