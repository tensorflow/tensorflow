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

"""## Variables

@@Variable

## Variable helper functions

TensorFlow provides a set of functions to help manage the set of variables
collected in the graph.

@@global_variables
@@local_variables
@@model_variables
@@trainable_variables
@@moving_average_variables

@@global_variables_initializer
@@local_variables_initializer
@@variables_initializer
@@is_variable_initialized
@@report_uninitialized_variables
@@assert_variables_initialized

@@assign
@@assign_add
@@assign_sub

## Saving and Restoring Variables

@@Saver

@@latest_checkpoint

@@get_checkpoint_state
@@update_checkpoint_state

## Sharing Variables

TensorFlow provides several classes and operations that you can use to
create variables contingent on certain conditions.

@@get_variable
@@VariableScope
@@variable_scope
@@variable_op_scope
@@get_variable_scope
@@make_template

@@no_regularizer

@@constant_initializer
@@random_normal_initializer
@@truncated_normal_initializer
@@random_uniform_initializer
@@uniform_unit_scaling_initializer
@@zeros_initializer
@@ones_initializer
@@orthogonal_initializer

## Variable Partitioners for Sharding

@@fixed_size_partitioner
@@variable_axis_size_partitioner
@@min_max_variable_partitioner

## Sparse Variable Updates

The sparse update ops modify a subset of the entries in a dense `Variable`,
either overwriting the entries or adding / subtracting a delta.  These are
useful for training embedding models and similar lookup-based networks, since
only a small subset of embedding vectors change in any given step.

Since a sparse update of a large tensor may be generated automatically during
gradient computation (as in the gradient of
[`tf.gather`](../../api_docs/python/array_ops.md#gather)),
an [`IndexedSlices`](#IndexedSlices) class is provided that encapsulates a set
of sparse indices and values.  `IndexedSlices` objects are detected and handled
automatically by the optimizers in most cases.

@@scatter_update
@@scatter_add
@@scatter_sub
@@scatter_mul
@@scatter_div
@@scatter_nd_update
@@scatter_nd_add
@@scatter_nd_sub
@@sparse_mask
@@IndexedSlices

### Read-only Lookup Tables

@@initialize_all_tables


## Exporting and Importing Meta Graphs

@@export_meta_graph
@@import_meta_graph

# Deprecated functions (removed after 2017-03-02). Please don't use them.

@@all_variables
@@initialize_all_variables
@@initialize_local_variables
@@initialize_variables

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    A variable tensor.1;5A
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
  if ref.op.type == "VarHandleOp":
    return gen_resource_variable_ops.var_is_initialized_op(ref.handle,
                                                           name=name)
