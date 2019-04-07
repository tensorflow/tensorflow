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
"""Functional operations for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def map_fn(fn,
           elems,
           dtype=None,
           parallel_iterations=None,
           back_prop=True,
           swap_memory=False,
           infer_shape=True,
           name=None):
  """map on the list of tensors unpacked from `elems` on dimension 0.

  The simplest version of `map_fn` repeatedly applies the callable `fn` to a
  sequence of elements from first to last. The elements are made of the
  tensors unpacked from `elems`. `dtype` is the data type of the return
  value of `fn`. Users must provide `dtype` if it is different from
  the data type of `elems`.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is `[values.shape[0]] + fn(values[0]).shape`.

  This method also allows multi-arity `elems` and output of `fn`.  If `elems`
  is a (possibly nested) list or tuple of tensors, then each of these tensors
  must have a matching first (unpack) dimension.  The signature of `fn` may
  match the structure of `elems`.  That is, if `elems` is
  `(t1, [t2, t3, [t4, t5]])`, then an appropriate signature for `fn` is:
  `fn = lambda (t1, [t2, t3, [t4, t5]]):`.

  Furthermore, `fn` may emit a different structure than its input.  For example,
  `fn` may look like: `fn = lambda t1: return (t1 + 1, t1 - 1)`.  In this case,
  the `dtype` parameter is not optional: `dtype` must be a type or (possibly
  nested) tuple of types matching the output of `fn`.

  To apply a functional operation to the nonzero elements of a SparseTensor
  one of the following methods is recommended. First, if the function is
  expressible as TensorFlow ops, use

  ```python
    result = SparseTensor(input.indices, fn(input.values), input.dense_shape)
  ```

  If, however, the function is not expressible as a TensorFlow op, then use

  ```python
  result = SparseTensor(
    input.indices, map_fn(fn, input.values), input.dense_shape)
  ```

  instead.

  When executing eagerly, map_fn does not execute in parallel even if
  `parallel_iterations` is set to a value > 1. You can still get the
  performance benefits of running a function in parallel by using the
  `tf.contrib.eager.defun` decorator,

  ```python
  # Assume the function being used in map_fn is fn.
  # To ensure map_fn calls fn in parallel, use the defun decorator.
  @tf.contrib.eager.defun
  def func(tensor):
    return tf.map_fn(fn, tensor)
  ```

  Note that if you use the defun decorator, any non-TensorFlow Python code
  that you may have written in your function won't get executed. See
  `tf.contrib.eager.defun` for more details. The recommendation would be to
  debug without defun but switch to defun to get performance benefits of
  running map_fn in parallel.

  Args:
    fn: The callable to be performed.  It accepts one argument, which will have
      the same (possibly nested) structure as `elems`.  Its output must have the
      same structure as `dtype` if one is provided, otherwise it must have the
      same structure as `elems`.
    elems: A tensor or (possibly nested) sequence of tensors, each of which will
      be unpacked along their first dimension.  The nested sequence of the
      resulting slices will be applied to `fn`.
    dtype: (optional) The output type(s) of `fn`.  If `fn` returns a structure
      of Tensors differing from the structure of `elems`, then `dtype` is not
      optional and must have the same structure as the output of `fn`. Use
      `RaggedTensorType` to declare an output of type `RaggedTensor`.
    parallel_iterations: (optional) The number of iterations allowed to run in
      parallel. When graph building, the default value is 10. While executing
      eagerly, the default value is set to 1.
    back_prop: (optional) True enables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    infer_shape: (optional) False disables tests for consistent output shapes.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A possibly nested sequence of potentially ragged tensors.  Each
    tensor packs the results of applying `fn` to tensors unpacked from `elems`
    along the first dimension, from first to last.

  Raises:
    TypeError: if `fn` is not callable or the structure of the output of
      `fn` and `dtype` do not match, or if elems is a SparseTensor.
    ValueError: if the lengths of the output of `fn` and `dtype` do not match.

  #### Examples:

    ```python
    elems = np.array([1, 2, 3, 4, 5, 6])
    squares = map_fn(lambda x: x * x, elems)
    # squares == [1, 4, 9, 16, 25, 36]
    ```

    ```python
    elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
    alternate = map_fn(lambda x: x[0] * x[1], elems, dtype=tf.int64)
    # alternate == [-1, 2, -3]
    ```

    ```python
    elems = np.array([1, 2, 3])
    alternates = map_fn(lambda x: (x, -x), elems, dtype=(tf.int64, tf.int64))
    # alternates[0] == [1, 2, 3]
    # alternates[1] == [-1, -2, -3]
    ```

    ```python
    elems=ragged.constant([[1, 2, 3], [4, 5], [6, 7]])
    mean = map_fn(tf.reduce_mean, elems)
    # mean == [2, 4, 6]
    ```

    ```python
    elems=ragged.constant([[1, 2, 3], [4, 5], [6, 7]], dtype=tf.int64)
    out = map_fn(fn=lambda x: x+1, elems,
      dtype=ragged.RaggedTensorType(type=tf.int64, ragged_rank=0))
    # out = ragged.constant([[2, 3, 4], [5, 6], [7, 8]])
    ```
  """
  if not callable(fn):
    raise TypeError("fn must be callable.")

  if isinstance(elems, sparse_tensor.SparseTensor):
    raise TypeError(
        "To perform a map on the values of a sparse tensor use either "
        " SparseTensor(input.indices, fn(input.values), input.dense_shape) or "
        " SparseTensor(input.indices, map_fn(fn, input.values), "
        "input.dense_shape)")

  in_graph_mode = not context.executing_eagerly()
  # Set the default number of parallel_iterations depending on graph/eager mode.
  if in_graph_mode and not parallel_iterations:
    parallel_iterations = 10
  elif not in_graph_mode and not parallel_iterations:
    parallel_iterations = 1

  if not in_graph_mode and parallel_iterations > 1:
    logging.log_first_n(logging.WARN, "Setting parallel_iterations > 1 has no "
                        "effect when executing eagerly. Consider calling map_fn"
                        " with tf.contrib.eager.defun to execute fn in "
                        "parallel.", 1)
    parallel_iterations = 1

  input_is_sequence = nest.is_sequence(elems)
  input_flatten = lambda x: nest.flatten(x) if input_is_sequence else [x]

  def input_pack(x):
    return nest.pack_sequence_as(elems, x) if input_is_sequence else x[0]

  elems_flat = input_flatten(elems)

  with ops.name_scope(name, "map", elems_flat):
    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode:
      # Any get_variable calls in fn will cache the first call locally
      # and not issue repeated network I/O requests for each iteration.
      varscope = vs.get_variable_scope()
      varscope_caching_device_was_none = False
      if varscope.caching_device is None:
        # TODO(ebrevdo): Change to using colocate_with here and in other
        # methods.
        varscope.set_caching_device(lambda op: op.device)
        varscope_caching_device_was_none = True

    elems_flat = [
        ragged_tensor.convert_to_tensor_or_ragged_tensor(elem, name="elem")
        for elem in elems_flat
    ]

    # We can either infer the output, or we can assume that it will be the same
    # as the input structure.
    dtype = dtype or input_pack([elem.dtype for elem in elems_flat])

    # Find the number of iterations, n may be known statically.
    if isinstance(elems_flat[0], ragged_tensor.RaggedTensor):
      n = elems_flat[0].nrows(out_type=dtypes.int32)
    else:
      static_shape = elems_flat[0].shape
      if static_shape.ndims is not None and static_shape.ndims < 1:
        if len(elems_flat) == 1:
          raise ValueError(
              "elems must be a 1+ dimensional Tensor, not a scalar")
        else:
          raise ValueError(
              "elements in elems must be 1+ dimensional Tensors, not scalars")
      n = (tensor_shape.dimension_value(static_shape[0]) or
           array_ops.shape(elems_flat[0])[0])

    n = math_ops.cast(n, dtype=dtypes.int32)
    # Create a flat list of TAs.

    # Flatten the dtype structure to a list.
    dtype_flat = nest.flatten(dtype)

    # decompose to components
    dtype_components = [_maybe_decompose_dtype(d) for d in dtype_flat]
    dtype_components_flat = nest.flatten(dtype_components)

    # Create TensorArrays.
    accs_ta = [
        tensor_array_ops.TensorArray(
            dtype=t, dynamic_size=False, infer_shape=infer_shape, size=n)
        for t in dtype_components_flat
    ]

    i = constant_op.constant(0, dtype=dtypes.int32)

    def compute(i, tas):
      """The loop body of map_fn.

      Args:
        i: the loop counter
        tas: the flat TensorArray accumulator list

      Returns:
        (i + 1, tas): the updated counter + updated TensorArrays

      Raises:
        TypeError: if dtype and packed_fn_values structure do not match
        ValueType: if dtype and packed_fn_values lengths do not match
      """
      # Get Tensors or RaggedTensors sliced at i, then pack it back to the
      # original structure.
      packed_values = input_pack([elem_flat[i] for elem_flat in elems_flat])
      packed_fn_values = fn(packed_values)

      # Check that the structure of the output matches what was declared or
      # inferred.
      # nest.assert_same_structure(dtype or elems, packed_fn_values)

      # Flatten and decompose to a list of Tensors
      flat_fn_values = nest.flatten(packed_fn_values)

      # If we declared that we are expecting a RaggedTensor output, but we get a
      # Tensor output. We should try to convert it to a RaggedTensor.
      flat_fn_composite_tensors = list(
          _convert_declared(flat_fn_values, dtype_flat))

      flat_fn_components = [
          _maybe_decompose_tensor(t) for t in flat_fn_composite_tensors
      ]
      flat_fn_tensors = nest.flatten(flat_fn_components)

      # Write to TAs.
      tas = [ta.write(i, value) for (ta, value) in zip(tas, flat_fn_tensors)]

      return (i + 1, tas)

    _, r_a = control_flow_ops.while_loop(
        lambda i, _: i < n, compute, (i, accs_ta),
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory,
        maximum_iterations=n)

    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode and varscope_caching_device_was_none:
      varscope.set_caching_device(None)

    # Pack back into a list of components
    results_as_components = nest.pack_sequence_as(dtype_components, r_a)

    # Stack TensorArrays for Tensor outputs, and concat RaggedTensor outputs.
    def _stack_or_concat(e):
      if isinstance(e, _RaggedTensorComponents):
        return _concat_ragged_tensor_components(e)
      else:
        result = e.stack()
        return result

    results_flat_components = [
        _stack_or_concat(e) for e in results_as_components
    ]

    results_packed = [
        _maybe_recompose_tensor(c) for c in results_flat_components
    ]
    results_packed = nest.pack_sequence_as(dtype, results_packed)
    return results_packed


class _RaggedTensorComponents(
    collections.namedtuple(
        "_RaggedTensorComponents",
        ["flat_values", "nested_row_lengths", "outer_row_length"])):
  """A namedtuple of components which represent a `RaggedTensor`.

  _RaggedTensorComponents is a list of components which can be used to create a
  `RaggedTensor`. Use this class to represent a `RaggedTensor` in situations
  where nest.flatten and nest.pack_sequence_as should decompose ragged tensors
  into their components..

  The following are a list of components for a `RaggedTensor`:

  flat_values: The flat and inner values of a RaggedTensor. This could be
    a `Tensor`, a `TensorArray`, or a data type.
  nested_row_lengths: a tuple containing the row lengths of each rank. The
    elements of the tuple could be `Tensor`s or `TensorArray`s.
  outer_row_length: a `Tensor` or `TensorArray` containing the row length of the
    `RaggedTensor`'s outermost dimension.

  See `RaggedTensor` for more details of the use of each component.
  """
  __slots__ = ()


def _concat_ragged_tensor_components(rt_ta):
  flat_values = rt_ta.flat_values.concat()
  nested_row_lengths = tuple(
      row_lengths_ta.concat() for row_lengths_ta in rt_ta.nested_row_lengths)
  outer_row_length = rt_ta.outer_row_length.concat()
  return _RaggedTensorComponents(
      flat_values=flat_values,
      nested_row_lengths=nested_row_lengths,
      outer_row_length=outer_row_length)


def _maybe_decompose_tensor(rt):
  """Decompose tensors to their composite tensors."""
  if not isinstance(rt, ragged_tensor.RaggedTensor):
    return rt

  # The three component pieces we need:
  # - inner values
  flat_values = rt.flat_values

  # - row_splits of the RT
  splits = rt.nested_row_splits
  nested_row_lengths = tuple(split[1:] - split[:-1] for split in splits)

  # - outer row length
  outer_row_length = array_ops.expand_dims(rt.nrows(), axis=0)

  return _RaggedTensorComponents(
      flat_values=flat_values,
      nested_row_lengths=nested_row_lengths,
      outer_row_length=outer_row_length,
  )


def _maybe_recompose_tensor(t):
  """Reconstructs a _RaggedTensorComponents into a RaggedTensor."""
  if not isinstance(t, _RaggedTensorComponents):
    return t

  values = t.flat_values
  nested_row_lengths = tuple(t.nested_row_lengths)
  for nested_row_length in reversed(nested_row_lengths):
    values = ragged_tensor.RaggedTensor.from_row_lengths(
        values, nested_row_length)
  return ragged_tensor.RaggedTensor.from_row_lengths(values, t.outer_row_length)


def _maybe_decompose_dtype(d):
  """Decompose dtypes into composite tensors (if necessary)."""
  if not isinstance(d, ragged_tensor.RaggedTensorType):
    return d

  result = _RaggedTensorComponents(
      flat_values=d.dtype,
      nested_row_lengths=tuple(dtypes.int64 for i in range(d.ragged_rank - 1)),
      outer_row_length=dtypes.int64,
  )
  return result


def _convert_declared(fn_output_flat, output_declared):
  """Convert outputs which are `Tensor`s into `_RaggedTensorComponents`."""
  for current, declared in zip(fn_output_flat, output_declared):
    if isinstance(declared, ragged_tensor.RaggedTensorType):
      if isinstance(current, ragged_tensor.RaggedTensor):
        # Check that the ragged ranks match up.
        # + 1 to account for the rank of the outermost dimension.
        if declared.ragged_rank != current.ragged_rank + 1:
          raise ValueError(
              "The declared ragged rank (%d) mismatches the result (%d)" %
              (declared.ragged_rank, current.ragged_rank))
        yield current
      else:
        # We the output is a Tensor, but the caller has declared that we are
        # expecting an RaggedTensor output.
        if declared.ragged_rank != 1:
          raise ValueError(
              "The declared ragged rank (%d) mismatches the result (1)" %
              declared.ragged_rank)

        if isinstance(current, ragged_tensor.RaggedTensor):
          nrows = current.nrows()
        else:
          nrows = array_ops.shape(current, out_type=dtypes.int64)[0]
        row_length = array_ops.expand_dims(nrows, axis=0)
        rt = _RaggedTensorComponents(
            flat_values=current,
            nested_row_lengths=(),
            outer_row_length=row_length)
        yield rt
    else:
      yield current
