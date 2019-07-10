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
# =============================================================================

"""Functional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export("map_fn")
def map_fn(fn, elems, dtype=None, parallel_iterations=None, back_prop=True,
           swap_memory=False, infer_shape=True, name=None):
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
    fn: The callable to be performed.  It accepts one argument, which will
      have the same (possibly nested) structure as `elems`.  Its output
      must have the same structure as `dtype` if one is provided, otherwise
      it must have the same structure as `elems`.
    elems: A tensor or (possibly nested) sequence of tensors, each of which
      will be unpacked along their first dimension.  The nested sequence
      of the resulting slices will be applied to `fn`.
    dtype: (optional) The output type(s) of `fn`.  If `fn` returns a structure
      of Tensors differing from the structure of `elems`, then `dtype` is not
      optional and must have the same structure as the output of `fn`.
    parallel_iterations: (optional) The number of iterations allowed to run
      in parallel. When graph building, the default value is 10. While executing
      eagerly, the default value is set to 1.
    back_prop: (optional) True enables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    infer_shape: (optional) False disables tests for consistent output shapes.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor or (possibly nested) sequence of tensors.  Each tensor packs the
    results of applying `fn` to tensors unpacked from `elems` along the first
    dimension, from first to last.

  Raises:
    TypeError: if `fn` is not callable or the structure of the output of
      `fn` and `dtype` do not match, or if elems is a SparseTensor.
    ValueError: if the lengths of the output of `fn` and `dtype` do not match.

  Examples:
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

  if dtype is None:
    output_is_sequence = input_is_sequence
    output_flatten = input_flatten
    output_pack = input_pack
  else:
    output_is_sequence = nest.is_sequence(dtype)
    output_flatten = lambda x: nest.flatten(x) if output_is_sequence else [x]
    def output_pack(x):
      return (nest.pack_sequence_as(dtype, x)
              if output_is_sequence else x[0])

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
        ops.convert_to_tensor(elem, name="elem") for elem in elems_flat]

    dtype = dtype or input_pack([elem.dtype for elem in elems_flat])
    dtype_flat = output_flatten(dtype)

    # Convert elems to tensor array. n may be known statically.
    static_shape = elems_flat[0].shape
    if static_shape.ndims is not None and static_shape.ndims < 1:
      if len(elems_flat) == 1:
        raise ValueError("elems must be a 1+ dimensional Tensor, not a scalar")
      else:
        raise ValueError(
            "elements in elems must be 1+ dimensional Tensors, not scalars"
        )
    n = (tensor_shape.dimension_value(static_shape[0])
         or array_ops.shape(elems_flat[0])[0])

    # TensorArrays are always flat
    elems_ta = [
        tensor_array_ops.TensorArray(dtype=elem.dtype,
                                     size=n,
                                     dynamic_size=False,
                                     infer_shape=True)
        for elem in elems_flat]
    # Unpack elements
    elems_ta = [
        elem_ta.unstack(elem) for elem_ta, elem in zip(elems_ta, elems_flat)]

    i = constant_op.constant(0)

    accs_ta = [
        tensor_array_ops.TensorArray(dtype=dt,
                                     size=n,
                                     dynamic_size=False,
                                     infer_shape=infer_shape)
        for dt in dtype_flat]

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
      packed_values = input_pack([elem_ta.read(i) for elem_ta in elems_ta])
      packed_fn_values = fn(packed_values)
      nest.assert_same_structure(dtype or elems, packed_fn_values)
      flat_fn_values = output_flatten(packed_fn_values)
      tas = [ta.write(i, value) for (ta, value) in zip(tas, flat_fn_values)]
      return (i + 1, tas)

    _, r_a = control_flow_ops.while_loop(
        lambda i, _: i < n, compute, (i, accs_ta),
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory,
        maximum_iterations=n)
    results_flat = [r.stack() for r in r_a]

    n_static = tensor_shape.Dimension(tensor_shape.dimension_value(
        elems_flat[0].get_shape().with_rank_at_least(1)[0]))
    for elem in elems_flat[1:]:
      n_static.merge_with(tensor_shape.Dimension(tensor_shape.dimension_value(
          elem.get_shape().with_rank_at_least(1)[0])))
    for r in results_flat:
      r.set_shape(tensor_shape.TensorShape(n_static).concatenate(
          r.get_shape()[1:]))

    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode and varscope_caching_device_was_none:
      varscope.set_caching_device(None)

    return output_pack(results_flat)
