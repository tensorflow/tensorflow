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

"""Functional operations.

See the @{$python/functional_ops} guide.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
# pylint: disable=unused-import
from tensorflow.python.ops.gen_functional_ops import remote_call
# pylint: enable=unused-import
from tensorflow.python.ops.gen_functional_ops import symbolic_gradient
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


# TODO(yuanbyu, mrry): Handle stride to support sliding windows.
@tf_export("foldl")
def foldl(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
          swap_memory=False, name=None):
  """foldl on the list of tensors unpacked from `elems` on dimension 0.

  This foldl operator repeatedly applies the callable `fn` to a sequence
  of elements from first to last. The elements are made of the tensors
  unpacked from `elems` on dimension 0. The callable fn takes two tensors as
  arguments. The first argument is the accumulated value computed from the
  preceding invocation of fn. If `initializer` is None, `elems` must contain
  at least one element, and its first element is used as the initializer.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is fn(initializer, values[0]).shape`.

  This method also allows multi-arity `elems` and output of `fn`.  If `elems`
  is a (possibly nested) list or tuple of tensors, then each of these tensors
  must have a matching first (unpack) dimension.  The signature of `fn` may
  match the structure of `elems`.  That is, if `elems` is
  `(t1, [t2, t3, [t4, t5]])`, then an appropriate signature for `fn` is:
  `fn = lambda (t1, [t2, t3, [t4, t5]]):`.

  Args:
    fn: The callable to be performed.
    elems: A tensor or (possibly nested) sequence of tensors, each of which
      will be unpacked along their first dimension.  The nested sequence
      of the resulting slices will be the first argument to `fn`.
    initializer: (optional) A tensor or (possibly nested) sequence of tensors,
      as the initial value for the accumulator.
    parallel_iterations: (optional) The number of iterations allowed to run
      in parallel.
    back_prop: (optional) True enables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor or (possibly nested) sequence of tensors, resulting from applying
    `fn` consecutively to the list of tensors unpacked from `elems`, from first
    to last.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = [1, 2, 3, 4, 5, 6]
    sum = foldl(lambda a, x: a + x, elems)
    # sum == 21
    ```
  """
  if not callable(fn):
    raise TypeError("fn must be callable.")

  def create_ta(elem):
    return tensor_array_ops.TensorArray(
        dtype=elem.dtype, size=n, dynamic_size=False,
        infer_shape=True).unstack(elem)

  in_graph_mode = not context.executing_eagerly()
  with ops.name_scope(name, "foldl", [elems]):
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

    # Convert elems to tensor array. n may be known statically.
    elems_flat = [
        ops.convert_to_tensor(elem, name="elem") for elem in nest.flatten(elems)
    ]
    n = elems_flat[0].shape[0].value or array_ops.shape(elems_flat[0])[0]

    elems_ta = nest.map_structure(create_ta, elems)

    if initializer is None:
      a = nest.map_structure(lambda elem: elem.read(0), elems_ta)
      i = constant_op.constant(1)
    else:
      a = initializer
      i = constant_op.constant(0)

    def compute(i, a):
      elem_i = nest.map_structure(lambda elem: elem.read(i), elems_ta)
      a = fn(a, elem_i)
      return [i + 1, a]

    _, r_a = control_flow_ops.while_loop(
        lambda i, a: i < n, compute, [i, a],
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory)

    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode and varscope_caching_device_was_none:
      varscope.set_caching_device(None)

    return r_a


@tf_export("foldr")
def foldr(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
          swap_memory=False, name=None):
  """foldr on the list of tensors unpacked from `elems` on dimension 0.

  This foldr operator repeatedly applies the callable `fn` to a sequence
  of elements from last to first. The elements are made of the tensors
  unpacked from `elems`. The callable fn takes two tensors as arguments.
  The first argument is the accumulated value computed from the preceding
  invocation of fn. If `initializer` is None, `elems` must contain at least
  one element, and its first element is used as the initializer.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is `fn(initializer, values[0]).shape`.

  This method also allows multi-arity `elems` and output of `fn`.  If `elems`
  is a (possibly nested) list or tuple of tensors, then each of these tensors
  must have a matching first (unpack) dimension.  The signature of `fn` may
  match the structure of `elems`.  That is, if `elems` is
  `(t1, [t2, t3, [t4, t5]])`, then an appropriate signature for `fn` is:
  `fn = lambda (t1, [t2, t3, [t4, t5]]):`.

  Args:
    fn: The callable to be performed.
    elems: A tensor or (possibly nested) sequence of tensors, each of which
      will be unpacked along their first dimension.  The nested sequence
      of the resulting slices will be the first argument to `fn`.
    initializer: (optional) A tensor or (possibly nested) sequence of tensors,
      as the initial value for the accumulator.
    parallel_iterations: (optional) The number of iterations allowed to run
      in parallel.
    back_prop: (optional) True enables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor or (possibly nested) sequence of tensors, resulting from applying
    `fn` consecutively to the list of tensors unpacked from `elems`, from last
    to first.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = [1, 2, 3, 4, 5, 6]
    sum = foldr(lambda a, x: a + x, elems)
    # sum == 21
    ```
  """
  if not callable(fn):
    raise TypeError("fn must be callable.")

  def create_ta(elem):
    return tensor_array_ops.TensorArray(
        dtype=elem.dtype, size=n, dynamic_size=False,
        infer_shape=True).unstack(elem)

  in_graph_mode = not context.executing_eagerly()
  with ops.name_scope(name, "foldr", [elems]):
    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode:
      # Any get_variable calls in fn will cache the first call locally and not
      # issue repeated network I/O requests for each iteration.
      varscope = vs.get_variable_scope()
      varscope_caching_device_was_none = False
      if varscope.caching_device is None:
        # TODO(ebrevdo): Change to using colocate_with here and in other
        # methods.
        varscope.set_caching_device(lambda op: op.device)
        varscope_caching_device_was_none = True

    # Convert elems to tensor array. n may be known statically.
    elems_flat = [
        ops.convert_to_tensor(elem, name="elem") for elem in nest.flatten(elems)
    ]
    n = elems_flat[0].shape[0].value or array_ops.shape(elems_flat[0])[0]

    elems_ta = nest.map_structure(create_ta, elems)

    if initializer is None:
      i = n - 1
      a = nest.map_structure(lambda elem: elem.read(i), elems_ta)
    else:
      i = n
      a = initializer

    def compute(i, a):
      i -= 1
      elem = nest.map_structure(lambda elem: elem.read(i), elems_ta)
      a_out = fn(a, elem)
      return [i, a_out]

    _, r_a = control_flow_ops.while_loop(
        lambda i, a: i > 0,
        compute, [i, a],
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory)

    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode and varscope_caching_device_was_none:
      varscope.set_caching_device(None)

    return r_a


@tf_export("map_fn")
def map_fn(fn, elems, dtype=None, parallel_iterations=10, back_prop=True,
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
      in parallel.
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

  in_graph_mode = not context.executing_eagerly()
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
    n = static_shape[0].value or array_ops.shape(elems_flat[0])[0]

    # TensorArrays are always flat
    elems_ta = [
        tensor_array_ops.TensorArray(dtype=elem.dtype, size=n,
                                     dynamic_size=False,
                                     infer_shape=True)
        for elem in elems_flat]
    # Unpack elements
    elems_ta = [
        elem_ta.unstack(elem) for elem_ta, elem in zip(elems_ta, elems_flat)]

    i = constant_op.constant(0)

    accs_ta = [
        tensor_array_ops.TensorArray(dtype=dt, size=n,
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

    n_static = elems_flat[0].get_shape().with_rank_at_least(1)[0]
    for elem in elems_flat[1:]:
      n_static.merge_with(elem.get_shape().with_rank_at_least(1)[0])
    for r in results_flat:
      r.set_shape(tensor_shape.TensorShape(n_static).concatenate(
          r.get_shape()[1:]))

    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode and varscope_caching_device_was_none:
      varscope.set_caching_device(None)

    return output_pack(results_flat)


@tf_export("scan")
def scan(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
         swap_memory=False, infer_shape=True, reverse=False, name=None):
  """scan on the list of tensors unpacked from `elems` on dimension 0.

  The simplest version of `scan` repeatedly applies the callable `fn` to a
  sequence of elements from first to last. The elements are made of the tensors
  unpacked from `elems` on dimension 0. The callable fn takes two tensors as
  arguments. The first argument is the accumulated value computed from the
  preceding invocation of fn. If `initializer` is None, `elems` must contain
  at least one element, and its first element is used as the initializer.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is `[len(values)] + fn(initializer, values[0]).shape`.
  If reverse=True, it's fn(initializer, values[-1]).shape.

  This method also allows multi-arity `elems` and accumulator.  If `elems`
  is a (possibly nested) list or tuple of tensors, then each of these tensors
  must have a matching first (unpack) dimension.  The second argument of
  `fn` must match the structure of `elems`.

  If no `initializer` is provided, the output structure and dtypes of `fn`
  are assumed to be the same as its input; and in this case, the first
  argument of `fn` must match the structure of `elems`.

  If an `initializer` is provided, then the output of `fn` must have the same
  structure as `initializer`; and the first argument of `fn` must match
  this structure.

  For example, if `elems` is `(t1, [t2, t3])` and `initializer` is
  `[i1, i2]` then an appropriate signature for `fn` in `python2` is:
  `fn = lambda (acc_p1, acc_p2), (t1, [t2, t3]):` and `fn` must return a list,
  `[acc_n1, acc_n2]`.  An alternative correct signature for `fn`, and the
   one that works in `python3`, is:
  `fn = lambda a, t:`, where `a` and `t` correspond to the input tuples.

  Args:
    fn: The callable to be performed.  It accepts two arguments.  The first
      will have the same structure as `initializer` if one is provided,
      otherwise it will have the same structure as `elems`.  The second
      will have the same (possibly nested) structure as `elems`.  Its output
      must have the same structure as `initializer` if one is provided,
      otherwise it must have the same structure as `elems`.
    elems: A tensor or (possibly nested) sequence of tensors, each of which
      will be unpacked along their first dimension.  The nested sequence
      of the resulting slices will be the first argument to `fn`.
    initializer: (optional) A tensor or (possibly nested) sequence of tensors,
      initial value for the accumulator, and the expected output type of `fn`.
    parallel_iterations: (optional) The number of iterations allowed to run
      in parallel.
    back_prop: (optional) True enables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    infer_shape: (optional) False disables tests for consistent output shapes.
    reverse: (optional) True scans the tensor last to first (instead of first
      to last).
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor or (possibly nested) sequence of tensors.  Each tensor packs the
    results of applying `fn` to tensors unpacked from `elems` along the first
    dimension, and the previous accumulator value(s), from first to last (or
    last to first, if `reverse=True`).

  Raises:
    TypeError: if `fn` is not callable or the structure of the output of
      `fn` and `initializer` do not match.
    ValueError: if the lengths of the output of `fn` and `initializer`
      do not match.

  Examples:
    ```python
    elems = np.array([1, 2, 3, 4, 5, 6])
    sum = scan(lambda a, x: a + x, elems)
    # sum == [1, 3, 6, 10, 15, 21]
    sum = scan(lambda a, x: a + x, elems, reverse=True)
    # sum == [22, 21, 18, 15, 11, 6]
    ```

    ```python
    elems = np.array([1, 2, 3, 4, 5, 6])
    initializer = np.array(0)
    sum_one = scan(
        lambda a, x: x[0] - x[1] + a, (elems + 1, elems), initializer)
    # sum_one == [1, 2, 3, 4, 5, 6]
    ```

    ```python
    elems = np.array([1, 0, 0, 0, 0, 0])
    initializer = (np.array(0), np.array(1))
    fibonaccis = scan(lambda a, _: (a[1], a[0] + a[1]), elems, initializer)
    # fibonaccis == ([1, 1, 2, 3, 5, 8], [1, 2, 3, 5, 8, 13])
    ```
  """
  if not callable(fn):
    raise TypeError("fn must be callable.")

  input_is_sequence = nest.is_sequence(elems)
  input_flatten = lambda x: nest.flatten(x) if input_is_sequence else [x]
  def input_pack(x):
    return nest.pack_sequence_as(elems, x) if input_is_sequence else x[0]

  if initializer is None:
    output_is_sequence = input_is_sequence
    output_flatten = input_flatten
    output_pack = input_pack
  else:
    output_is_sequence = nest.is_sequence(initializer)
    output_flatten = lambda x: nest.flatten(x) if output_is_sequence else [x]
    def output_pack(x):
      return (nest.pack_sequence_as(initializer, x)
              if output_is_sequence else x[0])

  elems_flat = input_flatten(elems)

  in_graph_mode = not context.executing_eagerly()
  with ops.name_scope(name, "scan", elems_flat):
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

    # Convert elems to tensor array.
    elems_flat = [
        ops.convert_to_tensor(elem, name="elem") for elem in elems_flat]

    # Convert elems to tensor array. n may be known statically.
    n = elems_flat[0].shape[0].value or array_ops.shape(elems_flat[0])[0]

    # TensorArrays are always flat
    elems_ta = [
        tensor_array_ops.TensorArray(dtype=elem.dtype, size=n,
                                     dynamic_size=False,
                                     infer_shape=True)
        for elem in elems_flat]
    # Unpack elements
    elems_ta = [
        elem_ta.unstack(elem) for elem_ta, elem in zip(elems_ta, elems_flat)]

    if initializer is None:
      a_flat = [elem.read(n - 1 if reverse else 0) for elem in elems_ta]
      i = constant_op.constant(1)
    else:
      initializer_flat = output_flatten(initializer)
      a_flat = [ops.convert_to_tensor(init) for init in initializer_flat]
      i = constant_op.constant(0)

    # Create a tensor array to store the intermediate values.
    accs_ta = [
        tensor_array_ops.TensorArray(
            dtype=init.dtype, size=n,
            element_shape=init.shape if infer_shape else None,
            dynamic_size=False,
            infer_shape=infer_shape)
        for init in a_flat]

    if initializer is None:
      accs_ta = [acc_ta.write(n - 1 if reverse else 0, a)
                 for (acc_ta, a) in zip(accs_ta, a_flat)]

    def compute(i, a_flat, tas):
      """The loop body of scan.

      Args:
        i: the loop counter.
        a_flat: the accumulator value(s), flattened.
        tas: the output accumulator TensorArray(s), flattened.

      Returns:
        [i + 1, a_flat, tas]: the updated counter + new accumulator values +
          updated TensorArrays

      Raises:
        TypeError: if initializer and fn() output structure do not match
        ValueType: if initializer and fn() output lengths do not match
      """
      packed_elems = input_pack([elem_ta.read(i) for elem_ta in elems_ta])
      packed_a = output_pack(a_flat)
      a_out = fn(packed_a, packed_elems)
      nest.assert_same_structure(
          elems if initializer is None else initializer, a_out)
      flat_a_out = output_flatten(a_out)
      tas = [ta.write(i, value) for (ta, value) in zip(tas, flat_a_out)]
      if reverse:
        next_i = i - 1
      else:
        next_i = i + 1
      return (next_i, flat_a_out, tas)

    if reverse:
      initial_i = n - 1 - i
      condition = lambda i, _1, _2: i >= 0
    else:
      initial_i = i
      condition = lambda i, _1, _2: i < n
    _, _, r_a = control_flow_ops.while_loop(
        condition, compute, (initial_i, a_flat, accs_ta),
        parallel_iterations=parallel_iterations,
        back_prop=back_prop, swap_memory=swap_memory,
        maximum_iterations=n)

    results_flat = [r.stack() for r in r_a]

    n_static = elems_flat[0].get_shape().with_rank_at_least(1)[0]
    for elem in elems_flat[1:]:
      n_static.merge_with(elem.get_shape().with_rank_at_least(1)[0])
    for r in results_flat:
      r.set_shape(tensor_shape.TensorShape(n_static).concatenate(
          r.get_shape()[1:]))

    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode and varscope_caching_device_was_none:
      varscope.set_caching_device(None)

    return output_pack(results_flat)


# pylint: disable=invalid-name
def If(cond, inputs, then_branch, else_branch, name=None):
  r"""output = Cond(inputs) ? then_branch(inputs) : else_branch(inputs).

  Args:
    cond: A `Tensor`. A scalar. If the scalar is not a boolean, the scalar is
      converted to a boolean according to the following rule: if the
      scalar is a numerical value, non-zero means True and zero means
      False; if the scalar is a string, non-empty means True and empty
      means False.
    inputs: A list of input tensors.
    then_branch: A function takes 'inputs' and returns a list of tensors,
        whose types are the same as what else_branch returns.
    else_branch: A function takes 'inputs' and returns a list of tensors.
        whose types are the same as what then_branch returns.
    name: A name for the operation (optional).

  Returns:
    A list of tensors returned by either then_branch(inputs)
    or else_branch(inputs).
  """
  # pylint: disable=protected-access
  return gen_functional_ops._if(
      cond,
      inputs, [_.type for _ in then_branch.definition.signature.output_arg],
      then_branch,
      else_branch,
      name=name)


def Gradient(inputs, f, name=None):
  r"""Computes the gradient function for function f via backpropagation.

  Args:
    inputs: A list of tensors of size N + M.
    f: The function we want to compute the gradient for.

      The function 'f' must be a numerical function which takes N inputs and
      produces M outputs. Its gradient function 'g', which is  a function
      taking N + M inputs and produces N outputs.

      I.e. if we have
         (y1, y2, ..., yM) = f(x1, x2, ..., xN),
      then, g is
         (dL/dx1, dL/dx2, ..., dL/dxN) = g(x1, x2, ..., xN,
                                           dL/dy1, dL/dy2, ..., dL/dyM),

      where L is a scalar-value function of (x1, x2, ..., xN) (e.g., the
      loss function). dL/dxi is the partial derivative of L with respect
      to xi.

    name: A name for the operation (optional).

  Returns:
    A list of tensors of size N.
  """
  # TODO(zhifengc): Pretty-print the above spec in latex.
  # TODO(zhfiengc): Needs some math expert to say the comment above better.
  tlist = [_.type for _ in f.definition.signature.input_arg]
  return symbolic_gradient(input=inputs, Tout=tlist, f=f, name=name)


# pylint: disable=invalid-name,protected-access
def While(input_, cond, body, name=None, hostmem=None):
  r"""output = input; While (Cond(output)) { output = Body(output) }.

  Args:
    input_: A list of `Tensor` objects.
      A list of input tensors whose types are T.
    cond: . A function takes 'input' and returns a tensor.  If the tensor is
      a scalar of non-boolean, the scalar is converted to a boolean
      according to the following rule: if the scalar is a numerical
      value, non-zero means True and zero means False; if the scalar is
      a string, non-empty means True and empty means False. If the
      tensor is not a scalar, non-emptiness means True and False
      otherwise.
    body: . A function takes a list of tensors and returns another
      list tensors. Both lists have the same types as specified
      by T.
    name: A name for the operation (optional).
    hostmem: A list of integer. If i is in the list, input[i] is a
      host memory tensor.

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
    A list of output tensors whose types are T.
  """
  ret = gen_functional_ops._while(input_, cond, body, name=name)
  if hostmem:
    input_attr = attr_value_pb2.AttrValue()
    input_attr.list.i.extend(hostmem)
    ret[0].op._set_attr("_input_hostmem", input_attr)  # pylint: disable=protected-access

    output_attr = attr_value_pb2.AttrValue()
    output_attr.list.i.extend(hostmem)
    ret[0].op._set_attr("_output_hostmem", output_attr)  # pylint: disable=protected-access
  return ret


# b/36459430
#
# Ideally, we do not need this rewrite For loop into a While loop.
# However, today, if a While runs on GPU and the condition returns a
# boolean, the While kernel crashes. Even if we fix the crash, the
# bool needs to be copied between GPU and CPU. So, a for loop is much
# preferred when running on GPU.
#
# On the other hand, For op has no directly XLA kernel. So, when we run
# a for loop, we need to rewrite it using a While op.
#
# It should be possible and probably better to write a XLA C++ kernel
# implementing the logic in _ForUsingWhile.
def _ForUsingWhile(start,
                   limit,
                   delta,
                   inputs,
                   forbody,
                   name=None,
                   hostmem=None):
  """Helper to implement a For loop using a While."""
  # To support negative delta (e.g., range(100, 0, -3)), we iterate
  # over the range(n) and use iter * delta + start as the real
  # iteration index. (e.g., for i in range(34): iter = i * (-3) +
  # 100).
  d = math_ops.abs(delta)
  # XLA on TPUs doesn't support integer division
  n = math_ops.cast(
      math_ops.cast((math_ops.abs(limit - start) + d - 1), dtypes.float32) /
      math_ops.cast(d, dtypes.float32), dtypes.int32)

  # Carried loop variables ("extra_args") are implicitly added to the input list
  # of the WhileBody function. WhileCond does not call forbody, and so does not
  # depend on any of forbody's extra_args. Since WhileCond and WhileBody
  # must have identical inputs, we have to augment the cond signature to take
  # the same types as the carried loop variables.
  body_sig = [dtypes.int32] * 4 + list(forbody.declared_input_types)[1:]
  cond_sig = body_sig + [t.dtype for t in forbody.captured_inputs]

  cond_name = "%s_Cond" % forbody.name

  @function.Defun(*cond_sig, func_name=cond_name)
  def WhileCond(i, n, *args):
    del args
    return i < n

  body_name = "%s_Body" % forbody.name

  @function.Defun(*body_sig, func_name=body_name)
  def WhileBody(i, n, start, delta, *args):
    """A While wrapper for forbody that handles loop-carried captured inputs."""
    for_result = forbody(start + i * delta, *args)
    # Nullary functions return an Operation. Normal functions can't do this
    # because their return values are converted to Tensors.
    if isinstance(for_result, ops.Operation):
      for_result = ()
    # Unary functions return a single Tensor value.
    elif isinstance(for_result, ops.Tensor):
      for_result = (for_result,)
    extra_args = tuple(function.get_extra_args())
    return (i + 1, n, start, delta) + tuple(for_result) + extra_args

  if hostmem is not None:
    hostmem = [0, 1, 2, 3] + [(4 + _) for _ in hostmem]
  else:
    hostmem = [0, 1, 2, 3]

  results = While(
      input_=[0, n, start, delta] + inputs + WhileBody.captured_inputs,
      cond=WhileCond,
      body=WhileBody,
      name=name,
      hostmem=hostmem)
  # Slice off the loop-carried captured inputs.
  return list(results[4:len(results) - len(WhileBody.captured_inputs)])


def For(start,
        limit,
        delta,
        inputs,
        body,
        name=None,
        hostmem=None,
        rewrite_with_while=None):
  r"""out = input; for i in range(start, limit, delta) out = body(i, out).

  Args:
    start: A `Tensor` of type `int32`.
    limit: A `Tensor` of type `int32`.
    delta: A `Tensor` of type `int32`.
    inputs: A list of `Tensor` objects.
      A list of input tensors whose types are T.
    body: A function takes a list of tensors and returns another
      list of tensors. Both lists have the same types as (int32, T...).
    name: A name for the operation (optional).
    hostmem: A list of integer. If i is in the list, inputs[i] is a
      host memory tensor. In other words, (i+1)-th argument of the body
      function is expecting a host memory.
    rewrite_with_while: If True, using While op to implement the For.

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
    A list of output tensors whose types are T.
  """
  if rewrite_with_while:
    return _ForUsingWhile(start, limit, delta, inputs, body, name, hostmem)
  if body.captured_inputs:
    wrapper_name = "%s_BodyWrapper" % body.name

    @function.Defun(*body.declared_input_types, func_name=wrapper_name)
    def BodyWrapper(*args):
      """A wrapper for body that handles loop-carried captured inputs."""
      body_result = body(*args)
      extra_args = tuple(function.get_extra_args())
      # Nullary functions return an Operation. Normal functions can't do this
      # because their return values are converted to Tensors.
      if isinstance(body_result, ops.Operation):
        return extra_args
      # Unary functions return a single Tensor value.
      elif not isinstance(body_result, tuple):
        return (body_result,) + extra_args
      # N-ary functions return a tuple of Tensors.
      else:
        return body_result + extra_args

    inputs += BodyWrapper.captured_inputs
    ret = gen_functional_ops._for(
        start, limit, delta, inputs, BodyWrapper, name=name)
    # Slice off the loop-carried captured inputs.
    ret = ret[:-len(BodyWrapper.captured_inputs)]
  else:
    ret = gen_functional_ops._for(start, limit, delta, inputs, body, name=name)
  if hostmem:
    num_for_params = 3  # start/limit/delta

    input_attr = attr_value_pb2.AttrValue()
    input_attr.list.i.extend([num_for_params + i for i in hostmem])
    ret[0].op._set_attr("_input_hostmem", input_attr)  # pylint: disable=protected-access

    output_attr = attr_value_pb2.AttrValue()
    output_attr.list.i.extend(hostmem)
    ret[0].op._set_attr("_output_hostmem", output_attr)  # pylint: disable=protected-access
  return ret
# pylint: enable=invalid-name,protected-access


def partitioned_call(args, f, tout=None, executing_eagerly=None):
  """Executes a function while respecting device annotations.

  Currently, only those functions that execute within the same address space
  can be executed.

  Args:
    args: The arguments of the function, including captured inputs.
    f: The function to execute; an instance of `_DefinedFunction` or
      `_EagerDefinedFunction`.
    tout: a list containing the output dtypes enums; if `None`, inferred from
      the signature of `f`.
    executing_eagerly: (Optional) A boolean indicating whether the context is
      executing eagerly. If `None`, fetched from the global context.

  Returns:
    The list of `Tensor`s returned by invoking `f(args)`. If the function does
    not return anything, then returns `None` if eager execution is enabled, or
    the `Operation` if not.
  """

  if tout is None:
    tout = tuple(x.type for x in f.definition.signature.output_arg)

  if executing_eagerly is None:
    executing_eagerly = context.executing_eagerly()

  if executing_eagerly or len(tout):
    if f.stateful_ops:
      outputs = gen_functional_ops.stateful_partitioned_call(
          args=args, Tout=tout, f=f)
    else:
      outputs = gen_functional_ops.partitioned_call(args=args, Tout=tout, f=f)
    return outputs if outputs else None

  # The generated binding returns an empty list for functions that don't
  # return any Tensors, hence the need to use `create_op` directly.
  args = [ops.internal_convert_to_tensor(x) for x in args]
  tin_attr = attr_value_pb2.AttrValue(
      list=attr_value_pb2.AttrValue.ListValue(
          type=[x.dtype.as_datatype_enum for x in args]))
  tout_attr = attr_value_pb2.AttrValue(
      list=attr_value_pb2.AttrValue.ListValue(type=tout))
  func_attr = attr_value_pb2.AttrValue(
      func=attr_value_pb2.NameAttrList(name=f.name))

  graph = ops.get_default_graph()
  f.add_to_graph(graph)
  op_name = "StatefulPartitionedCall" if f.stateful_ops else "PartitionedCall"
  op = graph.create_op(
      op_name,
      args,
      tout,
      compute_shapes=False,
      name="PartitionedFunctionCall",
      attrs={"Tin": tin_attr, "Tout": tout_attr, "f": func_attr})
  outputs = op.outputs
  return outputs if outputs else op
