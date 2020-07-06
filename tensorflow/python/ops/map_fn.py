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


import re

from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["map_fn"])
@deprecation.deprecated_args(None, "Use fn_output_signature instead", "dtype")
def map_fn(fn,
           elems,
           dtype=None,
           parallel_iterations=None,
           back_prop=True,
           swap_memory=False,
           infer_shape=True,
           name=None,
           fn_output_signature=None):
  """Transforms `elems` by applying `fn` to each element unstacked on axis 0.

  See also `tf.scan`.

  `map_fn` unstacks `elems` on axis 0 to obtain a sequence of elements;
  calls `fn` to transform each element; and then stacks the transformed
  values back together.

  #### Mapping functions with single-Tensor inputs and outputs

  If `elems` is a single tensor and `fn`'s signature is `tf.Tensor->tf.Tensor`,
  then `map_fn(fn, elems)` is equivalent to
  `tf.stack([fn(elem) for elem in tf.unstack(elems)])`.  E.g.:

  >>> tf.map_fn(fn=lambda t: tf.range(t, t + 3), elems=tf.constant([3, 5, 2]))
  <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[3, 4, 5],
           [5, 6, 7],
           [2, 3, 4]], dtype=int32)>

  `map_fn(fn, elems).shape = [elems.shape[0]] + fn(elems[0]).shape`.

  #### Mapping functions with multi-arity inputs and outputs

  `map_fn` also supports functions with multi-arity inputs and outputs:

  * If `elems` is a tuple (or nested structure) of tensors, then those tensors
    must all have the same outer-dimension size (`num_elems`); and `fn` is
    used to transform each tuple (or structure) of corresponding slices from
    `elems`.  E.g., if `elems` is a tuple `(t1, t2, t3)`, then `fn` is used to
    transform each tuple of slices `(t1[i], t2[i], t3[i])`
    (where `0 <= i < num_elems`).

  * If `fn` returns a tuple (or nested structure) of tensors, then the
    result is formed by stacking corresponding elements from those structures.

  #### Specifying `fn`'s output signature

  If `fn`'s input and output signatures are different, then the output
  signature must be specified using `fn_output_signature`.  (The input and
  output signatures are differ if their structures, dtypes, or tensor types do
  not match).  E.g.:

  >>> tf.map_fn(fn=tf.strings.length,  # input & output have different dtypes
  ...           elems=tf.constant(["hello", "moon"]),
  ...           fn_output_signature=tf.int32)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([5, 4], dtype=int32)>
  >>> tf.map_fn(fn=tf.strings.join,  # input & output have different structures
  ...           elems=[tf.constant(['The', 'A']), tf.constant(['Dog', 'Cat'])],
  ...           fn_output_signature=tf.string)
  <tf.Tensor: shape=(2,), dtype=string,
   numpy=array([b'TheDog', b'ACat'], dtype=object)>

  `fn_output_signature` can be specified using any of the following:

  * A `tf.DType` or `tf.TensorSpec` (to describe a `tf.Tensor`)
  * A `tf.RaggedTensorSpec` (to describe a `tf.RaggedTensor`)
  * A `tf.SparseTensorSpec` (to describe a `tf.sparse.SparseTensor`)
  * A (possibly nested) tuple, list, or dict containing the above types.

  #### RaggedTensors

  `map_fn` supports `tf.RaggedTensor` inputs and outputs.  In particular:

  * If `elems` is a `RaggedTensor`, then `fn` will be called with each
    row of that ragged tensor.
    * If `elems` has only one ragged dimension, then the values passed to
      `fn` will be `tf.Tensor`s.
    * If `elems` has multiple ragged dimensions, then the values passed to
      `fn` will be `tf.RaggedTensor`s with one fewer ragged dimension.

  * If the result of `map_fn` should be a `RaggedTensor`, then use a
    `tf.RaggedTensorSpec` to specify `fn_output_signature`.
    * If `fn` returns `tf.Tensor`s with varying sizes, then use a
      `tf.RaggedTensorSpec` with `ragged_rank=0` to combine them into a
      single ragged tensor (which will have ragged_rank=1).
    * If `fn` returns `tf.RaggedTensor`s, then use a `tf.RaggedTensorSpec`
      with the same `ragged_rank`.

  >>> # Example: RaggedTensor input
  >>> rt = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])
  >>> tf.map_fn(tf.reduce_sum, rt, fn_output_signature=tf.int32)
  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([6, 0, 9, 6], dtype=int32)>

  >>> # Example: RaggedTensor output
  >>> elems = tf.constant([3, 5, 0, 2])
  >>> tf.map_fn(tf.range, elems,
  ...           fn_output_signature=tf.RaggedTensorSpec(shape=[None],
  ...                                                   dtype=tf.int32))
  <tf.RaggedTensor [[0, 1, 2], [0, 1, 2, 3, 4], [], [0, 1]]>

  Note: `map_fn` should only be used if you need to map a function over the
  *rows* of a `RaggedTensor`.  If you wish to map a function over the
  individual values, then you should use:

  * `tf.ragged.map_flat_values(fn, rt)`
    (if fn is expressible as TensorFlow ops)
  * `rt.with_flat_values(map_fn(fn, rt.flat_values))`
    (otherwise)

  E.g.:

  >>> rt = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])
  >>> tf.ragged.map_flat_values(lambda x: x + 2, rt)
  <tf.RaggedTensor [[3, 4, 5], [], [6, 7], [8]]>

  #### SparseTensors

  `map_fn` supports `tf.sparse.SparseTensor` inputs and outputs.  In particular:

  * If `elems` is a `SparseTensor`, then `fn` will be called with each row
    of that sparse tensor. In particular, the value passed to `fn` will be a
    `tf.sparse.SparseTensor` with one fewer dimension than `elems`.

  * If the result of `map_fn` should be a `SparseTensor`, then use a
    `tf.SparseTensorSpec` to specify `fn_output_signature`.  The individual
    `SparseTensor`s returned by `fn` will be stacked into a single
    `SparseTensor` with one more dimension.

  >>> # Example: SparseTensor input
  >>> st = tf.sparse.SparseTensor([[0, 0], [2, 0], [2, 1]], [2, 3, 4], [4, 4])
  >>> tf.map_fn(tf.sparse.reduce_sum, st, fn_output_signature=tf.int32)
  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([2, 0, 7, 0], dtype=int32)>

  >>> # Example: SparseTensor output
  >>> tf.sparse.to_dense(
  ...     tf.map_fn(tf.sparse.eye, tf.constant([2, 3]),
  ...               fn_output_signature=tf.SparseTensorSpec(None, tf.float32)))
  <tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
    array([[[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.]],
           [[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]]], dtype=float32)>

  Note: `map_fn` should only be used if you need to map a function over the
  *rows* of a `SparseTensor`.  If you wish to map a function over the nonzero
  values, then you should use:

  * If the function is expressible as TensorFlow ops, use:
    ```python
    tf.sparse.SparseTensor(st.indices, fn(st.values), st.dense_shape)
    ```
  * Otherwise, use:
    ```python
    tf.sparse.SparseTensor(st.indices, tf.map_fn(fn, st.values),
                           st.dense_shape)
    ```

  #### `map_fn` vs. vectorized operations

  `map_fn` will apply the operations used by `fn` to each element of `elems`,
  resulting in `O(elems.shape[0])` total operations.  This is somewhat
  mitigated by the fact that `map_fn` can process elements in parallel.
  However, a transform expressed using `map_fn` is still typically less
  efficient than an equivalent transform expressed using vectorized operations.

  `map_fn` should typically only be used if one of the following is true:

  * It is difficult or expensive to express the desired transform with
    vectorized operations.
  * `fn` creates large intermediate values, so an equivalent vectorized
    transform would take too much memory.
  * Processing elements in parallel is more efficient than an equivalent
    vectorized transform.
  * Efficiency of the transform is not critical, and using `map_fn` is
    more readable.

  E.g., the example given above that maps `fn=lambda t: tf.range(t, t + 3)`
  across `elems` could be rewritten more efficiently using vectorized ops:

  >>> elems = tf.constant([3, 5, 2])
  >>> tf.range(3) + tf.expand_dims(elems, 1)
  <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[3, 4, 5],
           [5, 6, 7],
           [2, 3, 4]], dtype=int32)>

  In some cases, `tf.vectorized_map` can be used to automatically convert a
  function to a vectorized eqivalent.

  #### Eager execution

  When executing eagerly, `map_fn` does not execute in parallel even if
  `parallel_iterations` is set to a value > 1. You can still get the
  performance benefits of running a function in parallel by using the
  `tf.function` decorator:

  >>> fn=lambda t: tf.range(t, t + 3)
  >>> @tf.function
  ... def func(elems):
  ...   return tf.map_fn(fn, elems, parallel_iterations=3)
  >>> func(tf.constant([3, 5, 2]))
  <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[3, 4, 5],
           [5, 6, 7],
           [2, 3, 4]], dtype=int32)>


  Note: if you use the `tf.function` decorator, any non-TensorFlow Python
  code that you may have written in your function won't get executed. See
  `tf.function` for more  details. The recommendation would be to debug without
  `tf.function` but switch to it to get performance benefits of running `map_fn`
  in parallel.

  Args:
    fn: The callable to be performed.  It accepts one argument, which will have
      the same (possibly nested) structure as `elems`.  Its output must have the
      same structure as `fn_output_signature` if one is provided; otherwise it
      must have the same structure as `elems`.
    elems: A tensor or (possibly nested) sequence of tensors, each of which will
      be unstacked along their first dimension.  `fn` will be applied to the
      nested sequence of the resulting slices.  `elems` may include ragged and
      sparse tensors. `elems` must consist of at least one tensor.
    dtype: Deprecated: Equivalent to `fn_output_signature`.
    parallel_iterations: (optional) The number of iterations allowed to run in
      parallel. When graph building, the default value is 10. While executing
      eagerly, the default value is set to 1.
    back_prop: (optional) False disables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    infer_shape: (optional) False disables tests for consistent output shapes.
    name: (optional) Name prefix for the returned tensors.
    fn_output_signature: The output signature of `fn`. Must be specified if
      `fn`'s input and output signatures are different (i.e., if their
      structures, dtypes, or tensor types do not match).
      `fn_output_signature` can be specified using any of the following:

      * A `tf.DType` or `tf.TensorSpec` (to describe a `tf.Tensor`)
      * A `tf.RaggedTensorSpec` (to describe a `tf.RaggedTensor`)
      * A `tf.SparseTensorSpec` (to describe a `tf.sparse.SparseTensor`)
      * A (possibly nested) tuple, list, or dict containing the above types.

  Returns:
    A tensor or (possibly nested) sequence of tensors.  Each tensor stacks the
    results of applying `fn` to tensors unstacked from `elems` along the first
    dimension, from first to last.  The result may include ragged and sparse
    tensors.

  Raises:
    TypeError: if `fn` is not callable or the structure of the output of
      `fn` and `fn_output_signature` do not match.
    ValueError: if the lengths of the output of `fn` and `fn_output_signature`
      do not match, or if the `elems` does not contain any tensor.

  Examples:

    >>> elems = np.array([1, 2, 3, 4, 5, 6])
    >>> tf.map_fn(lambda x: x * x, elems)
    <tf.Tensor: shape=(6,), dtype=int64, numpy=array([ 1,  4,  9, 16, 25, 36])>

    >>> elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
    >>> tf.map_fn(lambda x: x[0] * x[1], elems, fn_output_signature=tf.int64)
    <tf.Tensor: shape=(3,), dtype=int64, numpy=array([-1,  2, -3])>

    >>> elems = np.array([1, 2, 3])
    >>> tf.map_fn(lambda x: (x, -x), elems,
    ...          fn_output_signature=(tf.int64, tf.int64))
    (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([1, 2, 3])>,
     <tf.Tensor: shape=(3,), dtype=int64, numpy=array([-1, -2, -3])>)
  """
  # This function uses a `while_loop` to call `fn` on each value of the input
  # tensor(s) (unstacked on dimension 0).  The following sequence of variables
  # are used to transform the input tensor(s) (`elems`) into the output
  # tensor(s) (`result`):
  #
  #   - Preparing and unstacking input values for the while_loop:
  #     - elems: The input tensor(s) to map_fn. May include composite tensors.
  #     - elems_flat: Flattened list of tensors from elems (using nest.flatten)
  #                   May include composite tensors.
  #     - elems_batchable: Concatenation of "batchable tensor lists" for each
  #                        tensor in elems_flat.  This "boxes" composite tensors
  #                        into sliceable tf.Tensor objects.  For more info see:
  #                        TensorSpec._to_batched_tensor_list
  #     - elems_batchable_ta: List of TensorArrays used to unstack each Tensor
  #                           in elems_batchable into elems_value_batchable.
  #
  #   - Calling `fn` on each unstacked value in the body of the while_loop:
  #     - elems_value_batchable: Single unstacked value from elems_batchable.
  #     - elems_value_flat: Single unstacked value from elems_flat,
  #                         constructed from elems_value_batchable (using
  #                         TensorSpec._from_tensor_list).
  #     - elems_value: Single unstacked value from elems (the input to fn).
  #     - result_value: Result of calling `fn(elems_value)`.  May contain
  #                     composite tensors.
  #     - result_value_flat: Flattened list of tensors from result_value.
  #                          May contain composite tensors.
  #     - result_value_batchable: Concatenation of batchable tensor lists for
  #                               each tensor in result_value_flat
  #                               (using TensorSpec._to_tensor_list).
  #
  #   - Collecting and stacking output values from the while_loop:
  #     - result_batchable_ta: List of TensorArrays used to stack each tensor
  #                            ta result_value_batchable into result_batchable.
  #     - result_batchable: Stacked tensors from result_batchable_ta.
  #     - result_flat: Flat list of tensors for the result, constructed from
  #                    results bactchable (using TensorSpec._from_tensor_list).
  #     - result: Structured result value packed from results flat
  #               (using nest.pack_sequence_as).

  if fn_output_signature is None:
    fn_output_signature = dtype

  if not callable(fn):
    raise TypeError("fn must be callable.")

  in_graph_mode = not context.executing_eagerly()
  # Set the default number of parallel_iterations depending on graph/eager mode.
  if in_graph_mode and not parallel_iterations:
    parallel_iterations = 10
  elif not in_graph_mode and not parallel_iterations:
    parallel_iterations = 1
  elif not in_graph_mode and parallel_iterations > 1:
    logging.log_first_n(
        logging.WARN, "Setting parallel_iterations > 1 has no "
        "effect when executing eagerly. Consider calling map_fn"
        " with tf.function to execute fn in "
        "parallel.", 1)
    parallel_iterations = 1

  # Flatten the input tensors, and get the TypeSpec for each one.
  elems_flat = nest.flatten(elems)

  # Check in case this is an empty list
  if len(elems_flat) == 0:
    raise ValueError(
        "elems must be a Tensor or (possibly nested) sequence of Tensors. "
        "Got {}, which does not contain any Tensors.".format(elems))

  elems_flat_signature = [type_spec.type_spec_from_value(e) for e in elems_flat]
  elems_unflatten = lambda x: nest.pack_sequence_as(elems, x)

  # Flatten fn's output signature.
  if fn_output_signature is None:
    # If fn_output_signature was not specified, then assume that it matches the
    # input signature.
    result_flat_signature = [
        _most_general_compatible_type(s)._unbatch()  # pylint: disable=protected-access
        for s in elems_flat_signature
    ]
    result_unflatten = elems_unflatten
  else:
    result_flat_signature = [
        _dtype_to_spec(d) for d in nest.flatten(fn_output_signature)
    ]
    result_unflatten = lambda x: nest.pack_sequence_as(fn_output_signature, x)

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
        ops.convert_to_tensor_or_composite(t, name="elem") for t in elems_flat
    ]

    # Check that inputs are not scalars.
    elems_static_shape = elems_flat[0].shape
    if elems_static_shape.ndims is not None and elems_static_shape.ndims < 1:
      if len(elems_flat) == 1:
        raise ValueError("elems must be a 1+ dimensional Tensor, not a scalar")
      else:
        raise ValueError(
            "elements in elems must be 1+ dimensional Tensors, not scalars"
        )

    # Box any composite tensors into tensor lists.
    elems_batchable = _elems_flat_to_batchable(elems_flat)

    # Find the number of iterations, n.  (may be known statically.)
    n_static = tensor_shape.Dimension(
        tensor_shape.dimension_value(
            elems_batchable[0].get_shape().with_rank_at_least(1)[0]))
    for tensor in elems_batchable[1:]:
      n_static.merge_with(
          tensor_shape.Dimension(
              tensor_shape.dimension_value(
                  tensor.get_shape().with_rank_at_least(1)[0])))
    n = n_static.value or array_ops.shape(elems_batchable[0])[0]

    # Convert elems to tensor array.
    # TODO(edloper): Should we set infer_shape=False for composite tensors?
    elems_batchable_ta = [
        tensor_array_ops.TensorArray(
            dtype=t.dtype, size=n, dynamic_size=False, infer_shape=True)
        for t in elems_batchable
    ]
    # Unpack elements
    elems_batchable_ta = [
        ta.unstack(t) for (ta, t) in zip(elems_batchable_ta, elems_batchable)
    ]

    i = constant_op.constant(0)

    # Prepare result tensor array.
    # TODO(edloper): Should we set infer_shape=False for composite tensors?
    result_batchable_tensor_spec = (
        _result_flat_signature_to_batchable_tensor_spec(result_flat_signature))
    result_batchable_ta = []
    for spec in result_batchable_tensor_spec:
      result_batchable_ta.append(
          tensor_array_ops.TensorArray(
              dtype=spec.dtype, size=n, dynamic_size=False,
              infer_shape=infer_shape, element_shape=spec.shape))

    def compute(i, tas):
      """The loop body of map_fn.

      Args:
        i: the loop counter
        tas: the flat TensorArray accumulator list

      Returns:
        (i + 1, tas): the updated counter + updated TensorArrays

      Raises:
        TypeError: if fn_output_signature and result_value structure don't match
        ValueType: if fn_output_signature and result_value lengths don't match
      """
      elems_value_batchable = [ta.read(i) for ta in elems_batchable_ta]
      elems_value_flat = _elems_value_batchable_to_flat(elems_value_batchable,
                                                        elems_flat_signature)
      elems_value = elems_unflatten(elems_value_flat)
      ag_ctx = autograph_ctx.control_status_ctx()
      autographed_fn = autograph.tf_convert(fn, ag_ctx)
      result_value = autographed_fn(elems_value)
      nest.assert_same_structure(fn_output_signature or elems, result_value)
      result_value_flat = nest.flatten(result_value)
      result_value_batchable = _result_value_flat_to_batchable(
          result_value_flat, result_flat_signature)
      tas = [
          ta.write(i, value) for (ta, value) in zip(tas, result_value_batchable)
      ]
      return (i + 1, tas)

    _, r_a = control_flow_ops.while_loop(
        lambda i, _: i < n,
        compute, (i, result_batchable_ta),
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory,
        maximum_iterations=n)
    result_batchable = [r.stack() for r in r_a]

    # Update each output tensor w/ static shape info about the outer dimension.
    for r in result_batchable:
      r.set_shape(tensor_shape.TensorShape(n_static).concatenate(
          r.get_shape()[1:]))

    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode and varscope_caching_device_was_none:
      varscope.set_caching_device(None)

    result_flat = _result_batchable_to_flat(result_batchable,
                                            result_flat_signature)
    result = result_unflatten(result_flat)
    return result


def _dtype_to_spec(d):
  if not isinstance(d, type_spec.TypeSpec):
    d = tensor_spec.TensorSpec(None, d)
  return d


def _most_general_compatible_type(spec):
  """Returns the most general TypeSpec compatible with `spec`."""
  # TODO(edloper): Consider adding most_general_compatible_type to TypeSpec API
  if isinstance(spec, tensor_spec.TensorSpec):
    return tensor_spec.TensorSpec(None, spec.dtype)
  elif isinstance(spec, ragged_tensor.RaggedTensorSpec):
    # pylint: disable=protected-access
    return ragged_tensor.RaggedTensorSpec(None, spec._dtype, spec._ragged_rank,
                                          spec._row_splits_dtype)
  elif isinstance(spec, sparse_tensor.SparseTensorSpec):
    # pylint: disable=protected-access
    return sparse_tensor.SparseTensorSpec(None, spec.dtype)
  else:
    return spec


def _result_flat_signature_to_batchable_tensor_spec(result_flat_signature):
  """Converts result_flat_signature -> result_batchable_tensor_specs."""
  tensor_specs = []
  for spec in result_flat_signature:
    if not isinstance(spec, type_spec.BatchableTypeSpec):
      raise TypeError("map_fn can not generate %s outputs" % (spec,))
    tensor_specs.extend(spec._flat_tensor_specs)  # pylint: disable=protected-access
  return tensor_specs


def _elems_flat_to_batchable(elems_flat):
  """Converts elems_flat -> elems_batchable."""
  elems_batchable = []
  for elems_tensor in elems_flat:
    spec = type_spec.type_spec_from_value(elems_tensor)
    if not isinstance(spec, type_spec.BatchableTypeSpec):
      raise TypeError("map_fn can not consume %s inputs: got %r" %
                      (spec, elems_tensor))
    # pylint: disable=protected-access
    elems_batchable.extend(spec._to_batched_tensor_list(elems_tensor))
  return elems_batchable


def _elems_value_batchable_to_flat(elems_value_batchable, elems_flat_signature):
  """Converts elems_value_batchable -> elems_value_flat."""
  elems_value_flat = []
  i = 0
  for spec in elems_flat_signature:
    # pylint: disable=protected-access
    spec = spec._unbatch()
    tensor_list = elems_value_batchable[i:i + len(spec._flat_tensor_specs)]
    elems_value_flat.append(spec._from_compatible_tensor_list(tensor_list))
    i += len(tensor_list)
  assert i == len(elems_value_batchable)
  return elems_value_flat


def _result_value_flat_to_batchable(result_value_flat, result_flat_signature):
  """Converts result_value_flat -> result_value_batchable."""
  result_value_batchable = []
  for (r_value, r_spec) in zip(result_value_flat, result_flat_signature):
    if isinstance(r_spec, tensor_spec.TensorSpec):
      result_value_batchable.append(r_value)
    else:
      if not r_spec.is_compatible_with(r_value):
        raise ValueError(
            "Error in map_fn:\n  Expected `fn` to return a:\n    %s\n"
            "  But it returned a:\n    %s\n    (value=%s)\n"
            "  To fix, update the `fn_output_signature` (or `dtype`) "
            "argument to `map_fn`." %
            (r_spec, type_spec.type_spec_from_value(r_value), r_value))
      result_value_batchable.extend(r_spec._to_tensor_list(r_value))  # pylint: disable=protected-access
  return result_value_batchable


def _result_batchable_to_flat(result_batchable, result_flat_signature):
  """Converts result_batchable -> result_flat."""
  result_flat = []
  i = 0
  for spec in result_flat_signature:
    # pylint: disable=protected-access
    num_tensors = len(spec._flat_tensor_specs)
    result_flat.append(
        spec._batch(None)._from_compatible_tensor_list(
            result_batchable[i:i + num_tensors]))
    i += num_tensors
  assert i == len(result_batchable)
  return result_flat


@tf_export("map_fn", v1=[])
@deprecation.deprecated_arg_values(
    None,
    """back_prop=False is deprecated. Consider using tf.stop_gradient instead.
Instead of:
results = tf.map_fn(fn, elems, back_prop=False)
Use:
results = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(fn, elems))""",
    warn_once=True,
    back_prop=False)
@deprecation.deprecated_args(None, "Use fn_output_signature instead", "dtype")
def map_fn_v2(fn,
              elems,
              dtype=None,
              parallel_iterations=None,
              back_prop=True,
              swap_memory=False,
              infer_shape=True,
              name=None,
              fn_output_signature=None):
  """Transform `elems` by applying `fn` to each element unstacked on axis 0."""
  if fn_output_signature is None:
    fn_output_signature = dtype
  return map_fn(
      fn=fn,
      elems=elems,
      fn_output_signature=fn_output_signature,
      parallel_iterations=parallel_iterations,
      back_prop=back_prop,
      swap_memory=swap_memory,
      infer_shape=infer_shape,
      name=name)


# Docstring for v2 is the same as v1, except that back_prop is deprecated.
map_fn_v2.__doc__ = re.sub(
    r"(  back_prop: \(optional\) )(.*)",
    r"\1Deprecated: prefer using `tf.stop_gradient` instead.  \2",
    map_fn.__doc__)
assert "prefer using `tf.stop_gradient` instead" in map_fn_v2.__doc__
