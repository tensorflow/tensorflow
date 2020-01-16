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

# pylint: disable=g-short-docstring-punctuation
"""Sparse Tensor Representation.

See also `tf.SparseTensor`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_sparse_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import tf_export


def _convert_to_sparse_tensor(sp_input):
  """Convert `sp_input` to `SparseTensor` and return it.

  Args:
    sp_input: `SparseTensor` or `SparseTensorValue`.

  Returns:
    `sp_input` converted to `SparseTensor`.

  Raises:
    ValueError: if `sp_input` is neither `SparseTensor` nor `SparseTensorValue`.
  """
  if isinstance(sp_input, sparse_tensor.SparseTensorValue):
    return sparse_tensor.SparseTensor.from_value(sp_input)
  if not isinstance(sp_input, sparse_tensor.SparseTensor):
    raise TypeError("Input must be a SparseTensor.")
  return sp_input


def _convert_to_sparse_tensors(sp_inputs):
  """Convert `sp_inputs` to `SparseTensor` objects and return them.

  Args:
    sp_inputs: `list` or `tuple` of `SparseTensor` or `SparseTensorValue`
      objects.

  Returns:
    `sp_inputs` converted to `SparseTensor` objects.

  Raises:
    ValueError: if any item in `sp_inputs` is neither `SparseTensor` nor
      `SparseTensorValue`.
  """
  if isinstance(sp_inputs, list):
    return [_convert_to_sparse_tensor(sp_input) for sp_input in sp_inputs]
  if isinstance(sp_inputs, tuple):
    return (_convert_to_sparse_tensor(sp_input) for sp_input in sp_inputs)
  raise TypeError("Inputs must be a list or tuple.")


def _make_int64_tensor(value, name):
  if isinstance(value, compat.integral_types):
    return ops.convert_to_tensor(value, name=name, dtype=dtypes.int64)
  if not isinstance(value, ops.Tensor):
    raise TypeError("{} must be an integer value".format(name))
  if value.dtype == dtypes.int64:
    return value
  return math_ops.cast(value, dtypes.int64)


@tf_export("sparse.from_dense")
def from_dense(tensor, name=None):
  """Converts a dense tensor into a sparse tensor.

  Only elements not equal to zero will be present in the result. The resulting
  `SparseTensor` has the same dtype and shape as the input.

  Args:
    tensor: A dense `Tensor` to be converted to a `SparseTensor`.
    name: Optional name for the op.

  Returns:
    The `SparseTensor`.
  """
  with ops.name_scope(name, "dense_to_sparse"):
    tensor = ops.convert_to_tensor(tensor)
    indices = array_ops.where_v2(
        math_ops.not_equal(tensor, array_ops.constant(0, tensor.dtype)))
    values = array_ops.gather_nd(tensor, indices)
    shape = array_ops.shape(tensor, out_type=dtypes.int64)
    return sparse_tensor.SparseTensor(indices, values, shape)


@tf_export("sparse.expand_dims")
def sparse_expand_dims(sp_input, axis=None, name=None):
  """Inserts a dimension of 1 into a tensor's shape.

  Given a tensor `sp_input`, this operation inserts a dimension of 1 at the
  dimension index `axis` of `sp_input`'s shape. The dimension index `axis`
  starts at zero; if you specify a negative number for `axis` it is counted
  backwards from the end.

  Args:
    sp_input: A `SparseTensor`.
    axis: 0-D (scalar). Specifies the dimension index at which to expand the
      shape of `input`. Must be in the range `[-rank(sp_input) - 1,
      rank(sp_input)]`.
    name: The name of the output `SparseTensor`.

  Returns:
    A `SparseTensor` with the same data as `sp_input`, but its shape has an
    additional dimension of size 1 added.
  """
  rank = sp_input.dense_shape.get_shape()[0]
  axis = -1 if axis is None else axis

  with ops.name_scope(name, default_name="expand_dims", values=[sp_input]):
    if isinstance(axis, compat.integral_types):
      axis = ops.convert_to_tensor(axis, name="axis", dtype=dtypes.int32)
    elif not isinstance(axis, ops.Tensor):
      raise TypeError("axis must be an integer value in range [-rank(sp_input)"
                      " - 1, rank(sp_input)]")

    # Convert axis to a positive value if it is negative.
    axis = array_ops.where_v2(axis >= 0, axis, axis + rank + 1)

    # Create the new column of indices for the sparse tensor by slicing
    # the indices and inserting a new column of indices for the new dimension.
    column_size = array_ops.shape(sp_input.indices)[0]
    new_index = array_ops.zeros([column_size, 1], dtype=dtypes.int64)
    indices_before = array_ops.slice(sp_input.indices, [0, 0], [-1, axis])
    indices_after = array_ops.slice(sp_input.indices, [0, axis], [-1, -1])
    indices = array_ops.concat(
        [indices_before, new_index, indices_after], axis=1)

    # Create the new dense shape by splicing the tensor [1] in the correct
    # dimension of the existing shape.
    shape_before = array_ops.slice(sp_input.dense_shape, [0], [axis])
    shape_after = array_ops.slice(sp_input.dense_shape, [axis], [-1])
    new_shape = ops.convert_to_tensor([1], name="new_shape", dtype=dtypes.int64)
    shape = array_ops.concat([shape_before, new_shape, shape_after], axis=0)

    # Create the output sparse tensor.
    return sparse_tensor.SparseTensor(
        indices=indices, values=sp_input.values, dense_shape=shape)


@tf_export("sparse.eye")
def sparse_eye(num_rows,
               num_columns=None,
               dtype=dtypes.float32,
               name=None):
  """Creates a two-dimensional sparse tensor with ones along the diagonal.

  Args:
    num_rows: Non-negative integer or `int32` scalar `tensor` giving the number
      of rows in the resulting matrix.
    num_columns: Optional non-negative integer or `int32` scalar `tensor` giving
      the number of columns in the resulting matrix. Defaults to `num_rows`.
    dtype: The type of element in the resulting `Tensor`.
    name: A name for this `Op`. Defaults to "eye".

  Returns:
    A `SparseTensor` of shape [num_rows, num_columns] with ones along the
    diagonal.
  """
  with ops.name_scope(name, default_name="eye", values=[num_rows, num_columns]):
    num_rows = _make_int64_tensor(num_rows, "num_rows")
    num_columns = num_rows if num_columns is None else _make_int64_tensor(
        num_columns, "num_columns")

    # Create the sparse tensor.
    diag_size = math_ops.minimum(num_rows, num_columns)
    diag_range = math_ops.range(diag_size, dtype=dtypes.int64)

    return sparse_tensor.SparseTensor(
        indices=array_ops.stack([diag_range, diag_range], axis=1),
        values=array_ops.ones(diag_size, dtype=dtype),
        dense_shape=[num_rows, num_columns])


# pylint: disable=protected-access
@tf_export(v1=["sparse.concat", "sparse_concat"])
@deprecation.deprecated_endpoints("sparse_concat")
@deprecation.deprecated_args(
    None, "concat_dim is deprecated, use axis instead", "concat_dim")
def sparse_concat(axis,
                  sp_inputs,
                  name=None,
                  expand_nonconcat_dim=False,
                  concat_dim=None,
                  expand_nonconcat_dims=None):
  """Concatenates a list of `SparseTensor` along the specified dimension.

  Concatenation is with respect to the dense versions of each sparse input.
  It is assumed that each inputs is a `SparseTensor` whose elements are ordered
  along increasing dimension number.

  If expand_nonconcat_dim is False, all inputs' shapes must match, except for
  the concat dimension. If expand_nonconcat_dim is True, then inputs' shapes are
  allowed to vary among all inputs.

  The `indices`, `values`, and `shapes` lists must have the same length.

  If expand_nonconcat_dim is False, then the output shape is identical to the
  inputs', except along the concat dimension, where it is the sum of the inputs'
  sizes along that dimension.

  If expand_nonconcat_dim is True, then the output shape along the non-concat
  dimensions will be expand to be the largest among all inputs, and it is the
  sum of the inputs sizes along the concat dimension.

  The output elements will be resorted to preserve the sort order along
  increasing dimension number.

  This op runs in `O(M log M)` time, where `M` is the total number of non-empty
  values across all inputs. This is due to the need for an internal sort in
  order to concatenate efficiently across an arbitrary dimension.

  For example, if `axis = 1` and the inputs are

      sp_inputs[0]: shape = [2, 3]
      [0, 2]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      sp_inputs[1]: shape = [2, 4]
      [0, 1]: "d"
      [0, 2]: "e"

  then the output will be

      shape = [2, 7]
      [0, 2]: "a"
      [0, 4]: "d"
      [0, 5]: "e"
      [1, 0]: "b"
      [1, 1]: "c"

  Graphically this is equivalent to doing

      [    a] concat [  d e  ] = [    a   d e  ]
      [b c  ]        [       ]   [b c          ]

  Another example, if 'axis = 1' and the inputs are

      sp_inputs[0]: shape = [3, 3]
      [0, 2]: "a"
      [1, 0]: "b"
      [2, 1]: "c"

      sp_inputs[1]: shape = [2, 4]
      [0, 1]: "d"
      [0, 2]: "e"

  if expand_nonconcat_dim = False, this will result in an error. But if
  expand_nonconcat_dim = True, this will result in:

      shape = [3, 7]
      [0, 2]: "a"
      [0, 4]: "d"
      [0, 5]: "e"
      [1, 0]: "b"
      [2, 1]: "c"

  Graphically this is equivalent to doing

      [    a] concat [  d e  ] = [    a   d e  ]
      [b    ]        [       ]   [b            ]
      [  c  ]                    [  c          ]


  Args:
    axis: Dimension to concatenate along. Must be in range [-rank, rank),
      where rank is the number of dimensions in each input `SparseTensor`.
    sp_inputs: List of `SparseTensor` to concatenate.
    name: A name prefix for the returned tensors (optional).
    expand_nonconcat_dim: Whether to allow the expansion in the non-concat
      dimensions. Defaulted to False.
    concat_dim: The old (deprecated) name for axis.
    expand_nonconcat_dims: alias for expand_nonconcat_dim

  Returns:
    A `SparseTensor` with the concatenated output.

  Raises:
    TypeError: If `sp_inputs` is not a list of `SparseTensor`.
  """
  expand_nonconcat_dim = deprecation.deprecated_argument_lookup(
      "expand_nonconcat_dims", expand_nonconcat_dims,
      "expand_nonconcat_dim", expand_nonconcat_dim)
  if expand_nonconcat_dims is not None:
    expand_nonconcat_dim = expand_nonconcat_dims
  axis = deprecation.deprecated_argument_lookup("axis", axis, "concat_dim",
                                                concat_dim)
  return sparse_concat_v2(axis, sp_inputs, expand_nonconcat_dim, name)


@tf_export("sparse.concat", v1=[])
def sparse_concat_v2(axis, sp_inputs, expand_nonconcat_dims=False, name=None):  # pylint: disable=missing-docstring
  sp_inputs = _convert_to_sparse_tensors(sp_inputs)

  if len(sp_inputs) == 1:  # Degenerate case of one tensor.
    return sp_inputs[0]

  inds = [sp_input.indices for sp_input in sp_inputs]
  vals = [sp_input.values for sp_input in sp_inputs]
  shapes = [sp_input.dense_shape for sp_input in sp_inputs]

  if expand_nonconcat_dims:
    max_shape = math_ops.reduce_max(
        array_ops.concat(
            [array_ops.reshape(shape, [1, -1]) for shape in shapes], 0), 0)
    shapes = [
        array_ops.concat([
            max_shape[:axis], shape[-1:]
            if axis == -1 else shape[axis:axis + 1], []
            if axis == -1 else max_shape[axis + 1:]
        ], 0) for shape in shapes
    ]

  output_ind, output_val, output_shape = (
      gen_sparse_ops.sparse_concat(inds, vals, shapes, axis, name=name))

  shapes_value = [tensor_util.constant_value(shape) for shape in shapes]
  if shapes_value and all(shape is not None for shape in shapes_value):
    dim = sum(shape[axis] for shape in shapes_value)
    output_shape = shapes_value[0]
    output_shape[axis] = dim
    output_shape = ops.convert_to_tensor(output_shape)
  return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)


sparse_concat_v2.__doc__ = sparse_concat.__doc__.replace(
    "    concat_dim: The old (deprecated) name for axis.\n", "")


@tf_export(v1=["sparse.add", "sparse_add"])
@deprecation.deprecated_endpoints("sparse_add")
@deprecation.deprecated_args(
    None, "thresh is deprecated, use threshold instead", "thresh")
def sparse_add(a, b, threshold=None, thresh=None):
  """Adds two tensors, at least one of each is a `SparseTensor`.

  If one `SparseTensor` and one `Tensor` are passed in, returns a `Tensor`.  If
  both arguments are `SparseTensor`s, this returns a `SparseTensor`.  The order
  of arguments does not matter.  Use vanilla `tf.add()` for adding two dense
  `Tensor`s.

  The shapes of the two operands must match: broadcasting is not supported.

  The indices of any input `SparseTensor` are assumed ordered in standard
  lexicographic order.  If this is not the case, before this step run
  `SparseReorder` to restore index ordering.

  If both arguments are sparse, we perform "clipping" as follows.  By default,
  if two values sum to zero at some index, the output `SparseTensor` would still
  include that particular location in its index, storing a zero in the
  corresponding value slot.  To override this, callers can specify `thresh`,
  indicating that if the sum has a magnitude strictly smaller than `thresh`, its
  corresponding value and index would then not be included.  In particular,
  `thresh == 0.0` (default) means everything is kept and actual thresholding
  happens only for a positive value.

  For example, suppose the logical sum of two sparse operands is (densified):

      [       2]
      [.1     0]
      [ 6   -.2]

  Then,

  * `thresh == 0` (the default): all 5 index/value pairs will be returned.
  * `thresh == 0.11`: only .1 and 0 will vanish, and the remaining three
      index/value pairs will be returned.
  * `thresh == 0.21`: .1, 0, and -.2 will vanish.

  Args:
    a: The first operand; `SparseTensor` or `Tensor`.
    b: The second operand; `SparseTensor` or `Tensor`. At least one operand
      must be sparse.
    threshold: An optional 0-D `Tensor` (defaults to `0`). The magnitude
      threshold that determines if an output value/index pair takes space. Its
      dtype should match that of the values if they are real; if the latter are
      complex64/complex128, then the dtype should be float32/float64,
      correspondingly.
    thresh: Deprecated alias for `threshold`.

  Returns:
    A `SparseTensor` or a `Tensor`, representing the sum.

  Raises:
    TypeError: If both `a` and `b` are `Tensor`s.  Use `tf.add()` instead.
  """
  threshold = deprecation.deprecated_argument_lookup("threshold", threshold,
                                                     "thresh", thresh)
  if threshold is None:
    threshold = 0
  return sparse_add_v2(a, b, threshold)


@tf_export("sparse.add", v1=[])
def sparse_add_v2(a, b, threshold=0):
  """Adds two tensors, at least one of each is a `SparseTensor`.

  If one `SparseTensor` and one `Tensor` are passed in, returns a `Tensor`.  If
  both arguments are `SparseTensor`s, this returns a `SparseTensor`.  The order
  of arguments does not matter.  Use vanilla `tf.add()` for adding two dense
  `Tensor`s.

  The shapes of the two operands must match: broadcasting is not supported.

  The indices of any input `SparseTensor` are assumed ordered in standard
  lexicographic order.  If this is not the case, before this step run
  `SparseReorder` to restore index ordering.

  If both arguments are sparse, we perform "clipping" as follows.  By default,
  if two values sum to zero at some index, the output `SparseTensor` would still
  include that particular location in its index, storing a zero in the
  corresponding value slot.  To override this, callers can specify `threshold`,
  indicating that if the sum has a magnitude strictly smaller than `threshold`,
  its corresponding value and index would then not be included.  In particular,
  `threshold == 0.0` (default) means everything is kept and actual thresholding
  happens only for a positive value.

  For example, suppose the logical sum of two sparse operands is (densified):

      [       2]
      [.1     0]
      [ 6   -.2]

  Then,

  * `threshold == 0` (the default): all 5 index/value pairs will be
      returned.
  * `threshold == 0.11`: only .1 and 0 will vanish, and the remaining three
      index/value pairs will be returned.
  * `threshold == 0.21`: .1, 0, and -.2 will vanish.

  Args:
    a: The first operand; `SparseTensor` or `Tensor`.
    b: The second operand; `SparseTensor` or `Tensor`. At least one operand
      must be sparse.
    threshold: A 0-D `Tensor`. The magnitude threshold that determines if an
      output value/index pair takes space. Its dtype should match that of the
      values if they are real; if the latter are complex64/complex128, then the
      dtype should be float32/float64, correspondingly.

  Returns:
    A `SparseTensor` or a `Tensor`, representing the sum.

  Raises:
    TypeError: If both `a` and `b` are `Tensor`s.  Use `tf.add()` instead.
  """
  sparse_classes = (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)
  if not any(isinstance(inp, sparse_classes) for inp in [a, b]):
    raise TypeError("At least one input should be SparseTensor; do you mean to"
                    " use tf.add()?")

  if all(isinstance(inp, sparse_classes) for inp in [a, b]):
    a = _convert_to_sparse_tensor(a)
    b = _convert_to_sparse_tensor(b)
    threshold = ops.convert_to_tensor(
        threshold, dtype=a.values.dtype.real_dtype.base_dtype, name="threshold")
    output_ind, output_val, output_shape = (
        gen_sparse_ops.sparse_add(a.indices, a.values, a.dense_shape,
                                  b.indices, b.values, b.dense_shape,
                                  threshold))

    # Attempt to get output_shape statically.
    a.get_shape().assert_is_compatible_with(b.get_shape())
    static_shape = array_ops.broadcast_static_shape(a.get_shape(),
                                                    b.get_shape())
    if static_shape.is_fully_defined():
      output_shape = static_shape.as_list()

    return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)
  else:
    # swap to make `a` the SparseTensor.
    if isinstance(b, sparse_classes):
      a, b = b, a
    return gen_sparse_ops.sparse_tensor_dense_add(a.indices, a.values,
                                                  a.dense_shape, b)


@tf_export("sparse.cross")
def sparse_cross(inputs, name=None):
  """Generates sparse cross from a list of sparse and dense tensors.

  For example, if the inputs are

      * inputs[0]: SparseTensor with shape = [2, 2]
        [0, 0]: "a"
        [1, 0]: "b"
        [1, 1]: "c"
      * inputs[1]: SparseTensor with shape = [2, 1]
        [0, 0]: "d"
        [1, 0]: "e"
      * inputs[2]: Tensor [["f"], ["g"]]

  then the output will be:

      shape = [2, 2]
      [0, 0]: "a_X_d_X_f"
      [1, 0]: "b_X_e_X_g"
      [1, 1]: "c_X_e_X_g"

  Args:
    inputs: An iterable of `Tensor` or `SparseTensor`.
    name: Optional name for the op.

  Returns:
    A `SparseTensor` of type `string`.
  """
  return _sparse_cross_internal(inputs=inputs, hashed_output=False, name=name)


_sparse_cross = sparse_cross


@tf_export("sparse.cross_hashed")
def sparse_cross_hashed(inputs, num_buckets=0, hash_key=None, name=None):
  """Generates hashed sparse cross from a list of sparse and dense tensors.

  For example, if the inputs are

      * inputs[0]: SparseTensor with shape = [2, 2]
        [0, 0]: "a"
        [1, 0]: "b"
        [1, 1]: "c"
      * inputs[1]: SparseTensor with shape = [2, 1]
        [0, 0]: "d"
        [1, 0]: "e"
      * inputs[2]: Tensor [["f"], ["g"]]

  then the output will be:

      shape = [2, 2]
      [0, 0]: FingerprintCat64(
                  Fingerprint64("f"), FingerprintCat64(
                      Fingerprint64("d"), Fingerprint64("a")))
      [1, 0]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("b")))
      [1, 1]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("c")))

  Args:
    inputs: An iterable of `Tensor` or `SparseTensor`.
    num_buckets: An `int` that is `>= 0`.
      output = hashed_value%num_buckets if num_buckets > 0 else hashed_value.
    hash_key: Integer hash_key that will be used by the `FingerprintCat64`
      function. If not given, will use a default key.
    name: Optional name for the op.

  Returns:
    A `SparseTensor` of type `int64`.
  """
  return _sparse_cross_internal(
      inputs=inputs,
      hashed_output=True,
      num_buckets=num_buckets,
      hash_key=hash_key,
      name=name)


_sparse_cross_hashed = sparse_cross_hashed

_DEFAULT_HASH_KEY = 0xDECAFCAFFE


def _sparse_cross_internal(inputs,
                           hashed_output=False,
                           num_buckets=0,
                           hash_key=None,
                           name=None):
  """See gen_sparse_ops.sparse_cross."""
  if not isinstance(inputs, list):
    raise TypeError("Inputs must be a list")
  if not all(
      isinstance(i, sparse_tensor.SparseTensor) or isinstance(i, ops.Tensor)
      for i in inputs):
    raise TypeError("All inputs must be SparseTensors")

  sparse_inputs = [
      i for i in inputs if isinstance(i, sparse_tensor.SparseTensor)
  ]
  dense_inputs = [
      i for i in inputs if not isinstance(i, sparse_tensor.SparseTensor)
  ]

  indices = [sp_input.indices for sp_input in sparse_inputs]
  values = [sp_input.values for sp_input in sparse_inputs]
  shapes = [sp_input.dense_shape for sp_input in sparse_inputs]
  out_type = dtypes.int64 if hashed_output else dtypes.string

  internal_type = dtypes.string
  for i in range(len(values)):
    if values[i].dtype != dtypes.string:
      values[i] = math_ops.cast(values[i], dtypes.int64)
      internal_type = dtypes.int64
  for i in range(len(dense_inputs)):
    if dense_inputs[i].dtype != dtypes.string:
      dense_inputs[i] = math_ops.cast(dense_inputs[i], dtypes.int64)
      internal_type = dtypes.int64

  indices_out, values_out, shape_out = gen_sparse_ops.sparse_cross(
      indices=indices,
      values=values,
      shapes=shapes,
      dense_inputs=dense_inputs,
      hashed_output=hashed_output,
      num_buckets=num_buckets,
      hash_key=hash_key or _DEFAULT_HASH_KEY,
      out_type=out_type,
      internal_type=internal_type,
      name=name)

  return sparse_tensor.SparseTensor(indices_out, values_out, shape_out)


def sparse_dense_cwise_add(sp_t, dense_t):
  """Adds up a SparseTensor and a dense Tensor, using these special rules:

  (1) Broadcasts the dense side to have the same shape as the sparse side, if
      eligible;
  (2) Then, only the dense values pointed to by the indices of the SparseTensor
      participate in the cwise addition.

  By the rules, the result is a logical SparseTensor with exactly the same
  indices and shape, but possibly with different non-zero values.  The output of
  this Op is the resultant non-zero values.

  Args:
    sp_t: the SparseTensor operand.
    dense_t: the dense Tensor operand; must have the same dtype and a
      broadcast-compatible shape as `sp_t`.

  Returns:
    output: the SparseTensor output.
  """
  result = gen_sparse_ops.sparse_dense_cwise_add(sp_t.indices, sp_t.values,
                                                 sp_t.dense_shape, dense_t)
  return sparse_tensor.SparseTensor(sp_t.indices, result, sp_t.dense_shape)


@tf_export("sparse.reorder", v1=["sparse.reorder", "sparse_reorder"])
@deprecation.deprecated_endpoints("sparse_reorder")
def sparse_reorder(sp_input, name=None):
  """Reorders a `SparseTensor` into the canonical, row-major ordering.

  Note that by convention, all sparse ops preserve the canonical ordering
  along increasing dimension number. The only time ordering can be violated
  is during manual manipulation of the indices and values to add entries.

  Reordering does not affect the shape of the `SparseTensor`.

  For example, if `sp_input` has shape `[4, 5]` and `indices` / `values`:

      [0, 3]: b
      [0, 1]: a
      [3, 1]: d
      [2, 0]: c

  then the output will be a `SparseTensor` of shape `[4, 5]` and
  `indices` / `values`:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d

  Args:
    sp_input: The input `SparseTensor`.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A `SparseTensor` with the same shape and non-empty values, but in
    canonical ordering.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)

  reordered_ind, reordered_val = (
      gen_sparse_ops.sparse_reorder(
          sp_input.indices, sp_input.values, sp_input.dense_shape, name=name))

  if sp_input.get_shape().is_fully_defined():
    dense_shape = sp_input.get_shape().as_list()
  else:
    dense_shape = array_ops.identity(sp_input.dense_shape)

  return sparse_tensor.SparseTensor(reordered_ind, reordered_val, dense_shape)


@tf_export("sparse.reshape", v1=["sparse.reshape", "sparse_reshape"])
@deprecation.deprecated_endpoints("sparse_reshape")
def sparse_reshape(sp_input, shape, name=None):
  """Reshapes a `SparseTensor` to represent values in a new dense shape.

  This operation has the same semantics as `reshape` on the represented dense
  tensor.  The indices of non-empty values in `sp_input` are recomputed based
  on the new dense shape, and a new `SparseTensor` is returned containing the
  new indices and new shape.  The order of non-empty values in `sp_input` is
  unchanged.

  If one component of `shape` is the special value -1, the size of that
  dimension is computed so that the total dense size remains constant.  At
  most one component of `shape` can be -1.  The number of dense elements
  implied by `shape` must be the same as the number of dense elements
  originally represented by `sp_input`.

  For example, if `sp_input` has shape `[2, 3, 6]` and `indices` / `values`:

      [0, 0, 0]: a
      [0, 0, 1]: b
      [0, 1, 0]: c
      [1, 0, 0]: d
      [1, 2, 3]: e

  and `shape` is `[9, -1]`, then the output will be a `SparseTensor` of
  shape `[9, 4]` and `indices` / `values`:

      [0, 0]: a
      [0, 1]: b
      [1, 2]: c
      [4, 2]: d
      [8, 1]: e

  Args:
    sp_input: The input `SparseTensor`.
    shape: A 1-D (vector) int64 `Tensor` specifying the new dense shape of the
      represented `SparseTensor`.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A `SparseTensor` with the same non-empty values but with indices calculated
    by the new dense shape.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
    ValueError:  If argument `shape` requests a `SparseTensor` with a different
      number of elements than `sp_input`.
    ValueError:  If `shape` has more than one inferred (== -1) dimension.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  shape = math_ops.cast(shape, dtype=dtypes.int64)

  with ops.name_scope(name, "SparseReshape", [sp_input]) as name:
    reshaped_ind, reshaped_shape = gen_sparse_ops.sparse_reshape(
        sp_input.indices, sp_input.dense_shape, shape, name=name)

    reshaped_shape_const = tensor_util.constant_value_as_shape(shape)
    reshaped_shape_const = (
        reshaped_shape_const.as_list() if reshaped_shape_const.ndims is not None
        else None)

    if (reshaped_shape_const is not None
        and sp_input.shape.is_fully_defined()):
      # constant_value_as_shape tends to get more information about the partial
      # shape values, but here we specifically need to know if the *user* passed
      # a shape with 2+ unknown dimensions; and for that constant_value
      # provides either the user's direct value or None if only partial elements
      # are known via the python shape inference code.
      shape_const_by_user = tensor_util.constant_value(shape)
      if shape_const_by_user is not None:
        num_implied_by_user = sum(d == -1 for d in shape_const_by_user)
        if num_implied_by_user > 1:
          raise ValueError(
              "At most one dimension can be inferred (-1). Found: %s"
              % shape_const_by_user)
      original_reshaped_shape = list(reshaped_shape_const)  # A copy
      in_shape_size = np.prod(sp_input.shape.as_list())
      num_implied = sum(dim is None for dim in reshaped_shape_const)
      if num_implied == 1:
        implied_idx = original_reshaped_shape.index(None)
        non_implied_idx = (
            original_reshaped_shape[:implied_idx] +
            original_reshaped_shape[implied_idx + 1:])
        reshaped_shape_const[implied_idx] = int(
            in_shape_size // np.prod(non_implied_idx))
      if num_implied <= 1:
        reshaped_size = np.prod(reshaped_shape_const)
        if reshaped_size != in_shape_size:
          raise ValueError(
              "Cannot reshape a tensor with %d elements to shape %s "
              "(%d elements)." %
              (in_shape_size, original_reshaped_shape, reshaped_size))
        reshaped_shape = constant_op.constant(
            reshaped_shape_const, dtype=dtypes.int64)

    return sparse_tensor.SparseTensor(reshaped_ind,
                                      array_ops.identity(sp_input.values),
                                      reshaped_shape)


# TODO(aselle): Remove keyword required once for 1.0 final
class KeywordRequired(object):

  def __repr__(self):
    # This is needed to make documentation without fully qualified module paths
    return "KeywordRequired()"


@tf_export(v1=["sparse.split", "sparse_split"])
@deprecation.deprecated_endpoints("sparse_split")
@deprecation.deprecated_args(
    None, "split_dim is deprecated, use axis instead", "split_dim")
def sparse_split(keyword_required=KeywordRequired(),
                 sp_input=None,
                 num_split=None,
                 axis=None,
                 name=None,
                 split_dim=None):
  """Split a `SparseTensor` into `num_split` tensors along `axis`.

  If the `sp_input.dense_shape[axis]` is not an integer multiple of `num_split`
  each slice starting from 0:`shape[axis] % num_split` gets extra one
  dimension. For example, if `axis = 1` and `num_split = 2` and the
  input is:

      input_tensor = shape = [2, 7]
      [    a   d e  ]
      [b c          ]

  Graphically the output tensors are:

      output_tensor[0] =
      [    a   ]
      [b c     ]

      output_tensor[1] =
      [ d e  ]
      [      ]

  Args:
    keyword_required: Python 2 standin for * (temporary for argument reorder)
    sp_input: The `SparseTensor` to split.
    num_split: A Python integer. The number of ways to split.
    axis: A 0-D `int32` `Tensor`. The dimension along which to split.
    name: A name for the operation (optional).
    split_dim: Deprecated old name for axis.

  Returns:
    `num_split` `SparseTensor` objects resulting from splitting `value`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
    ValueError: If the deprecated `split_dim` and `axis` are both non None.
  """
  if not isinstance(keyword_required, KeywordRequired):
    raise ValueError("Keyword arguments are required for this function.")
  if sp_input is None:
    raise ValueError("sp_input is required")
  if num_split is None:
    raise ValueError("num_split is required")
  if axis is None:
    raise ValueError("axis is required")
  axis = deprecation.deprecated_argument_lookup("axis", axis, "split_dim",
                                                split_dim)
  sp_input = _convert_to_sparse_tensor(sp_input)

  output_inds, output_vals, output_shapes = (
      gen_sparse_ops.sparse_split(
          axis,
          sp_input.indices,
          sp_input.values,
          sp_input.dense_shape,
          num_split,
          name=name))
  sparse_tensors = []
  for i in range(0, num_split):
    sparse_tensors.append(
        sparse_tensor.SparseTensor(output_inds[i], output_vals[i],
                                   output_shapes[i]))
  return sparse_tensors


@tf_export("sparse.split", v1=[])
def sparse_split_v2(sp_input=None,
                    num_split=None,
                    axis=None,
                    name=None):
  """Split a `SparseTensor` into `num_split` tensors along `axis`.

  If the `sp_input.dense_shape[axis]` is not an integer multiple of `num_split`
  each slice starting from 0:`shape[axis] % num_split` gets extra one
  dimension. For example, if `axis = 1` and `num_split = 2` and the
  input is:

      input_tensor = shape = [2, 7]
      [    a   d e  ]
      [b c          ]

  Graphically the output tensors are:

      output_tensor[0] =
      [    a ]
      [b c   ]

      output_tensor[1] =
      [ d e  ]
      [      ]

  Args:
    sp_input: The `SparseTensor` to split.
    num_split: A Python integer. The number of ways to split.
    axis: A 0-D `int32` `Tensor`. The dimension along which to split.
    name: A name for the operation (optional).

  Returns:
    `num_split` `SparseTensor` objects resulting from splitting `value`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  return sparse_split(sp_input=sp_input,
                      num_split=num_split,
                      axis=axis,
                      name=name,
                      split_dim=None)


@tf_export("sparse.slice", v1=["sparse.slice", "sparse_slice"])
@deprecation.deprecated_endpoints("sparse_slice")
def sparse_slice(sp_input, start, size, name=None):
  """Slice a `SparseTensor` based on the `start` and `size.

  For example, if the input is

      input_tensor = shape = [2, 7]
      [    a   d e  ]
      [b c          ]

  Graphically the output tensors are:

      sparse.slice([0, 0], [2, 4]) = shape = [2, 4]
      [    a  ]
      [b c    ]

      sparse.slice([0, 4], [2, 3]) = shape = [2, 3]
      [ d e  ]
      [      ]

  Args:
    sp_input: The `SparseTensor` to split.
    start: 1-D. tensor represents the start of the slice.
    size: 1-D. tensor represents the size of the slice.
    name: A name for the operation (optional).

  Returns:
    A `SparseTensor` objects resulting from splicing.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  start = ops.convert_to_tensor(start, dtypes.int64)
  size = ops.convert_to_tensor(size, dtypes.int64)

  with ops.name_scope(name, "SparseSlice", [sp_input]) as name:
    output_indices, output_values, output_shape = gen_sparse_ops.sparse_slice(
        sp_input.indices,
        sp_input.values,
        sp_input.dense_shape,
        start,
        size,
        name=name)

    return sparse_tensor.SparseTensor(output_indices, output_values,
                                      output_shape)


@tf_export(v1=["sparse_to_dense"])
@deprecation.deprecated(
    None,
    "Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.")
def sparse_to_dense(sparse_indices,
                    output_shape,
                    sparse_values,
                    default_value=0,
                    validate_indices=True,
                    name=None):
  """Converts a sparse representation into a dense tensor.

  Builds an array `dense` with shape `output_shape` such that

  ```python
  # If sparse_indices is scalar
  dense[i] = (i == sparse_indices ? sparse_values : default_value)

  # If sparse_indices is a vector, then for each i
  dense[sparse_indices[i]] = sparse_values[i]

  # If sparse_indices is an n by d matrix, then for each i in [0, n)
  dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
  ```

  All other values in `dense` are set to `default_value`.  If `sparse_values`
  is a scalar, all sparse indices are set to this single value.

  Indices should be sorted in lexicographic order, and indices must not
  contain any repeats. If `validate_indices` is True, these properties
  are checked during execution.

  Args:
    sparse_indices: A 0-D, 1-D, or 2-D `Tensor` of type `int32` or `int64`.
      `sparse_indices[i]` contains the complete index where `sparse_values[i]`
      will be placed.
    output_shape: A 1-D `Tensor` of the same type as `sparse_indices`.  Shape
      of the dense output tensor.
    sparse_values: A 0-D or 1-D `Tensor`.  Values corresponding to each row of
      `sparse_indices`, or a scalar value to be used for all sparse indices.
    default_value: A 0-D `Tensor` of the same type as `sparse_values`.  Value
      to set for indices not specified in `sparse_indices`.  Defaults to zero.
    validate_indices: A boolean value.  If True, indices are checked to make
      sure they are sorted in lexicographic order and that there are no repeats.
    name: A name for the operation (optional).

  Returns:
    Dense `Tensor` of shape `output_shape`.  Has the same type as
    `sparse_values`.
  """
  return gen_sparse_ops.sparse_to_dense(
      sparse_indices,
      output_shape,
      sparse_values,
      default_value=default_value,
      validate_indices=validate_indices,
      name=name)


@tf_export("sparse.reduce_max", v1=[])
def sparse_reduce_max_v2(
    sp_input, axis=None, keepdims=None, output_is_sparse=False, name=None):
  """Computes the max of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_max()`.  In particular, this Op also returns a dense `Tensor`
  if `output_is_sparse` is `False`, or a `SparseTensor` if `output_is_sparse`
  is `True`.

  Note: A gradient is not defined for this function, so it can't be used
  in training models that need gradient descent.

  Reduces `sp_input` along the dimensions given in `axis`.  Unless
  `keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keepdims` is true, the reduced dimensions are retained
  with length 1.

  If `axis` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  similar to the indexing rules in Python.

  The values not defined in `sp_input` don't participate in the reduce max,
  as opposed to be implicitly assumed 0 -- hence it can return negative values
  for sparse `axis`. But, in case there are no values in
  `axis`, it will reduce to 0. See second example below.

  For example:

  ```python
  # 'x' represents [[1, ?, 2]
  #                 [?, 3, ?]]
  # where ? is implicitly-zero.
  tf.sparse.reduce_max(x) ==> 3
  tf.sparse.reduce_max(x, 0) ==> [1, 3, 2]
  tf.sparse.reduce_max(x, 1) ==> [2, 3]  # Can also use -1 as the axis.
  tf.sparse.reduce_max(x, 1, keepdims=True) ==> [[2], [3]]
  tf.sparse.reduce_max(x, [0, 1]) ==> 3

  # 'y' represents [[-7, ?]
  #                 [ 4, 3]
  #                 [ ?, ?]
  tf.sparse.reduce_max(x, 1) ==> [-7, 4, 0]
  ```

  Args:
    sp_input: The SparseTensor to reduce. Should have numeric type.
    axis: The dimensions to reduce; list or scalar. If `None` (the
      default), reduces all dimensions.
    keepdims: If true, retain reduced dimensions with length 1.
    output_is_sparse: If true, returns a `SparseTensor` instead of a dense
      `Tensor` (the default).
    name: A name for the operation (optional).

  Returns:
    The reduced Tensor or the reduced SparseTensor if `output_is_sparse` is
    True.
  """
  if keepdims is None:
    keepdims = False

  # reduction_axes is the deprecated name for axis.
  reduction_axes = None

  if output_is_sparse:
    output_ind, output_val, output_shape = (
        gen_sparse_ops.sparse_reduce_max_sparse(
            sp_input.indices, sp_input.values, sp_input.dense_shape,
            math_ops._ReductionDims(sp_input, axis, reduction_axes), keepdims,
            name=name))

    return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)

  return gen_sparse_ops.sparse_reduce_max(
      sp_input.indices, sp_input.values, sp_input.dense_shape,
      math_ops._ReductionDims(sp_input, axis, reduction_axes), keepdims,
      name=name)


@tf_export(v1=["sparse.reduce_max", "sparse_reduce_max"])
@deprecation.deprecated_endpoints("sparse_reduce_max")
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
@deprecation.deprecated_args(
    None, "reduction_axes is deprecated, use axis instead",
    "reduction_axes")
def sparse_reduce_max(sp_input, axis=None, keepdims=None,
                      reduction_axes=None, keep_dims=None):
  """Computes the max of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_max()`.  In particular, this Op also returns a dense `Tensor`
  instead of a sparse one.

  Note: A gradient is not defined for this function, so it can't be used
  in training models that need gradient descent.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keepdims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  similar to the indexing rules in Python.

  The values not defined in `sp_input` don't participate in the reduce max,
  as opposed to be implicitly assumed 0 -- hence it can return negative values
  for sparse `reduction_axes`. But, in case there are no values in
  `reduction_axes`, it will reduce to 0. See second example below.

  For example:

  ```python
  # 'x' represents [[1, ?, 2]
  #                 [?, 3, ?]]
  # where ? is implicitly-zero.
  tf.sparse.reduce_max(x) ==> 3
  tf.sparse.reduce_max(x, 0) ==> [1, 3, 2]
  tf.sparse.reduce_max(x, 1) ==> [2, 3]  # Can also use -1 as the axis.
  tf.sparse.reduce_max(x, 1, keepdims=True) ==> [[2], [3]]
  tf.sparse.reduce_max(x, [0, 1]) ==> 3

  # 'y' represents [[-7, ?]
  #                 [ 4, 3]
  #                 [ ?, ?]
  tf.sparse.reduce_max(x, 1) ==> [-7, 4, 0]
  ```

  Args:
    sp_input: The SparseTensor to reduce. Should have numeric type.
    axis: The dimensions to reduce; list or scalar. If `None` (the
      default), reduces all dimensions.
    keepdims: If true, retain reduced dimensions with length 1.
    reduction_axes: Deprecated name of `axis`.
    keep_dims:  Deprecated alias for `keepdims`.

  Returns:
    The reduced Tensor.
  """
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  if keepdims is None:
    keepdims = False

  return gen_sparse_ops.sparse_reduce_max(
      sp_input.indices, sp_input.values, sp_input.dense_shape,
      math_ops._ReductionDims(sp_input, axis, reduction_axes), keepdims)


@tf_export(v1=["sparse.reduce_max_sparse", "sparse_reduce_max_sparse"])
@deprecation.deprecated_endpoints("sparse_reduce_max_sparse")
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
def sparse_reduce_max_sparse(sp_input,
                             axis=None,
                             keepdims=None,
                             reduction_axes=None,
                             keep_dims=None):
  """Computes the max of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_max()`.  In contrast to SparseReduceSum, this Op returns a
  SparseTensor.

  Note: A gradient is not defined for this function, so it can't be used
  in training models that need gradient descent.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keepdims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  which are interpreted according to the indexing rules in Python.

  Args:
    sp_input: The SparseTensor to reduce. Should have numeric type.
    axis: The dimensions to reduce; list or scalar. If `None` (the
      default), reduces all dimensions.
    keepdims: If true, retain reduced dimensions with length 1.
    reduction_axes: Deprecated name of axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced SparseTensor.
  """
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  if keepdims is None:
    keepdims = False

  output_ind, output_val, output_shape = (
      gen_sparse_ops.sparse_reduce_max_sparse(
          sp_input.indices, sp_input.values, sp_input.dense_shape,
          math_ops._ReductionDims(sp_input, axis, reduction_axes), keepdims))

  return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)


@tf_export("sparse.reduce_sum", v1=[])
def sparse_reduce_sum_v2(
    sp_input, axis=None, keepdims=None, output_is_sparse=False, name=None):
  """Computes the sum of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
  if `output_is_sparse` is `False`, or a `SparseTensor` if `output_is_sparse`
  is `True`.

  Note: if `output_is_sparse` is True, a gradient is not defined for this
  function, so it can't be used in training models that need gradient descent.

  Reduces `sp_input` along the dimensions given in `axis`.  Unless `keepdims` is
  true, the rank of the tensor is reduced by 1 for each entry in `axis`. If
  `keepdims` is true, the reduced dimensions are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  similar to the indexing rules in Python.

  For example:

  ```python
  # 'x' represents [[1, ?, 1]
  #                 [?, 1, ?]]
  # where ? is implicitly-zero.
  tf.sparse.reduce_sum(x) ==> 3
  tf.sparse.reduce_sum(x, 0) ==> [1, 1, 1]
  tf.sparse.reduce_sum(x, 1) ==> [2, 1]  # Can also use -1 as the axis.
  tf.sparse.reduce_sum(x, 1, keepdims=True) ==> [[2], [1]]
  tf.sparse.reduce_sum(x, [0, 1]) ==> 3
  ```

  Args:
    sp_input: The SparseTensor to reduce. Should have numeric type.
    axis: The dimensions to reduce; list or scalar. If `None` (the
      default), reduces all dimensions.
    keepdims: If true, retain reduced dimensions with length 1.
    output_is_sparse: If true, returns a `SparseTensor` instead of a dense
      `Tensor` (the default).
    name: A name for the operation (optional).

  Returns:
    The reduced Tensor or the reduced SparseTensor if `output_is_sparse` is
    True.
  """
  if keepdims is None:
    keepdims = False

  # reduction_axes is the deprecated name for axis.
  reduction_axes = None

  if output_is_sparse:
    output_ind, output_val, output_shape = (
        gen_sparse_ops.sparse_reduce_sum_sparse(
            sp_input.indices, sp_input.values, sp_input.dense_shape,
            math_ops._ReductionDims(sp_input, axis, reduction_axes), keepdims,
            name=name))
    return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)

  return gen_sparse_ops.sparse_reduce_sum(
      sp_input.indices, sp_input.values, sp_input.dense_shape,
      math_ops._ReductionDims(sp_input, axis, reduction_axes), keepdims,
      name=name)


@tf_export(v1=["sparse.reduce_sum", "sparse_reduce_sum"])
@deprecation.deprecated_endpoints("sparse_reduce_sum")
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
@deprecation.deprecated_args(
    None, "reduction_axes is deprecated, use axis instead",
    "reduction_axes")
def sparse_reduce_sum(sp_input, axis=None, keepdims=None,
                      reduction_axes=None, keep_dims=None):
  """Computes the sum of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
  instead of a sparse one.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keepdims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  similar to the indexing rules in Python.

  For example:

  ```python
  # 'x' represents [[1, ?, 1]
  #                 [?, 1, ?]]
  # where ? is implicitly-zero.
  tf.sparse.reduce_sum(x) ==> 3
  tf.sparse.reduce_sum(x, 0) ==> [1, 1, 1]
  tf.sparse.reduce_sum(x, 1) ==> [2, 1]  # Can also use -1 as the axis.
  tf.sparse.reduce_sum(x, 1, keepdims=True) ==> [[2], [1]]
  tf.sparse.reduce_sum(x, [0, 1]) ==> 3
  ```

  Args:
    sp_input: The SparseTensor to reduce. Should have numeric type.
    axis: The dimensions to reduce; list or scalar. If `None` (the
      default), reduces all dimensions.
    keepdims: If true, retain reduced dimensions with length 1.
    reduction_axes: Deprecated name of `axis`.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced Tensor.
  """
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  if keepdims is None:
    keepdims = False

  return gen_sparse_ops.sparse_reduce_sum(
      sp_input.indices, sp_input.values, sp_input.dense_shape,
      math_ops._ReductionDims(sp_input, axis, reduction_axes), keepdims)


@tf_export(v1=["sparse.reduce_sum_sparse", "sparse_reduce_sum_sparse"])
@deprecation.deprecated_endpoints("sparse_reduce_sum_sparse")
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
def sparse_reduce_sum_sparse(sp_input,
                             axis=None,
                             keepdims=None,
                             reduction_axes=None,
                             keep_dims=None):
  """Computes the sum of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
  SparseTensor.

  Note: A gradient is not defined for this function, so it can't be used
  in training models that need gradient descent.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keepdims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  which are interpreted according to the indexing rules in Python.

  Args:
    sp_input: The SparseTensor to reduce. Should have numeric type.
    axis: The dimensions to reduce; list or scalar. If `None` (the
      default), reduces all dimensions.
    keepdims: If true, retain reduced dimensions with length 1.
    reduction_axes: Deprecated name of axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced SparseTensor.
  """
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  if keepdims is None:
    keepdims = False

  output_ind, output_val, output_shape = (
      gen_sparse_ops.sparse_reduce_sum_sparse(
          sp_input.indices, sp_input.values, sp_input.dense_shape,
          math_ops._ReductionDims(sp_input, axis, reduction_axes), keepdims))

  return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)


@tf_export("sparse.to_dense", v1=["sparse.to_dense", "sparse_tensor_to_dense"])
@deprecation.deprecated_endpoints("sparse_tensor_to_dense")
def sparse_tensor_to_dense(sp_input,
                           default_value=None,
                           validate_indices=True,
                           name=None):
  """Converts a `SparseTensor` into a dense tensor.

  This op is a convenience wrapper around `sparse_to_dense` for `SparseTensor`s.

  For example, if `sp_input` has shape `[3, 5]` and non-empty string values:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c

  and `default_value` is `x`, then the output will be a dense `[3, 5]`
  string tensor with values:

      [[x a x b x]
       [x x x x x]
       [c x x x x]]

  Indices must be without repeats.  This is only
  tested if `validate_indices` is `True`.

  Args:
    sp_input: The input `SparseTensor`.
    default_value: Scalar value to set for indices not specified in
      `sp_input`.  Defaults to zero.
    validate_indices: A boolean value.  If `True`, indices are checked to make
      sure they are sorted in lexicographic order and that there are no repeats.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A dense tensor with shape `sp_input.dense_shape` and values specified by
    the non-empty values in `sp_input`. Indices not in `sp_input` are assigned
    `default_value`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  if default_value is None:
    default_value = array_ops.zeros([], dtype=sp_input.dtype)

  return gen_sparse_ops.sparse_to_dense(
      sp_input.indices,
      sp_input.dense_shape,
      sp_input.values,
      default_value=default_value,
      validate_indices=validate_indices,
      name=name)


@tf_export(
    "sparse.to_indicator", v1=["sparse.to_indicator", "sparse_to_indicator"])
@deprecation.deprecated_endpoints("sparse_to_indicator")
def sparse_to_indicator(sp_input, vocab_size, name=None):
  """Converts a `SparseTensor` of ids into a dense bool indicator tensor.

  The last dimension of `sp_input.indices` is discarded and replaced with
  the values of `sp_input`.  If `sp_input.dense_shape = [D0, D1, ..., Dn, K]`,
  then `output.shape = [D0, D1, ..., Dn, vocab_size]`, where

      output[d_0, d_1, ..., d_n, sp_input[d_0, d_1, ..., d_n, k]] = True

  and False elsewhere in `output`.

  For example, if `sp_input.dense_shape = [2, 3, 4]` with non-empty values:

      [0, 0, 0]: 0
      [0, 1, 0]: 10
      [1, 0, 3]: 103
      [1, 1, 1]: 150
      [1, 1, 2]: 149
      [1, 1, 3]: 150
      [1, 2, 1]: 121

  and `vocab_size = 200`, then the output will be a `[2, 3, 200]` dense bool
  tensor with False everywhere except at positions

      (0, 0, 0), (0, 1, 10), (1, 0, 103), (1, 1, 149), (1, 1, 150),
      (1, 2, 121).

  Note that repeats are allowed in the input SparseTensor.
  This op is useful for converting `SparseTensor`s into dense formats for
  compatibility with ops that expect dense tensors.

  The input `SparseTensor` must be in row-major order.

  Args:
    sp_input: A `SparseTensor` with `values` property of type `int32` or
      `int64`.
    vocab_size: A scalar int64 Tensor (or Python int) containing the new size
      of the last dimension, `all(0 <= sp_input.values < vocab_size)`.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A dense bool indicator tensor representing the indices with specified value.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)

  with ops.name_scope(name, "SparseToIndicator", [sp_input]) as name:
    num_entries = array_ops.shape(sp_input.indices)[0]
    new_values = array_ops.fill(array_ops.expand_dims(num_entries, 0), True)
    sp_values = sparse_tensor.SparseTensor(sp_input.indices, new_values,
                                           sp_input.dense_shape)

    sp_new = sparse_merge_impl(sp_input, sp_values, vocab_size, name)

    # validate_indices may be False because we allow duplicates in new_indices:
    # repeated indices are allowed when creating an indicator matrix.
    return sparse_tensor_to_dense(
        sp_new, default_value=False, validate_indices=False, name=name)


@tf_export(v1=["sparse.merge", "sparse_merge"])
@deprecation.deprecated(None, "No similar op available at this time.")
def sparse_merge(sp_ids, sp_values, vocab_size, name=None,
                 already_sorted=False):
  """Combines a batch of feature ids and values into a single `SparseTensor`.

  The most common use case for this function occurs when feature ids and
  their corresponding values are stored in `Example` protos on disk.
  `parse_example` will return a batch of ids and a batch of values, and this
  function joins them into a single logical `SparseTensor` for use in
  functions such as `sparse_tensor_dense_matmul`, `sparse_to_dense`, etc.

  The `SparseTensor` returned by this function has the following properties:

    - `indices` is equivalent to `sp_ids.indices` with the last
      dimension discarded and replaced with `sp_ids.values`.
    - `values` is simply `sp_values.values`.
    - If `sp_ids.dense_shape = [D0, D1, ..., Dn, K]`, then
      `output.shape = [D0, D1, ..., Dn, vocab_size]`.

  For example, consider the following feature vectors:

  ```python
    vector1 = [-3, 0, 0, 0, 0, 0]
    vector2 = [ 0, 1, 0, 4, 1, 0]
    vector3 = [ 5, 0, 0, 9, 0, 0]
  ```

  These might be stored sparsely in the following Example protos by storing
  only the feature ids (column number if the vectors are treated as a matrix)
  of the non-zero elements and the corresponding values:

  ```python
    examples = [Example(features={
                    "ids": Feature(int64_list=Int64List(value=[0])),
                    "values": Feature(float_list=FloatList(value=[-3]))}),
                Example(features={
                    "ids": Feature(int64_list=Int64List(value=[1, 4, 3])),
                    "values": Feature(float_list=FloatList(value=[1, 1, 4]))}),
                Example(features={
                    "ids": Feature(int64_list=Int64List(value=[0, 3])),
                    "values": Feature(float_list=FloatList(value=[5, 9]))})]
  ```

  The result of calling parse_example on these examples will produce a
  dictionary with entries for "ids" and "values". Passing those two objects
  to this function along with vocab_size=6, will produce a `SparseTensor` that
  sparsely represents all three instances. Namely, the `indices` property will
  contain the coordinates of the non-zero entries in the feature matrix (the
  first dimension is the row number in the matrix, i.e., the index within the
  batch, and the second dimension is the column number, i.e., the feature id);
  `values` will contain the actual values. `shape` will be the shape of the
  original matrix, i.e., (3, 6). For our example above, the output will be
  equal to:

  ```python
    SparseTensor(indices=[[0, 0], [1, 1], [1, 3], [1, 4], [2, 0], [2, 3]],
                 values=[-3, 1, 4, 1, 5, 9],
                 dense_shape=[3, 6])
  ```

  This method generalizes to higher-dimensions by simply providing a list for
  both the sp_ids as well as the vocab_size.
  In this case the resulting `SparseTensor` has the following properties:
    - `indices` is equivalent to `sp_ids[0].indices` with the last
      dimension discarded and concatenated with
      `sp_ids[0].values, sp_ids[1].values, ...`.
    - `values` is simply `sp_values.values`.
    - If `sp_ids.dense_shape = [D0, D1, ..., Dn, K]`, then
      `output.shape = [D0, D1, ..., Dn] + vocab_size`.

  Args:
    sp_ids: A single `SparseTensor` with `values` property of type `int32`
      or `int64` or a Python list of such `SparseTensor`s or a list thereof.
    sp_values: A `SparseTensor` of any type.
    vocab_size: A scalar `int64` Tensor (or Python int) containing the new size
      of the last dimension, `all(0 <= sp_ids.values < vocab_size)`.
      Or a list thereof with `all(0 <= sp_ids[i].values < vocab_size[i])` for
      all `i`.
    name: A name prefix for the returned tensors (optional)
    already_sorted: A boolean to specify whether the per-batch values in
     `sp_values` are already sorted. If so skip sorting, False by default
     (optional).

  Returns:
    A `SparseTensor` compactly representing a batch of feature ids and values,
    useful for passing to functions that expect such a `SparseTensor`.

  Raises:
    TypeError: If `sp_values` is not a `SparseTensor`. Or if `sp_ids` is neither
      a `SparseTensor` nor a list thereof. Or if `vocab_size` is not a
      `Tensor` or a Python int and `sp_ids` is a `SparseTensor`. Or if
      `vocab_size` is not a or list thereof and `sp_ids` is a list.
    ValueError: If `sp_ids` and `vocab_size` are lists of different lengths.
  """
  return sparse_merge_impl(sp_ids, sp_values, vocab_size, name, already_sorted)


def sparse_merge_impl(sp_ids,
                      sp_values,
                      vocab_size,
                      name=None,
                      already_sorted=False):
  """Internal implementation for sparse_merge to avoid deprecation warnings."""
  if isinstance(sp_ids, sparse_tensor.SparseTensorValue) or isinstance(
      sp_ids, sparse_tensor.SparseTensor):
    sp_ids = [sp_ids]
    if not (isinstance(vocab_size, ops.Tensor) or
            isinstance(vocab_size, numbers.Integral)):
      raise TypeError("vocab_size has to be a Tensor or Python int. Found %s" %
                      type(vocab_size))
    vocab_size = [vocab_size]
  else:
    if not isinstance(sp_ids, collections_abc.Iterable):
      raise TypeError("sp_ids has to be a SparseTensor or list thereof. "
                      "Found %s" % type(sp_ids))
    if not isinstance(vocab_size, collections_abc.Iterable):
      raise TypeError("vocab_size has to be a list of Tensors or Python ints. "
                      "Found %s" % type(vocab_size))
    for dim in vocab_size:
      if not (isinstance(dim, ops.Tensor) or isinstance(dim, numbers.Integral)):
        raise TypeError(
            "vocab_size has to be a list of Tensors or Python ints. Found %s" %
            type(dim))
  if len(sp_ids) != len(vocab_size):
    raise ValueError("sp_ids and vocab_size have to have equal lengths.")

  with ops.name_scope(name, "SparseMerge", [sp_ids, sp_values]):
    sp_ids = [_convert_to_sparse_tensor(sp_ids_dim) for sp_ids_dim in sp_ids]
    sp_values = _convert_to_sparse_tensor(sp_values)
    ids = []
    for sp_ids_dim in sp_ids:
      ids_dim = sp_ids_dim.values
      if sp_ids_dim.dtype != dtypes.int64:
        ids_dim = math_ops.cast(ids_dim, dtypes.int64)
      ids += [array_ops.expand_dims(ids_dim, axis=1)]

    vocab_size = [math_ops.cast(x, dtypes.int64) for x in vocab_size]

    # Slice off the last dimension of indices, then tack on the ids
    indices_columns_to_preserve = sp_ids[0].indices[:, :-1]
    new_indices = array_ops.concat([indices_columns_to_preserve] + ids, 1)

    new_values = sp_values.values
    new_shape = array_ops.concat([sp_ids[0].dense_shape[:-1], vocab_size], 0)

    result = sparse_tensor.SparseTensor(new_indices, new_values, new_shape)
    if already_sorted:
      return result
    sorted_result = sparse_reorder(result)
    return sparse_tensor.SparseTensor(
        sorted_result.indices, sorted_result.values, new_shape)


@tf_export("sparse.retain", v1=["sparse.retain", "sparse_retain"])
@deprecation.deprecated_endpoints("sparse_retain")
def sparse_retain(sp_input, to_retain):
  """Retains specified non-empty values within a `SparseTensor`.

  For example, if `sp_input` has shape `[4, 5]` and 4 non-empty string values:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d

  and `to_retain = [True, False, False, True]`, then the output will
  be a `SparseTensor` of shape `[4, 5]` with 2 non-empty values:

      [0, 1]: a
      [3, 1]: d

  Args:
    sp_input: The input `SparseTensor` with `N` non-empty elements.
    to_retain: A bool vector of length `N` with `M` true values.

  Returns:
    A `SparseTensor` with the same shape as the input and `M` non-empty
    elements corresponding to the true positions in `to_retain`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)

  to_retain = ops.convert_to_tensor(to_retain)

  # Shape checking, if shape is known at graph construction time
  retain_shape = to_retain.get_shape()
  retain_shape.assert_has_rank(1)
  if sp_input.values.get_shape().dims is not None:
    sp_input.values.get_shape().dims[0].merge_with(
        tensor_shape.dimension_at_index(retain_shape, 0))

  where_true = array_ops.reshape(array_ops.where_v2(to_retain), [-1])
  new_indices = array_ops.gather(sp_input.indices, where_true)
  new_values = array_ops.gather(sp_input.values, where_true)
  return sparse_tensor.SparseTensor(new_indices, new_values,
                                    array_ops.identity(sp_input.dense_shape))


@tf_export(
    "sparse.reset_shape", v1=["sparse.reset_shape", "sparse_reset_shape"])
@deprecation.deprecated_endpoints("sparse_reset_shape")
def sparse_reset_shape(sp_input, new_shape=None):
  """Resets the shape of a `SparseTensor` with indices and values unchanged.

  If `new_shape` is None, returns a copy of `sp_input` with its shape reset
  to the tight bounding box of `sp_input`. This will be a shape consisting of
  all zeros if sp_input has no values.

  If `new_shape` is provided, then it must be larger or equal in all dimensions
  compared to the shape of `sp_input`. When this condition is met, the returned
  SparseTensor will have its shape reset to `new_shape` and its indices and
  values unchanged from that of `sp_input.`

  For example:

    Consider a `sp_input` with shape [2, 3, 5]:

      [0, 0, 1]: a
      [0, 1, 0]: b
      [0, 2, 2]: c
      [1, 0, 3]: d

    - It is an error to set `new_shape` as [3, 7] since this represents a
      rank-2 tensor while `sp_input` is rank-3. This is either a ValueError
      during graph construction (if both shapes are known) or an OpError during
      run time.

    - Setting `new_shape` as [2, 3, 6] will be fine as this shape is larger or
      equal in every dimension compared to the original shape [2, 3, 5].

    - On the other hand, setting new_shape as [2, 3, 4] is also an error: The
      third dimension is smaller than the original shape [2, 3, 5] (and an
      `InvalidArgumentError` will be raised).

    - If `new_shape` is None, the returned SparseTensor will have a shape
      [2, 3, 4], which is the tight bounding box of `sp_input`.

  Args:
    sp_input: The input `SparseTensor`.
    new_shape: None or a vector representing the new shape for the returned
      `SparseTensor`.

  Returns:
    A `SparseTensor` indices and values unchanged from `input_sp`. Its shape is
      `new_shape` if that is set. Otherwise it is the tight bounding box of
       `input_sp`

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
    ValueError: If `new_shape` represents a tensor with a different rank from
      that of `sp_input` (if shapes are known when graph is constructed).
    ValueError:  If `new_shape` is determined during graph build to have
      dimension sizes that are too small.
    OpError:
      - If `new_shape` has dimension sizes that are too small.
      - If shapes are not known during graph construction time, and during run
        time it is found out that the ranks do not match.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)

  in_indices = array_ops.identity(sp_input.indices)
  in_values = array_ops.identity(sp_input.values)
  in_shape = array_ops.identity(sp_input.dense_shape)

  if new_shape is None:
    dim_low_bound = math_ops.reduce_max(in_indices, axis=0)
    output_shape_tensor = math_ops.maximum(
        array_ops.constant(0, dtype=dtypes.int64),
        math_ops.add(dim_low_bound, array_ops.ones_like(in_shape)))
  else:
    output_shape_tensor = ops.convert_to_tensor(new_shape)
    output_shape_tensor.get_shape().assert_has_rank(1)
    output_shape_tensor = math_ops.cast(output_shape_tensor, dtypes.int64)
    # For cases when shape is known during graph construction, this catches the
    # error before the sparse_tensor.SparseTensor catches it.
    if output_shape_tensor.get_shape().rank is not None:
      output_shape_tensor.get_shape().dims[0].merge_with(
          in_shape.get_shape().dims[0])

    output_shape_tensor_const = tensor_util.constant_value(output_shape_tensor)
    # For cases where all shapes are known during graph construction
    if (output_shape_tensor_const is not None and
        sp_input.get_shape().is_fully_defined()):
      in_shape_const = np.array(sp_input.get_shape().as_list())
      if not np.all(in_shape_const <= output_shape_tensor_const):
        raise ValueError(
            "Requested new_shape should have dimension sizes >= sp_input.shape."
            "  Found new_shape (%s), sp_input.shape (%s)." %
            (in_shape_const, output_shape_tensor_const))
      output_shape_tensor = output_shape_tensor_const
    else:
      # For cases where shape is not known during graph construction.
      output_shape_tensor = control_flow_ops.with_dependencies([
          check_ops.assert_equal(
              array_ops.shape(in_shape), array_ops.shape(output_shape_tensor))
      ], output_shape_tensor)
      output_shape_tensor = control_flow_ops.with_dependencies(
          [check_ops.assert_less_equal(in_shape, output_shape_tensor)],
          output_shape_tensor)

  return sparse_tensor.SparseTensor(in_indices, in_values, output_shape_tensor)


@tf_export(
    "sparse.fill_empty_rows",
    v1=["sparse.fill_empty_rows", "sparse_fill_empty_rows"])
@deprecation.deprecated_endpoints("sparse_fill_empty_rows")
def sparse_fill_empty_rows(sp_input, default_value, name=None):
  """Fills empty rows in the input 2-D `SparseTensor` with a default value.

  This op adds entries with the specified `default_value` at index
  `[row, 0]` for any row in the input that does not already have a value.

  For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d

  Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:

      [0, 1]: a
      [0, 3]: b
      [1, 0]: default_value
      [2, 0]: c
      [3, 1]: d
      [4, 0]: default_value

  Note that the input may have empty columns at the end, with no effect on
  this op.

  The output `SparseTensor` will be in row-major order and will have the
  same shape as the input.

  This op also returns an indicator vector such that

      empty_row_indicator[i] = True iff row i was an empty row.

  Args:
    sp_input: A `SparseTensor` with shape `[N, M]`.
    default_value: The value to fill for empty rows, with the same type as
      `sp_input.`
    name: A name prefix for the returned tensors (optional)

  Returns:
    sp_ordered_output: A `SparseTensor` with shape `[N, M]`, and with all empty
      rows filled in with `default_value`.
    empty_row_indicator: A bool vector of length `N` indicating whether each
      input row was empty.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  with ops.name_scope(name, "SparseFillEmptyRows", [sp_input]):
    default_value = ops.convert_to_tensor(
        default_value, dtype=sp_input.values.dtype)
    (output_indices, output_values, empty_row_indicator,
     unused_reverse_index_map) = gen_sparse_ops.sparse_fill_empty_rows(
         indices=sp_input.indices,
         values=sp_input.values,
         dense_shape=sp_input.dense_shape,
         default_value=default_value)
    return (sparse_tensor.SparseTensor(
        indices=output_indices,
        values=output_values,
        dense_shape=sp_input.dense_shape), empty_row_indicator)


@tf_export(v1=["io.serialize_sparse", "serialize_sparse"])
@deprecation.deprecated_endpoints("serialize_sparse")
def serialize_sparse(sp_input, name=None, out_type=dtypes.string):
  """Serialize a `SparseTensor` into a 3-vector (1-D `Tensor`) object.

  Args:
    sp_input: The input `SparseTensor`.
    name: A name prefix for the returned tensors (optional).
    out_type: The `dtype` to use for serialization.

  Returns:
    A 3-vector (1-D `Tensor`), with each column representing the serialized
    `SparseTensor`'s indices, values, and shape (respectively).

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  return serialize_sparse_v2(sp_input, out_type, name)


@tf_export("io.serialize_sparse", v1=[])
def serialize_sparse_v2(sp_input, out_type=dtypes.string, name=None):
  """Serialize a `SparseTensor` into a 3-vector (1-D `Tensor`) object.

  Args:
    sp_input: The input `SparseTensor`.
    out_type: The `dtype` to use for serialization.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A 3-vector (1-D `Tensor`), with each column representing the serialized
    `SparseTensor`'s indices, values, and shape (respectively).

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)

  return gen_sparse_ops.serialize_sparse(
      sp_input.indices,
      sp_input.values,
      sp_input.dense_shape,
      name=name,
      out_type=out_type)


@tf_export(v1=["io.serialize_many_sparse", "serialize_many_sparse"])
@deprecation.deprecated_endpoints("serialize_many_sparse")
def serialize_many_sparse(sp_input, name=None, out_type=dtypes.string):
  """Serialize `N`-minibatch `SparseTensor` into an `[N, 3]` `Tensor`.

  The `SparseTensor` must have rank `R` greater than 1, and the first dimension
  is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The serialized
  `SparseTensor` objects going into each row of the output `Tensor` will have
  rank `R-1`.

  The minibatch size `N` is extracted from `sparse_shape[0]`.

  Args:
    sp_input: The input rank `R` `SparseTensor`.
    name: A name prefix for the returned tensors (optional).
    out_type: The `dtype` to use for serialization.

  Returns:
    A matrix (2-D `Tensor`) with `N` rows and `3` columns. Each column
    represents serialized `SparseTensor`'s indices, values, and shape
    (respectively).

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  return serialize_many_sparse_v2(sp_input, out_type, name)


@tf_export("io.serialize_many_sparse", v1=[])
def serialize_many_sparse_v2(sp_input, out_type=dtypes.string, name=None):
  """Serialize `N`-minibatch `SparseTensor` into an `[N, 3]` `Tensor`.

  The `SparseTensor` must have rank `R` greater than 1, and the first dimension
  is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The serialized
  `SparseTensor` objects going into each row of the output `Tensor` will have
  rank `R-1`.

  The minibatch size `N` is extracted from `sparse_shape[0]`.

  Args:
    sp_input: The input rank `R` `SparseTensor`.
    out_type: The `dtype` to use for serialization.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A matrix (2-D `Tensor`) with `N` rows and `3` columns. Each column
    represents serialized `SparseTensor`'s indices, values, and shape
    (respectively).

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)

  return gen_sparse_ops.serialize_many_sparse(
      sp_input.indices,
      sp_input.values,
      sp_input.dense_shape,
      name=name,
      out_type=out_type)


def deserialize_sparse(serialized_sparse, dtype, rank=None, name=None):
  """Deserialize `SparseTensor` objects.

  The input `serialized_sparse` must have the shape `[?, ?, ..., ?, 3]` where
  the last dimension stores serialized `SparseTensor` objects and the other N
  dimensions (N >= 0) correspond to a batch. The ranks of the original
  `SparseTensor` objects must all match. When the final `SparseTensor` is
  created, its rank is the rank of the incoming `SparseTensor` objects plus N;
  the sparse tensors have been concatenated along new dimensions, one for each
  batch.

  The output `SparseTensor` object's shape values for the original dimensions
  are the max across the input `SparseTensor` objects' shape values for the
  corresponding dimensions. The new dimensions match the size of the batch.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `SparseReorder` to restore index ordering.

  For example, if the serialized input is a `[2 x 3]` matrix representing two
  original `SparseTensor` objects:

      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]

  and

      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]

  then the final deserialized `SparseTensor` will be:

      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]

  Args:
    serialized_sparse: The serialized `SparseTensor` objects.
      The last dimension must have 3 columns.
    dtype: The `dtype` of the serialized `SparseTensor` objects.
    rank: (optional) Python int, the rank of the `SparseTensor` objects.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A `SparseTensor` representing the deserialized `SparseTensor` objects.

  """
  output_indices, output_values, output_shape = (
      gen_sparse_ops.deserialize_sparse(serialized_sparse, dtype, name=name))

  # Feed rank data back in, if available
  output_indices.set_shape([None, rank])
  output_shape.set_shape([rank])

  return sparse_tensor.SparseTensor(output_indices, output_values, output_shape)


@tf_export(
    "io.deserialize_many_sparse",
    v1=["io.deserialize_many_sparse", "deserialize_many_sparse"])
@deprecation.deprecated_endpoints("deserialize_many_sparse")
def deserialize_many_sparse(serialized_sparse, dtype, rank=None, name=None):
  """Deserialize and concatenate `SparseTensors` from a serialized minibatch.

  The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
  `N` is the minibatch size and the rows correspond to packed outputs of
  `serialize_sparse`.  The ranks of the original `SparseTensor` objects
  must all match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects (they have been
  concatenated along a new row dimension).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `sparse.reorder` to restore index ordering.

  For example, if the serialized input is a `[2, 3]` matrix representing two
  original `SparseTensor` objects:

      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]

  and

      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]

  then the final deserialized `SparseTensor` will be:

      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]

  Args:
    serialized_sparse: 2-D `Tensor` of type `string` of shape `[N, 3]`.
      The serialized and packed `SparseTensor` objects.
    dtype: The `dtype` of the serialized `SparseTensor` objects.
    rank: (optional) Python int, the rank of the `SparseTensor` objects.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A `SparseTensor` representing the deserialized `SparseTensor`s,
    concatenated along the `SparseTensor`s' first dimension.

    All of the serialized `SparseTensor`s must have had the same rank and type.
  """
  output_indices, output_values, output_shape = (
      gen_sparse_ops.deserialize_many_sparse(
          serialized_sparse, dtype, name=name))

  # Feed rank data back in, if available
  output_indices.set_shape([None, rank])
  output_shape.set_shape([rank])

  return sparse_tensor.SparseTensor(output_indices, output_values, output_shape)


@tf_export("sparse.sparse_dense_matmul",
           v1=["sparse.sparse_dense_matmul", "sparse.matmul",
               "sparse_tensor_dense_matmul"])
@deprecation.deprecated_endpoints("sparse_tensor_dense_matmul")
def sparse_tensor_dense_matmul(sp_a,
                               b,
                               adjoint_a=False,
                               adjoint_b=False,
                               name=None):
  # pylint: disable=line-too-long
  """Multiply SparseTensor (or dense Matrix) (of rank 2) "A" by dense matrix
  (or SparseTensor) "B". Please note that one and only one of the inputs MUST
  be a SparseTensor and the other MUST be a dense matrix.

  No validity checking is performed on the indices of `A`.  However, the
  following input format is recommended for optimal behavior:

  * If `adjoint_a == false`: `A` should be sorted in lexicographically
    increasing order.  Use `sparse.reorder` if you're not sure.
  * If `adjoint_a == true`: `A` should be sorted in order of increasing
    dimension 1 (i.e., "column major" order instead of "row major" order).

  Using `tf.nn.embedding_lookup_sparse` for sparse multiplication:

  It's not obvious but you can consider `embedding_lookup_sparse` as another
  sparse and dense multiplication. In some situations, you may prefer to use
  `embedding_lookup_sparse` even though you're not dealing with embeddings.

  There are two questions to ask in the decision process: Do you need gradients
  computed as sparse too? Is your sparse data represented as two
  `SparseTensor`s: ids and values? There is more explanation about data format
  below. If you answer any of these questions as yes, consider using
  `tf.nn.embedding_lookup_sparse`.

  Following explains differences between the expected SparseTensors:
  For example if dense form of your sparse data has shape `[3, 5]` and values:

      [[  a      ]
       [b       c]
       [    d    ]]


  `SparseTensor` format expected by `sparse_tensor_dense_matmul`:
   `sp_a` (indices, values):

      [0, 1]: a
      [1, 0]: b
      [1, 4]: c
      [2, 2]: d

  `SparseTensor` format expected by `embedding_lookup_sparse`:
   `sp_ids`                 `sp_weights`

      [0, 0]: 1                [0, 0]: a
      [1, 0]: 0                [1, 0]: b
      [1, 1]: 4                [1, 1]: c
      [2, 0]: 2                [2, 0]: d


  Deciding when to use `sparse_tensor_dense_matmul` vs.
  `matmul`(a_is_sparse=True):

  There are a number of questions to ask in the decision process, including:

  * Will the SparseTensor `A` fit in memory if densified?
  * Is the column count of the product large (>> 1)?
  * Is the density of `A` larger than approximately 15%?

  If the answer to several of these questions is yes, consider
  converting the `SparseTensor` to a dense one and using `tf.matmul` with
  `a_is_sparse=True`.

  This operation tends to perform well when `A` is more sparse, if the column
  size of the product is small (e.g. matrix-vector multiplication), if
  `sp_a.dense_shape` takes on large values.

  Below is a rough speed comparison between `sparse_tensor_dense_matmul`,
  labeled 'sparse', and `matmul`(a_is_sparse=True), labeled 'dense'.  For
  purposes of the comparison, the time spent converting from a `SparseTensor` to
  a dense `Tensor` is not included, so it is overly conservative with respect to
  the time ratio.

  Benchmark system:
  CPU: Intel Ivybridge with HyperThreading (6 cores) dL1:32KB dL2:256KB dL3:12MB
  GPU: NVidia Tesla k40c

  Compiled with:
  `-c opt --config=cuda --copt=-mavx`

  ```
  tensorflow/python/sparse_tensor_dense_matmul_op_test --benchmarks
  A sparse [m, k] with % nonzero values between 1% and 80%
  B dense [k, n]

  % nnz  n   gpu   m     k     dt(dense)     dt(sparse)   dt(sparse)/dt(dense)
  0.01   1   True  100   100   0.000221166   0.00010154   0.459112
  0.01   1   True  100   1000  0.00033858    0.000109275  0.322745
  0.01   1   True  1000  100   0.000310557   9.85661e-05  0.317385
  0.01   1   True  1000  1000  0.0008721     0.000100875  0.115669
  0.01   1   False 100   100   0.000208085   0.000107603  0.51711
  0.01   1   False 100   1000  0.000327112   9.51118e-05  0.290762
  0.01   1   False 1000  100   0.000308222   0.00010345   0.335635
  0.01   1   False 1000  1000  0.000865721   0.000101397  0.117124
  0.01   10  True  100   100   0.000218522   0.000105537  0.482958
  0.01   10  True  100   1000  0.000340882   0.000111641  0.327506
  0.01   10  True  1000  100   0.000315472   0.000117376  0.372064
  0.01   10  True  1000  1000  0.000905493   0.000123263  0.136128
  0.01   10  False 100   100   0.000221529   9.82571e-05  0.44354
  0.01   10  False 100   1000  0.000330552   0.000112615  0.340687
  0.01   10  False 1000  100   0.000341277   0.000114097  0.334324
  0.01   10  False 1000  1000  0.000819944   0.000120982  0.147549
  0.01   25  True  100   100   0.000207806   0.000105977  0.509981
  0.01   25  True  100   1000  0.000322879   0.00012921   0.400181
  0.01   25  True  1000  100   0.00038262    0.00014158   0.370035
  0.01   25  True  1000  1000  0.000865438   0.000202083  0.233504
  0.01   25  False 100   100   0.000209401   0.000104696  0.499979
  0.01   25  False 100   1000  0.000321161   0.000130737  0.407076
  0.01   25  False 1000  100   0.000377012   0.000136801  0.362856
  0.01   25  False 1000  1000  0.000861125   0.00020272   0.235413
  0.2    1   True  100   100   0.000206952   9.69219e-05  0.46833
  0.2    1   True  100   1000  0.000348674   0.000147475  0.422959
  0.2    1   True  1000  100   0.000336908   0.00010122   0.300439
  0.2    1   True  1000  1000  0.001022      0.000203274  0.198898
  0.2    1   False 100   100   0.000207532   9.5412e-05   0.459746
  0.2    1   False 100   1000  0.000356127   0.000146824  0.41228
  0.2    1   False 1000  100   0.000322664   0.000100918  0.312764
  0.2    1   False 1000  1000  0.000998987   0.000203442  0.203648
  0.2    10  True  100   100   0.000211692   0.000109903  0.519165
  0.2    10  True  100   1000  0.000372819   0.000164321  0.440753
  0.2    10  True  1000  100   0.000338651   0.000144806  0.427596
  0.2    10  True  1000  1000  0.00108312    0.000758876  0.70064
  0.2    10  False 100   100   0.000215727   0.000110502  0.512231
  0.2    10  False 100   1000  0.000375419   0.0001613    0.429653
  0.2    10  False 1000  100   0.000336999   0.000145628  0.432132
  0.2    10  False 1000  1000  0.00110502    0.000762043  0.689618
  0.2    25  True  100   100   0.000218705   0.000129913  0.594009
  0.2    25  True  100   1000  0.000394794   0.00029428   0.745402
  0.2    25  True  1000  100   0.000404483   0.0002693    0.665788
  0.2    25  True  1000  1000  0.0012002     0.00194494   1.62052
  0.2    25  False 100   100   0.000221494   0.0001306    0.589632
  0.2    25  False 100   1000  0.000396436   0.000297204  0.74969
  0.2    25  False 1000  100   0.000409346   0.000270068  0.659754
  0.2    25  False 1000  1000  0.00121051    0.00193737   1.60046
  0.5    1   True  100   100   0.000214981   9.82111e-05  0.456836
  0.5    1   True  100   1000  0.000415328   0.000223073  0.537101
  0.5    1   True  1000  100   0.000358324   0.00011269   0.314492
  0.5    1   True  1000  1000  0.00137612    0.000437401  0.317851
  0.5    1   False 100   100   0.000224196   0.000101423  0.452386
  0.5    1   False 100   1000  0.000400987   0.000223286  0.556841
  0.5    1   False 1000  100   0.000368825   0.00011224   0.304318
  0.5    1   False 1000  1000  0.00136036    0.000429369  0.31563
  0.5    10  True  100   100   0.000222125   0.000112308  0.505608
  0.5    10  True  100   1000  0.000461088   0.00032357   0.701753
  0.5    10  True  1000  100   0.000394624   0.000225497  0.571422
  0.5    10  True  1000  1000  0.00158027    0.00190898   1.20801
  0.5    10  False 100   100   0.000232083   0.000114978  0.495418
  0.5    10  False 100   1000  0.000454574   0.000324632  0.714146
  0.5    10  False 1000  100   0.000379097   0.000227768  0.600817
  0.5    10  False 1000  1000  0.00160292    0.00190168   1.18638
  0.5    25  True  100   100   0.00023429    0.000151703  0.647501
  0.5    25  True  100   1000  0.000497462   0.000598873  1.20386
  0.5    25  True  1000  100   0.000460778   0.000557038  1.20891
  0.5    25  True  1000  1000  0.00170036    0.00467336   2.74845
  0.5    25  False 100   100   0.000228981   0.000155334  0.678371
  0.5    25  False 100   1000  0.000496139   0.000620789  1.25124
  0.5    25  False 1000  100   0.00045473    0.000551528  1.21287
  0.5    25  False 1000  1000  0.00171793    0.00467152   2.71927
  0.8    1   True  100   100   0.000222037   0.000105301  0.47425
  0.8    1   True  100   1000  0.000410804   0.000329327  0.801664
  0.8    1   True  1000  100   0.000349735   0.000131225  0.375212
  0.8    1   True  1000  1000  0.00139219    0.000677065  0.48633
  0.8    1   False 100   100   0.000214079   0.000107486  0.502085
  0.8    1   False 100   1000  0.000413746   0.000323244  0.781261
  0.8    1   False 1000  100   0.000348983   0.000131983  0.378193
  0.8    1   False 1000  1000  0.00136296    0.000685325  0.50282
  0.8    10  True  100   100   0.000229159   0.00011825   0.516017
  0.8    10  True  100   1000  0.000498845   0.000532618  1.0677
  0.8    10  True  1000  100   0.000383126   0.00029935   0.781336
  0.8    10  True  1000  1000  0.00162866    0.00307312   1.88689
  0.8    10  False 100   100   0.000230783   0.000124958  0.541452
  0.8    10  False 100   1000  0.000493393   0.000550654  1.11606
  0.8    10  False 1000  100   0.000377167   0.000298581  0.791642
  0.8    10  False 1000  1000  0.00165795    0.00305103   1.84024
  0.8    25  True  100   100   0.000233496   0.000175241  0.75051
  0.8    25  True  100   1000  0.00055654    0.00102658   1.84458
  0.8    25  True  1000  100   0.000463814   0.000783267  1.68875
  0.8    25  True  1000  1000  0.00186905    0.00755344   4.04132
  0.8    25  False 100   100   0.000240243   0.000175047  0.728625
  0.8    25  False 100   1000  0.000578102   0.00104499   1.80763
  0.8    25  False 1000  100   0.000485113   0.000776849  1.60138
  0.8    25  False 1000  1000  0.00211448    0.00752736   3.55992
  ```

  Args:
    sp_a: SparseTensor (or dense Matrix) A, of rank 2.
    b: dense Matrix (or SparseTensor) B, with the same dtype as sp_a.
    adjoint_a: Use the adjoint of A in the matrix multiply.  If A is complex,
      this is transpose(conj(A)).  Otherwise it's transpose(A).
    adjoint_b: Use the adjoint of B in the matrix multiply.  If B is complex,
      this is transpose(conj(B)).  Otherwise it's transpose(B).
    name: A name prefix for the returned tensors (optional)

  Returns:
    A dense matrix (pseudo-code in dense np.matrix notation):
      `A = A.H if adjoint_a else A`
      `B = B.H if adjoint_b else B`
      `return A*B`
  """
  # pylint: enable=line-too-long

  if isinstance(b, sparse_tensor.SparseTensor) \
          or isinstance(b, sparse_tensor.SparseTensorValue):
    # We can do C * D where C is sparse but if we want to do A * B when
    # B is sparse we have to transpose. But AB = (B'A')' so we have to feed in
    # the transpose of the arguments as well.
    if adjoint_a != adjoint_b:
      return array_ops.transpose(sparse_tensor_dense_matmul(b, sp_a,
                                               adjoint_a, adjoint_b))
    else:
      return array_ops.transpose(sparse_tensor_dense_matmul(b, sp_a,
                                            adjoint_a=not adjoint_a,
                                            adjoint_b=not adjoint_b))

  else:
    sp_a = _convert_to_sparse_tensor(sp_a)
    with ops.name_scope(name, "SparseTensorDenseMatMul",
                        [sp_a.indices, sp_a.values, b]) as name:
      b = ops.convert_to_tensor(b, name="b")
      return gen_sparse_ops.sparse_tensor_dense_mat_mul(
          a_indices=sp_a.indices,
          a_values=sp_a.values,
          a_shape=sp_a.dense_shape,
          b=b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)

@tf_export("sparse.softmax", v1=["sparse.softmax", "sparse_softmax"])
@deprecation.deprecated_endpoints("sparse_softmax")
def sparse_softmax(sp_input, name=None):
  """Applies softmax to a batched N-D `SparseTensor`.

  The inputs represent an N-D SparseTensor with logical shape `[..., B, C]`
  (where `N >= 2`), and with indices sorted in the canonical lexicographic
  order.

  This op is equivalent to applying the normal `tf.nn.softmax()` to each
  innermost logical submatrix with shape `[B, C]`, but with the catch that *the
  implicitly zero elements do not participate*.  Specifically, the algorithm is
  equivalent to:

    (1) Applies `tf.nn.softmax()` to a densified view of each innermost
        submatrix with shape `[B, C]`, along the size-C dimension;
    (2) Masks out the original implicitly-zero locations;
    (3) Renormalizes the remaining elements.

  Hence, the `SparseTensor` result has exactly the same non-zero indices and
  shape.

  Example:

  ```python
  # First batch:
  # [?   e.]
  # [1.  ? ]
  # Second batch:
  # [e   ? ]
  # [e   e ]
  shape = [2, 2, 2]  # 3-D SparseTensor
  values = np.asarray([[[0., np.e], [1., 0.]], [[np.e, 0.], [np.e, np.e]]])
  indices = np.vstack(np.where(values)).astype(np.int64).T

  result = tf.sparse.softmax(tf.SparseTensor(indices, values, shape))
  # ...returning a 3-D SparseTensor, equivalent to:
  # [?   1.]     [1    ?]
  # [1.  ? ] and [.5  .5]
  # where ? means implicitly zero.
  ```

  Args:
    sp_input: N-D `SparseTensor`, where `N >= 2`.
    name: optional name of the operation.
  Returns:
    output: N-D `SparseTensor` representing the results.
  """
  with ops.name_scope(name, "SparseSoftmax",
                      [sp_input.indices, sp_input.values]) as name:
    out_vals = gen_sparse_ops.sparse_softmax(sp_input.indices, sp_input.values,
                                             sp_input.dense_shape)
    return sparse_tensor.SparseTensor(sp_input.indices, out_vals,
                                      sp_input.dense_shape)


@tf_export("sparse.maximum", v1=["sparse.maximum", "sparse_maximum"])
@deprecation.deprecated_endpoints("sparse_maximum")
def sparse_maximum(sp_a, sp_b, name=None):
  """Returns the element-wise max of two SparseTensors.

  Assumes the two SparseTensors have the same shape, i.e., no broadcasting.
  Example:

  ```python
  sp_zero = sparse_tensor.SparseTensor([[0]], [0], [7])
  sp_one = sparse_tensor.SparseTensor([[1]], [1], [7])
  res = tf.sparse.maximum(sp_zero, sp_one).eval()
  # "res" should be equal to SparseTensor([[0], [1]], [0, 1], [7]).
  ```

  Args:
    sp_a: a `SparseTensor` operand whose dtype is real, and indices
      lexicographically ordered.
    sp_b: the other `SparseTensor` operand with the same requirements (and the
      same shape).
    name: optional name of the operation.
  Returns:
    output: the output SparseTensor.
  """
  with ops.name_scope(
      name, "SparseSparseMaximum",
      [sp_a.indices, sp_a.values, sp_b.indices, sp_b.values]) as name:
    out_indices, out_values = gen_sparse_ops.sparse_sparse_maximum(
        sp_a.indices,
        sp_a.values,
        sp_a.dense_shape,
        sp_b.indices,
        sp_b.values,
        sp_b.dense_shape,
        name=name)
  return sparse_tensor.SparseTensor(out_indices, out_values, sp_a.dense_shape)


@tf_export("sparse.minimum", v1=["sparse.minimum", "sparse_minimum"])
@deprecation.deprecated_endpoints("sparse_minimum")
def sparse_minimum(sp_a, sp_b, name=None):
  """Returns the element-wise min of two SparseTensors.

  Assumes the two SparseTensors have the same shape, i.e., no broadcasting.
  Example:

  ```python
  sp_zero = sparse_tensor.SparseTensor([[0]], [0], [7])
  sp_one = sparse_tensor.SparseTensor([[1]], [1], [7])
  res = tf.sparse.minimum(sp_zero, sp_one).eval()
  # "res" should be equal to SparseTensor([[0], [1]], [0, 0], [7]).
  ```

  Args:
    sp_a: a `SparseTensor` operand whose dtype is real, and indices
      lexicographically ordered.
    sp_b: the other `SparseTensor` operand with the same requirements (and the
      same shape).
    name: optional name of the operation.
  Returns:
    output: the output SparseTensor.
  """
  with ops.name_scope(
      name, "SparseSparseMinimum",
      [sp_a.indices, sp_a.values, sp_b.indices, sp_b.values]) as name:
    out_indices, out_values = gen_sparse_ops.sparse_sparse_minimum(
        sp_a.indices,
        sp_a.values,
        sp_a.dense_shape,
        sp_b.indices,
        sp_b.values,
        sp_b.dense_shape,
        name=name)
  return sparse_tensor.SparseTensor(out_indices, out_values, sp_a.dense_shape)


@tf_export("sparse.transpose", v1=["sparse.transpose", "sparse_transpose"])
@deprecation.deprecated_endpoints("sparse_transpose")
def sparse_transpose(sp_input, perm=None, name=None):
  """Transposes a `SparseTensor`

  The returned tensor's dimension i will correspond to the input dimension
  `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
  the rank of the input tensor. Hence by default, this operation performs a
  regular matrix transpose on 2-D input Tensors.

  For example, if `sp_input` has shape `[4, 5]` and `indices` / `values`:

      [0, 3]: b
      [0, 1]: a
      [3, 1]: d
      [2, 0]: c

  then the output will be a `SparseTensor` of shape `[5, 4]` and
  `indices` / `values`:

      [0, 2]: c
      [1, 0]: a
      [1, 3]: d
      [3, 0]: b

  Args:
    sp_input: The input `SparseTensor`.
    perm: A permutation of the dimensions of `sp_input`.
    name: A name prefix for the returned tensors (optional)
  Returns:
    A transposed `SparseTensor`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  with ops.name_scope(name, "SparseTranspose", [sp_input]) as name:
    if perm is None:
      if sp_input.shape.is_fully_defined():
        rank = len(sp_input.shape)
        perm = (rank - 1) - np.arange(0, rank, 1)
      else:
        rank = array_ops.rank(sp_input)
        perm = (rank - 1) - math_ops.range(0, rank, 1)
    indices = sp_input.indices
    transposed_indices = array_ops.transpose(
        array_ops.gather(array_ops.transpose(indices), perm))

    perm_ = tensor_util.constant_value(ops.convert_to_tensor(perm))
    if perm_ is not None and sp_input.get_shape().is_fully_defined():
      old_shape_ = sp_input.get_shape().as_list()
      transposed_dense_shape = list(old_shape_)  # Copy.
      for i, p in enumerate(perm_):
        transposed_dense_shape[i] = old_shape_[p]
    else:
      dense_shape = sp_input.dense_shape
      transposed_dense_shape = array_ops.gather(dense_shape, perm)
    transposed_st = sparse_tensor.SparseTensor(
        transposed_indices, sp_input.values, transposed_dense_shape)
    transposed_st = sparse_reorder(transposed_st)
    return transposed_st


def _add_sparse_to_tensors_map(sp_input,
                               container=None,
                               shared_name=None,
                               name=None):
  """Add a `SparseTensor` to a `SparseTensorsMap` and return its handle.

  Args:
    sp_input: The input `SparseTensor`.
    container: The container for the underlying `SparseTensorsMap` (optional).
    shared_name: The shared name for the underlying `SparseTensorsMap`
      (optional, defaults to the name of the newly created op).
    name: A name prefix for the returned tensors (optional).

  Returns:
    A string 1-vector (1D `Tensor`), with the single element representing the
    a unique handle to a `SparseTensor` stored by the `SparseTensorMap`
    underlying this op.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)

  return gen_sparse_ops.add_sparse_to_tensors_map(
      sp_input.indices,
      sp_input.values,
      sp_input.dense_shape,
      container=container,
      shared_name=shared_name,
      name=name)


def _add_many_sparse_to_tensors_map(sp_input,
                                    container=None,
                                    shared_name=None,
                                    name=None):
  """Add a minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.

  The `SparseTensor` must have rank `R` greater than 1, and the first dimension
  is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The serialized
  `SparseTensor` objects going into each row of the output `Tensor` will have
  rank `R-1`.

  The minibatch size `N` is extracted from `sparse_shape[0]`.

  Args:
    sp_input: The input rank `R` `SparseTensor`.
    container: The container for the underlying `SparseTensorsMap` (optional).
    shared_name: The shared name for the underlying `SparseTensorsMap`
      (optional, defaults to the name of the newly created op).
    name: A name prefix for the returned tensors (optional).

  Returns:
    A string matrix (2-D `Tensor`) with `N` rows and `1` column.
    Each row represents a unique handle to a `SparseTensor` stored by
    the `SparseTensorMap` underlying this op.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)

  return gen_sparse_ops.add_many_sparse_to_tensors_map(
      sp_input.indices,
      sp_input.values,
      sp_input.dense_shape,
      container=container,
      shared_name=shared_name,
      name=name)


def _take_many_sparse_from_tensors_map(sparse_map_op,
                                       sparse_handles,
                                       rank=None,
                                       name=None):
  """Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.

  The input `sparse_handles` must be a string matrix of shape `[N, 1]` where
  `N` is the minibatch size and the rows correspond to packed outputs of
  `add_sparse_to_tensors_map`.  The ranks of the original `SparseTensor` objects
  must all match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects (they have been
  concatenated along a new row dimension).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `sparse.reorder` to restore index ordering.

  For example, if the serialized input is a `[2, 3]` matrix representing two
  original `SparseTensor` objects:

      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]

  and

      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]

  then the final deserialized `SparseTensor` will be:

      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]

  Args:
    sparse_map_op: The `Operation` that created the original handles.
      Usually this is, e.g., `add_sparse_to_tensors_map(...).op`.
    sparse_handles: 2-D `Tensor` of type `string` of shape `[N, 1]`.
      The serialized and packed `SparseTensor` objects.
    rank: (optional) Python int, the rank of the `SparseTensor` objects.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A `SparseTensor` representing the deserialized `SparseTensor`s,
    concatenated along the `SparseTensor`s' first dimension.

    All of the serialized `SparseTensor`s must have had the same rank and type.
  """
  if not isinstance(sparse_map_op, ops.Operation):
    raise TypeError("sparse_map_op be an Operation")
  if sparse_map_op.type not in ("AddSparseToTensorsMap",
                                "AddManySparseToTensorsMap"):
    raise TypeError(
        "sparse_map_op must be one of AddSparseToTensorsMap or "
        "AddSparseToTensorsMap. Instead, found `%s`." % sparse_map_op.type)
  with ops.colocate_with(sparse_map_op):
    shared_name = sparse_map_op.get_attr("shared_name") or sparse_map_op.name
    output_indices, output_values, output_shape = (
        gen_sparse_ops.take_many_sparse_from_tensors_map(
            sparse_handles,
            dtype=sparse_map_op.get_attr("T"),
            container=sparse_map_op.get_attr("container"),
            shared_name=shared_name,
            name=name))

  # Feed rank data back in, if available
  output_indices.set_shape([None, rank])
  output_shape.set_shape([rank])

  return sparse_tensor.SparseTensor(output_indices, output_values, output_shape)


class _UnaryMapValueDispatcher(dispatch.OpDispatcher):
  """OpDispatcher for unary ops that maps base function across sparse values."""

  def __init__(self, original_func):
    self._original_func = original_func
    func_name = get_canonical_name_for_symbol(original_func)
    arg_names = tf_inspect.getfullargspec(original_func)[0]
    self._x = arg_names[0]
    original_func.__doc__ = (
        original_func.__doc__.rstrip() + "\n\n" +
        ("    If `{x}` is a `SparseTensor`, returns\n"
         "    `SparseTensor({x}.indices, tf.{func}({x}.values, ...), "
         "{x}.dense_shape)`").format(x=self._x, func=func_name))

  def handle(self, args, kwargs):
    if args:
      x, args = args[0], args[1:]
    else:
      kwargs = kwargs.copy()
      x = kwargs.pop(self._x, None)
    if isinstance(x, sparse_tensor.SparseTensor):
      return sparse_tensor.SparseTensor(
          indices=x.indices,
          values=self._original_func(x.values, *args, **kwargs),
          dense_shape=x.dense_shape)
    else:
      return self.NOT_SUPPORTED


_UNARY_OPS = [
    # TODO(b/120307967) Add dispatchers for additional TensorFlow ops.
    math_ops.abs,
    math_ops.negative,
    math_ops.sign,
    math_ops.square,
    math_ops.sqrt,
    math_ops.erf,
    math_ops.tanh,
    math_ops.bessel_i0e,
    math_ops.bessel_i1e,
]
for unary_op in _UNARY_OPS:
  _UnaryMapValueDispatcher(unary_op).register(unary_op)
