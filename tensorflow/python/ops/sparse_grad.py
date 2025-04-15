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
"""Gradients for operators defined in sparse_ops.py."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

# TODO(b/31222613): This op may be differentiable, and there may be
# latent bugs here.
ops.NotDifferentiable("SparseAddGrad")
ops.NotDifferentiable("SparseConcat")


@ops.RegisterGradient("SparseReorder")
def _SparseReorderGrad(
    op: ops.Operation, unused_output_indices_grad, output_values_grad
):
  """Gradients for the SparseReorder op.

  Args:
    op: the SparseReorder op
    unused_output_indices_grad: the incoming gradients of the output indices
    output_values_grad: the incoming gradients of the output values

  Returns:
    Gradient for each of the 3 input tensors:
      (input_indices, input_values, input_shape)
    The gradients for input_indices and input_shape is None.
  """
  input_indices = op.inputs[0]
  input_shape = op.inputs[2]

  num_entries = array_ops.shape(input_indices)[0]
  entry_indices = math_ops.range(num_entries)
  sp_unordered = sparse_tensor.SparseTensor(input_indices, entry_indices,
                                            input_shape)
  sp_ordered = sparse_ops.sparse_reorder(sp_unordered)
  inverted_permutation = array_ops.invert_permutation(sp_ordered.values)

  return (None, array_ops.gather(output_values_grad,
                                 inverted_permutation), None)


@ops.RegisterGradient("SparseAdd")
def _SparseAddGrad(op: ops.Operation, *grads):
  """The backward operator for the SparseAdd op.

  The SparseAdd op calculates A + B, where A, B, and the sum are all represented
  as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
  non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
  values of A and B.

  Args:
    op: the SparseAdd op
    *grads: the incoming gradients, one element per output of `op`

  Returns:
    Gradient for each of the 6 input tensors of SparseAdd:
      (a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh)
    The gradients for the indices, shapes, and the threshold are None.
  """
  val_grad = grads[1]
  a_indices = op.inputs[0]
  b_indices = op.inputs[3]
  sum_indices = op.outputs[0]
  # NOTE: we do not need to take `thresh` into account, since it simply affects
  # the non-zero elements of the sum, and we will peek into `sum_indices` in the
  # gradient op.

  a_val_grad, b_val_grad = gen_sparse_ops.sparse_add_grad(
      val_grad, a_indices, b_indices, sum_indices)
  a_val_grad.set_shape(op.inputs[1].get_shape())
  b_val_grad.set_shape(op.inputs[4].get_shape())
  # (a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh)
  return (None, a_val_grad, None, None, b_val_grad, None, None)


@ops.RegisterGradient("SparseTensorDenseAdd")
def _SparseTensorDenseAddGrad(op: ops.Operation, out_grad):
  sp_indices = op.inputs[0]
  # (sparse_indices, sparse_values, sparse_shape, dense)
  return (None, array_ops.gather_nd(out_grad, sp_indices), None, out_grad)


@ops.RegisterGradient("SparseReduceSum")
def _SparseReduceSumGrad(op: ops.Operation, out_grad):
  """Similar to gradient for the Sum Op (i.e. tf.reduce_sum())."""
  sp_indices = op.inputs[0]
  sp_shape = op.inputs[2]
  output_shape_kept_dims = math_ops.reduced_shape(sp_shape, op.inputs[3])
  out_grad_reshaped = array_ops.reshape(out_grad, output_shape_kept_dims)
  scale = sp_shape // math_ops.cast(output_shape_kept_dims, dtypes.int64)
  # (sparse_indices, sparse_values, sparse_shape, reduction_axes)
  return (None, array_ops.gather_nd(out_grad_reshaped,
                                    sp_indices // scale), None, None)


@ops.RegisterGradient("SparseSlice")
def _SparseSliceGrad(op: ops.Operation, *grads):
  """The backward operator for the SparseSlice op.

  This op takes in the upstream gradient w.r.t. non-empty values of
  the sliced `SparseTensor`, and outputs the gradients w.r.t.
  the non-empty values of input `SparseTensor`.

  Args:
    op: the SparseSlice op
    *grads: the incoming gradients, one element per output of `op`

  Returns:
    Gradient for each of the 5 input tensors of SparseSlice:
      (indices, values, shape, start, size)
    The gradients for the indices, shape, start and the size are None.
  """
  backprop_val_grad = grads[1]
  input_indices = op.inputs[0]
  input_start = op.inputs[3]
  output_indices = op.outputs[0]

  val_grad = gen_sparse_ops.sparse_slice_grad(backprop_val_grad, input_indices,
                                              input_start, output_indices)
  val_grad.set_shape(op.inputs[1].get_shape())
  # (indices, values, shape, start, size)
  return (None, val_grad, None, None, None)


@ops.RegisterGradient("SparseTensorDenseMatMul")
def _SparseTensorDenseMatMulGrad(op: ops.Operation, grad):
  """Gradients for the dense tensor in the SparseTensorDenseMatMul op.

  Args:
    op: the SparseTensorDenseMatMul op
    grad: the incoming gradient

  Returns:
    Gradient for each of the 4 input tensors:
      (sparse_indices, sparse_values, sparse_shape, dense_tensor)
    The gradients for indices and shape are None.

  Raises:
    TypeError: When the two operands don't have the same type.
  """
  a_indices, a_values, a_shape = op.inputs[:3]
  b = op.inputs[3]
  adj_a = op.get_attr("adjoint_a")
  adj_b = op.get_attr("adjoint_b")

  a_type = a_values.dtype.base_dtype
  b_type = b.dtype.base_dtype
  if a_type != b_type:
    raise TypeError(
        f"SparseTensorDenseMatMul op received operands with different types: "
        f"`{a_type}` and `{b_type}`.")

  # gradient w.r.t. dense
  b_grad = gen_sparse_ops.sparse_tensor_dense_mat_mul(
      a_indices, a_values, a_shape, grad, adjoint_a=not adj_a)
  if adj_b:
    b_grad = array_ops.matrix_transpose(b_grad, conjugate=True)

  # gradient w.r.t. sparse values

  # TODO(zongheng): these gather calls could potentially duplicate rows/cols in
  # memory.  If there is a need, we should look into implementing this more
  # intelligently to avoid duplicating data.

  # With no adjoints, a_grad is matmul(grad, adjoint(b)). Since a is sparse, we
  # just want to compute that matmul at the rows/columns of non-zero values. The
  # (r, c) value is sum(grad[r, :] * adjoint(b)[:, c]), where the latter term is
  # more conveniently written as conj(b)[c, :]. That expression is more
  # efficient to calculate as a matmul, after expanding the two terms to be 2D
  # (i.e. a row vector and a column vector).
  #
  # If adj_b then we replace conj(b) by transpose(b); if adj_a we need to
  # adjoint the result, which is equivalent to swapping r and c and taking
  # conjugates.

  # Get grad[r, :] and b[c, :] (or with r and c swapped if adj_a, or with
  # transpose(b) if adj_b), as batches of vectors (with the batch dimension
  # corresponding to the non-zero indices of a).
  rows = a_indices[:, 0]
  cols = a_indices[:, 1]
  parts_a = array_ops.gather(grad, rows if not adj_a else cols)
  parts_b = array_ops.gather(
      b if not adj_b else array_ops.transpose(b), cols if not adj_a else rows)

  if not adj_a and not adj_b:
    # grad[r, :] * conj(b[c, :]) = row(grad[r, :]) @ adjoint(row(b[c, :]))
    a_values_grad = math_ops.matmul(
        array_ops.expand_dims(parts_a, -2),
        array_ops.expand_dims(parts_b, -2),
        adjoint_b=True)
  elif adj_a and not adj_b:
    # conj(grad[c, :] * conj(b[r, :])) = adjoint(col(grad[c, :])) @ col(b[r, :])
    a_values_grad = math_ops.matmul(
        array_ops.expand_dims(parts_a, -1),
        array_ops.expand_dims(parts_b, -1),
        adjoint_a=True)
  elif not adj_a and adj_b:
    # grad[r, :] * transpose(b)[c, :] =
    #     row(grad[r, :]) @ col(transpose(b)[c, :])
    a_values_grad = math_ops.matmul(
        array_ops.expand_dims(parts_a, -2), array_ops.expand_dims(parts_b, -1))
  elif adj_a and adj_b:
    # conj(grad[c, :] * transpose(b)[r, :]) =
    #     adjoint(col(grad[c, :])) @ adjoint(row(transpose(b)[r, :])
    a_values_grad = math_ops.matmul(
        array_ops.expand_dims(parts_a, -1),
        array_ops.expand_dims(parts_b, -2),
        adjoint_a=True,
        adjoint_b=True)

  # gradients w.r.t. (a_indices, a_values, a_shape, b)
  return (None, array_ops.squeeze(a_values_grad, axis=[-2, -1]), None, b_grad)


@ops.RegisterGradient("SparseDenseCwiseAdd")
def _SparseDenseCwiseAddGrad(unused_op, unused_grad):
  raise NotImplementedError(
      "Gradient for SparseDenseCwiseAdd is not implemented.")


def _SparseDenseCwiseMulOrDivGrad(op: ops.Operation, grad, is_mul):
  """Common code for SparseDenseCwise{Mul,Div} gradients."""
  x_indices = op.inputs[0]
  x_shape = op.inputs[2]
  y = op.inputs[3]

  y_shape = math_ops.cast(array_ops.shape(y), dtypes.int64)
  num_added_dims = array_ops.expand_dims(
      array_ops.size(x_shape) - array_ops.size(y_shape), 0)
  augmented_y_shape = array_ops.concat(
      [array_ops.ones(num_added_dims, ops.dtypes.int64), y_shape], 0)

  scaling = x_shape // augmented_y_shape
  scaled_indices = x_indices // scaling
  scaled_indices = array_ops.slice(scaled_indices,
                                   array_ops.concat([[0], num_added_dims], 0),
                                   [-1, -1])
  dense_vals = array_ops.gather_nd(y, scaled_indices)

  if is_mul:
    dx = grad * dense_vals
    dy_val = grad * op.inputs[1]
  else:
    dx = grad / dense_vals
    dy_val = grad * (-op.inputs[1] / math_ops.square(dense_vals))
  # indices can repeat after scaling, so we can't use sparse_to_dense().
  dy = sparse_ops.sparse_add(
      array_ops.zeros_like(y),
      sparse_tensor.SparseTensor(scaled_indices, dy_val, y_shape))

  # (sp_indices, sp_vals, sp_shape, dense)
  return (None, dx, None, dy)


@ops.RegisterGradient("SparseDenseCwiseMul")
def _SparseDenseCwiseMulGrad(op: ops.Operation, grad):
  """Gradients for SparseDenseCwiseMul."""
  return _SparseDenseCwiseMulOrDivGrad(op, grad, True)


@ops.RegisterGradient("SparseDenseCwiseDiv")
def _SparseDenseCwiseDivGrad(op: ops.Operation, grad):
  """Gradients for SparseDenseCwiseDiv."""
  return _SparseDenseCwiseMulOrDivGrad(op, grad, False)


@ops.RegisterGradient("SparseSoftmax")
def _SparseSoftmaxGrad(op: ops.Operation, grad):
  """Gradients for SparseSoftmax.

  The calculation is the same as SoftmaxGrad:

    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

  where we now only operate on the non-zero values present in the SparseTensors.

  Args:
    op: the SparseSoftmax op.
    grad: the upstream gradient w.r.t. the non-zero SparseSoftmax output values.

  Returns:
    Gradients w.r.t. the input (sp_indices, sp_values, sp_shape).
  """
  indices, shape = op.inputs[0], op.inputs[2]
  out_vals = op.outputs[0]
  sp_output = sparse_tensor.SparseTensor(indices, out_vals, shape)
  sp_grad = sparse_tensor.SparseTensor(indices, grad, shape)
  sp_product = sparse_tensor.SparseTensor(indices,
                                          sp_output.values * sp_grad.values,
                                          shape)

  # [..., B, 1], dense.
  sum_reduced = -sparse_ops.sparse_reduce_sum(sp_product, [-1], keepdims=True)
  # sparse [..., B, C] + dense [..., B, 1] with broadcast; outputs sparse.
  sp_sum = sparse_ops.sparse_dense_cwise_add(sp_grad, sum_reduced)

  grad_x = sp_sum.values * sp_output.values
  return [None, grad_x, None]


@ops.RegisterGradient("SparseSparseMaximum")
def _SparseSparseMaximumGrad(unused_op: ops.Operation, unused_grad):
  raise NotImplementedError(
      "Gradient for SparseSparseMaximum is not implemented."
  )


@ops.RegisterGradient("SparseSparseMinimum")
def _SparseSparseMinimumGrad(unused_op: ops.Operation, unused_grad):
  raise NotImplementedError(
      "Gradient for SparseSparseMinimum is not implemented."
  )


@ops.RegisterGradient("SparseFillEmptyRows")
def _SparseFillEmptyRowsGrad(
    op: ops.Operation,
    unused_grad_output_indices,
    output_grad_values,
    unused_grad_empty_row_indicator,
    unused_grad_reverse_index_map,
):
  """Gradients for SparseFillEmptyRows."""
  reverse_index_map = op.outputs[3]

  d_values, d_default_value = gen_sparse_ops.sparse_fill_empty_rows_grad(
      reverse_index_map=reverse_index_map, grad_values=output_grad_values
  )

  # d_indices, d_values, d_dense_shape, d_default_value.
  return [None, d_values, None, d_default_value]


@ops.RegisterGradient("SparseToDense")
def _SparseToDenseGrad(op: ops.Operation, grad):
  sparse_indices, output_shape, _, _ = op.inputs

  sparse_values_grad = array_ops.gather_nd(grad, sparse_indices)
  default_value_grad = math_ops.reduce_sum(grad) - math_ops.reduce_sum(
      sparse_values_grad)
  return [
      array_ops.zeros_like(sparse_indices),
      array_ops.zeros_like(output_shape), sparse_values_grad, default_value_grad
  ]
