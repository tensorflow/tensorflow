# Copyright 2015 Google Inc. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


ops.NoGradient("SparseAddGrad")
ops.NoGradient("SparseConcat")
ops.NoGradient("SparseReduceSum")
ops.NoGradient("SparseToDense")


@ops.RegisterGradient("SparseReorder")
def _SparseReorderGrad(op, unused_output_indices_grad, output_values_grad):
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
  sp_unordered = ops.SparseTensor(input_indices, entry_indices, input_shape)
  sp_ordered = sparse_ops.sparse_reorder(sp_unordered)
  inverted_permutation = array_ops.invert_permutation(sp_ordered.values)

  return (None,
          array_ops.gather(output_values_grad, inverted_permutation),
          None)


@ops.RegisterGradient("SparseAdd")
def _SparseAddGrad(op, *grads):
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

  # pylint: disable=protected-access
  a_val_grad, b_val_grad = gen_sparse_ops._sparse_add_grad(val_grad, a_indices,
                                                           b_indices,
                                                           sum_indices)
  a_val_grad.set_shape(op.inputs[1].get_shape())
  b_val_grad.set_shape(op.inputs[4].get_shape())
  # (a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh)
  return (None, a_val_grad, None, None, b_val_grad, None, None)


@ops.RegisterGradient("SparseTensorDenseAdd")
def _SparseTensorDenseAdd(op, out_grad):
  sp_indices = op.inputs[0]
  #  (sparse_indices, sparse_values, sparse_shape, dense)
  return (None, array_ops.gather_nd(out_grad, sp_indices), None, out_grad)


@ops.RegisterGradient("SparseTensorDenseMatMul")
def _SparseTensorDenseMatMulGrad(op, grad):
  """Gradients for the dense tensor in the SparseTensorDenseMatMul op.

  If either input is complex, no gradient is provided.

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
  sp_t = ops.SparseTensor(*op.inputs[:3])
  adj_a = op.get_attr("adjoint_a")
  adj_b = op.get_attr("adjoint_b")

  a_type = sp_t.values.dtype.base_dtype
  b_type = op.inputs[3].dtype.base_dtype
  if a_type != b_type:
    raise TypeError("SparseTensorDenseMatMul op received operands with "
                    "different types: ", a_type, " and ", b_type)
  is_complex = a_type == ops.dtypes.complex64
  if is_complex:
    raise NotImplementedError("SparseTensorDenseMatMul op does not support "
                              "complex gradients.")

  # gradient w.r.t. dense
  b_grad = sparse_ops.sparse_tensor_dense_matmul(sp_t, grad,
                                                 adjoint_a=not adj_a)
  if adj_b:
    b_grad = array_ops.transpose(b_grad)

  # gradient w.r.t. sparse values
  a_indices = op.inputs[0]
  b = op.inputs[3]

  rows = a_indices[:, 0]
  cols = a_indices[:, 1]

  # TODO(zongheng, ebrevdo): add conjugates in the right places when complex
  # values are allowed.
  # TODO(zongheng): these gather calls could potentially duplicate rows/cols in
  # memory.  If there is a need, we should look into implementing this more
  # intelligently to avoid duplicating data.
  parts_a = array_ops.gather(grad, rows if not adj_a else cols)
  parts_b = array_ops.gather(b if not adj_b else array_ops.transpose(b),
                             cols if not adj_a else rows)
  a_values_grad = math_ops.reduce_sum(parts_a * parts_b, reduction_indices=1)

  # gradients w.r.t. (a_indices, a_values, a_shape, b)
  return (None, a_values_grad, None, b_grad)
