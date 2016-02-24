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
from tensorflow.python.ops import sparse_ops


ops.NoGradient("SparseToDense")


ops.NoGradient("SparseConcat")


ops.NoGradient("SparseReorder")


@ops.RegisterGradient("SparseTensorDenseMatMul")
def _SparseTensorDenseMatMulGrad(op, grad):
  """Gradients for the dense tensor in the SparseTensorDenseMatMul op.

  Gradients are only provided for the dense tensor.

  If either input is complex, no gradient is provided.

  Args:
    op: the SparseTensorDenseMatMul op
    grad: the incoming gradient

  Returns:
    Gradient for each of the 4 input tensors:
      (sparse_indices, sparse_values, sparse_shape, dense_tensor)
    The sparse tensor gradients are always None.
  """
  sp_t = ops.SparseTensor(*op.inputs[:3])
  adj_a = op.get_attr("adjoint_a")
  adj_b = op.get_attr("adjoint_b")

  a_type = sp_t.values.dtype
  b_type = op.inputs[3].dtype
  assert a_type == b_type
  is_complex = a_type == ops.dtypes.complex64
  if is_complex:
    raise NotImplementedError("SparseTensorDenseMatMul op does not support "
                              "complex gradients.")

  b_grad = sparse_ops.sparse_tensor_dense_matmul(sp_t, grad,
                                                 adjoint_a=not adj_a)
  if adj_b:
    b_grad = array_ops.transpose(b_grad)

  return (None, None, None, b_grad)
