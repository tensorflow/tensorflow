// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations under
// the License.
// ==============================================================================

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("WALSComputePartialLhsAndRhs")
    .Input("factors: float32")
    .Input("factor_weights: float32")
    .Input("unobserved_weights: float32")
    .Input("input_weights: float32")
    .Input("input_indices: int64")
    .Input("input_values: float32")
    .Input("input_block_size: int64")
    .Input("input_is_transpose: bool")
    .Output("partial_lhs: float32")
    .Output("partial_rhs: float32")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"(
Computes the partial left-hand side and right-hand side of WALS update.

factors: Matrix of size m * k.
factor_weights: Vector of size m. Corresponds to column weights
unobserved_weights: Scalar. Weight for unobserved input entries.
input_weights: Vector of size n. Corresponds to row weights.
input_indices: Indices for the input SparseTensor.
input_values: Values for the input SparseTensor.
input_block_size: Scalar. Number of rows spanned by input.
input_is_transpose: If true, logically transposes the input for processing.
partial_lhs: 3-D tensor with size input_block_size x k x k.
partial_rhs: Matrix with size input_block_size x k.
)");

REGISTER_OP("MaskedMatmul")
    .Input("a: float32")
    .Input("b: float32")
    .Input("mask_indices: int64")
    .Input("transpose_a: bool")
    .Input("transpose_b: bool")
    .Output("prod_values: float32")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"(
Computes the product a * b, but only for indices (i, j) in mask_indices. The
result is stored in prod_values, a rank 1 tensor, such that for all i,
prod_values[i] = (a * b)[mask_indices[i, 0], mask_indices[i, 1]].
Note that the shapes of the input matrices a, b should be compatible (after
transposing as specified by the arguments transpose_a and transpose_b).

Input arguments:
a: A rank 2 tensor of shape [m, n].
b: A rank 2 tensor of shape [s, t]. The inner dimensions of a and b should match
  after transposition.
mask_indices: A rank 2 tensor, of shape [nnz, 2] where nnz is the number of
  non-zero elements in the output. The indices are not assumed to be in
  lexicographic, or any particular order.
  For all i, mask_indices[i, :] should represent a valid index of the product
  matrix (a * b) (after transposition). That is:
  mask_indices[i, 0] should be in [0, m) if !transpose_a, and in [0, n)
    otherwise.
  mask_indices[i, 1] should be in [0, t) if !transpose_b, and in [0, s)
    otherwise.
transpose_a: A boolean, specifies whether to transpose the matrix a.
transpose_b: A boolean, specifies whether to transpose the matrix b.

Output arguments:
prod_values: A rank 1 tensor of shape [nnz], representing the values of the
  non-zero elements in the product, such that for all i,
  prod_values[i] = (a * b)[mask_indices[i, 0], mask_indices[i, 1]].
)");

}  // namespace tensorflow
