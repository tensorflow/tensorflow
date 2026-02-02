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

}  // namespace tensorflow
