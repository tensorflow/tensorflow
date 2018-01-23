/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_TF2XLA_LIB_TRIANGULAR_SOLVE_H_
#define TENSORFLOW_COMPILER_TF2XLA_LIB_TRIANGULAR_SOLVE_H_

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"

namespace tensorflow {

// Solves systems of linear equations with upper or lower triangular matrices by
// backsubstitution.
//
// `a` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
// square matrices. The strictly upper triangular part of each inner-most matrix
// is assumed to be zero and not accessed.
// `b` is a tensor of shape `[..., M, K]`.
//
// The innermost matrices in the output satisfy matrix equations
// `output[..., i, j] * adjoint(a[..., k, j]) = b[..., i, k]`.
//
// Uses a blocked algorithm if `block_size` is > 1; if block_size == 1 then no
// blocking is used.
// TODO(phawkins): equivalent to the BLAS TRSM routine with side=right,
// kind=lower, and transposed_a=true. Implement the other possible combinations
// of side, kind and transposed_a.
xla::StatusOr<xla::ComputationDataHandle> TriangularSolve(
    xla::ComputationBuilder* builder, const xla::ComputationDataHandle& a,
    xla::ComputationDataHandle b, int64 block_size = 256);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_TRIANGULAR_SOLVE_H_
