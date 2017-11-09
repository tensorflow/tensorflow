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

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATVEC_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATVEC_H_

#include "tensorflow/core/platform/types.h"

namespace xla {

// Performs a matrix-vector multiplication using Eigen. 'lhs' and 'rhs' are
// pointers to buffers containing input matrices in column-major order. 'out' is
// a pointer to a buffer sufficiently large to hold the result of the
// operation. Following standard nomenclature: lhs is m x k, rhs is k x n, and
// out is m x n.
//
// This requires that m = 1 or n = 1.
//
// TODO(b/64684907): Compare runtime performance of these functions with dot
// simplification.
void EigenMatVecF32(float* out, float* lhs, float* rhs, tensorflow::int64 m,
                    tensorflow::int64 n, tensorflow::int64 k,
                    tensorflow::int32 transpose_lhs,
                    tensorflow::int32 transpose_rhs);

void EigenMatVecF64(double* out, double* lhs, double* rhs, tensorflow::int64 m,
                    tensorflow::int64 n, tensorflow::int64 k,
                    tensorflow::int32 transpose_lhs,
                    tensorflow::int32 transpose_rhs);

}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATVEC_H_
