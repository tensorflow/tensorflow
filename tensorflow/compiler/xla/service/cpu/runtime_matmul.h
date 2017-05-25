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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATMUL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATMUL_H_

#include "tensorflow/core/platform/types.h"

extern "C" {

// Performs a multi-threaded matrix multiplication using Eigen. 'lhs' and 'rhs'
// are pointers to buffers containing input matrices in column-major
// order. 'out' is a pointer to a buffer sufficiently large to hold the result
// of the operation. Following standard nomenclature: lhs is m x k,
// rhs is k x n, and out is m x n.
extern void __xla_cpu_runtime_EigenMatMulF32(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, float* out,
    float* lhs, float* rhs, tensorflow::int64 m, tensorflow::int64 n,
    tensorflow::int64 k, tensorflow::int32 transpose_lhs,
    tensorflow::int32 transpose_rhs);

extern void __xla_cpu_runtime_EigenMatMulF64(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, double* out,
    double* lhs, double* rhs, tensorflow::int64 m, tensorflow::int64 n,
    tensorflow::int64 k, tensorflow::int32 transpose_lhs,
    tensorflow::int32 transpose_rhs);

}  // extern "C"

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATMUL_H_
