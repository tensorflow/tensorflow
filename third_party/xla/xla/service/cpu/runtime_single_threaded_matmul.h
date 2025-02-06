/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_MATMUL_H_
#define XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_MATMUL_H_

#include <complex>
#include <cstdint>

#include "Eigen/Core"
#include "tsl/platform/ml_dtypes.h"

extern "C" {

// Performs a single-threaded matrix multiplication using Eigen. 'lhs' and 'rhs'
// are pointers to buffers containing input matrices in column-major order.
// 'out' is a pointer to a buffer sufficiently large to hold the result of the
// operation. Following standard nomenclature: lhs is m x k, rhs is k x n, and
// out is m x n.
extern void __xla_cpu_runtime_EigenSingleThreadedMatMulF16(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    Eigen::half* out, Eigen::half* lhs, Eigen::half* rhs, int64_t m, int64_t n,
    int64_t k, int32_t transpose_lhs, int32_t transpose_rhs);

extern void __xla_cpu_runtime_EigenSingleThreadedMatMulF32(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, float* out,
    float* lhs, float* rhs, int64_t m, int64_t n, int64_t k,
    int32_t transpose_lhs, int32_t transpose_rhs);

extern void __xla_cpu_runtime_EigenSingleThreadedMatMulF64(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, double* out,
    double* lhs, double* rhs, int64_t m, int64_t n, int64_t k,
    int32_t transpose_lhs, int32_t transpose_rhs);

extern void __xla_cpu_runtime_EigenSingleThreadedMatMulC64(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    std::complex<float>* out, std::complex<float>* lhs,
    std::complex<float>* rhs, int64_t m, int64_t n, int64_t k,
    int32_t transpose_lhs, int32_t transpose_rhs);

extern void __xla_cpu_runtime_EigenSingleThreadedMatMulC128(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    std::complex<double>* out, std::complex<double>* lhs,
    std::complex<double>* rhs, int64_t m, int64_t n, int64_t k,
    int32_t transpose_lhs, int32_t transpose_rhs);

extern void __xla_cpu_runtime_EigenSingleThreadedMatMulS32(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, int32_t* out,
    int32_t* lhs, int32_t* rhs, int64_t m, int64_t n, int64_t k,
    int32_t transpose_lhs, int32_t transpose_rhs);

extern void __xla_cpu_runtime_EigenSingleThreadedMatMulU8(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, uint8_t* out,
    uint8_t* lhs, uint8_t* rhs, int64_t m, int64_t n, int64_t k,
    int32_t transpose_lhs, int32_t transpose_rhs);

extern void __xla_cpu_runtime_EigenSingleThreadedMatMulF8E5M2(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    tsl::float8_e5m2* out, tsl::float8_e5m2* lhs, tsl::float8_e5m2* rhs,
    int64_t m, int64_t n, int64_t k, int32_t transpose_lhs,
    int32_t transpose_rhs);

extern void __xla_cpu_runtime_EigenSingleThreadedMatMulF8E4M3FN(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    tsl::float8_e4m3fn* out, tsl::float8_e4m3fn* lhs, tsl::float8_e4m3fn* rhs,
    int64_t m, int64_t n, int64_t k, int32_t transpose_lhs,
    int32_t transpose_rhs);

}  // extern "C"

#endif  // XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_MATMUL_H_
