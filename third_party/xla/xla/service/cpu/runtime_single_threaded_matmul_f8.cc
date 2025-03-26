/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>

#include "absl/base/attributes.h"
#include "xla/service/cpu/runtime_single_threaded_matmul.h"
#include "xla/service/cpu/runtime_single_threaded_matmul_common.h"
#include "tsl/platform/ml_dtypes.h"

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_EigenSingleThreadedMatMulF8E5M2(
    const void* run_options_ptr, tsl::float8_e5m2* out, tsl::float8_e5m2* lhs,
    tsl::float8_e5m2* rhs, int64_t m, int64_t n, int64_t k,
    int32_t transpose_lhs, int32_t transpose_rhs) {
  xla::SingleThreadedMatMulDispatch<tsl::float8_e5m2>(
      run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_EigenSingleThreadedMatMulF8E4M3FN(
    const void* run_options_ptr, tsl::float8_e4m3fn* out,
    tsl::float8_e4m3fn* lhs, tsl::float8_e4m3fn* rhs, int64_t m, int64_t n,
    int64_t k, int32_t transpose_lhs, int32_t transpose_rhs) {
  xla::SingleThreadedMatMulDispatch<tsl::float8_e4m3fn>(
      run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
}
