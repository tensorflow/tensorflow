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

#include "xla/service/cpu/runtime_single_threaded_matmul.h"

#include <cstdint>

#include "absl/base/attributes.h"
#include "xla/service/cpu/runtime_single_threaded_matmul_common.h"

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_EigenSingleThreadedMatMulS32(const void* run_options_ptr,
                                               int32_t* out, int32_t* lhs,
                                               int32_t* rhs, int64_t m,
                                               int64_t n, int64_t k,
                                               int32_t transpose_lhs,
                                               int32_t transpose_rhs) {
  xla::SingleThreadedMatMulDispatch<int32_t>(
      run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
}
