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

#include "xla/backends/cpu/runtime/dot_thunk.h"  // NOLINT IWYU pragma: keep

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "xla/tsl/framework/contraction/eigen_contraction_kernel.h"  // IWYU pragma: keep
#endif

template void ::xla::cpu::DotThunk::TypedMatMul<float>(
    const Eigen::ThreadPoolDevice* device, void* out, void* lhs, void* rhs,
    int64_t m, int64_t n, int64_t k, bool transpose_lhs, bool transpose_rhs,
    DoneCallback done);
