/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Minimal stand-in for the PyTorch header FlashAttention-2 includes.
// Instead of throwing, the first error is recorded so the caller (fa2_api.cu)
// can report it through TF/XLA.
#pragma once  // NOLINT(build/header_guard): stands in for a PyTorch header.
#include <cuda_runtime.h>

#include <cstdio>  // FA2 sources rely on these arriving via the real PyTorch headers.
#include <cstdlib>

namespace fa2_shim {
inline thread_local cudaError_t last_error = cudaSuccess;
inline void record(cudaError_t err) {
  if (err != cudaSuccess && last_error == cudaSuccess) last_error = err;
}
}  // namespace fa2_shim

#define C10_CUDA_CHECK(EXPR) fa2_shim::record(EXPR)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() fa2_shim::record(cudaGetLastError())
