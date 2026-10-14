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
#pragma once  // NOLINT(build/header_guard): stands in for a PyTorch header.
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <tuple>

namespace at::cuda::philox {
// As in PyTorch: when captured_, seed/offset live in device memory (read here,
// inside the kernel); otherwise they were passed by value.
__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(
    at::PhiloxCudaState arg) {
  if (arg.captured_) {
    return std::make_tuple(
        static_cast<uint64_t>(*arg.seed_.ptr),
        static_cast<uint64_t>(*arg.offset_.ptr) + arg.offset_intragraph_);
  }
  return std::make_tuple(arg.seed_.val, arg.offset_.val);
}
}  // namespace at::cuda::philox
