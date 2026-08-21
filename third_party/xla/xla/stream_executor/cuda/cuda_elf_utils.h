/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_ELF_UTILS_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_ELF_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/types/span.h"

namespace stream_executor::cuda {

// Helper function to parse CUDA Fatbinary containers or raw ELF CUBINs from
// a FatbinWrapper pointer and return an absl::Span pointing to the binary
// payload.
//
// The fatbinary wrapper structure and container headers are defined in:
//  - Official CUDA Toolkit header: <fatbinary_section.h> (__fatBinC_Wrapper_t)
//  - NVIDIA CUDA Binary Utilities:
//  https://docs.nvidia.com/cuda/cuda-binary-utilities/
//  - GPU Ocelot FatBinaryContext: https://github.com/gtcasl/gpuocelot
std::optional<absl::Span<const uint8_t>> ParseFatBinaryOrElf(
    const void* fat_cubin);

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_ELF_UTILS_H_
