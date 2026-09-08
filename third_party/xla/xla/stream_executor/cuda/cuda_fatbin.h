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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_FATBIN_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_FATBIN_H_

#include <cstdint>

namespace stream_executor::cuda {

// Standard uncompressed CUDA Fatbinary container magic number (NVCC).
inline constexpr uint32_t kFatbinMagicUncompressed = 0xba55d10a;

// Compressed CUDA Fatbinary container magic number (CUDA 11+ / 12+ NVCC).
inline constexpr uint32_t kFatbinMagicCompressed = 0xba55ed50;

// Legacy/alternative CUDA Fatbinary payload header magic number.
inline constexpr uint32_t kFatbinMagicLegacy = 0x00101001;

// CUDA fatbin wrapper structure passed to __cudaRegisterFatBinary.
//
// The fatbinary wrapper structure and container headers are defined in:
//  - Official CUDA Toolkit header: <fatbinary_section.h> (__fatBinC_Wrapper_t)
//  - NVIDIA CUDA Binary Utilities:
//    https://docs.nvidia.com/cuda/cuda-binary-utilities/
//  - GPU Ocelot FatBinaryContext: https://github.com/gtcasl/gpuocelot
struct FatbinWrapper {
  uint32_t magic;
  uint32_t version;
  const void* data;
  void* filename_or_fatbins;
};

// CUDA fatbinary header structure at the beginning of the payload.
struct FatHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint64_t fat_size;
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_FATBIN_H_
