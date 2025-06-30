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

#include "xla/service/gpu/llvm_gpu_backend/ptx_version_util.h"

namespace xla::gpu::nvptx {

namespace {
constexpr stream_executor::SemanticVersion kFallbackPtxVersion{6, 5, 0};
constexpr stream_executor::SemanticVersion kMaxPtxVersion{8, 8, 0};
}  // namespace

stream_executor::SemanticVersion
DetermineHighestSupportedPtxVersionFromCudaVersion(
    stream_executor::SemanticVersion cuda_version) {
  if (cuda_version < stream_executor::SemanticVersion{11, 0, 0}) {
    // For everything below CUDA 11 we just fall back to PTX 6.5.
    // We don't support CUDA below 11 anymore.
    return kFallbackPtxVersion;
  }

  // Mapping determined from
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes
  // Examples:
  // CUDA 11.0 -> PTX 7.0
  // CUDA 11.1 -> PTX 7.1
  // CUDA 12.0 -> PTX 8.0
  // CUDA 12.4 -> PTX 8.4
  // This versioning scheme is valid until CUDA 12.6
  if (cuda_version < stream_executor::SemanticVersion{12, 6, 0}) {
    return {cuda_version.major() - 4, cuda_version.minor(), 0};
  }
  // CUDA 12.6 -> PTX 8.5
  // CUDA 12.8 -> PTX 8.7
  // CUDA 12.9 -> PTX 8.8
  if (cuda_version < stream_executor::SemanticVersion{12, 10, 0}) {
    return {cuda_version.major() - 4, cuda_version.minor() - 1, 0};
  }

  // Return maximum known PTX version.
  return kMaxPtxVersion;
}
}  // namespace xla::gpu::nvptx
