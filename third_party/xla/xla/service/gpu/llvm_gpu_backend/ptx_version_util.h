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

#ifndef XLA_SERVICE_GPU_LLVM_GPU_BACKEND_PTX_VERSION_UTIL_H_
#define XLA_SERVICE_GPU_LLVM_GPU_BACKEND_PTX_VERSION_UTIL_H_

#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla::gpu::nvptx {

// Determine PTX version from CUDA version.
stream_executor::SemanticVersion
DetermineHighestSupportedPtxVersionFromCudaVersion(
    stream_executor::SemanticVersion cuda_version);

// Returns the minimum PTX version required for the given compute capability.
// Used to ensure we emit PTX that the target GPU supports (e.g. sm_120/sm_120a
// require PTX 8.7+ per NVIDIA; PTX 8.5 does not support them).
stream_executor::SemanticVersion GetMinimumPtxVersionForComputeCapability(
    stream_executor::CudaComputeCapability compute_capability);

}  // namespace xla::gpu::nvptx

#endif  // XLA_SERVICE_GPU_LLVM_GPU_BACKEND_PTX_VERSION_UTIL_H_
