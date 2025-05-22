/* Copyright 2023 The OpenXLA Authors.

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

#include <cstddef>

#include "absl/base/casts.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/make_batch_pointers_kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"

namespace stream_executor::rocm {
namespace {
__global__ void MakeBatchPointers(char* base, size_t stride, size_t n,
                                  void** ptrs_out) {
  size_t idx = size_t(threadIdx.x) + size_t(blockIdx.x) * size_t(blockDim.x);
  if (idx >= n) return;
  ptrs_out[idx] = base + idx * stride;
}
}  // namespace
}  // namespace stream_executor::rocm

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    MakeBatchPointersKernelRocm, stream_executor::gpu::MakeBatchPointersKernel,
    stream_executor::rocm::kROCmPlatformId, ([](size_t arity) {
      stream_executor::MultiKernelLoaderSpec spec(arity);
      spec.AddInProcessSymbol(
          absl::bit_cast<void*>(&stream_executor::rocm::MakeBatchPointers),
          "make_batch_pointers");
      return spec;
    }));
