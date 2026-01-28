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

#include <cstddef>

#include "absl/base/casts.h"
#include "xla/stream_executor/gpu/collective_signal.cu.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.cu.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/rocm/collective_signal_rocm.cu.h"  // IWYU pragma: keep
#include "xla/stream_executor/rocm/rocm_platform_id.h"

namespace stream_executor::gpu {

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    MultiGpuBarrierKernelRocm,                    // 1. Identifier
    stream_executor::gpu::MultiGpuBarrierKernel,  // 2. KernelTrait
    stream_executor::rocm::kROCmPlatformId,       // 3. Platform ID
    ([](size_t arity) {                           // 4. Kernel Spec Creator
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(
              &MultiGpuBarrierKernelImpl<PlatformType::kRocm>),
          "multi_gpu_barrier_kernel", arity);
    }));

}  // namespace stream_executor::gpu
