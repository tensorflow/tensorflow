/* Copyright 2025 The OpenXLA Authors.

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
#include <cstdint>

#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/gpu_test_kernel_traits.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_lib.cu.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    AddI32KernelRocm, stream_executor::gpu::internal::AddI32Kernel,
    stream_executor::rocm::kROCmPlatformId, ([](size_t arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::AddI32),

          "AddI32", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    MulI32KernelRocm, stream_executor::gpu::internal::MulI32Kernel,
    stream_executor::rocm::kROCmPlatformId, ([](size_t arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::MulI32),

          "MulI32", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    IncAndCmpKernelRocm, stream_executor::gpu::internal::IncAndCmpKernel,
    stream_executor::rocm::kROCmPlatformId, ([](size_t arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::IncAndCmp),

          "IncAndCmp", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    AddI32Ptrs3KernelRocm, stream_executor::gpu::internal::AddI32Ptrs3Kernel,
    stream_executor::rocm::kROCmPlatformId, ([](size_t arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::AddI32Ptrs3),
          "AddI32Ptrs3", arity,
          [&](const stream_executor::Kernel& kernel,
              const stream_executor::KernelArgs& args) {
            auto bufs = stream_executor::Cast<
                            stream_executor::KernelArgsDeviceMemoryArray>(&args)
                            ->device_memory_args();
            auto cast = [](auto m) {
              return reinterpret_cast<int32_t*>(m.opaque());
            };
            return stream_executor::PackKernelArgs(
                /*shmem_bytes=*/0, stream_executor::gpu::Ptrs3<int32_t>{
                                       cast(bufs[0]),
                                       cast(bufs[1]),
                                       cast(bufs[2]),
                                   });
          });
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    CopyKernelRocm, stream_executor::gpu::internal::CopyKernel,
    stream_executor::rocm::kROCmPlatformId, ([](size_t arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::CopyKernel),

          "CopyKernel", arity);
    }));
