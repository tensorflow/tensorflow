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

#ifndef XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_ALLOCATOR_CONFIG_H_
#define XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_ALLOCATOR_CONFIG_H_

#include <cstddef>
#include <cstdint>
#include <optional>

namespace xla {

struct GpuAllocatorConfig {
  enum class Kind {
    kDefault,   // Client picks the best option for the platform.
    kPlatform,  // The platform's default.
    kBFC,  // Allocator using a "Best-Fit with Coalescing" algorithm. Currently
           // only available for GPU.
    kCudaAsync,  // Use the CUDA async allocator.
  };
  Kind kind = Kind::kDefault;

  // Only used if kind == kBFC. The maximum fraction of available memory to
  // allocate. This is the default value of XLA_CLIENT_MEM_FRACTION.
  //
  // If `gpu_system_memory_size` is set, it determines memory allocation.
  // `memory_fraction` won't be used in this case.
  double memory_fraction = 0.75;

  // Only used if kind == kBFC. The absolute size of reserved memory space for
  // GPU system in bytes.
  //
  // If null, the default value `memory_fraction` will be used.
  std::optional<int64_t> gpu_system_memory_size = std::nullopt;

  // Only used if kind == kBFC. If true, the allocator will immediately allocate
  // the maximum amount allowed by `memory_fraction`. This reduces
  // fragmentation, allowing more of the total memory to be used. If false, the
  // allocator will allocate more memory as allocations are requested.
  bool preallocate = true;

  // Amount of collective memory (ncclMemAlloc) to preallocate. If this value is
  // 0, collective memory space will be grown as needed to fit the application's
  // usage, with the drawback of potentially higher fragmentation. If set,
  // should be set to a multiple of 512MB to avoid wasting memory due to
  // granularity requirements.
  size_t collective_memory_size = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_ALLOCATOR_CONFIG_H_
