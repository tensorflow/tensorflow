/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_PJRT_GPU_GPU_HELPERS_H_
#define XLA_PJRT_GPU_GPU_HELPERS_H_

#include <memory>
#include <optional>
#include <set>
#include <string>

#include "absl/types/span.h"
#include "xla/client/local_client.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "tsl/framework/bfc_allocator.h"

namespace xla {

// Builds an xla::LocalClient for the GPU platform.
StatusOr<LocalClient*> GetGpuXlaClient(
    const std::optional<std::string>& platform_name,
    const std::optional<std::set<int>>& allowed_devices);

// Enables peer access between all pairs of GPUs where possible.
void EnablePeerAccess(absl::Span<se::StreamExecutor* const> executors);

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
  // allocate. This is the default value of XLA_PYTHON_CLIENT_MEM_FRACTION.
  double memory_fraction = 0.75;

  // Only used if kind == kBFC. If true, the allocator will immediately allocate
  // the maximum amount allowed by `memory_fraction`. This reduces
  // fragmentation, allowing more of the total memory to be used. If false, the
  // allocator will allocate more memory as allocations are requested.
  bool preallocate = true;
};

std::unique_ptr<tsl::BFCAllocator> GetGpuHostAllocator(
    se::StreamExecutor* executor);

// Builds a BFCAllocator for all local GPUs.
StatusOr<std::unique_ptr<tsl::BFCAllocator>> CreateBFCAllocator(
    se::StreamExecutor* executor, double memory_fraction, bool preallocate);

}  // namespace xla

#endif  // XLA_PJRT_GPU_GPU_HELPERS_H_
