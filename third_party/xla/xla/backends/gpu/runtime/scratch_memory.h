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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SCRATCH_MEMORY_H_
#define XLA_BACKENDS_GPU_RUNTIME_SCRATCH_MEMORY_H_

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_cache.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {
// Holds a scratch device memory per HLO executable which can be used
// by collective operations such as RaggedAllToAll.
class ScratchMemory {
 public:
  explicit ScratchMemory(
      std::shared_ptr<stream_executor::MemoryAllocation> memory_allocation,
      std::shared_ptr<xla::SymmetricMemory> symmetric_memory)
      : memory_allocation_(std::move(memory_allocation)),
        symmetric_memory_(std::move(symmetric_memory)) {}

  absl::StatusOr<stream_executor::DeviceAddressBase> GetScratchMemoryAddress() {
    return memory_allocation_->address();
  }

  absl::StatusOr<std::shared_ptr<xla::SymmetricMemory>> GetSymmetricMemory() {
    return symmetric_memory_;
  }

 private:
  std::shared_ptr<stream_executor::MemoryAllocation> memory_allocation_;
  std::shared_ptr<xla::SymmetricMemory> symmetric_memory_;
};

absl::StatusOr<ScratchMemory> AcquireScratchMemory(
    const CollectiveParams& collective_params,
    const ScratchMemoryRequests& scratch_memory_requests,
    CollectiveMemoryCache& collective_memory_cache,
    stream_executor::StreamExecutor* executor, CollectiveCliques& cliques);
}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_SCRATCH_MEMORY_H_
