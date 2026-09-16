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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_COLLECTIVES_TEST_UTIL_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_COLLECTIVES_TEST_UTIL_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"

namespace xla::gpu {

absl::StatusOr<std::vector<se::StreamExecutor*>> CreateExecutors(
    se::Platform* platform, size_t n);

std::vector<std::unique_ptr<GpuCommunicator>> DowncastComms(
    std::vector<std::unique_ptr<Communicator>> comms);

absl::StatusOr<std::vector<std::unique_ptr<GpuCommunicator>>>
CreateCommunicators(absl::Span<se::StreamExecutor* const> executors,
                    std::vector<GlobalDeviceId> device_ids,
                    bool blocking = true, size_t num_ids = 1,
                    GpuCollectives* custom_backend = nullptr);

absl::StatusOr<std::vector<std::unique_ptr<GpuCommunicator>>>
SplitCommunicators(
    absl::Span<se::StreamExecutor* const> executors,
    absl::Span<const std::unique_ptr<GpuCommunicator>> existing_comms,
    std::vector<GlobalDeviceId> device_ids, bool blocking = true);

absl::StatusOr<std::vector<std::unique_ptr<se::MemoryAllocator>>>
CreateMemoryAllocators(absl::Span<se::StreamExecutor* const> executors);

absl::StatusOr<std::vector<std::unique_ptr<se::MemoryAllocation>>> Allocate(
    absl::Span<const std::unique_ptr<se::MemoryAllocator>> allocators,
    size_t num_bytes);

std::vector<Future<std::unique_ptr<SymmetricMemory>>> CreateSymmetricMemory(
    tsl::Executor& exec,
    absl::Span<const std::unique_ptr<GpuCommunicator>> comms,
    absl::Span<const std::unique_ptr<se::MemoryAllocation>> allocs);

absl::StatusOr<std::vector<std::unique_ptr<SymmetricMemory>>>
AwaitSymmetricMemory(
    std::vector<Future<std::unique_ptr<SymmetricMemory>>> futures);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_COLLECTIVES_TEST_UTIL_H_
