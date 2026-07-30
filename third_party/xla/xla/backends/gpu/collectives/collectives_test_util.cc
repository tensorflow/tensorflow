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

#include "xla/backends/gpu/collectives/collectives_test_util.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"

namespace xla::gpu {

absl::StatusOr<std::vector<se::StreamExecutor*>> CreateExecutors(
    se::Platform* platform, size_t n) {
  std::vector<se::StreamExecutor*> executors(n);
  for (size_t d = 0; d < n; ++d) {
    ASSIGN_OR_RETURN(executors[d], platform->ExecutorForDevice(d));
  }
  return executors;
}

std::vector<std::unique_ptr<GpuCommunicator>> DowncastComms(
    std::vector<std::unique_ptr<Communicator>> comms) {
  std::vector<std::unique_ptr<GpuCommunicator>> gpu_comms;
  gpu_comms.reserve(comms.size());
  for (size_t i = 0; i < comms.size(); ++i) {
    gpu_comms.emplace_back(dynamic_cast<GpuCommunicator*>(comms[i].release()));
  }
  return gpu_comms;
}

absl::StatusOr<std::vector<std::unique_ptr<GpuCommunicator>>>
CreateCommunicators(absl::Span<se::StreamExecutor* const> executors,
                    std::vector<GlobalDeviceId> device_ids, bool blocking,
                    size_t num_ids, GpuCollectives* custom_backend) {
  CHECK_EQ(executors.size(), device_ids.size());

  GpuCollectives* collectives = custom_backend;
  if (custom_backend == nullptr) {
    collectives = GpuCollectives::Default("GPU");
  }

  std::vector<GpuCollectives::Device> devices;
  devices.reserve(executors.size());
  for (se::StreamExecutor* executor : executors) {
    devices.emplace_back(executor);
  }

  std::vector<GpuCollectives::DeviceRank> device_ranks;
  device_ranks.reserve(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    device_ranks.emplace_back(&devices[i], RankId(i));
  }

  CliqueIds clique_ids;
  for (size_t i = 0; i < num_ids; ++i) {
    ASSIGN_OR_RETURN(CliqueId clique_id, collectives->CreateUniqueCliqueId());
    clique_ids.Add(clique_id);
  }

  GpuCliqueKey clique_key(device_ids, executors.size());

  GpuCollectives::Config config;
  config.blocking_communicators = blocking;
  config.async_execution = !blocking;

  ASSIGN_OR_RETURN(auto comms, collectives->CreateCommunicatorsWithCancel(
                                   clique_key, clique_ids, device_ranks, config,
                                   std::make_shared<CancellationToken>()));
  return DowncastComms(std::move(comms));
}

absl::StatusOr<std::vector<std::unique_ptr<GpuCommunicator>>>
SplitCommunicators(
    absl::Span<se::StreamExecutor* const> executors,
    absl::Span<const std::unique_ptr<GpuCommunicator>> existing_comms,
    std::vector<GlobalDeviceId> device_ids, bool blocking) {
  CHECK_EQ(executors.size(), device_ids.size());

  GpuCollectives* collectives = GpuCollectives::Default("GPU");

  std::vector<GpuCollectives::Device> devices;
  devices.reserve(executors.size());
  for (se::StreamExecutor* executor : executors) {
    devices.emplace_back(executor);
  }

  std::vector<GpuCollectives::DeviceRank> device_ranks;
  device_ranks.reserve(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    device_ranks.emplace_back(&devices[i], RankId(i));
  }

  GpuCollectives::Config config;
  config.blocking_communicators = blocking;
  config.async_execution = !blocking;

  std::vector<RankId> keys(existing_comms.size());
  std::vector<const Communicator*> existing_comms_ptrs(existing_comms.size());
  for (size_t i = 0; i < existing_comms.size(); ++i) {
    keys[i] = RankId(i);
    existing_comms_ptrs[i] = existing_comms[i].get();
  }

  ASSIGN_OR_RETURN(auto comms,
                   collectives->SplitCommunicatorsWithCancel(
                       existing_comms_ptrs, /*color=*/0, keys, config,
                       device_ranks, std::make_shared<CancellationToken>()));
  return DowncastComms(std::move(comms));
}

absl::StatusOr<std::vector<std::unique_ptr<se::MemoryAllocator>>>
CreateMemoryAllocators(absl::Span<se::StreamExecutor* const> executors) {
  std::vector<std::unique_ptr<se::MemoryAllocator>> allocators;
  allocators.reserve(executors.size());
  for (se::StreamExecutor* executor : executors) {
    ASSIGN_OR_RETURN(
        allocators.emplace_back(),
        executor->CreateMemoryAllocator(se::MemorySpace::kCollective));
  }
  return allocators;
}

absl::StatusOr<std::vector<std::unique_ptr<se::MemoryAllocation>>> Allocate(
    absl::Span<const std::unique_ptr<se::MemoryAllocator>> allocators,
    size_t num_bytes) {
  std::vector<std::unique_ptr<se::MemoryAllocation>> allocations;
  allocations.reserve(allocators.size());
  for (auto& allocator : allocators) {
    ASSIGN_OR_RETURN(allocations.emplace_back(),
                     allocator->Allocate(num_bytes));
  }
  return allocations;
}

std::vector<Future<std::unique_ptr<SymmetricMemory>>> CreateSymmetricMemory(
    tsl::Executor& exec,
    absl::Span<const std::unique_ptr<GpuCommunicator>> comms,
    absl::Span<const std::unique_ptr<se::MemoryAllocation>> allocs) {
  CHECK_EQ(comms.size(), allocs.size());

  std::vector<Future<std::unique_ptr<SymmetricMemory>>> futures;
  futures.reserve(allocs.size());
  for (size_t i = 0; i < comms.size(); ++i) {
    futures.emplace_back(MakeFutureOn(exec, [=] {
      return comms[i]->CreateSymmetricMemory(allocs[i]->address());
    }));
  }

  return futures;
}

absl::StatusOr<std::vector<std::unique_ptr<SymmetricMemory>>>
AwaitSymmetricMemory(
    std::vector<Future<std::unique_ptr<SymmetricMemory>>> futures) {
  std::vector<std::unique_ptr<SymmetricMemory>> symm;
  symm.reserve(futures.size());

  for (auto& future : futures) {
    ASSIGN_OR_RETURN(symm.emplace_back(), std::move(future).Await());
  }

  return symm;
}

}  // namespace xla::gpu
