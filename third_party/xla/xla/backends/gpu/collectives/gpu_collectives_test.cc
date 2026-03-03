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

#include "xla/backends/gpu/collectives/gpu_collectives.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr GlobalDeviceId kD0(0);
static constexpr GlobalDeviceId kD1(1);
static constexpr GlobalDeviceId kD2(2);
static constexpr GlobalDeviceId kD3(2);

static absl::StatusOr<std::vector<se::StreamExecutor*>> CreateExecutors(
    se::Platform* platform, size_t n) {
  std::vector<se::StreamExecutor*> executors(n);
  for (size_t d = 0; d < n; ++d) {
    TF_ASSIGN_OR_RETURN(executors[d], platform->ExecutorForDevice(d));
  }
  return executors;
}

// Creates communicators for the given executors.
static absl::StatusOr<std::vector<std::unique_ptr<GpuCommunicator>>>
CreateCommunicators(absl::Span<se::StreamExecutor* const> executors,
                    std::vector<GlobalDeviceId> device_ids,
                    bool blocking = true, size_t num_ids = 1) {
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

  CliqueIds clique_ids;
  for (size_t i = 0; i < num_ids; ++i) {
    TF_ASSIGN_OR_RETURN(CliqueId clique_id,
                        collectives->CreateUniqueCliqueId());
    clique_ids.Add(clique_id);
  }

  GpuCliqueKey clique_key(device_ids, executors.size());

  GpuCollectives::Config config;
  config.blocking_communicators = blocking;
  config.async_execution = !blocking;

  TF_ASSIGN_OR_RETURN(auto comms,
                      collectives->CreateCommunicatorsWithCancel(
                          clique_key, clique_ids, device_ranks, config,
                          std::make_shared<CancellationToken>()));
  CHECK_EQ(comms.size(), executors.size());

  std::vector<std::unique_ptr<GpuCommunicator>> gpu_comms;
  gpu_comms.reserve(comms.size());
  for (size_t i = 0; i < comms.size(); ++i) {
    gpu_comms.emplace_back(dynamic_cast<GpuCommunicator*>(comms[i].release()));
  }
  return gpu_comms;
}

// Creates memory allocators that allocate physical memory in the collective
// memory space, which makes them compatible with symmetric memory requirements.
static absl::StatusOr<std::vector<std::unique_ptr<se::MemoryAllocator>>>
CreateMemoryAllocators(absl::Span<se::StreamExecutor* const> executors) {
  std::vector<std::unique_ptr<se::MemoryAllocator>> allocators;
  allocators.reserve(executors.size());
  for (se::StreamExecutor* executor : executors) {
    TF_ASSIGN_OR_RETURN(
        allocators.emplace_back(),
        executor->CreateMemoryAllocator(se::MemorySpace::kCollective));
  }
  return allocators;
}

// Allocate `num_bytes` on each allocator.
static absl::StatusOr<std::vector<std::unique_ptr<se::MemoryAllocation>>>
Allocate(absl::Span<const std::unique_ptr<se::MemoryAllocator>> allocators,
         size_t num_bytes) {
  std::vector<std::unique_ptr<se::MemoryAllocation>> allocations;
  allocations.reserve(allocators.size());
  for (auto& allocator : allocators) {
    TF_ASSIGN_OR_RETURN(allocations.emplace_back(),
                        allocator->Allocate(num_bytes));
  }
  return allocations;
}

// Create symmetric memory with given comms and allocations.
static std::vector<Future<std::unique_ptr<SymmetricMemory>>>
CreateSymmetricMemory(
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

// Wait for symmetric memory futures to become available.
static absl::StatusOr<std::vector<std::unique_ptr<SymmetricMemory>>>
AwaitSymmetricMemory(
    std::vector<Future<std::unique_ptr<SymmetricMemory>>> futures) {
  std::vector<std::unique_ptr<SymmetricMemory>> symm;
  symm.reserve(futures.size());

  for (auto& future : futures) {
    TF_ASSIGN_OR_RETURN(symm.emplace_back(), std::move(future).Await());
  }

  return symm;
}

TEST(GpuCollectivesTest, CreateWithMultipleIds) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));

  ASSERT_OK_AND_ASSIGN(
      auto comms, CreateCommunicators(executors, {kD0, kD1}, /*blocking=*/true,
                                      /*num_ids=*/2));

  EXPECT_TRUE(comms[0]->platform_comm().handle);
  EXPECT_TRUE(comms[1]->platform_comm().handle);
}

TEST(GpuCollectivesTest, CreateSymmetricMemory) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1}));

  EXPECT_TRUE(comms[0]->platform_comm().handle);
  EXPECT_TRUE(comms[1]->platform_comm().handle);

  if (!absl::c_all_of(comms, [](auto& c) { return c->SupportsDeviceComm(); })) {
    GTEST_SKIP() << "GPU communicators do not suppoort symmetric memory";
  }

  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(auto allocs, Allocate(allocators, 1024));

  // Because creating symmetric memory is a collective operation, we must call
  // it from a thead pool to avoid deadlocks.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 2);
  tsl::Executor& exec = *pool.AsExecutor();

  // Register allocated buffers as symmetric memory.
  auto fsymm = CreateSymmetricMemory(exec, comms, allocs);
  ASSERT_OK_AND_ASSIGN(auto symm, AwaitSymmetricMemory(std::move(fsymm)));
}

TEST(GpuCollectivesTest, CreateSymmetricMemoryOnDifferentComms) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 4) {
    GTEST_SKIP() << "Test requires at least 4 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors_vec,
                       CreateExecutors(platform, 4));
  absl::Span<se::StreamExecutor*> executors(executors_vec);

  ASSERT_OK_AND_ASSIGN(auto comms0123,
                       CreateCommunicators(executors, {kD0, kD1, kD2, kD3}));
  ASSERT_OK_AND_ASSIGN(auto comms01,
                       CreateCommunicators(executors.first(2), {kD0, kD1}));
  ASSERT_OK_AND_ASSIGN(auto comms23,
                       CreateCommunicators(executors.last(2), {kD2, kD3}));

  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(auto allocs_vec, Allocate(allocators, 1024));
  auto allocs = absl::MakeSpan(allocs_vec);

  // Because creating symmetric memory is a collective operation, we must call
  // it from a thead pool to avoid deadlocks.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 4);
  tsl::Executor& exec = *pool.AsExecutor();

  // In the test below we create multiple symmetric allocation on different
  // groups of communicators from the same physical memory.

  // Create symmetric memory on [0,1,2,3].
  auto fsymm0123 = CreateSymmetricMemory(exec, comms0123, allocs);
  ASSERT_OK_AND_ASSIGN(auto symm0123,
                       AwaitSymmetricMemory(std::move(fsymm0123)));

  // Create symmetric memory on [0,1].
  auto fsymm01 = CreateSymmetricMemory(exec, comms01, allocs.first(2));
  ASSERT_OK_AND_ASSIGN(auto symm01, AwaitSymmetricMemory(std::move(fsymm01)));

  // Create symmetric memory on [2,3].
  auto fsymm23 = CreateSymmetricMemory(exec, comms23, allocs.last(2));
  ASSERT_OK_AND_ASSIGN(auto symm23, AwaitSymmetricMemory(std::move(fsymm23)));
}

TEST(GpuCollectivesTest, CreateDeviceComm) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1}));

  if (!comms[0]->SupportsDeviceComm() || !comms[1]->SupportsDeviceComm()) {
    GTEST_SKIP() << "GPU communicators do not suppoort device-initiated comms";
  }

  GpuDeviceCommunicator::Requirements reqs;
  reqs.lsa_barrier_count = 8;

  // Because creating device comms is a collective operation, we must call
  // it from a thead pool to avoid deadlocks.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 2);
  tsl::Executor& exec = *pool.AsExecutor();

  auto fdev_comm0 =
      MakeFutureOn(exec, [&] { return comms[0]->CreateDeviceComm(reqs); });
  auto fdev_comm1 =
      MakeFutureOn(exec, [&] { return comms[1]->CreateDeviceComm(reqs); });

  ASSERT_OK_AND_ASSIGN(auto dev_comm0, std::move(fdev_comm0).Await());
  ASSERT_OK_AND_ASSIGN(auto dev_comm1, std::move(fdev_comm1).Await());

  EXPECT_TRUE(dev_comm0->platform_comm().handle);
  EXPECT_TRUE(dev_comm1->platform_comm().handle);
}

// Test that GPU communicators can be safely aborted and after they are aborted
// they stay in valid state and reject all API calls.
class GpuAbortCollectivesTest : public ::testing::TestWithParam<bool> {};

static void AssertAborted(absl::Status s) {
  ASSERT_THAT(s, absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                        testing::HasSubstr("aborted")));
};

static void AssertEventAborted(Future<> future) {
  return AssertAborted(future.Await());
};

TEST_P(GpuAbortCollectivesTest, Abort) {
  bool blocking = GetParam();

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));

  ASSERT_OK_AND_ASSIGN(auto comms,
                       CreateCommunicators(executors, {kD0, kD1}, blocking));

  // First time we call Abort it must succeed.
  ASSERT_OK(comms[0]->Abort());
  ASSERT_OK(comms[1]->Abort());

  // We can't abort already aborted communicator.
  AssertAborted(comms[0]->Abort());
  AssertAborted(comms[1]->Abort());

  // Operations with communicator should return abort error.
  AssertAborted(comms[0]->HealthCheck());
  AssertAborted(comms[0]->NumRanks().status());

  // A fake executor and device address to test collective operations.
  GpuCollectives::Executor executor(nullptr);
  se::DeviceAddressBase addr;

  AssertAborted(comms[0]->RegisterBufferOnce(addr, 0, false));
  AssertEventAborted(
      comms[0]->AllReduce(addr, addr, U64, 0, ReductionKind::SUM, executor));
  AssertEventAborted(
      comms[0]->Broadcast(addr, addr, U64, 0, RankId(0), executor));
  AssertEventAborted(comms[0]->ReduceScatter(addr, addr, U64, 0,
                                             ReductionKind::SUM, executor));
  AssertEventAborted(comms[0]->AllGather(addr, addr, U64, 0, executor));
  AssertEventAborted(comms[0]->AllToAll({}, {}, U64, 0, executor));
  AssertEventAborted(
      comms[0]->CollectivePermute(addr, addr, U64, 0, {}, {}, executor));
  AssertEventAborted(comms[0]->Send(addr, U64, 0, RankId(0), executor));
  AssertEventAborted(comms[0]->Recv(addr, U64, 0, RankId(0), executor));
}

INSTANTIATE_TEST_SUITE_P(GpuAbortCollectives, GpuAbortCollectivesTest,
                         testing::Values(true, false));

}  // namespace
}  // namespace xla::gpu
