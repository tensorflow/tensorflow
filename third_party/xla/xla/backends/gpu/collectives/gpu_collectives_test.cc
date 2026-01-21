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

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
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

// Creates a pair of communicators for the given executors.
static absl::StatusOr<std::vector<std::unique_ptr<GpuCommunicator>>>
CreateCommunicators(se::StreamExecutor* executor0,
                    se::StreamExecutor* executor1, bool blocking = true) {
  GpuCollectives::Device device0(executor0);
  GpuCollectives::Device device1(executor1);

  GpuCollectives* collectives = GpuCollectives::Default("GPU");

  TF_ASSIGN_OR_RETURN(CliqueId clique_id, collectives->CreateUniqueCliqueId());
  CliqueIds clique_ids(clique_id);

  GpuCliqueKey clique_key({GlobalDeviceId(0), GlobalDeviceId(1)},
                          /*num_local_participants=*/2);

  Collectives::DeviceRank rank0(&device0, RankId(0));
  Collectives::DeviceRank rank1(&device1, RankId(1));

  GpuCollectives::Config config;
  config.blocking_communicators = blocking;
  config.async_execution = !blocking;

  TF_ASSIGN_OR_RETURN(auto comms,
                      collectives->CreateCommunicators(clique_key, clique_ids,
                                                       {rank0, rank1}, config));
  CHECK_EQ(comms.size(), 2);

  std::vector<std::unique_ptr<GpuCommunicator>> gpu_comms;
  gpu_comms.emplace_back(dynamic_cast<GpuCommunicator*>(comms[0].release()));
  gpu_comms.emplace_back(dynamic_cast<GpuCommunicator*>(comms[1].release()));
  return gpu_comms;
}

TEST(GpuCollectivesTest, CreateSymmetricMemory) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor0,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor1,
                       platform->ExecutorForDevice(1));

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executor0, executor1));

  EXPECT_TRUE(comms[0]->platform_comm().handle);
  EXPECT_TRUE(comms[1]->platform_comm().handle);

  if (!comms[0]->SupportsDeviceComm() || !comms[1]->SupportsDeviceComm()) {
    GTEST_SKIP() << "GPU communicators do not suppoort symmetric memory";
  }

  // Create memory allocators that allocate physical memory in the collective
  // memory space, which makes them compatible with symmetric memory
  // requirements.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<se::MemoryAllocator> allocator0,
      executor0->CreateMemoryAllocator(se::MemorySpace::kCollective));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<se::MemoryAllocator> allocator1,
      executor1->CreateMemoryAllocator(se::MemorySpace::kCollective));

  // Allocate device memory on each participating rank.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::MemoryAllocation> alloc0,
                       allocator0->Allocate(1024));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::MemoryAllocation> alloc1,
                       allocator1->Allocate(1024));

  // Because creating symmetric memory is a collective operation, we must call
  // it from a thead pool to avoid deadlocks.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 2);
  tsl::Executor& exec = *pool.AsExecutor();

  // Register allocated buffers as symmetric memory.
  auto fsymm0 = MakeFutureOn(
      exec, [&] { return comms[0]->CreateSymmetricMemory(alloc0->address()); });
  auto fsymm1 = MakeFutureOn(
      exec, [&] { return comms[1]->CreateSymmetricMemory(alloc1->address()); });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<SymmetricMemory> symm0,
                       std::move(fsymm0).Await());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<SymmetricMemory> symm1,
                       std::move(fsymm1).Await());
}

TEST(GpuCollectivesTest, CreateDeviceComm) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor0,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor1,
                       platform->ExecutorForDevice(1));

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executor0, executor1));

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

  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor0,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor1,
                       platform->ExecutorForDevice(1));

  ASSERT_OK_AND_ASSIGN(auto comms,
                       CreateCommunicators(executor0, executor1, blocking));

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
