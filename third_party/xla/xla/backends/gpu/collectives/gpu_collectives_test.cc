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

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/collectives/allocator_memory_registration.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/collectives_test_util.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/registered_memory.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr GlobalDeviceId kD0(0);
static constexpr GlobalDeviceId kD1(1);
static constexpr GlobalDeviceId kD2(2);
static constexpr GlobalDeviceId kD3(3);

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

TEST(GpuCollectivesTest, SplitCommunicators) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1},
                                                       /*blocking=*/true));

  EXPECT_TRUE(comms[0]->platform_comm().handle);
  EXPECT_TRUE(comms[1]->platform_comm().handle);

  ASSERT_OK_AND_ASSIGN(
      auto split_comms,
      SplitCommunicators(executors, comms, {kD0, kD1}, /*blocking=*/true));
  EXPECT_TRUE(split_comms[0]->platform_comm().handle);
  EXPECT_TRUE(split_comms[1]->platform_comm().handle);
}

TEST(GpuCollectivesTest, GroupLaunchMultipleCommunicators) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 4) {
    GTEST_SKIP() << "Test requires at least 4 GPUs";
  }

  GpuCollectives* collectives = GpuCollectives::Default("GPU");

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors_vec,
                       CreateExecutors(platform, 4));
  absl::Span<se::StreamExecutor*> executors(executors_vec);

  ASSERT_OK_AND_ASSIGN(auto comms01,
                       CreateCommunicators(executors.first(2), {kD0, kD1}));
  ASSERT_OK_AND_ASSIGN(auto comms23,
                       CreateCommunicators(executors.last(2), {kD2, kD3}));

  std::vector<std::unique_ptr<se::Stream>> streams;
  streams.reserve(executors.size());
  for (se::StreamExecutor* executor : executors) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                         executor->CreateStream());
    streams.push_back(std::move(stream));
  }

  constexpr size_t kCount = 4;
  constexpr size_t kNumBytes = kCount * sizeof(float);
  std::vector<std::array<float, kCount>> inputs = {
      std::array<float, kCount>{1.0f, 2.0f, 3.0f, 4.0f},
      std::array<float, kCount>{10.0f, 20.0f, 30.0f, 40.0f},
      std::array<float, kCount>{5.0f, 6.0f, 7.0f, 8.0f},
      std::array<float, kCount>{50.0f, 60.0f, 70.0f, 80.0f},
  };

  std::vector<se::DeviceAddress<float>> send_buffers;
  std::vector<se::DeviceAddress<float>> recv_buffers;
  send_buffers.reserve(executors.size());
  recv_buffers.reserve(executors.size());
  for (size_t i = 0; i < executors.size(); ++i) {
    send_buffers.push_back(executors[i]->AllocateArray<float>(kCount));
    recv_buffers.push_back(executors[i]->AllocateArray<float>(kCount));
    ASSERT_OK(
        streams[i]->Memcpy(&send_buffers[i], inputs[i].data(), kNumBytes));
    ASSERT_OK(streams[i]->MemZero(&recv_buffers[i], kNumBytes));
    ASSERT_OK(streams[i]->BlockHostUntilDone());
  }

  std::vector<const GpuCommunicator*> group_comms = {
      comms01[0].get(), comms01[1].get(), comms23[0].get(), comms23[1].get()};

  ASSERT_OK(collectives->GroupLaunch(group_comms, [&]() -> absl::Status {
    GpuCollectives::Executor executor0(streams[0].get());
    GpuCollectives::Executor executor1(streams[1].get());
    GpuCollectives::Executor executor2(streams[2].get());
    GpuCollectives::Executor executor3(streams[3].get());

    RETURN_IF_ERROR(comms01[0]->LaunchAllReduce(send_buffers[0],
                                                recv_buffers[0], F32, kCount,
                                                ReductionKind::SUM, executor0));
    RETURN_IF_ERROR(comms01[1]->LaunchAllReduce(send_buffers[1],
                                                recv_buffers[1], F32, kCount,
                                                ReductionKind::SUM, executor1));
    RETURN_IF_ERROR(comms23[0]->LaunchAllReduce(send_buffers[2],
                                                recv_buffers[2], F32, kCount,
                                                ReductionKind::SUM, executor2));
    return comms23[1]->LaunchAllReduce(send_buffers[3], recv_buffers[3], F32,
                                       kCount, ReductionKind::SUM, executor3);
  }));

  std::vector<std::array<float, kCount>> expected = {
      std::array<float, kCount>{11.0f, 22.0f, 33.0f, 44.0f},
      std::array<float, kCount>{11.0f, 22.0f, 33.0f, 44.0f},
      std::array<float, kCount>{55.0f, 66.0f, 77.0f, 88.0f},
      std::array<float, kCount>{55.0f, 66.0f, 77.0f, 88.0f},
  };

  for (size_t i = 0; i < executors.size(); ++i) {
    ASSERT_OK(streams[i]->BlockHostUntilDone());
    std::array<float, kCount> output;
    ASSERT_OK(streams[i]->Memcpy(output.data(), recv_buffers[i], kNumBytes));
    ASSERT_OK(streams[i]->BlockHostUntilDone());
    EXPECT_THAT(output, testing::ElementsAreArray(expected[i]));
  }

  for (size_t i = 0; i < executors.size(); ++i) {
    executors[i]->Deallocate(&send_buffers[i]);
    executors[i]->Deallocate(&recv_buffers[i]);
  }
}

TEST(GpuCollectivesTest, CreateSymmetricMemory) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1}));

  EXPECT_TRUE(comms[0]->platform_comm().handle);
  EXPECT_TRUE(comms[1]->platform_comm().handle);

  if (!absl::c_all_of(comms, [](auto& c) { return c->SupportsDeviceComm(); })) {
    GTEST_SKIP() << "GPU communicators do not support symmetric memory";
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

TEST(GpuCollectivesTest, CreateRegisteredMemory) {
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

  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(auto allocs, Allocate(allocators, 1024));

  // Unlike symmetric memory, buffer registration is not a collective operation:
  // each rank may register independently. The returned handle keeps the buffer
  // registered until it is destroyed.
  std::vector<std::unique_ptr<RegisteredMemory>> registered;
  for (size_t i = 0; i < comms.size(); ++i) {
    ASSERT_OK_AND_ASSIGN(
        auto reg, comms[i]->CreateRegisteredMemory(allocs[i]->address()));
    EXPECT_EQ(reg->addr(), allocs[i]->address());
    registered.push_back(std::move(reg));
  }
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

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1}));

  if (!comms[0]->SupportsDeviceComm() || !comms[1]->SupportsDeviceComm()) {
    GTEST_SKIP() << "GPU communicators do not support device-initiated comms";
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

TEST(GpuCollectivesTest, PutAndWaitSignal) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));
  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }
  if (!executors[0]
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1}));

  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));

  constexpr size_t kNumFloats = 4;
  constexpr size_t kNumBytes = kNumFloats * sizeof(float);

  ASSERT_OK_AND_ASSIGN(auto send_allocs, Allocate(allocators, kNumBytes));
  ASSERT_OK_AND_ASSIGN(auto recv_allocs, Allocate(allocators, kNumBytes));

  ASSERT_OK_AND_ASSIGN(auto stream0, executors[0]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream1, executors[1]->CreateStream());

  float h_send0[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float h_send1[] = {5.0f, 6.0f, 7.0f, 8.0f};

  se::DeviceAddressBase send0_addr = send_allocs[0]->address();
  se::DeviceAddressBase send1_addr = send_allocs[1]->address();
  se::DeviceAddressBase recv0_addr = recv_allocs[0]->address();
  se::DeviceAddressBase recv1_addr = recv_allocs[1]->address();

  ASSERT_OK(stream0->Memcpy(&send0_addr, h_send0, kNumBytes));
  ASSERT_OK(stream1->Memcpy(&send1_addr, h_send1, kNumBytes));
  ASSERT_OK(stream0->MemZero(&recv0_addr, kNumBytes));
  ASSERT_OK(stream1->MemZero(&recv1_addr, kNumBytes));
  ASSERT_OK(stream0->BlockHostUntilDone());
  ASSERT_OK(stream1->BlockHostUntilDone());

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 2);
  tsl::Executor& exec = *pool.AsExecutor();

  auto fsymm_send = CreateSymmetricMemory(exec, comms, send_allocs);
  ASSERT_OK_AND_ASSIGN(auto symm_send,
                       AwaitSymmetricMemory(std::move(fsymm_send)));

  auto fsymm_recv = CreateSymmetricMemory(exec, comms, recv_allocs);
  ASSERT_OK_AND_ASSIGN(auto symm_recv,
                       AwaitSymmetricMemory(std::move(fsymm_recv)));

  GpuSignalDesc signal_desc(0, 0);

  auto f0 = MakeFutureOn<void>(exec, [&]() -> absl::Status {
    GpuCollectives::Executor gpu_exec(stream0.get());
    RETURN_IF_ERROR(comms[0]
                        ->Put(send0_addr, symm_recv[0].get(), 0, kNumBytes,
                              RankId(1), gpu_exec)
                        .Await());
    return comms[0]->WaitSignal(RankId(1), 1, signal_desc, gpu_exec).Await();
  });

  auto f1 = MakeFutureOn<void>(exec, [&]() -> absl::Status {
    GpuCollectives::Executor gpu_exec(stream1.get());
    RETURN_IF_ERROR(comms[1]
                        ->Put(send1_addr, symm_recv[1].get(), 0, kNumBytes,
                              RankId(0), gpu_exec)
                        .Await());
    return comms[1]->WaitSignal(RankId(0), 1, signal_desc, gpu_exec).Await();
  });

  ASSERT_OK(f0.Await());
  ASSERT_OK(f1.Await());

  ASSERT_OK(stream0->BlockHostUntilDone());
  ASSERT_OK(stream1->BlockHostUntilDone());

  float h_recv0[kNumFloats];
  float h_recv1[kNumFloats];
  ASSERT_OK(stream0->Memcpy(h_recv0, recv0_addr, kNumBytes));
  ASSERT_OK(stream1->Memcpy(h_recv1, recv1_addr, kNumBytes));
  ASSERT_OK(stream0->BlockHostUntilDone());
  ASSERT_OK(stream1->BlockHostUntilDone());

  EXPECT_THAT(h_recv0, testing::ElementsAre(5.0f, 6.0f, 7.0f, 8.0f));
  EXPECT_THAT(h_recv1, testing::ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
}

// Verifies that AllocatorMemoryRegistration registers recorded allocator ranges
// with the clique communicator that runs on the same device.
TEST(GpuCollectivesTest, AllocatorMemoryRegistrationRegistersWithClique) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1}));

  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(auto allocs, Allocate(allocators, 1024));

  // Record one allocator range per device through the suballocator visitor, the
  // same way the BFC preallocation path would.
  auto registration = std::make_shared<AllocatorMemoryRegistration>();
  auto alloc_visitor = registration->alloc_visitor();
  for (size_t i = 0; i < executors.size(); ++i) {
    alloc_visitor(allocs[i]->address().opaque(), executors[i]->device_ordinal(),
                  allocs[i]->address().size());
  }

  // Build a clique from the real communicators and register memory with it.
  absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators;
  for (size_t i = 0; i < comms.size(); ++i) {
    communicators.emplace(RankId(i), std::move(comms[i]));
  }
  GpuClique clique(GpuCliqueKey({kD0, kD1}, /*num_local_participants=*/2),
                   std::nullopt, std::move(communicators),
                   /*peer_access_enabled=*/true,
                   std::make_shared<CancellationToken>());

  ASSERT_OK(registration->RegisterWithClique(clique));
}

}  // namespace
}  // namespace xla::gpu
