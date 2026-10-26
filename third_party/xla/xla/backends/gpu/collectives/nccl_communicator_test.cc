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
#include <functional>
#include <memory>
#include <thread>  // NOLINT
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/synchronization/barrier.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/collectives_test_util.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
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
#include "xla/tsl/util/tied_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Not;

static constexpr GlobalDeviceId kD0(0);
static constexpr GlobalDeviceId kD1(1);

// Test checks that there are no deadlocks when NCCL API is used concurrently
// from multiple threads.
// Test launches two parallel threads which should use the same NCCL
// communicator asynchronously. The first thread should perform memory
// registration and the second thread should perform AllReduce.
// Memory registration operations should wait for the AllReduce to complete
// using a barrier.
TEST(NcclCommunicatorTest, AsyncApiCalls) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  constexpr int kNumDevices = 2;
  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, kNumDevices));

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }

  if (!executors[0]
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1},
                                                       /*blocking=*/true));

  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(auto sym_allocs, Allocate(allocators, 1024));
  ASSERT_OK_AND_ASSIGN(auto send_allocs, Allocate(allocators, 1024));
  ASSERT_OK_AND_ASSIGN(auto recv_allocs, Allocate(allocators, 1024));

  ASSERT_OK_AND_ASSIGN(auto stream0, executors[0]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream1, executors[1]->CreateStream());

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl_test", kNumThreads);

  constexpr int kNumIterations = 1000;
  std::vector<std::unique_ptr<absl::Barrier>> registration_barriers;
  registration_barriers.reserve(kNumIterations);
  std::vector<std::unique_ptr<absl::Barrier>> all_reduce_barriers;
  all_reduce_barriers.reserve(kNumIterations);
  for (int i = 0; i < kNumIterations; ++i) {
    registration_barriers.push_back(
        std::make_unique<absl::Barrier>(kNumThreads));
    all_reduce_barriers.push_back(std::make_unique<absl::Barrier>(kNumThreads));
  }

  // Register memory, synchronize with barrier, unregister memory
  // asynchronously.
  std::function<absl::Status(int)> memory_registration_fn =
      [&](int rank) -> absl::Status {
    for (int i = 0; i < kNumIterations; ++i) {
      if (i > 0) {
        all_reduce_barriers[i - 1]->Block();
      }
      ABSL_ASSIGN_OR_RETURN(auto sym_mem, comms[rank]->CreateSymmetricMemory(
                                         sym_allocs[rank]->address()));
      registration_barriers[i]->Block();
    }
    all_reduce_barriers[kNumIterations - 1]->Block();
    return absl::OkStatus();
  };

  // Perform AllReduce.
  std::function<absl::Status(int)> all_reduce_fn =
      [&](int rank) -> absl::Status {
    se::Stream* stream = rank == 0 ? stream0.get() : stream1.get();
    GpuCollectives::Executor gpu_exec(stream);
    for (int i = 0; i < kNumIterations; ++i) {
      registration_barriers[i]->Block();
      ABSL_RETURN_IF_ERROR(comms[rank]
                          ->AllReduce(send_allocs[rank]->address(),
                                      recv_allocs[rank]->address(), F32, 256,
                                      ReductionKind::SUM, gpu_exec)
                          .Await());
      all_reduce_barriers[i]->Block();
    }
    return absl::OkStatus();
  };

  tsl::Executor& exec = *pool.AsExecutor();
  std::vector<Future<>> futures;
  futures.reserve(kNumThreads);
  futures.push_back(
      MakeFutureOn(exec, [&]() { return memory_registration_fn(0); }));
  futures.push_back(
      MakeFutureOn(exec, [&]() { return memory_registration_fn(1); }));
  futures.push_back(MakeFutureOn(exec, [&]() { return all_reduce_fn(0); }));
  futures.push_back(MakeFutureOn(exec, [&]() { return all_reduce_fn(1); }));

  for (auto& f : futures) {
    EXPECT_OK(f.Await());
  }
}

TEST(NcclCommunicatorTest, CliqueSemaphore) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  constexpr int kNumDevices = 2;
  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, kNumDevices));

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }

  if (!executors[0]
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1},
                                                       /*blocking=*/true));

  // 1. Initially, clique semaphore should not be initialized on either
  // communicator.
  EXPECT_FALSE(comms[0]->IsCliqueSemaphoreInitialized());
  EXPECT_FALSE(comms[1]->IsCliqueSemaphoreInitialized());

  // Querying before initialization should return NotFoundError.
  EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[1]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[1]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));

  // 2. Allocate collective memory for semaphores on both devices.
  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(
      auto sem_allocs,
      Allocate(allocators, sizeof(uint32_t) *
                               GpuCommunicator::kMaxCollectiveIds *
                               GpuCommunicator::kMaxCliqueSemaphores));

  ASSERT_OK_AND_ASSIGN(auto stream0, executors[0]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream1, executors[1]->CreateStream());
  se::DeviceAddressBase addr0 = sem_allocs[0]->address();
  se::DeviceAddressBase addr1 = sem_allocs[1]->address();
  ASSERT_OK(stream0->MemZero(&addr0, addr0.size()));
  ASSERT_OK(stream1->MemZero(&addr1, addr1.size()));
  ASSERT_OK(stream0->BlockHostUntilDone());
  ASSERT_OK(stream1->BlockHostUntilDone());

  // 3. Create symmetric memory concurrently across both communicators.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl_test", kNumDevices);
  tsl::Executor& exec = *pool.AsExecutor();

  auto symm_futures = CreateSymmetricMemory(exec, comms, sem_allocs);
  ASSERT_OK_AND_ASSIGN(auto sym_mems,
                       AwaitSymmetricMemory(std::move(symm_futures)));

  // 4. Tie and initialize clique semaphores in a scoped TiedAny container.
  {
    tsl::TiedAny tied;
    auto tied_sem0 = tied.Tie(std::move(sem_allocs[0]));
    auto tied_sym0 = tied.Tie(std::move(sym_mems[0]));
    comms[0]->InitializeCliqueSemaphore(std::move(tied_sem0),
                                        std::move(tied_sym0));

    auto tied_sem1 = tied.Tie(std::move(sem_allocs[1]));
    auto tied_sym1 = tied.Tie(std::move(sym_mems[1]));
    comms[1]->InitializeCliqueSemaphore(std::move(tied_sem1),
                                        std::move(tied_sym1));

    // After initialization, both communicators report initialized.
    EXPECT_TRUE(comms[0]->IsCliqueSemaphoreInitialized());
    EXPECT_TRUE(comms[1]->IsCliqueSemaphoreInitialized());

    constexpr uint64_t kSemaphoreSliceBytes =
        GpuCommunicator::kMaxCliqueSemaphores * sizeof(uint32_t);

    // 5. Query valid peer ranks (both self and peer) and collective IDs.
    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem0_for_rank0_c0,
        comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0));
    EXPECT_NE(sem0_for_rank0_c0.opaque(), nullptr);
    EXPECT_EQ(sem0_for_rank0_c0.size(), kSemaphoreSliceBytes);

    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem0_for_rank0_c1,
        comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/1));
    EXPECT_EQ(
        static_cast<char*>(sem0_for_rank0_c1.opaque()),
        static_cast<char*>(sem0_for_rank0_c0.opaque()) + kSemaphoreSliceBytes);
    EXPECT_EQ(sem0_for_rank0_c1.size(), kSemaphoreSliceBytes);

    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem0_for_rank1,
        comms[0]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0));
    EXPECT_NE(sem0_for_rank1.opaque(), nullptr);
    EXPECT_EQ(sem0_for_rank1.size(), kSemaphoreSliceBytes);

    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem1_for_rank0,
        comms[1]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0));
    EXPECT_NE(sem1_for_rank0.opaque(), nullptr);
    EXPECT_EQ(sem1_for_rank0.size(), kSemaphoreSliceBytes);

    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem1_for_rank1,
        comms[1]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0));
    EXPECT_NE(sem1_for_rank1.opaque(), nullptr);
    EXPECT_EQ(sem1_for_rank1.size(), kSemaphoreSliceBytes);

    // 6. Query invalid rank out of bounds or out-of-range collective_id.
    EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(999), /*collective_id=*/0),
                Not(IsOk()));
    EXPECT_THAT(comms[1]->GetCliqueSemaphore(RankId(999), /*collective_id=*/0),
                Not(IsOk()));
    EXPECT_THAT(comms[0]->GetCliqueSemaphore(
                    RankId(0),
                    /*collective_id=*/GpuCommunicator::kMaxCollectiveIds),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/-1),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 7. After TiedAny is destroyed, the tied references expire and
  // GetCliqueSemaphore should return NotFoundError.
  EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[1]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[1]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));

  // 8. Verify bounds check when the symmetric memory buffer is smaller than
  // required for higher collective IDs (including partial slice overflow).
  constexpr uint64_t kSemaphoreSliceBytes =
      GpuCommunicator::kMaxCliqueSemaphores * sizeof(uint32_t);
  constexpr uint64_t kSmallBufferSize = kSemaphoreSliceBytes + sizeof(uint32_t);
  ASSERT_OK_AND_ASSIGN(auto small_sem_allocs,
                       Allocate(allocators, kSmallBufferSize));
  std::vector<Future<std::unique_ptr<SymmetricMemory>>> small_symm_futures;
  small_symm_futures.reserve(kNumDevices);
  for (int i = 0; i < kNumDevices; ++i) {
    se::DeviceAddressBase small_addr(small_sem_allocs[i]->address().opaque(),
                                     kSmallBufferSize);
    small_symm_futures.push_back(MakeFutureOn(exec, [&, i, small_addr] {
      return comms[i]->CreateSymmetricMemory(small_addr);
    }));
  }
  ASSERT_OK_AND_ASSIGN(auto small_sym_mems,
                       AwaitSymmetricMemory(std::move(small_symm_futures)));
  {
    tsl::TiedAny tied;
    comms[0]->InitializeCliqueSemaphore(
        tied.Tie(std::move(small_sem_allocs[0])),
        tied.Tie(std::move(small_sym_mems[0])));
    comms[1]->InitializeCliqueSemaphore(
        tied.Tie(std::move(small_sem_allocs[1])),
        tied.Tie(std::move(small_sym_mems[1])));

    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem0_c0,
        comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0));
    EXPECT_EQ(sem0_c0.size(), kSemaphoreSliceBytes);

    // collective_id=1 starts within the allocation (offset < size), but the
    // full slice exceeds the allocation size.
    EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/1),
                StatusIs(absl::StatusCode::kInternal));
    // collective_id=2 starts past the end of the allocation.
    EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/2),
                StatusIs(absl::StatusCode::kInternal));
  }
}

class NcclCommunicatorDestructorTest : public ::testing::TestWithParam<bool> {};

TEST_P(NcclCommunicatorDestructorTest, DestroyWithVlogEnabled) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 1));
  ASSERT_OK_AND_ASSIGN(auto comms,
                       CreateCommunicators(executors, {kD0}, GetParam()));
  ASSERT_EQ(comms.size(), 1);

  const int old_vlog_level =
      absl::SetVLogLevel("nccl_communicator", /*log_level=*/1);

  auto destroyed = std::make_shared<absl::Notification>();
  std::shared_ptr<GpuCommunicator> comm = std::move(comms[0]);
  std::thread destroyer([destroyed, comm = std::move(comm)]() mutable {
    comm.reset();
    destroyed->Notify();
  });

  const bool finished =
      destroyed->WaitForNotificationWithTimeout(absl::Seconds(60));
  absl::SetVLogLevel("nccl_communicator", old_vlog_level);
  if (finished) {
    destroyer.join();
  } else {
    destroyer.detach();
  }
  EXPECT_TRUE(finished) << "NcclCommunicator destructor deadlocked";
}

INSTANTIATE_TEST_SUITE_P(Blocking, NcclCommunicatorDestructorTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Blocking" : "Async";
                         });

}  // namespace
}  // namespace xla::gpu
