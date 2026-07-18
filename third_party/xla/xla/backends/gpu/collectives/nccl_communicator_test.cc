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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/barrier.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/collectives_test_util.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
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
      ASSIGN_OR_RETURN(auto sym_mem, comms[rank]->CreateSymmetricMemory(
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
      RETURN_IF_ERROR(comms[rank]
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

}  // namespace
}  // namespace xla::gpu
