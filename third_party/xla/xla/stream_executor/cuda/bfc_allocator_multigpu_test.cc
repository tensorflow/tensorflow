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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/bfc_allocator.h"
#include "xla/tsl/framework/device_id.h"
#include "xla/tsl/platform/env.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"

namespace stream_executor::gpu {
namespace {

// Registration is collective, whereas extension allocation is rank-local.
class CudaBfcSymmetricGrowthTest : public ::testing::Test {
 protected:
  template <typename F>
  void OnBothRanks(F function) {
    std::unique_ptr<tsl::Thread> first(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(), "rank_0", [&] { function(0); }));
    std::unique_ptr<tsl::Thread> second(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(), "rank_1", [&] { function(1); }));
  }

  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        auto* platform, PlatformManager::PlatformWithId(cuda::kCudaPlatformId));
    if (platform->VisibleDeviceCount() < 2) {
      GTEST_SKIP() << "Requires two CUDA devices";
    }
    for (int rank = 0; rank < 2; ++rank) {
      ASSERT_OK_AND_ASSIGN(executors_[rank], platform->ExecutorForDevice(rank));
    }
    if (!executors_[0]->CanEnablePeerAccessTo(executors_[1]) ||
        !executors_[1]->CanEnablePeerAccessTo(executors_[0])) {
      GTEST_SKIP() << "Symmetric-window test requires CUDA peer access";
    }
    for (int rank = 0; rank < 2; ++rank) {
      ASSERT_OK_AND_ASSIGN(uint64_t granularity,
                           executors_[rank]->GetCollectiveMemoryGranularity());
      if (rank == 0) {
        page_ = granularity;
      }
      ASSERT_EQ(granularity, page_);
      auto sub = std::make_unique<DeviceMemAllocator>(
          executors_[rank], tsl::PlatformDeviceId(rank));
      ASSERT_OK(sub->ReserveMemory(16 * page_));
      tsl::BFCAllocator::Options opts;
      opts.allow_growth = true;
      opts.allow_retry_on_failure = false;
      opts.enable_spatial_partitioning = true;
      opts.initial_region_bytes = 4 * page_;
      opts.lower_end_policy = {tsl::BFCAllocator::HoleOrder::kAscendingAddress,
                               tsl::BFCAllocator::SplitPolicy::kExact,
                               tsl::BFCAllocator::SplitPolicy::kExact};
      opts.upper_end_policy = {tsl::BFCAllocator::HoleOrder::kDescendingAddress,
                               tsl::BFCAllocator::SplitPolicy::kRetainPadding,
                               tsl::BFCAllocator::SplitPolicy::kRetainPadding};
      allocators_[rank] = std::make_unique<tsl::BFCAllocator>(
          std::move(sub), 16 * page_, "symmetric_growth", opts);
      buffers_[rank][0] = allocators_[rank]->AllocateRaw(page_, page_, lower_);
      ASSERT_NE(buffers_[rank][0], nullptr);
      ordinary_[rank] = allocators_[rank]->AllocateRaw(256, page_, upper_);
      ASSERT_NE(ordinary_[rank], nullptr);
      auto activation = executors_[rank]->Activate();
      ASSERT_EQ(
          cuMemsetD32(reinterpret_cast<CUdeviceptr>(ordinary_[rank]), 1234, 1),
          CUDA_SUCCESS);
      ASSERT_EQ(cuStreamCreate(&streams_[rank], CU_STREAM_NON_BLOCKING),
                CUDA_SUCCESS);
    }
    const int devices[] = {0, 1};
    ASSERT_EQ(ncclCommInitAll(comms_.data(), 2, devices), ncclSuccess);
  }

  void TearDown() override {
    OnBothRanks([&](int rank) {
      if (!executors_[rank]) {
        return;
      }
      auto activation = executors_[rank]->Activate();
      for (auto window : windows_[rank]) {
        if (window) {
          EXPECT_EQ(ncclCommWindowDeregister(comms_[rank], window),
                    ncclSuccess);
        }
      }
      if (comms_[rank]) {
        EXPECT_EQ(ncclCommDestroy(comms_[rank]), ncclSuccess);
      }
      if (streams_[rank]) {
        EXPECT_EQ(cuStreamDestroy(streams_[rank]), CUDA_SUCCESS);
      }
      allocators_[rank].reset();
    });
  }

  void Register(int index) {
    OnBothRanks([&](int rank) {
      auto activation = executors_[rank]->Activate();
      EXPECT_EQ(ncclCommWindowRegister(comms_[rank], buffers_[rank][index],
                                       page_, &windows_[rank][index],
                                       NCCL_WIN_COLL_SYMMETRIC),
                ncclSuccess);
    });
    for (int rank = 0; rank < 2; ++rank) {
      ASSERT_NE(windows_[rank][index], nullptr);
    }
  }

  void AllReduce(int index) {
    for (int rank = 0; rank < 2; ++rank) {
      auto activation = executors_[rank]->Activate();
      const float input = rank + 1;
      ASSERT_EQ(
          cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(buffers_[rank][index]),
                       &input, sizeof(input)),
          CUDA_SUCCESS);
    }
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for (int rank = 0; rank < 2; ++rank) {
      auto activation = executors_[rank]->Activate();
      EXPECT_EQ(ncclAllReduce(buffers_[rank][index], buffers_[rank][index], 1,
                              ncclFloat, ncclSum, comms_[rank], streams_[rank]),
                ncclSuccess);
    }
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
    for (int rank = 0; rank < 2; ++rank) {
      auto activation = executors_[rank]->Activate();
      ASSERT_EQ(cuStreamSynchronize(streams_[rank]), CUDA_SUCCESS);
      float output;
      ASSERT_EQ(
          cuMemcpyDtoH(&output,
                       reinterpret_cast<CUdeviceptr>(buffers_[rank][index]),
                       sizeof(output)),
          CUDA_SUCCESS);
      EXPECT_EQ(output, 3.0f);
    }
  }

  size_t page_ = 0;
  std::array<StreamExecutor*, 2> executors_ = {};
  std::array<std::unique_ptr<tsl::BFCAllocator>, 2> allocators_;
  std::array<ncclComm_t, 2> comms_ = {};
  std::array<CUstream, 2> streams_ = {};
  std::array<void*, 2> ordinary_ = {};
  std::array<std::array<void*, 2>, 2> buffers_ = {};
  std::array<std::array<ncclWindow_t, 2>, 2> windows_ = {};
  const tsl::AllocationAttributes lower_{false, false, nullptr,
                                         tsl::AllocationEnd::kLower};
  const tsl::AllocationAttributes upper_{false, false, nullptr,
                                         tsl::AllocationEnd::kUpper};
};

TEST_F(CudaBfcSymmetricGrowthTest, WindowsSurviveAsymmetricGrowth) {
  ASSERT_NO_FATAL_FAILURE(Register(0));
  ASSERT_NO_FATAL_FAILURE(AllReduce(0));
  // Only rank zero grows, while its first symmetric window remains registered.
  void* extension = allocators_[0]->AllocateRaw(256, 8 * page_, upper_);
  ASSERT_NE(extension, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(extension),
            reinterpret_cast<uintptr_t>(buffers_[0][0]) + 4 * page_);
  // Pointer range queries describe individual physical mappings, not the VA
  // reservation. A single fabric handle cannot export a buffer spanning them.
  auto* gpu_executor = static_cast<GpuExecutor*>(executors_[0]);
  ASSERT_OK_AND_ASSIGN(auto initial_range,
                       gpu_executor->GetAllocationRange(buffers_[0][0]));
  ASSERT_OK_AND_ASSIGN(auto extension_range,
                       gpu_executor->GetAllocationRange(extension));
  EXPECT_EQ(initial_range.opaque(), buffers_[0][0]);
  EXPECT_EQ(initial_range.size(), 4 * page_);
  EXPECT_EQ(extension_range.opaque(), extension);
  EXPECT_EQ(extension_range.size(), 8 * page_);
  EXPECT_GT(allocators_[0]->GetStats()->pool_bytes.value(), 4 * page_);
  EXPECT_EQ(allocators_[1]->GetStats()->pool_bytes.value(), 4 * page_);
  ASSERT_NO_FATAL_FAILURE(AllReduce(0));
  {
    auto activation = executors_[0]->Activate();
    uint32_t value;
    ASSERT_EQ(cuMemcpyDtoH(&value, reinterpret_cast<CUdeviceptr>(ordinary_[0]),
                           sizeof(value)),
              CUDA_SUCCESS);
    EXPECT_EQ(value, 1234);
    ASSERT_EQ(cuMemsetD32(reinterpret_cast<CUdeviceptr>(extension), 5678, 1),
              CUDA_SUCCESS);
    ASSERT_EQ(cuMemcpyDtoH(&value, reinterpret_cast<CUdeviceptr>(extension),
                           sizeof(value)),
              CUDA_SUCCESS);
    EXPECT_EQ(value, 5678);
  }
  for (int rank = 0; rank < 2; ++rank) {
    buffers_[rank][1] = allocators_[rank]->AllocateRaw(page_, page_, lower_);
    ASSERT_NE(buffers_[rank][1], nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffers_[rank][1]) -
                  reinterpret_cast<uintptr_t>(buffers_[rank][0]),
              page_);
  }
  ASSERT_NO_FATAL_FAILURE(Register(1));
  ASSERT_NO_FATAL_FAILURE(AllReduce(1));
  allocators_[0]->DeallocateRaw(extension);
  // The released extension remains unavailable to collective allocations.
  EXPECT_EQ(allocators_[0]->AllocateRaw(page_, 2 * page_, lower_), nullptr);

  // Free the old upper buffer, then reuse a span crossing the initial mapping
  // boundary without allocating any more physical memory. Both S(1) windows
  // stay registered throughout this coalescing and reuse.
  allocators_[0]->DeallocateRaw(ordinary_[0]);
  ordinary_[0] = nullptr;
  const int64_t mapped_bytes = allocators_[0]->GetStats()->pool_bytes.value();
  void* spanning = allocators_[0]->AllocateRaw(256, 9 * page_, upper_);
  ASSERT_NE(spanning, nullptr);
  const uintptr_t old_end =
      reinterpret_cast<uintptr_t>(buffers_[0][0]) + 4 * page_;
  ASSERT_LT(reinterpret_cast<uintptr_t>(spanning), old_end);
  ASSERT_GT(reinterpret_cast<uintptr_t>(spanning) + 9 * page_, old_end);
  EXPECT_EQ(allocators_[0]->GetStats()->pool_bytes.value(), mapped_bytes);
  {
    auto activation = executors_[0]->Activate();
    ASSERT_EQ(cuMemsetD32(old_end - sizeof(uint32_t), 9012, 2), CUDA_SUCCESS);
    std::array<uint32_t, 2> values;
    ASSERT_EQ(
        cuMemcpyDtoH(values.data(), old_end - sizeof(uint32_t), sizeof(values)),
        CUDA_SUCCESS);
    EXPECT_EQ(values[0], 9012);
    EXPECT_EQ(values[1], 9012);
  }
  ASSERT_NO_FATAL_FAILURE(AllReduce(0));
  ASSERT_NO_FATAL_FAILURE(AllReduce(1));
  allocators_[0]->DeallocateRaw(spanning);
}

}  // namespace
}  // namespace stream_executor::gpu
