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

#include "xla/tsl/framework/bfc_allocator.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/device_id.h"

namespace stream_executor::gpu {
namespace {

TEST(CudaBfcGrowthTest, FirstRequestCanSpanInitialBackingAndExtensionAtCap) {
  ASSERT_OK_AND_ASSIGN(auto* platform,
                       PlatformManager::PlatformWithId(cuda::kCudaPlatformId));
  if (platform->VisibleDeviceCount() < 1) {
    GTEST_SKIP() << "Requires CUDA";
  }
  ASSERT_OK_AND_ASSIGN(auto* executor, platform->ExecutorForDevice(0));
  auto sub =
      std::make_unique<DeviceMemAllocator>(executor, tsl::PlatformDeviceId(0));
  ASSERT_OK_AND_ASSIGN(uint64_t page,
                       executor->GetCollectiveMemoryGranularity());
  ASSERT_OK(sub->ReserveMemory(8 * page));
  tsl::BFCAllocator::Options opts;
  opts.allow_growth = true;
  opts.allow_retry_on_failure = false;
  opts.enable_spatial_partitioning = true;
  opts.initial_region_bytes = 4 * page;
  tsl::BFCAllocator allocator(std::move(sub), 8 * page, "first_growth", opts);
  const tsl::AllocationAttributes upper{false, false, nullptr,
                                        tsl::AllocationEnd::kUpper};
  void* ptr = allocator.AllocateRaw(page, 8 * page, upper);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(allocator.GetStats()->pool_bytes.value(), 8 * page);
  auto activation = executor->Activate();
  const CUdeviceptr boundary = reinterpret_cast<CUdeviceptr>(ptr) + 4 * page;
  ASSERT_EQ(cuMemsetD32(boundary - sizeof(uint32_t), 42, 2), CUDA_SUCCESS);
  std::array<uint32_t, 2> values;
  ASSERT_EQ(
      cuMemcpyDtoH(values.data(), boundary - sizeof(uint32_t), sizeof(values)),
      CUDA_SUCCESS);
  EXPECT_EQ(values[0], 42);
  EXPECT_EQ(values[1], 42);
  EXPECT_EQ(allocator.AllocateRaw(page, page, upper), nullptr);
  allocator.DeallocateRaw(ptr);
}

}  // namespace
}  // namespace stream_executor::gpu
