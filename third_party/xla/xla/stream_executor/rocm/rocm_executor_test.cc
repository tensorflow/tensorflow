/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_executor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
using testing::IsEmpty;
using testing::Not;

TEST(RocmExecutorTest, CreateDeviceDescription) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeviceDescription> result,
                       RocmExecutor::CreateDeviceDescription(0));

  constexpr SemanticVersion kNullVersion{0, 0, 0};
  EXPECT_NE(result->runtime_version(), kNullVersion);
  EXPECT_NE(result->driver_version(), kNullVersion);
  EXPECT_NE(result->compile_time_toolkit_version(), kNullVersion);

  EXPECT_THAT(result->platform_version(), Not(IsEmpty()));
  EXPECT_THAT(result->name(), Not(IsEmpty()));
  EXPECT_THAT(result->model_str(), Not(IsEmpty()));
  EXPECT_THAT(result->device_vendor(), "Advanced Micro Devices, Inc");

  EXPECT_THAT(result->gpu_compute_capability()
                  .rocm_compute_capability()
                  ->gcn_arch_name(),
              Not(IsEmpty()));

  EXPECT_GT(result->pcie_bandwidth(), 1024 * 1024);
}

TEST(RocmExecutorTest, GetRocmKernel) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(KernelLoaderSpec add_kernel,
                       GetAddI32TestKernelSpec(rocm::kROCmPlatformId));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                       executor->LoadKernel(add_kernel));

  auto rocm_executor = dynamic_cast<RocmExecutor*>(executor);
  ASSERT_NE(rocm_executor, nullptr);
  EXPECT_THAT(rocm_executor->GetRocmKernel(kernel.get()),
              absl_testing::IsOkAndHolds(kernel.get()));

  rocm_executor->UnloadKernel(kernel.get());
  EXPECT_THAT(rocm_executor->GetRocmKernel(kernel.get()),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));

  EXPECT_THAT(rocm_executor->GetRocmKernel(nullptr),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(RocmExecutorTest, CreateUnifiedMemoryAllocatorWorks) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocator> allocator,
                       executor->CreateMemoryAllocator(MemorySpace::kUnified));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
  allocation.reset();
}

TEST(RocmExecutorTest, CreateHostMemoryAllocatorWorks) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocator> allocator,
                       executor->CreateMemoryAllocator(MemorySpace::kHost));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
  allocation.reset();
}

TEST(RocmExecutorTest, CreateCollectiveMemoryAllocatorWorks) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MemoryAllocator> allocator,
      executor->CreateMemoryAllocator(MemorySpace::kCollective));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
  allocation.reset();
}

TEST(RocmExecutorTest, CreateUnsupportedMemoryAllocatorsFail) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  EXPECT_THAT(executor->CreateMemoryAllocator(MemorySpace::kDevice),
              Not(absl_testing::IsOk()));
}

TEST(RocmExecutorTest, AllocateArrayReturnsRequestedSize) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  // AllocateArray exposes the logical array byte range, not any padding the
  // backend allocator may have added.
  DeviceAddress<uint64_t> buffer = executor->AllocateScalar<uint64_t>();
  ASSERT_NE(buffer.opaque(), nullptr);
  EXPECT_EQ(buffer.size(), sizeof(uint64_t));
  executor->Deallocate(&buffer);
}

TEST(RocmExecutorTest, GetPointerMemorySpaceWorksWithHostMemory) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       executor->HostMemoryAllocate(256));
  EXPECT_THAT(executor->GetPointerMemorySpace(allocation->address().opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kHost));
}

TEST(RocmExecutorTest, GetPointerMemorySpaceWorksWithDeviceAddress) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  DeviceAddressBase allocation = executor->Allocate(256);
  ASSERT_NE(allocation.opaque(), nullptr);
  EXPECT_THAT(executor->GetPointerMemorySpace(allocation.opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kDevice));
  executor->Deallocate(&allocation);
}

TEST(RocmExecutorTest, IsHostMemoryPinnedWorksWithHostMemory) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       executor->HostMemoryAllocate(256));
  EXPECT_TRUE(
      executor->IsHostMemoryPinned(allocation->address().opaque(), 256));
}

TEST(RocmExecutorTest, GetPointerMemorySpaceWorksWithUnifiedMemory) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  ASSERT_OK_AND_ASSIGN(auto unified_memory_allocator,
                       executor->CreateMemoryAllocator(MemorySpace::kUnified));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       unified_memory_allocator->Allocate(256));
  EXPECT_THAT(executor->GetPointerMemorySpace(allocation->address().opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kUnified));
}

TEST(RocmExecutorTest, IsHostMemoryPinnedWorksWithRegisteredHostMemory) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  constexpr size_t kSize = 64 * 1024;
  std::vector<uint8_t> buffer(kSize);
  // Pageable host memory isn't pinned until we register it.
  EXPECT_FALSE(executor->IsHostMemoryPinned(buffer.data(), buffer.size()));

  ASSERT_TRUE(executor->HostMemoryRegister(buffer.data(), buffer.size()));
  EXPECT_TRUE(executor->IsHostMemoryPinned(buffer.data(), buffer.size()));
  EXPECT_TRUE(executor->IsHostMemoryPinned(buffer.data() + 1024, 4096));
  EXPECT_FALSE(executor->IsHostMemoryPinned(buffer.data(), 0));
  EXPECT_FALSE(executor->IsHostMemoryPinned(buffer.data() + kSize / 2, kSize));

  ASSERT_TRUE(executor->HostMemoryUnregister(buffer.data()));
  EXPECT_FALSE(executor->IsHostMemoryPinned(buffer.data(), buffer.size()));
}

TEST(RocmExecutorTest, IsHostMemoryPinnedIsFalseForDeviceAddress) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  DeviceAddressBase allocation = executor->Allocate(256);
  ASSERT_NE(allocation.opaque(), nullptr);
  EXPECT_FALSE(executor->IsHostMemoryPinned(allocation.opaque(), 256));
  executor->Deallocate(&allocation);
}

TEST(RocmExecutorTest, MultipleExecutorsForSameDevice) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_GT(platform->VisibleDeviceCount(), 0);

  // Create multiple executors for device 0, bypassing the platform cache.
  // This simulates the scenario where multiple PjRt clients are created
  // without preallocation for the same physical GPU.
  constexpr int kNumExecutors = 3;
  std::vector<std::unique_ptr<RocmExecutor>> executors;
  for (int i = 0; i < kNumExecutors; ++i) {
    auto executor =
        std::make_unique<RocmExecutor>(platform, /*device_ordinal=*/0);
    ASSERT_THAT(executor->Init(), absl_testing::IsOk());
    executors.push_back(std::move(executor));
  }

  // Verify all executors can create device descriptions and allocate memory.
  for (int i = 0; i < kNumExecutors; ++i) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeviceDescription> desc,
                         executors[i]->CreateDeviceDescription());
    EXPECT_THAT(desc->name(), Not(IsEmpty()));

    DeviceAddressBase ptr = executors[i]->Allocate(1024, /*memory_space=*/0);
    EXPECT_NE(ptr.opaque(), nullptr);
    executors[i]->Deallocate(&ptr);

    // Allocate collective memory through the device allocator.
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<MemoryAllocator> collective_allocator,
        executors[i]->CreateMemoryAllocator(MemorySpace::kCollective));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<MemoryAllocation> collective_allocation,
        collective_allocator->Allocate(4096));
    EXPECT_NE(collective_allocation->address().opaque(), nullptr);
  }
}

}  // namespace
}  // namespace stream_executor::gpu
