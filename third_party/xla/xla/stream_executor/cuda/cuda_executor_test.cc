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

#include "xla/stream_executor/cuda/cuda_executor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
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
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::testing::_;
using ::testing::AnyOf;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

TEST(CudaExecutorTest, CreateDeviceDescription) {
  CudaPlatform platform;
  ASSERT_GT(platform.VisibleDeviceCount(), 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeviceDescription> result,
                          CudaExecutor::CreateDeviceDescription(0));

  EXPECT_TRUE(result->runtime_version().IsValid());
  EXPECT_TRUE(result->driver_version().IsValid());
  EXPECT_TRUE(result->compile_time_toolkit_version().IsValid());
  EXPECT_TRUE(result->cub_version().IsValid());

  EXPECT_GT(result->kernel_mode_driver_version().major(), 300);  // NOLINT

  EXPECT_GT(result->pcie_bandwidth(), 1024 * 1024);
  EXPECT_THAT(result->platform_version(), Not(IsEmpty()));
  EXPECT_THAT(result->name(), Not(IsEmpty()));
  EXPECT_THAT(result->model_str(), Not(IsEmpty()));
  EXPECT_THAT(result->device_vendor(), "NVIDIA Corporation");

  EXPECT_THAT(*result->gpu_compute_capability().cuda_compute_capability(),
              ::testing::Field("major", &CudaComputeCapability::major, Ge(1)));

  DeviceInterconnectInfo info = result->device_interconnect_info();
  // nvmlDeviceGetGpuFabricInfoV is only available in driver r545+
  if (result->cuda_compute_capability().IsAtLeastHopper() &&
      result->kernel_mode_driver_version().major() >= 545 &&
      info.active_links) {
    EXPECT_GE(info.active_links, 18);

    EXPECT_THAT(info.clique_id, Not(IsEmpty()));
    EXPECT_THAT(info.cluster_uuid, Not(IsEmpty()));
  }

  EXPECT_THAT(DeviceDescription::FromProto(result->ToProto()),
              IsOkAndHolds(*result));
}

TEST(CudaExecutorTest, GetCudaKernel) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  auto cuda_executor = dynamic_cast<CudaExecutor*>(executor);
  ASSERT_NE(cuda_executor, nullptr);

  auto verify_kernel = [&](const KernelLoaderSpec& spec) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                            executor->LoadKernel(spec));
    EXPECT_THAT(cuda_executor->GetCudaKernel(kernel.get()),
                absl_testing::IsOkAndHolds(kernel.get()));

    cuda_executor->UnloadKernel(kernel.get());
    EXPECT_THAT(cuda_executor->GetCudaKernel(kernel.get()),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));

    EXPECT_THAT(cuda_executor->GetCudaKernel(nullptr),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));
  };

  TF_ASSERT_OK_AND_ASSIGN(KernelLoaderSpec add,
                          GetAddI32TestKernelSpec(cuda::kCudaPlatformId));
  verify_kernel(add);
  verify_kernel(GetAddI32PtxKernelSpec());
}

TEST(CudaExecutorTest, CreateUnifiedMemoryAllocatorWorks) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MemoryAllocator> allocator,
      executor->CreateMemoryAllocator(MemorySpace::kUnified));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                          allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
}

TEST(CudaExecutorTest, CreateHostMemoryAllocatorWorks) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocator> allocator,
                          executor->CreateMemoryAllocator(MemorySpace::kHost));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                          allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
}

TEST(CudaExecutorTest, CreateCollectiveMemoryAllocatorWorks) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MemoryAllocator> allocator,
      executor->CreateMemoryAllocator(MemorySpace::kCollective));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                          allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
}

// TODO: b/420735471 - Enable test once fixed.
TEST(CudaExecutorTest,
     DISABLED_CreateCollectiveMemoryAllocatorFailsForExcessiveSize) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MemoryAllocator> allocator,
      executor->CreateMemoryAllocator(MemorySpace::kCollective));
  constexpr uint64_t kTooBig = 1125899906842624;  // 1 PiB
  EXPECT_THAT(
      allocator->Allocate(kTooBig),
      absl_testing::StatusIs(
          _, AnyOf(HasSubstr("failed to allocate 1.00PiB (1125899906842624 "
                             "bytes) from device collective memory:"),
                   HasSubstr("out of memory"))));
}

TEST(CudaExecutorTest, CreateUnsupportedMemoryAllocatorsFail) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  EXPECT_THAT(executor->CreateMemoryAllocator(MemorySpace::kDevice),
              Not(absl_testing::IsOk()));
}

TEST(CudaExecutorTest, GetPointerMemorySpaceWorksWithUnifiedMemory) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  TF_ASSERT_OK_AND_ASSIGN(
      auto unified_memory_allocator,
      executor->CreateMemoryAllocator(MemorySpace::kUnified));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                          unified_memory_allocator->Allocate(256));
  EXPECT_THAT(executor->GetPointerMemorySpace(allocation->address().opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kUnified));
}

TEST(CudaExecutorTest, GetPointerMemorySpaceWorksWithHostMemory) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                          executor->HostMemoryAllocate(256));
  EXPECT_THAT(executor->GetPointerMemorySpace(allocation->address().opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kHost));
}

TEST(CudaExecutorTest, GetPointerMemorySpaceWorksWithDeviceAddress) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  DeviceAddressBase allocation = executor->Allocate(256);
  EXPECT_NE(allocation.opaque(), nullptr);
  EXPECT_THAT(executor->GetPointerMemorySpace(allocation.opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kDevice));
}

TEST(CudaExecutorTest, AllocateMemoryWithVmmApi) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  auto cuda_executor = dynamic_cast<CudaExecutor*>(executor);
  ASSERT_NE(cuda_executor, nullptr);
  DeviceAddressBase ptr =
      cuda_executor->Allocate(1024, static_cast<int>(MemorySpace::kP2P));

  EXPECT_NE(ptr.opaque(), nullptr);
  TF_ASSERT_OK_AND_ASSIGN(size_t granularity,
                          cuda_executor->GetVmmGranularity());
  EXPECT_EQ(ptr.size(), granularity);
  EXPECT_THAT(executor->GetPointerMemorySpace(ptr.opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kDevice));

  TF_ASSERT_OK_AND_ASSIGN(CudaExecutor::VmmMemoryHandle handle,
                          cuda_executor->RetainVmmMemoryHandle(ptr.opaque()));
  EXPECT_NE(handle.handle(), 0);
}

TEST(CudaExecutorTest, MultipleExecutorsForSameDevice) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  ASSERT_GT(platform->VisibleDeviceCount(), 0);

  // Create multiple executors for device 0, bypassing the platform cache.
  // This simulates the scenario where multiple PjRt clients are created
  // without preallocation for the same physical GPU.
  constexpr int kNumExecutors = 3;
  std::vector<std::unique_ptr<CudaExecutor>> executors;
  for (int i = 0; i < kNumExecutors; ++i) {
    auto executor =
        std::make_unique<CudaExecutor>(platform, /*device_ordinal=*/0);
    ASSERT_THAT(executor->Init(), absl_testing::IsOk());
    executors.push_back(std::move(executor));
  }

  // Verify all executors can create device descriptions and allocate memory.
  for (int i = 0; i < kNumExecutors; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto desc, executors[i]->CreateDeviceDescription());
    EXPECT_THAT(desc->name(), Not(IsEmpty()));

    DeviceAddressBase ptr = executors[i]->Allocate(1024, /*memory_space=*/0);
    EXPECT_NE(ptr.opaque(), nullptr);
    executors[i]->Deallocate(&ptr);

    // Allocate VMM memory (collective memory space uses the VMM allocator).
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<MemoryAllocator> vmm_allocator,
        executors[i]->CreateMemoryAllocator(MemorySpace::kCollective));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> vmm_allocation,
                            vmm_allocator->Allocate(4096));
    EXPECT_NE(vmm_allocation->address().opaque(), nullptr);
  }
}

TEST(CudaExecutorTest,
     RetainVmmMemoryHandleForTheMemoryAllocatedWithoutVmmApi) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  auto cuda_executor = dynamic_cast<CudaExecutor*>(executor);
  ASSERT_NE(cuda_executor, nullptr);
  DeviceAddressBase ptr =
      cuda_executor->Allocate(1024, static_cast<int>(MemorySpace::kDevice));

  EXPECT_NE(ptr.opaque(), nullptr);
  EXPECT_EQ(ptr.size(), 1024);

  EXPECT_THAT(cuda_executor->RetainVmmMemoryHandle(ptr.opaque()),
              absl_testing::StatusIs(absl::StatusCode::kInternal));
}
}  // namespace
}  // namespace stream_executor::gpu
