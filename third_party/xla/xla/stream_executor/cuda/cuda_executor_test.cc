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

#include <cstdint>
#include <memory>

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
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
using ::testing::_;
using ::testing::AnyOf;
using testing::Ge;
using ::testing::HasSubstr;
using testing::IsEmpty;
using testing::Not;
using testing::VariantWith;

TEST(CudaExecutorTest, CreateDeviceDescription) {
  CudaPlatform platform;
  ASSERT_GT(platform.VisibleDeviceCount(), 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeviceDescription> result,
                          CudaExecutor::CreateDeviceDescription(0));

  constexpr SemanticVersion kNullVersion{0, 0, 0};
  EXPECT_NE(result->runtime_version(), kNullVersion);
  EXPECT_NE(result->driver_version(), kNullVersion);
  EXPECT_NE(result->compile_time_toolkit_version(), kNullVersion);

  EXPECT_GT(result->pcie_bandwidth(), 1024 * 1024);
  EXPECT_THAT(result->platform_version(), Not(IsEmpty()));
  EXPECT_THAT(result->name(), Not(IsEmpty()));
  EXPECT_THAT(result->model_str(), Not(IsEmpty()));
  EXPECT_THAT(result->device_vendor(), "NVIDIA Corporation");

  EXPECT_THAT(*result->gpu_compute_capability().cuda_compute_capability(),
              ::testing::Field("major", &CudaComputeCapability::major, Ge(1)));

  DeviceInterconnectInfo info = result->device_interconnect_info();
  if (result->cuda_compute_capability().IsAtLeastBlackwell() &&
      info.active_links) {
    EXPECT_GE(info.active_links, 18);

    EXPECT_THAT(info.clique_id, Not(IsEmpty()));
    EXPECT_THAT(info.cluster_uuid, Not(IsEmpty()));
  }
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
  EXPECT_EQ(ptr.size(), 1024);
  EXPECT_THAT(executor->GetPointerMemorySpace(ptr.opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kDevice));

  TF_ASSERT_OK_AND_ASSIGN(CudaExecutor::VmmMemoryHandle handle,
                          cuda_executor->RetainVmmMemoryHandle(ptr.opaque()));
  EXPECT_NE(handle.handle(), 0);
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
