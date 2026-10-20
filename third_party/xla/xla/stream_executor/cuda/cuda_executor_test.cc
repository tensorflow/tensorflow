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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "absl/strings/match.h"
#include "xla/debug_options_flags.h"
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
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"

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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeviceDescription> result,
                       CudaExecutor::CreateDeviceDescription(0));

  EXPECT_TRUE(result->runtime_version().IsValid());
  EXPECT_TRUE(result->driver_version().IsValid());
  EXPECT_TRUE(result->compile_time_toolkit_version().IsValid());
  EXPECT_TRUE(result->cub_version().IsValid());

  EXPECT_GT(result->kernel_mode_driver_version().major_version(),
            300);  // NOLINT

  if (!absl::StartsWith(result->name(), "NVIDIA GB300")) {
    EXPECT_GT(result->pcie_bandwidth(), 1024 * 1024);
  }
  EXPECT_THAT(result->platform_version(), Not(IsEmpty()));
  EXPECT_THAT(result->name(), Not(IsEmpty()));
  EXPECT_THAT(result->model_str(), Not(IsEmpty()));
  EXPECT_THAT(result->device_vendor(), "NVIDIA Corporation");

  EXPECT_THAT(*result->gpu_compute_capability().cuda_compute_capability(),
              ::testing::Field("major", &CudaComputeCapability::major, Ge(1)));

  DeviceInterconnectInfo info = result->device_interconnect_info();
  if (result->cuda_compute_capability().IsAtLeastHopper() &&
      info.active_links) {
    const auto cc = result->cuda_compute_capability();
    if (cc.major == 10 && cc.minor == 7) {
      EXPECT_EQ(info.active_links, 36);
    } else if (cc.major == 10 && (cc.minor == 0 || cc.minor == 3)) {
      EXPECT_EQ(info.active_links, 18);
    } else {
      EXPECT_GE(info.active_links, 18);
    }
    // nvmlDeviceGetGpuFabricInfoV is only available in driver r545+
    if (result->kernel_mode_driver_version().major_version() >= 545) {
      EXPECT_THAT(info.clique_id, Not(IsEmpty()));
      EXPECT_THAT(info.cluster_uuid, Not(IsEmpty()));
    }
  }
  EXPECT_THAT(DeviceDescription::FromProto(result->ToProto()),
              IsOkAndHolds(*result));
}

TEST(CudaExecutorTest, GetCudaKernel) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  auto cuda_executor = dynamic_cast<CudaExecutor*>(executor);
  ASSERT_NE(cuda_executor, nullptr);

  auto verify_kernel = [&](const KernelLoaderSpec& spec) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                         executor->LoadKernel(spec));
    EXPECT_THAT(cuda_executor->GetCudaKernel(kernel.get()),
                absl_testing::IsOkAndHolds(kernel.get()));

    cuda_executor->UnloadKernel(kernel.get());
    EXPECT_THAT(cuda_executor->GetCudaKernel(kernel.get()),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));

    EXPECT_THAT(cuda_executor->GetCudaKernel(nullptr),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));
  };

  ASSERT_OK_AND_ASSIGN(KernelLoaderSpec add,
                       GetAddI32TestKernelSpec(cuda::kCudaPlatformId));
  verify_kernel(add);
  verify_kernel(GetAddI32PtxKernelSpec());
}

TEST(CudaExecutorTest, CreateUnifiedMemoryAllocatorWorks) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocator> allocator,
                       executor->CreateMemoryAllocator(MemorySpace::kUnified));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
}

TEST(CudaExecutorTest, CreateHostMemoryAllocatorWorks) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocator> allocator,
                       executor->CreateMemoryAllocator(MemorySpace::kHost));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
}

TEST(CudaExecutorTest, CreateCollectiveMemoryAllocatorWorks) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MemoryAllocator> allocator,
      executor->CreateMemoryAllocator(MemorySpace::kCollective));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_GE(allocation->address().size(), 1024);
}

TEST(CudaExecutorTest, AllocateArrayReturnsRequestedSize) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  DeviceAddress<uint64_t> buffer = executor->AllocateScalar<uint64_t>();
  ASSERT_NE(buffer.opaque(), nullptr);
  EXPECT_EQ(buffer.size(), sizeof(uint64_t));
  EXPECT_NE(buffer.payload(), 0);
  executor->Deallocate(&buffer);
}

// TODO: b/420735471 - Enable test once fixed.
TEST(CudaExecutorTest,
     DISABLED_CreateCollectiveMemoryAllocatorFailsForExcessiveSize) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(
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
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  EXPECT_THAT(executor->CreateMemoryAllocator(MemorySpace::kDevice),
              Not(absl_testing::IsOk()));
}

TEST(CudaExecutorTest, GetPointerMemorySpaceWorksWithUnifiedMemory) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  ASSERT_OK_AND_ASSIGN(auto unified_memory_allocator,
                       executor->CreateMemoryAllocator(MemorySpace::kUnified));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       unified_memory_allocator->Allocate(256));
  EXPECT_THAT(executor->GetPointerMemorySpace(allocation->address().opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kUnified));
}

TEST(CudaExecutorTest, GetPointerMemorySpaceWorksWithHostMemory) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       executor->HostMemoryAllocate(256));
  EXPECT_THAT(executor->GetPointerMemorySpace(allocation->address().opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kHost));
}

TEST(CudaExecutorTest, GetPointerMemorySpaceWorksWithDeviceAddress) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  DeviceAddressBase allocation = executor->Allocate(256);
  EXPECT_NE(allocation.opaque(), nullptr);
  EXPECT_THAT(executor->GetPointerMemorySpace(allocation.opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kDevice));
}

TEST(CudaExecutorTest, AllocateCollectiveMemoryWithDeviceAllocator) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  auto cuda_executor = dynamic_cast<CudaExecutor*>(executor);
  ASSERT_NE(cuda_executor, nullptr);
  DeviceAddressBase ptr =
      cuda_executor->Allocate(1024, static_cast<int>(MemorySpace::kCollective));

  EXPECT_NE(ptr.opaque(), nullptr);
  ASSERT_OK_AND_ASSIGN(size_t granularity, cuda_executor->GetVmmGranularity());
  EXPECT_EQ(ptr.size(), granularity);
  EXPECT_THAT(executor->GetPointerMemorySpace(ptr.opaque()),
              absl_testing::IsOkAndHolds(MemorySpace::kDevice));

  ASSERT_OK_AND_ASSIGN(CudaExecutor::VmmMemoryHandle handle,
                       cuda_executor->RetainVmmMemoryHandle(ptr.opaque()));
  EXPECT_NE(handle.handle(), 0);
  cuda_executor->Deallocate(&ptr);
}

TEST(CudaExecutorTest, MultipleExecutorsForSameDevice) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
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
    ASSERT_OK_AND_ASSIGN(auto desc, executors[i]->CreateDeviceDescription());
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

TEST(CudaExecutorTest, DisabledVmmRejectsReservationAndPhysicalAllocation) {
  const bool was_disabled =
      xla::GetDebugOptionsFromFlags().xla_gpu_experimental_vmm_disabled();
  std::vector<tsl::Flag> flags;
  xla::AppendDebugOptionsFlags(&flags);
  auto set_vmm_disabled = [&](bool disabled) {
    std::vector<std::string> args = {
        disabled ? "--xla_gpu_experimental_vmm_disabled=true"
                 : "--xla_gpu_experimental_vmm_disabled=false"};
    EXPECT_TRUE(tsl::Flags::Parse(args, flags));
    EXPECT_TRUE(args.empty());
  };
  absl::Cleanup restore_flag = [&] { set_vmm_disabled(was_disabled); };
  set_vmm_disabled(true);

  // Use a fresh executor: cached executors retain the options from Init().
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_GT(platform->VisibleDeviceCount(), 0);
  CudaExecutor executor(platform, /*device_ordinal=*/0);
  ASSERT_OK(executor.Init());
  EXPECT_THAT(executor.CreateMemoryReservation(4096),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(executor.CreatePhysicalMemoryAllocation(4096),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));

  // A caller falling back from reservation mode can still allocate normally.
  DeviceAddressBase memory = executor.Allocate(4096, /*memory_space=*/0);
  EXPECT_NE(memory.opaque(), nullptr);
  executor.Deallocate(&memory);
}

// Reservations and physical allocations created through the executor use its
// probed options, map together, and GetAllocationRange describes the mapping
// backing a pointer rather than the whole reservation.
TEST(CudaExecutorTest, ReservationMapsPhysicalMemoryAndReportsMappingRange) {
  if (xla::GetDebugOptionsFromFlags().xla_gpu_experimental_vmm_disabled()) {
    GTEST_SKIP() << "CUDA VMM is disabled";
  }
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  auto cuda_executor = dynamic_cast<CudaExecutor*>(executor);
  ASSERT_NE(cuda_executor, nullptr);

  // A one-byte reservation is padded to exactly one granule.
  ASSERT_OK_AND_ASSIGN(auto one_granule, executor->CreateMemoryReservation(1));
  const size_t granularity = one_granule->granularity();
  ASSERT_GT(granularity, 0);
  EXPECT_EQ(one_granule->address().size(), granularity);

  // Reserve two granules and back only the first with physical memory.
  ASSERT_OK_AND_ASSIGN(auto reservation,
                       executor->CreateMemoryReservation(2 * granularity));
  ASSERT_EQ(reservation->address().size(), 2 * granularity);
  ASSERT_OK_AND_ASSIGN(auto physical,
                       executor->CreatePhysicalMemoryAllocation(granularity));
  ASSERT_EQ(physical->address().size(), granularity);
  ASSERT_OK_AND_ASSIGN(
      auto mapping,
      reservation->MapTo(/*reservation_offset=*/0, /*allocation_offset=*/0,
                         granularity, *physical));
  DeviceAddressBase mapped = mapping.mapped_address();
  EXPECT_EQ(mapped.opaque(), reservation->address().opaque());
  EXPECT_EQ(mapped.size(), granularity);

  // The mapped prefix is accessible from the device.
  std::vector<uint8_t> pattern(granularity, 0xAB);
  ASSERT_OK(
      executor->SynchronousMemcpyH2D(pattern.data(), granularity, &mapped));
  std::vector<uint8_t> readback(granularity);
  ASSERT_OK(
      executor->SynchronousMemcpyD2H(mapped, granularity, readback.data()));
  EXPECT_EQ(readback, pattern);

  // RANGE_* attributes would report the whole two-granule reservation.
  ASSERT_OK_AND_ASSIGN(DeviceAddressBase range,
                       cuda_executor->GetAllocationRange(mapped.opaque()));
  EXPECT_EQ(range.opaque(), mapped.opaque());
  EXPECT_EQ(range.size(), granularity);
}

TEST(CudaExecutorTest, RetainVmmMemoryHandleForDefaultDeviceMemory) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  auto cuda_executor = dynamic_cast<CudaExecutor*>(executor);
  ASSERT_NE(cuda_executor, nullptr);
  DeviceAddressBase ptr =
      cuda_executor->Allocate(1024, static_cast<int>(MemorySpace::kDevice));

  EXPECT_NE(ptr.opaque(), nullptr);
  ASSERT_OK_AND_ASSIGN(size_t granularity, cuda_executor->GetVmmGranularity());
  EXPECT_EQ(ptr.size(), granularity);

  ASSERT_OK_AND_ASSIGN(CudaExecutor::VmmMemoryHandle handle,
                       cuda_executor->RetainVmmMemoryHandle(ptr.opaque()));
  EXPECT_NE(handle.handle(), 0);
  cuda_executor->Deallocate(&ptr);
}
}  // namespace
}  // namespace stream_executor::gpu
