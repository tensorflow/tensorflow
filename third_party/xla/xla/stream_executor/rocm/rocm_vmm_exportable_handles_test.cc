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

#include <unistd.h>

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_device_address_vmm_allocator.h"
#include "xla/stream_executor/rocm/rocm_raw_memory_allocation.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/rocm/rocm_vmm_allocator.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::Not;

// Exports `handle` as a POSIX file descriptor and immediately closes it.
absl::Status TryExportAsPosixFd(hipMemGenericAllocationHandle_t handle) {
  int fd = -1;
  ABSL_RETURN_IF_ERROR(
      ToStatus(hipMemExportToShareableHandle(
                   &fd, handle, hipMemHandleTypePosixFileDescriptor, 0ULL),
               "hipMemExportToShareableHandle"));
  if (fd < 0) {
    return absl::InternalError(
        "Export succeeded but produced an invalid descriptor.");
  }
  close(fd);
  return absl::OkStatus();
}

class RocmVmmExportableHandlesTest : public testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<Platform*> platform =
        PlatformManager::PlatformWithName("ROCM");
    if (!platform.ok()) {
      GTEST_SKIP() << "ROCM platform not available";
    }
    platform_ = *platform;

    absl::StatusOr<StreamExecutor*> executor = platform_->ExecutorForDevice(0);
    if (!executor.ok()) {
      GTEST_SKIP() << "ROCM executor not available: " << executor.status();
    }
    executor_ = *executor;
  }

  absl::StatusOr<hipMemAllocationHandleType> RequestedHandleType(
      const RocmVmmAllocator::Options& options) {
    std::unique_ptr<ActivateContext> activation = executor_->Activate();
    ABSL_ASSIGN_OR_RETURN(RocmVmmAllocationParams params,
                     GetRocmVmmAllocationParams(executor_, options));
    return params.prop.requestedHandleTypes;
  }

  // Callers must GTEST_SKIP() themselves: it returns from the function it
  // appears in, so a helper cannot skip the test body for them.
  bool ExportableHandlesSupported() {
    absl::StatusOr<hipMemAllocationHandleType> handle_type =
        RequestedHandleType(kExportable);
    return handle_type.ok() &&
           *handle_type == hipMemHandleTypePosixFileDescriptor;
  }

  static constexpr RocmVmmAllocator::Options kExportable = {
      .enable_posix_fd_handle = true};
  static constexpr RocmVmmAllocator::Options kNonExportable = {
      .enable_posix_fd_handle = false};

  Platform* platform_ = nullptr;
  StreamExecutor* executor_ = nullptr;
};

// On ROCm 10.0 and newer, a caller that asks for exportable handles gets them,
// unless this driver, container or device cannot export at all.
TEST_F(RocmVmmExportableHandlesTest, ExportableHandlesOnSupportedRocm) {
  // ROCm 10.0 ships HIP 7.15; runtime_version() is the HIP version.
  if (executor_->GetDeviceDescription().runtime_version() <
      SemanticVersion{7, 15, 0}) {
    GTEST_SKIP() << "Exportable VMM handles are gated off below ROCm 10.0.";
  }

  ASSERT_OK_AND_ASSIGN(hipMemAllocationHandleType handle_type,
                       RequestedHandleType(kExportable));
  if (handle_type == hipMemHandleTypeNone) {
    GTEST_SKIP() << "The HIP runtime is new enough, but the granularity query "
                    "fell back to non-exportable handles.";
  }
  EXPECT_EQ(handle_type, hipMemHandleTypePosixFileDescriptor);
}

// Asking for exportable handles must never cost an allocation that does not
// need them: where fd export is unavailable, the request is dropped instead.
TEST_F(RocmVmmExportableHandlesTest, ExportableRequestNeverBreaksAllocation) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmRawMemoryAllocation> allocation,
                       RocmRawMemoryAllocation::Create(
                           executor_, /*size=*/1024 * 1024, kExportable));
  EXPECT_NE(allocation->GetHandle(), nullptr);
}

// Both fallbacks land on hipMemHandleTypeNone properties, so those must
// allocate.
TEST_F(RocmVmmExportableHandlesTest, NonExportablePropertiesStillAllocate) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  ASSERT_OK_AND_ASSIGN(RocmVmmAllocationParams params,
                       GetRocmVmmAllocationParams(executor_, kNonExportable));
  ASSERT_EQ(params.prop.requestedHandleTypes, hipMemHandleTypeNone);

  ASSERT_OK_AND_ASSIGN(
      hipMemGenericAllocationHandle_t handle,
      CreateRocmVmmPhysicalAllocation(params.prop, params.granularity));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(hipMemRelease(handle), hipSuccess);
}

// Nothing can make a working driver refuse fd export, so the retry is walked
// with properties hipMemCreate rejects whatever the handle type: what it must
// not do is report the second failure as a handle.
TEST_F(RocmVmmExportableHandlesTest, RejectedExportableCreateIsReported) {
  if (!ExportableHandlesSupported()) {
    GTEST_SKIP() << "Exportable VMM handles require ROCm 10.0 or newer.";
  }

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  ASSERT_OK_AND_ASSIGN(RocmVmmAllocationParams params,
                       GetRocmVmmAllocationParams(executor_, kExportable));

  int device_count = 0;
  ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
  params.prop.location.id = device_count;  // One past the last device.

  EXPECT_THAT(CreateRocmVmmPhysicalAllocation(params.prop, params.granularity),
              Not(IsOk()));
}

// A caller that does not ask for exportable handles never gets them.
TEST_F(RocmVmmExportableHandlesTest, NonExportableOptionIsHonored) {
  EXPECT_THAT(RequestedHandleType(kNonExportable),
              IsOkAndHolds(hipMemHandleTypeNone));
}

TEST_F(RocmVmmExportableHandlesTest, ParamsDescribePinnedDeviceAllocation) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  ASSERT_OK_AND_ASSIGN(RocmVmmAllocationParams params,
                       GetRocmVmmAllocationParams(executor_, kExportable));

  EXPECT_EQ(params.prop.type, hipMemAllocationTypePinned);
  EXPECT_EQ(params.prop.location.type, hipMemLocationTypeDevice);
  EXPECT_EQ(params.prop.location.id, executor_->device_ordinal());
  EXPECT_EQ(static_cast<int>(params.prop.allocFlags.compressionType), 0);
  EXPECT_EQ(static_cast<int>(params.prop.allocFlags.gpuDirectRDMACapable), 0);
  EXPECT_EQ(static_cast<int>(params.prop.allocFlags.usage), 0);
  EXPECT_EQ(params.prop.win32HandleMetaData, nullptr);
  EXPECT_GT(params.granularity, 0u);
}

// Allocation sites and granularity queries would otherwise drift apart.
TEST_F(RocmVmmExportableHandlesTest, ParamsAreStableAcrossCalls) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  ASSERT_OK_AND_ASSIGN(RocmVmmAllocationParams first,
                       GetRocmVmmAllocationParams(executor_, kExportable));
  for (int i = 0; i < 4; ++i) {
    ASSERT_OK_AND_ASSIGN(RocmVmmAllocationParams params,
                         GetRocmVmmAllocationParams(executor_, kExportable));
    EXPECT_EQ(params.prop.requestedHandleTypes,
              first.prop.requestedHandleTypes);
    EXPECT_EQ(params.granularity, first.granularity);
  }
}

TEST_F(RocmVmmExportableHandlesTest, NonExportableAllocationCannotBeExported) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmRawMemoryAllocation> allocation,
                       RocmRawMemoryAllocation::Create(
                           executor_, /*size=*/1024 * 1024, kNonExportable));

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  EXPECT_THAT(TryExportAsPosixFd(allocation->GetHandle()), Not(IsOk()));
}

// RocmDeviceAddressVmmAllocator creates its physical memory through
// RocmRawMemoryAllocation::Create, so this covers both.
TEST_F(RocmVmmExportableHandlesTest, RawMemoryAllocationHandleIsExportable) {
  if (!ExportableHandlesSupported()) {
    GTEST_SKIP() << "Exportable VMM handles require ROCm 10.0 or newer.";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmRawMemoryAllocation> allocation,
                       RocmRawMemoryAllocation::Create(
                           executor_, /*size=*/1024 * 1024, kExportable));
  ASSERT_NE(allocation->GetHandle(), nullptr);

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  EXPECT_THAT(TryExportAsPosixFd(allocation->GetHandle()), IsOk())
      << "VMM allocation handle cannot be exported as a POSIX file "
         "descriptor.";
}

// A descriptor that cannot be imported back is of no use to whoever receives
// it, so check the full round trip.
TEST_F(RocmVmmExportableHandlesTest, ExportedDescriptorCanBeImported) {
  if (!ExportableHandlesSupported()) {
    GTEST_SKIP() << "Exportable VMM handles require ROCm 10.0 or newer.";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmRawMemoryAllocation> allocation,
                       RocmRawMemoryAllocation::Create(
                           executor_, /*size=*/1024 * 1024, kExportable));

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  int fd = -1;
  ASSERT_EQ(
      hipMemExportToShareableHandle(&fd, allocation->GetHandle(),
                                    hipMemHandleTypePosixFileDescriptor, 0ULL),
      hipSuccess);
  ASSERT_GE(fd, 0);

  hipMemGenericAllocationHandle_t imported = nullptr;
  hipError_t import_result = hipMemImportFromShareableHandle(
      &imported, reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
      hipMemHandleTypePosixFileDescriptor);
  close(fd);

  ASSERT_EQ(import_result, hipSuccess);
  ASSERT_NE(imported, nullptr);
  EXPECT_EQ(hipMemRelease(imported), hipSuccess);
}

// Default-constructed, so a default that stopped requesting exportable handles
// would fail here.
TEST_F(RocmVmmExportableHandlesTest, VmmAllocatorMemoryIsExportable) {
  if (!ExportableHandlesSupported()) {
    GTEST_SKIP() << "Exportable VMM handles require ROCm 10.0 or newer.";
  }

  RocmVmmAllocator allocator(executor_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(/*size=*/1024 * 1024));
  ASSERT_NE(allocation->address().opaque(), nullptr);

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(
      hipMemRetainAllocationHandle(&handle, allocation->address().opaque()),
      hipSuccess);

  EXPECT_THAT(TryExportAsPosixFd(handle), IsOk());
  EXPECT_EQ(hipMemRelease(handle), hipSuccess);
}

// The option reaches the mapped path as well, not just the raw one.
TEST_F(RocmVmmExportableHandlesTest, NonExportableVmmAllocatorMemory) {
  RocmVmmAllocator allocator(executor_, kNonExportable);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(/*size=*/1024 * 1024));
  ASSERT_NE(allocation->address().opaque(), nullptr);

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(
      hipMemRetainAllocationHandle(&handle, allocation->address().opaque()),
      hipSuccess);

  EXPECT_THAT(TryExportAsPosixFd(handle), Not(IsOk()));
  EXPECT_EQ(hipMemRelease(handle), hipSuccess);
}

// The path that matters in practice: a collective-space buffer from the kVmm
// allocator, retained and exported the way a collective library would. Takes
// the options DeviceAddressVmmAllocator::Create() leaves at their defaults.
TEST_F(RocmVmmExportableHandlesTest, CollectiveAllocationIsExportable) {
  if (!ExportableHandlesSupported()) {
    GTEST_SKIP() << "Exportable VMM handles require ROCm 10.0 or newer.";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor_->CreateStream());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmDeviceAddressVmmAllocator> allocator,
      RocmDeviceAddressVmmAllocator::Create(executor_, stream.get()));

  ASSERT_OK_AND_ASSIGN(
      ScopedDeviceAddress<uint8_t> scoped_address,
      allocator->Allocate(executor_->device_ordinal(), /*size=*/393216,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_FALSE(scoped_address.is_null());

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipMemRetainAllocationHandle(&handle, scoped_address->opaque()),
            hipSuccess);

  EXPECT_THAT(TryExportAsPosixFd(handle), IsOk())
      << "Collective-space VMM buffer cannot be exported as a POSIX file "
         "descriptor.";
  EXPECT_EQ(hipMemRelease(handle), hipSuccess);
}

// A one-byte request padded to exactly the reported granularity pins down
// that the querying site and the allocating site cannot disagree.
TEST_F(RocmVmmExportableHandlesTest, PaddingMatchesReportedGranularity) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();

  ASSERT_OK_AND_ASSIGN(RocmVmmAllocationParams params,
                       GetRocmVmmAllocationParams(executor_, kExportable));
  ASSERT_GT(params.granularity, 0u);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmRawMemoryAllocation> allocation,
      RocmRawMemoryAllocation::Create(executor_, /*size=*/1, kExportable));
  EXPECT_EQ(allocation->address().size(), params.granularity);
}

}  // namespace
}  // namespace stream_executor::gpu
