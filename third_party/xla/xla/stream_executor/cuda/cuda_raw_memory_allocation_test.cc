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

#include "xla/stream_executor/cuda/cuda_raw_memory_allocation.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_device_allocator.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace stream_executor::gpu {
namespace {

// 1 MB — will be rounded up to the VMM granularity (typically 2 MB).
static constexpr uint64_t kTestSize = 1024 * 1024;

class CudaRawMemoryAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(Platform * platform, PlatformManager::PlatformWithId(
                                                  cuda::kCudaPlatformId));
    ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
  }

  StreamExecutor* executor_ = nullptr;
};

// Verifies that Create returns a valid handle and that the allocation is at
// least as large as requested.
TEST_F(CudaRawMemoryAllocationTest, CreateAllocation) {
  ASSERT_OK_AND_ASSIGN(auto alloc,
                       CudaRawMemoryAllocation::Create(executor_, kTestSize));

  EXPECT_NE(alloc->GetHandle(), 0u);
  EXPECT_GE(alloc->address().size(), kTestSize);
}

// Verifies that address().opaque() is the handle cast to void* and that
// address().size() matches the padded allocation size.
TEST_F(CudaRawMemoryAllocationTest, AddressReflectsHandle) {
  ASSERT_OK_AND_ASSIGN(auto alloc,
                       CudaRawMemoryAllocation::Create(executor_, kTestSize));

  EXPECT_EQ(
      alloc->address().opaque(),
      reinterpret_cast<void*>(static_cast<uintptr_t>(alloc->GetHandle())));
  EXPECT_GE(alloc->address().size(), kTestSize);
}

// Verifies that callers can supply allocator options instead of probing them.
TEST_F(CudaRawMemoryAllocationTest, CreateWithExplicitOptions) {
  CudaDeviceAllocator::Options options;
  options.enable_posix_fd_handle = false;
  options.enable_fabric_handle = false;
  ASSERT_OK_AND_ASSIGN(auto alloc, CudaRawMemoryAllocation::Create(
                                       executor_, kTestSize, options));

  EXPECT_NE(alloc->GetHandle(), 0u);
  EXPECT_GE(alloc->address().size(), kTestSize);
}

// Requesting exportable handle types must succeed on every machine: either
// the driver supports them, or Create falls back to simpler handle types.
TEST_F(CudaRawMemoryAllocationTest,
       ExportableHandleTypesFallBackWhenUnsupported) {
  CudaDeviceAllocator::Options options;
  options.enable_posix_fd_handle = true;
  options.enable_fabric_handle = true;
  ASSERT_OK_AND_ASSIGN(auto alloc, CudaRawMemoryAllocation::Create(
                                       executor_, kTestSize, options));

  EXPECT_NE(alloc->GetHandle(), 0u);
  EXPECT_GE(alloc->address().size(), kTestSize);
}

// Verifies that a very small request is still satisfied (padded to
// granularity).
TEST_F(CudaRawMemoryAllocationTest, SizeIsAtLeastRequested) {
  ASSERT_OK_AND_ASSIGN(auto alloc,
                       CudaRawMemoryAllocation::Create(executor_, 1));

  EXPECT_NE(alloc->GetHandle(), 0u);
  EXPECT_GE(alloc->address().size(), 1u);
}

// The allocation is VMM-only, so options that opt out of VMM are rejected
// instead of silently issuing VMM driver calls.
TEST_F(CudaRawMemoryAllocationTest, DisabledVmmOptionsAreRejected) {
  CudaDeviceAllocator::Options options;
  options.use_vmm = false;
  EXPECT_THAT(CudaRawMemoryAllocation::Create(executor_, kTestSize, options),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

// As in CudaDeviceAllocator, the size is padded to the larger of
// options.alignment and the mapping granularity.
TEST_F(CudaRawMemoryAllocationTest, AlignmentLargerThanGranularityPadsSize) {
  // A one-byte request with probed options is padded to exactly one granule.
  ASSERT_OK_AND_ASSIGN(auto one_granule,
                       CudaRawMemoryAllocation::Create(executor_, 1));
  const uint64_t granularity = one_granule->address().size();

  CudaDeviceAllocator::Options options;
  options.alignment = 2 * granularity;
  ASSERT_OK_AND_ASSIGN(auto alloc,
                       CudaRawMemoryAllocation::Create(executor_, 1, options));

  EXPECT_EQ(alloc->address().size(), 2 * granularity);
}

}  // namespace
}  // namespace stream_executor::gpu
