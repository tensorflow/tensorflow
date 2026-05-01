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

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

using absl_testing::IsOk;

// 1 MB — will be rounded up to the VMM granularity (typically 2 MB).
static constexpr uint64_t kTestSize = 1024 * 1024;

class CudaRawMemoryAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(
        Platform * platform,
        PlatformManager::PlatformWithId(cuda::kCudaPlatformId));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
  }

  StreamExecutor* executor_ = nullptr;
};

// Verifies that Create returns a valid handle and that the allocation is at
// least as large as requested.
TEST_F(CudaRawMemoryAllocationTest, CreateAllocation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, CudaRawMemoryAllocation::Create(executor_, kTestSize));

  EXPECT_NE(alloc->GetHandle(), 0u);
  EXPECT_GE(alloc->address().size(), kTestSize);
}

// Verifies that address().opaque() is the handle cast to void* and that
// address().size() matches the padded allocation size.
TEST_F(CudaRawMemoryAllocationTest, AddressReflectsHandle) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, CudaRawMemoryAllocation::Create(executor_, kTestSize));

  EXPECT_EQ(
      alloc->address().opaque(),
      reinterpret_cast<void*>(static_cast<uintptr_t>(alloc->GetHandle())));
  EXPECT_GE(alloc->address().size(), kTestSize);
}

// Verifies that a very small request is still satisfied (padded to
// granularity).
TEST_F(CudaRawMemoryAllocationTest, SizeIsAtLeastRequested) {
  TF_ASSERT_OK_AND_ASSIGN(auto alloc,
                          CudaRawMemoryAllocation::Create(executor_, 1));

  EXPECT_NE(alloc->GetHandle(), 0u);
  EXPECT_GE(alloc->address().size(), 1u);
}

}  // namespace
}  // namespace stream_executor::gpu
