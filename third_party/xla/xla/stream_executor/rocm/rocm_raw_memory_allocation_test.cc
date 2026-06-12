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

#include "xla/stream_executor/rocm/rocm_raw_memory_allocation.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

using absl_testing::IsOk;

static constexpr uint64_t kTestSize = 1024 * 1024;

class RocmRawMemoryAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("ROCM");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "ROCM platform not available";
    }
    auto executor_or = platform_or.value()->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      GTEST_SKIP() << "ROCM executor not available: " << executor_or.status();
    }
    executor_ = executor_or.value();
  }

  StreamExecutor* executor_ = nullptr;
};

TEST_F(RocmRawMemoryAllocationTest, CreateAllocation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, RocmRawMemoryAllocation::Create(executor_, kTestSize));

  EXPECT_NE(alloc->GetHandle(), nullptr);
  EXPECT_GE(alloc->address().size(), kTestSize);
}

TEST_F(RocmRawMemoryAllocationTest, AddressReflectsHandle) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, RocmRawMemoryAllocation::Create(executor_, kTestSize));

  EXPECT_EQ(alloc->address().opaque(), static_cast<void*>(alloc->GetHandle()));
  EXPECT_GE(alloc->address().size(), kTestSize);
}

TEST_F(RocmRawMemoryAllocationTest, SizeIsAtLeastRequested) {
  TF_ASSERT_OK_AND_ASSIGN(auto alloc,
                          RocmRawMemoryAllocation::Create(executor_, 1));

  EXPECT_NE(alloc->GetHandle(), nullptr);
  EXPECT_GE(alloc->address().size(), 1u);
}

}  // namespace
}  // namespace stream_executor::gpu
