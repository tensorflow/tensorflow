/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/stream_executor/sycl/sycl_context.h"

#include <gtest/gtest.h>
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor::sycl {
namespace {

TEST(SyclContextTest, GetDeviceTotalMemory) {
  TF_ASSERT_OK_AND_ASSIGN(::sycl::device device,
                          SyclDevicePool::GetDevice(kDefaultDeviceOrdinal));
  TF_ASSERT_OK_AND_ASSIGN(uint64_t total_memory,
                          SyclContext::GetDeviceTotalMemory(device));
  EXPECT_GT(total_memory, 0)
      << "Total memory should be greater than 0, got " << total_memory;
}

TEST(SyclContextTest, CreateAndSynchronizeContext) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SyclContext> sycl_context_ptr,
                          SyclContext::Create(kDefaultDeviceOrdinal));
  EXPECT_NE(sycl_context_ptr, nullptr);
  EXPECT_EQ(sycl_context_ptr->device_ordinal(), kDefaultDeviceOrdinal);
  EXPECT_TRUE(sycl_context_ptr->Synchronize().ok());
}

}  // namespace
}  // namespace stream_executor::sycl
