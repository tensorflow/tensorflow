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

#include <gtest/gtest.h>
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::sycl {
namespace {

TEST(SyclPlatformTest, TestPlatformName) {
  TF_ASSERT_OK_AND_ASSIGN(
      Platform * platform,
      stream_executor::PlatformManager::PlatformWithId(kSyclPlatformId));
  EXPECT_EQ(platform->Name(), "SYCL");
}

TEST(SyclPlatformTest, FindExistingWorks) {
  TF_ASSERT_OK_AND_ASSIGN(
      Platform * platform,
      stream_executor::PlatformManager::PlatformWithId(kSyclPlatformId));
  EXPECT_GT(platform->VisibleDeviceCount(), 0);
  for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
    EXPECT_FALSE(platform->FindExisting(i).ok());
  }
  absl::flat_hash_map<int, StreamExecutor*> executors;
  for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto executor, platform->ExecutorForDevice(i));
    executors[i] = executor;
  }
  EXPECT_EQ(executors.size(), platform->VisibleDeviceCount());
  for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto executor, platform->FindExisting(i));
    EXPECT_EQ(executor, executors[i]);
  }
}

}  // namespace
}  // namespace stream_executor::sycl
