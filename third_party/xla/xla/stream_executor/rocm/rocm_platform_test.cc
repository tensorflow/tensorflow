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

#include "xla/stream_executor/rocm/rocm_platform.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {

TEST(RocmPlatformTest, TestPlatformName) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  EXPECT_EQ(platform->Name(), "ROCM");
}

// Miss before ExecutorForDevice; same pointer after.
TEST(RocmPlatformTest, FindExistingWorks) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  const int num_devices = platform->VisibleDeviceCount();
  ASSERT_GT(num_devices, 0);
  for (int i = 0; i < num_devices; ++i) {
    EXPECT_FALSE(platform->FindExisting(i).ok());
  }
  absl::flat_hash_map<int, StreamExecutor*> executors;
  for (int i = 0; i < num_devices; ++i) {
    ASSERT_OK_AND_ASSIGN(auto executor, platform->ExecutorForDevice(i));
    executors[i] = executor;
  }
  EXPECT_EQ(executors.size(), num_devices);
  for (int i = 0; i < num_devices; ++i) {
    ASSERT_OK_AND_ASSIGN(auto executor, platform->FindExisting(i));
    EXPECT_EQ(executor, executors[i]);
  }
}

}  // namespace
}  // namespace stream_executor::gpu
