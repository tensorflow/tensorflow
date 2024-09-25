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
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {

class GpuExecutorTest : public testing::Test {
 public:
  Platform* GetPlatform() {
    auto name = absl::AsciiStrToLower(
        xla::PlatformUtil::CanonicalPlatformName("gpu").value());
    return PlatformManager::PlatformWithName(name).value();
  }
};

using GetPointerMemorySpaceTest = GpuExecutorTest;

TEST_F(GetPointerMemorySpaceTest, Host) {
  StreamExecutor* executor = GetPlatform()->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto host_ptr, executor->HostMemoryAllocate(64));
  TF_ASSERT_OK_AND_ASSIGN(auto memory_space,
                          executor->GetPointerMemorySpace(host_ptr->opaque()))
  EXPECT_EQ(memory_space, MemoryType::kHost);
}

TEST_F(GetPointerMemorySpaceTest, Device) {
  StreamExecutor* executor = GetPlatform()->ExecutorForDevice(0).value();
  auto mem = executor->Allocate(64);
  ASSERT_NE(mem, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(auto memory_space,
                          executor->GetPointerMemorySpace(mem.opaque()))
  EXPECT_EQ(memory_space, MemoryType::kDevice);
  executor->Deallocate(&mem);
}

}  // namespace stream_executor
