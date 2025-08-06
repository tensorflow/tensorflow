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
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace gpu {

static Platform* NewPlatform() {
  Platform* platform = PlatformManager::PlatformWithName("SYCL").value();
  return platform;
}

TEST(SyclPlatformTest, Name) {
  auto platform = NewPlatform();
  auto name = platform->Name();
  EXPECT_EQ(name, "SYCL");
}

}  // namespace gpu
}  // namespace stream_executor
