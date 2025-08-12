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

#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {

TEST(GpuTestKernelsFatbinTest, GetGpuTestKernelsFatbin) {
  bool found_at_least_one_platform = false;
  for (const auto& platform_name : {"CUDA", "ROCM"}) {
    absl::StatusOr<Platform*> platform =
        PlatformManager::PlatformWithName(platform_name);
    if (platform.ok()) {
      found_at_least_one_platform = true;
      TF_ASSERT_OK_AND_ASSIGN(
          std::vector<uint8_t> fatbin,
          GetGpuTestKernelsFatbin(platform.value()->Name()));
      EXPECT_FALSE(fatbin.empty());
    }
  }

  if (!found_at_least_one_platform) {
    // This case is not necessarily a test error, therefore we mark the test
    // as skipped.
    GTEST_SKIP() << "No GPU platform was linked into this test binary.";
  }
}

}  // namespace
}  // namespace stream_executor::gpu
