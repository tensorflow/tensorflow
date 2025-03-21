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
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

TEST(GpuTestKernelsFatbinTest, GetGpuTestKernelsFatbin) {
  std::vector<uint8_t> fatbin;

  TF_ASSERT_OK_AND_ASSIGN(fatbin, GetGpuTestKernelsFatbin());
  EXPECT_FALSE(fatbin.empty());
}

}  // namespace
}  // namespace stream_executor::gpu
