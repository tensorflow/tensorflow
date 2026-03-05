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

#include "xla/stream_executor/sycl/sycl_device_description.h"

#include <gtest/gtest.h>
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::sycl {
namespace {

TEST(SyclDeviceDescriptionTest, CreateOneApiDeviceDescription) {
  std::unique_ptr<DeviceDescription> device_desc =
      CreateOneApiDeviceDescription(0).value_or(nullptr);
  // Sanity checks on fields that are part to DeviceDescription::ToString().
  ASSERT_NE(device_desc, nullptr);
  EXPECT_GT(device_desc->core_count(), 0);
  EXPECT_GT(device_desc->clock_rate_ghz(), 0.0f);
  EXPECT_GT(device_desc->threads_per_block_limit(), 0);
  EXPECT_GT(device_desc->threads_per_warp(), 0);
  EXPECT_GT(device_desc->threads_per_core_limit(), 0);
  EXPECT_GT(device_desc->l2_cache_size(), 0);
  EXPECT_GT(device_desc->device_memory_size(), 0);
  EXPECT_GT(device_desc->memory_bandwidth(), 0);
  EXPECT_GT(device_desc->shared_memory_per_block(), 0);
  EXPECT_GT(device_desc->shared_memory_per_core(), 0);
  EXPECT_GT(device_desc->fpus_per_core(), 0);
}

}  // namespace
}  // namespace stream_executor::sycl
