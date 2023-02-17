/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info_for_tests.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/tsl/platform/test.h"

namespace stream_executor {
namespace gpu {
namespace {

TEST(DeviceInfoTest, DeviceInfo) {
  ASSERT_FALSE(cuInit(/*Flags=*/0));
  std::unique_ptr<DeviceDescription> d =
      GpuExecutor::CreateDeviceDescription(/*device_ordinal=*/0).value();
  absl::string_view name(d->name());
  if (name == "NVIDIA RTX A6000") {
    xla::gpu::GpuDeviceInfo test_info =
        xla::gpu::TestGpuDeviceInfo::RTXA6000DeviceInfo();
    EXPECT_EQ(test_info.name, d->name());
    EXPECT_EQ(test_info.threads_per_block_limit, d->threads_per_block_limit());
    EXPECT_EQ(test_info.threads_per_warp, d->threads_per_warp());
    EXPECT_EQ(test_info.shared_memory_per_block, d->shared_memory_per_block());
    EXPECT_EQ(test_info.shared_memory_per_core, d->shared_memory_per_core());
    EXPECT_EQ(test_info.threads_per_core_limit, d->threads_per_core_limit());
    EXPECT_EQ(test_info.core_count, d->core_count());
    EXPECT_EQ(test_info.fpus_per_core, d->fpus_per_core());
    EXPECT_EQ(test_info.block_dim_limit_x, d->block_dim_limit().x);
    EXPECT_EQ(test_info.block_dim_limit_y, d->block_dim_limit().y);
    EXPECT_EQ(test_info.block_dim_limit_z, d->block_dim_limit().z);
    EXPECT_EQ(test_info.memory_bandwidth, d->memory_bandwidth());
    EXPECT_EQ(test_info.l2_cache_size, d->l2_cache_size());
    // Clock rate can vary between base and boost values.
    EXPECT_LE(test_info.clock_rate_ghz, d->clock_rate_ghz());
  } else if (name == "Quadro P1000") {
    EXPECT_EQ(d->fpus_per_core(), 128);
    EXPECT_EQ(d->l2_cache_size(), 1024 * 1024);
  } else if (name == "Tesla P100-SXM2-16GB") {
    EXPECT_EQ(d->fpus_per_core(), 64);
    EXPECT_EQ(d->l2_cache_size(), 4 * 1024 * 1024);
  } else {
    VLOG(1) << "Not tested for " << name;
  }
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
