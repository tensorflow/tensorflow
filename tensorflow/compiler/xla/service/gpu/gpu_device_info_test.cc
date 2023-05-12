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

#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info_for_tests.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/tsl/platform/test.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

namespace stream_executor {
namespace gpu {
namespace {

namespace se = stream_executor;

TEST(DeviceInfoTest, DeviceInfoIsCorrect) {
  string test_platform = "cuda";
#if TENSORFLOW_USE_ROCM
  test_platform = "rocm";
#endif
  se::Platform* platform = se::MultiPlatformManager::PlatformWithName(test_platform).value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  const xla::gpu::GpuDeviceInfo dev_info = xla::gpu::GetGpuDeviceInfo(executor);
  absl::string_view name(dev_info.name);
  if (name == "NVIDIA RTX A6000") {
    xla::gpu::GpuDeviceInfo test_info =
        xla::gpu::TestGpuDeviceInfo::RTXA6000DeviceInfo();
    EXPECT_THAT(
        dev_info,
        ::testing::FieldsAre(
            test_info.name, test_info.threads_per_block_limit,
            test_info.threads_per_warp, test_info.shared_memory_per_block,
            test_info.shared_memory_per_block_optin,
            test_info.shared_memory_per_core, test_info.threads_per_core_limit,
            test_info.core_count, test_info.fpus_per_core,
            test_info.block_dim_limit_x, test_info.block_dim_limit_y,
            test_info.block_dim_limit_z, test_info.memory_bandwidth,
            test_info.l2_cache_size,
            // Clock rate can vary between base and boost values.
            ::testing::Ge(test_info.clock_rate_ghz),
            dev_info.device_memory_size));
  } else if (name == "Quadro P1000") {
    EXPECT_THAT(
        dev_info,
        ::testing::FieldsAre(
            name, /*threads_per_block_limit=*/1024,
            /*threads_per_warp=*/32, /*shared_memory_per_block=*/48 * 1024,
            /*shared_memory_per_block_optin=*/48 * 1024,
            /*shared_memory_per_core=*/96 * 1024,
            /*threads_per_core_limit=*/2048, /*core_count=*/5,
            /*fpus_per_core=*/128,
            /*block_dim_limit_x=*/2'147'483'647,
            /*block_dim_limit_y=*/65535,
            /*block_dim_limit_z=*/65535,
            /*memory_bandwidth=*/80'160'000'000, /*l2_cache_size=*/1024 * 1024,
            /*clock_rate_ghz=*/::testing::Ge(1.4),
            /*device_memory_size=*/4'234'346'496));
  } else if (name == "Tesla P100-SXM2-16GB") {
    EXPECT_THAT(
        dev_info,
        ::testing::FieldsAre(name, /*threads_per_block_limit=*/1024,
                             /*threads_per_warp=*/32,
                             /*shared_memory_per_block=*/48 * 1024,
                             /*shared_memory_per_block_optin=*/48 * 1024,
                             /*shared_memory_per_core=*/64 * 1024,
                             /*threads_per_core_limit=*/2048, /*core_count=*/56,
                             /*fpus_per_core=*/64,
                             /*block_dim_limit_x=*/2'147'483'647,
                             /*block_dim_limit_y=*/65535,
                             /*block_dim_limit_z=*/65535,
                             /*memory_bandwidth=*/732'160'000'000,
                             /*l2_cache_size=*/4 * 1024 * 1024,
                             /*clock_rate_ghz=*/::testing::Ge(1.4),
                             /*device_memory_size=*/17'066'622'976));
  }
#if TF_ROCM_VERSION >= 50500
  else if (name == "AMD Instinct MI210") {
    xla::gpu::GpuDeviceInfo test_info =
        xla::gpu::TestGpuDeviceInfo::AMDMI210DeviceInfo();
    EXPECT_THAT(
            dev_info,
            ::testing::FieldsAre(
                test_info.name, test_info.threads_per_block_limit,
                test_info.threads_per_warp, test_info.shared_memory_per_block,
                test_info.shared_memory_per_block_optin,
                test_info.shared_memory_per_core, test_info.threads_per_core_limit,
                test_info.core_count, test_info.fpus_per_core,
                test_info.block_dim_limit_x, test_info.block_dim_limit_y,
                test_info.block_dim_limit_z, test_info.memory_bandwidth,
                test_info.l2_cache_size,
                ::testing::Ge(test_info.clock_rate_ghz),
                dev_info.device_memory_size));
  }
  else if (name == "AMD Instinct MI100"){
    EXPECT_THAT(
            dev_info,
            ::testing::FieldsAre(name, /*threads_per_block_limit=*/1024,
            /*threads_per_warp=*/64, /*shared_memory_per_block=*/64 * 1024,
            /*shared_memory_per_block_optin=*/0, 
            /*shared_memory_per_core=*/64 * 1024,
            /*threads_per_core_limit=*/2560, /*core_count=*/120, 
            /*fpus_per_core=*/0, /*block_dim_limit_x=*/2'147'483'647,
            /*block_dim_limit_y=*/2'147'483'647,
            /*block_dim_limit_z=*/2'147'483'647,
            /*memory_bandwidth=*/1228800000000, 
            /*l2_cache_size=*/8 * 1024 * 1024,
            /*clock_rate_ghz=*/::testing::Ge(1.5),
            /*device_memory_size=*/33'806'090'240));
  }  
  else if (name == "AMD Instinct MI50/MI60"){
    EXPECT_THAT(
            dev_info,
            ::testing::FieldsAre(name, /*threads_per_block_limit=*/1024,
            /*threads_per_warp=*/64, /*shared_memory_per_block=*/64 * 1024,
            /*shared_memory_per_block_optin=*/0,
            /*shared_memory_per_core=*/64 * 1024,
            /*threads_per_core_limit=*/2560, /*core_count=*/60,
            /*fpus_per_core=*/0, /*block_dim_limit_x=*/2'147'483'647,
            /*block_dim_limit_y=*/2'147'483'647,
            /*block_dim_limit_z=*/2'147'483'647,
            /*memory_bandwidth=*/256000000000,
            /*l2_cache_size=*/8 * 1024 * 1024,
            /*clock_rate_ghz=*/::testing::Ge(1.7),
            /*device_memory_size=*/17'163'091'968));
  }
#endif  
  else {
    VLOG(1) << "Not tested for " << name;
  }
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
