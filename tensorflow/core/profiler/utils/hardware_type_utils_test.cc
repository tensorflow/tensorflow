/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/hardware_type_utils.h"

#include "xla/tsl/profiler/utils/math_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(HardwareTypeUtilsTest, H100PeakComputTFlops) {
  DeviceCapabilities device_cap;
  // For NVIDIA H100 PCIe 80 GB, according to
  // https://resources.nvidia.com/en-us-data-center-overview/gtc22-whitepaper-hopper
  // https://www.techpowerup.com/gpu-specs/h100-pcie-80-gb.c3899
  device_cap.set_clock_rate_in_ghz(1.620);
  device_cap.set_num_cores(114);
  device_cap.set_memory_size_in_bytes(
      tsl::profiler::GibiToGiga(tsl::profiler::GigaToUni(80)));
  device_cap.set_memory_bandwidth(tsl::profiler::GigaToUni(2.04 * 1024));
  device_cap.set_device_vendor("Nvidia");
  device_cap.mutable_compute_capability()->set_major(9);
  device_cap.mutable_compute_capability()->set_minor(0);

  // Get target TFLOPS per SM and check.
  double peak_tflops =
      GetFlopMaxThroughputPerSM(device_cap) * device_cap.num_cores() / 1000.0;
  EXPECT_NEAR(peak_tflops, 756, /*abs_error=*/1.0);
}

TEST(HardwareTypeUtilsTest, A100PeakComputTFlops) {
  DeviceCapabilities device_cap;
  // For NVIDIA A100 SXM4 80 GB, according to:
  // https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
  // https://www.techpowerup.com/gpu-specs/a100-sxm4-80-gb.c3746
  device_cap.set_clock_rate_in_ghz(1.410);
  device_cap.set_num_cores(108);
  device_cap.set_memory_size_in_bytes(
      tsl::profiler::GibiToGiga(tsl::profiler::GigaToUni(80)));
  device_cap.set_memory_bandwidth(tsl::profiler::GigaToUni(2.04 * 1024));
  device_cap.set_device_vendor("Nvidia");
  device_cap.mutable_compute_capability()->set_major(8);
  device_cap.mutable_compute_capability()->set_minor(0);

  double peak_tflops =
      GetFlopMaxThroughputPerSM(device_cap) * device_cap.num_cores() / 1000.0;
  EXPECT_NEAR(peak_tflops, 312, /*abs_error=*/1.0);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
