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

#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(ConvertXPlaneToOpStats, PerfEnv) {
  XSpace xspace;
  constexpr double kMaxError = 0.01;
  constexpr int kClockRateKHz = 1530000;
  constexpr int kCoreCount = 80;
  constexpr uint64 kMemoryBandwidthBytesPerSecond = 900 * 1e9;
  // Volta.
  constexpr int kComputeCapMajor = 7;
  constexpr int kComputeCapMinor = 0;

  XPlaneBuilder device_plane(xspace.add_planes());
  device_plane.SetName(absl::StrCat(kGpuPlanePrefix, ":0"));
  device_plane.ParseAndAddStatValue(
      *device_plane.GetOrCreateStatMetadata("clock_rate"),
      absl::StrCat(kClockRateKHz));
  device_plane.ParseAndAddStatValue(
      *device_plane.GetOrCreateStatMetadata("core_count"),
      absl::StrCat(kCoreCount));
  device_plane.ParseAndAddStatValue(
      *device_plane.GetOrCreateStatMetadata("memory_bandwidth"),
      absl::StrCat(kMemoryBandwidthBytesPerSecond));
  device_plane.ParseAndAddStatValue(
      *device_plane.GetOrCreateStatMetadata("compute_cap_major"),
      absl::StrCat(kComputeCapMajor));
  device_plane.ParseAndAddStatValue(
      *device_plane.GetOrCreateStatMetadata("compute_cap_minor"),
      absl::StrCat(kComputeCapMinor));

  OpStats op_stats = ConvertXSpaceToOpStats(xspace);
  const PerfEnv& perf_env = op_stats.perf_env();
  EXPECT_NEAR(141, perf_env.peak_tera_flops_per_second(), kMaxError);
  EXPECT_NEAR(900, perf_env.peak_hbm_bw_giga_bytes_per_second(), kMaxError);
  EXPECT_NEAR(156.67, perf_env.ridge_point(), kMaxError);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
