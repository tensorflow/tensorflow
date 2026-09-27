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

#include "xla/tools/cost_model/gpu_bandwidth_benchmark.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::Gt;
using ::testing::Not;

TEST(GpuBandwidthBenchmarkTest, FormatBandwidthTableEmpty) {
  EXPECT_EQ(FormatBandwidthTable({}),
            "DMA Size (Bytes)    Bandwidth Fraction\n"
            "----------------    ------------------\n");
}

TEST(GpuBandwidthBenchmarkTest, FormatBandwidthTableMultipleEntries) {
  const std::vector<BandwidthEntry> entries = {
      {8192, 0.00043418f},
      {16384, 0.00092645f},
      {32768, 0.00184066f},
      {8589934592LL, 1.0f},
  };
  EXPECT_EQ(FormatBandwidthTable(entries),
            "DMA Size (Bytes)    Bandwidth Fraction\n"
            "----------------    ------------------\n"
            "            8192            0.00043418\n"
            "           16384            0.00092645\n"
            "           32768            0.00184066\n"
            "      8589934592            1.00000000\n");
}

TEST(GpuBandwidthBenchmarkTest, GetPeakBandwidthValidDevice) {
  const absl::StatusOr<double> peak_bw =
      GetPeakBandwidthBytesPerSec(/*device_id=*/0);
  if (!peak_bw.ok()) {
    GTEST_SKIP() << "No GPU device available: " << peak_bw.status();
  }
  EXPECT_THAT(peak_bw, IsOkAndHolds(Gt(0.0)));
}

TEST(GpuBandwidthBenchmarkTest, GetPeakBandwidthInvalidDevice) {
  EXPECT_THAT(GetPeakBandwidthBytesPerSec(/*device_id=*/-1), Not(IsOk()));
  EXPECT_THAT(GetPeakBandwidthBytesPerSec(/*device_id=*/9999), Not(IsOk()));
}

}  // namespace
}  // namespace xla::gpu
