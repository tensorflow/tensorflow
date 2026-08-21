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

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Gt;
using ::testing::Not;
using ::testing::SizeIs;

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
      GetPeakBandwidthBytesPerSec(/*ordinal=*/0);
  if (!peak_bw.ok()) {
    GTEST_SKIP() << "No GPU device available: " << peak_bw.status();
  }
  EXPECT_THAT(peak_bw, IsOkAndHolds(Gt(0.0)));
}

TEST(GpuBandwidthBenchmarkTest, GetPeakBandwidthInvalidDevice) {
  EXPECT_THAT(GetPeakBandwidthBytesPerSec(/*ordinal=*/-1), Not(IsOk()));
  EXPECT_THAT(GetPeakBandwidthBytesPerSec(/*ordinal=*/9999), Not(IsOk()));
}

TEST(GpuBandwidthBenchmarkTest, MeasureBandwidthTableInvalidArguments) {
  EXPECT_THAT(MeasureBandwidthTable(/*ordinal=*/-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(MeasureBandwidthTable(/*ordinal=*/0, /*min_exp=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      MeasureBandwidthTable(/*ordinal=*/0, /*min_exp=*/20, /*max_exp=*/13),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      MeasureBandwidthTable(/*ordinal=*/0, /*min_exp=*/13, /*max_exp=*/62),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(MeasureBandwidthTable(/*ordinal=*/0, /*min_exp=*/13,
                                    /*max_exp=*/15, /*warmup_runs=*/-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(MeasureBandwidthTable(/*ordinal=*/0, /*min_exp=*/13,
                                    /*max_exp=*/15, /*warmup_runs=*/2,
                                    /*measurement_runs=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GpuBandwidthBenchmarkTest, MeasureBandwidthTableScalesWithTransferSize) {
  constexpr int kMinExp = 13;
  constexpr int kMaxExp = 27;
  const absl::StatusOr<std::vector<BandwidthEntry>> entries =
      MeasureBandwidthTable(/*ordinal=*/0, kMinExp, kMaxExp,
                            /*warmup_runs=*/2, /*measurement_runs=*/5);
  if (!entries.ok() &&
      (entries.status().code() == absl::StatusCode::kNotFound ||
       entries.status().code() == absl::StatusCode::kUnavailable)) {
    GTEST_SKIP() << "No GPU platform or device available: " << entries.status();
  }
  ASSERT_THAT(entries, IsOkAndHolds(SizeIs(kMaxExp - kMinExp + 1)));
  EXPECT_EQ(entries->front().dma_size_bytes, int64_t{1} << kMinExp);
  EXPECT_EQ(entries->back().dma_size_bytes, int64_t{1} << kMaxExp);
  for (const BandwidthEntry& entry : *entries) {
    EXPECT_GT(entry.bandwidth_fraction, 0.0f);
    EXPECT_LE(entry.bandwidth_fraction, 1.0f);
  }
  for (size_t i = 1; i < entries->size(); ++i) {
    EXPECT_EQ((*entries)[i].dma_size_bytes,
              2 * (*entries)[i - 1].dma_size_bytes);
    EXPECT_GT((*entries)[i].bandwidth_fraction,
              (*entries)[i - 1].bandwidth_fraction);
  }
}

}  // namespace
}  // namespace xla::gpu
