/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_dynamic_histogram.h"

#include <cstdint>
#include <limits>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace odml {
namespace {

using ::testing::ElementsAre;

TEST(DynamicHistogramTest, InitializesFromFirstBatchWithPadding) {
  DynamicHistogram hist(/*max_bins=*/4);
  EXPECT_TRUE(hist.counts().empty());

  const float data[] = {-5.0f, 5.0f};
  hist.Add(data, 2, -5.0f, 5.0f);

  // Range [-5, 5] is padded by 10% on each side ([-6, 6]) and split into 4 bins
  // of width 3.0.
  EXPECT_DOUBLE_EQ(hist.lower_bound(), -6.0);
  EXPECT_DOUBLE_EQ(hist.bin_width(), 3.0);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 0, 0, 1));

  // A subsequent in-bounds batch (including exact boundary values `-6.0` and
  // `6.0`) bins into existing buckets without expanding.
  const float in_bounds[] = {-6.0f, 0.0f, 6.0f};
  hist.Add(in_bounds, 3, -6.0f, 6.0f);
  EXPECT_DOUBLE_EQ(hist.lower_bound(), -6.0);
  EXPECT_DOUBLE_EQ(hist.bin_width(), 3.0);
  EXPECT_THAT(hist.counts(), ElementsAre(2, 0, 1, 2));
}

TEST(DynamicHistogramTest, ExpandRightDoublesBinWidth) {
  DynamicHistogram hist(/*max_bins=*/4);
  const float data1[] = {-5.0f, 5.0f};
  hist.Add(data1, 2, -5.0f, 5.0f);

  const float data2[] = {7.0f};
  hist.Add(data2, 1, 7.0f, 7.0f);
  EXPECT_DOUBLE_EQ(hist.lower_bound(), -6.0);
  EXPECT_DOUBLE_EQ(hist.bin_width(), 6.0);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 1, 1));

  // One bin remains free (`max_bins = 4`), so expanding left to `-10.0` adds
  // one bin without doubling `bin_width`.
  const float data3[] = {-10.0f};
  hist.Add(data3, 1, -10.0f, -10.0f);
  EXPECT_DOUBLE_EQ(hist.lower_bound(), -12.0);
  EXPECT_DOUBLE_EQ(hist.bin_width(), 6.0);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 1, 1, 1));
}

TEST(DynamicHistogramTest, ExpandLeftDoublesBinWidth) {
  DynamicHistogram hist(/*max_bins=*/4);
  const float data1[] = {-5.0f, 5.0f};
  hist.Add(data1, 2, -5.0f, 5.0f);

  const float data2[] = {-8.0f};
  hist.Add(data2, 1, -8.0f, -8.0f);
  EXPECT_DOUBLE_EQ(hist.lower_bound(), -12.0);
  EXPECT_DOUBLE_EQ(hist.bin_width(), 6.0);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 1, 1));

  // One bin remains free (`max_bins = 4`), so expanding right to `10.0` adds
  // one bin without doubling `bin_width`.
  const float data3[] = {10.0f};
  hist.Add(data3, 1, 10.0f, 10.0f);
  EXPECT_DOUBLE_EQ(hist.lower_bound(), -12.0);
  EXPECT_DOUBLE_EQ(hist.bin_width(), 6.0);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 1, 1, 1));
}

TEST(DynamicHistogramTest, InferredBinWidthAndGoldenParityWithPython) {
  DynamicHistogram hist(/*max_bins=*/8);
  const float batch1[] = {-2.0f, 0.0f, 2.0f};
  hist.Add(batch1, 3, -2.0f, 2.0f);

  EXPECT_NEAR(hist.lower_bound(), -2.4, 1e-9);
  EXPECT_NEAR(hist.bin_width(), 0.6, 1e-9);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 0, 0, 0, 1, 0, 0, 1));

  // Second batch forces left expansion + doubling, then right expansion +
  // doubling (with odd-length padding before compaction).
  const float batch2[] = {-5.0f, 5.0f};
  hist.Add(batch2, 2, -5.0f, 5.0f);

  EXPECT_NEAR(hist.lower_bound(), -6.0, 1e-9);
  EXPECT_NEAR(hist.bin_width(), 2.4, 1e-9);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 1, 1, 1, 1));
}

TEST(DynamicHistogramTest, ZeroRangeFirstAdd) {
  DynamicHistogram hist(/*max_bins=*/4);
  const float data[] = {3.0f, 3.0f, 3.0f};
  hist.Add(data, 3, 3.0f, 3.0f);

  // Zero-range input at 3.0 is padded by 1e-4 on each side to [2.9999, 3.0001],
  // giving bin width 2e-4 / 4 = 5e-5.
  EXPECT_NEAR(hist.lower_bound(), 2.9999, 1e-9);
  EXPECT_NEAR(hist.bin_width(), 5e-5, 1e-9);
  EXPECT_THAT(hist.counts(), ElementsAre(0, 0, 3, 0));

  // When 2e-4 / max_bins < 1e-5, bin_width clamps to 1e-5.
  DynamicHistogram clamped_hist(/*max_bins=*/100);
  clamped_hist.Add(data, 3, 3.0f, 3.0f);
  EXPECT_DOUBLE_EQ(clamped_hist.bin_width(), 1e-5);
}

TEST(DynamicHistogramTest, FiltersNonFiniteAndOutOfRangeFloats) {
  DynamicHistogram hist(/*max_bins=*/10);
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float data[] = {0.5f, kNaN, 1.5f, kInf, 2.5f, -kInf, -10.0f, 10.0f};
  hist.Add(data, 8, /*min_val=*/0.5f, /*max_val=*/2.5f);

  // Only the three values inside [0.5, 2.5] are counted.
  EXPECT_EQ(hist.counts().front(), 1);
  EXPECT_EQ(hist.counts().back(), 1);
  const int64_t total =
      std::accumulate(hist.counts().begin(), hist.counts().end(), int64_t{0});
  EXPECT_EQ(total, 3);
}

TEST(DynamicHistogramTest, ExpandAcrossHugeDynamicRange) {
  // First batch clamps `bin_width` to 1e-5; second batch at +-1e20 makes
  // `(lower_bound - d_min) / bin_width` (~1e25) exceed 2^63 before compaction.
  DynamicHistogram hist(/*max_bins=*/4);
  const float batch1[] = {0.0f, 0.0f};
  hist.Add(batch1, 2, 0.0f, 0.0f);

  const float batch2[] = {-1e20f, 1e20f};
  hist.Add(batch2, 2, -1e20f, 1e20f);

  EXPECT_LE(hist.counts().size(), 4u);
  EXPECT_LE(hist.lower_bound(), -1e20);
  const double upper_bound =
      hist.lower_bound() + hist.counts().size() * hist.bin_width();
  EXPECT_GE(upper_bound, 1e20);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 0, 2, 1));
}

}  // namespace
}  // namespace odml
