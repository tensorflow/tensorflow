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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace odml {
namespace {

using ::testing::ElementsAre;

TEST(DynamicHistogramTest, InitializationWithExplicitBinWidth) {
  DynamicHistogram hist(/*max_bins=*/10, /*initial_bin_width=*/0.1);
  EXPECT_FALSE(hist.initialized());

  const float data[] = {0.0f, 0.5f};
  hist.Add(data, 2);
  EXPECT_TRUE(hist.initialized());
  EXPECT_DOUBLE_EQ(hist.bin_width(), 0.1);
  EXPECT_DOUBLE_EQ(hist.lower_bound(), 0.0);
  EXPECT_EQ(hist.counts().size(), 5);
  EXPECT_FLOAT_EQ(hist.global_min(), 0.0f);
  EXPECT_FLOAT_EQ(hist.global_max(), 0.5f);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 0, 0, 0, 1));
}

TEST(DynamicHistogramTest, AddWithinRange) {
  DynamicHistogram hist(/*max_bins=*/10, /*initial_bin_width=*/1.0);
  const float data[] = {0.5f, 1.5f, 2.5f};
  hist.Add(data, 3);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 2));
}

TEST(DynamicHistogramTest, ExpandRight) {
  DynamicHistogram hist(/*max_bins=*/10, /*initial_bin_width=*/1.0);
  const float data1[] = {0.0f};
  hist.Add(data1, 1);
  EXPECT_EQ(hist.counts().size(), 1);

  const float data2[] = {2.5f};
  hist.Add(data2, 1);
  EXPECT_DOUBLE_EQ(hist.lower_bound(), 0.0);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 0, 1));
}

TEST(DynamicHistogramTest, ExpandLeft) {
  DynamicHistogram hist(/*max_bins=*/10, /*initial_bin_width=*/1.0);
  const float data1[] = {2.0f};
  hist.Add(data1, 1);

  const float data2[] = {0.5f};
  hist.Add(data2, 1);
  EXPECT_DOUBLE_EQ(hist.lower_bound(), 0.0);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 0, 1));
}

TEST(DynamicHistogramTest, ExpandLeftAndRight) {
  DynamicHistogram hist(/*max_bins=*/10, /*initial_bin_width=*/1.0);
  const float data1[] = {2.0f};
  hist.Add(data1, 1);
  EXPECT_EQ(hist.counts().size(), 1);
  EXPECT_DOUBLE_EQ(hist.lower_bound(), 2.0);

  const float data2[] = {0.0f, 5.0f};
  hist.Add(data2, 2);
  EXPECT_DOUBLE_EQ(hist.lower_bound(), 0.0);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 0, 1, 0, 1));
}

TEST(DynamicHistogramTest, DoublingOnLeft) {
  DynamicHistogram hist(/*max_bins=*/4, /*initial_bin_width=*/1.0);
  const float data1[] = {2.0f};
  hist.Add(data1, 1);

  const float data2[] = {-2.0f};
  hist.Add(data2, 1);
  EXPECT_DOUBLE_EQ(hist.bin_width(), 2.0);
  EXPECT_DOUBLE_EQ(hist.lower_bound(), -2.0);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 0, 1));
}

TEST(DynamicHistogramTest, InferredBinWidthAndGoldenParityWithPython) {
  DynamicHistogram hist(/*max_bins=*/8);
  const float batch1[] = {-2.0f, 0.0f, 2.0f};
  hist.Add(batch1, 3);

  EXPECT_TRUE(hist.initialized());
  EXPECT_NEAR(hist.lower_bound(), -2.4, 1e-9);
  EXPECT_NEAR(hist.bin_width(), 0.6, 1e-9);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 0, 0, 0, 1, 0, 0, 1));

  // Second batch forces left expansion + doubling, then right expansion +
  // doubling (with odd-length padding before compaction).
  const float batch2[] = {-5.0f, 5.0f};
  hist.Add(batch2, 2);

  EXPECT_NEAR(hist.lower_bound(), -6.0, 1e-9);
  EXPECT_NEAR(hist.bin_width(), 2.4, 1e-9);
  EXPECT_FLOAT_EQ(hist.global_min(), -5.0f);
  EXPECT_FLOAT_EQ(hist.global_max(), 5.0f);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 1, 1, 1, 1));
}

TEST(DynamicHistogramTest, ZeroRangeFirstAdd) {
  DynamicHistogram hist(/*max_bins=*/4);
  const float data[] = {3.0f, 3.0f, 3.0f};
  hist.Add(data, 3);

  EXPECT_TRUE(hist.initialized());
  EXPECT_NEAR(hist.lower_bound(), 3.0 - 1e-4, 1e-9);
  EXPECT_NEAR(hist.bin_width(), 2e-4 / 4.0, 1e-9);
  int64_t total = 0;
  for (int64_t c : hist.counts()) {
    total += c;
  }
  EXPECT_EQ(total, 3);
}

TEST(DynamicHistogramTest, InfNaNAndValidRangeFiltering) {
  DynamicHistogram hist(/*max_bins=*/10, /*initial_bin_width=*/1.0);
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float data[] = {0.5f, kNaN, 1.5f, kInf, 2.5f, -kInf, -3.1e38f, 3.1e38f};
  hist.Add(data, 8);

  EXPECT_TRUE(hist.initialized());
  EXPECT_FLOAT_EQ(hist.global_min(), 0.5f);
  EXPECT_FLOAT_EQ(hist.global_max(), 2.5f);
  EXPECT_THAT(hist.counts(), ElementsAre(1, 2));
}

TEST(DynamicHistogramTest, AllInvalidLeavesHistogramUninitialized) {
  DynamicHistogram hist(/*max_bins=*/10);
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float data[] = {-kInf, kInf, kNaN, -3.1e38f, 3.1e38f};
  hist.Add(data, 5);

  EXPECT_FALSE(hist.initialized());
}

TEST(DynamicHistogramTest, IntegralValuesIgnoreValidRange) {
  DynamicHistogram hist(/*max_bins=*/10, /*initial_bin_width=*/1.0);
  const int32_t data[] = {0, 2, 5};
  hist.Add(data, 3, /*valid_min=*/1.0f, /*valid_max=*/3.0f);

  EXPECT_FLOAT_EQ(hist.global_min(), 0.0f);
  EXPECT_FLOAT_EQ(hist.global_max(), 5.0f);
  // The max value lands in the last bin via clamping.
  EXPECT_THAT(hist.counts(), ElementsAre(1, 0, 1, 0, 1));
}

TEST(DynamicHistogramTest, NullOrEmptyInputIsNoOp) {
  DynamicHistogram hist(/*max_bins=*/10);
  hist.Add(static_cast<const float*>(nullptr), 3);
  const float data[] = {1.0f};
  hist.Add(data, 0);

  EXPECT_FALSE(hist.initialized());
}

}  // namespace
}  // namespace odml
