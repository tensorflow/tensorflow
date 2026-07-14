/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/calibration_parameters.h"

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

namespace stablehlo::quantization {
namespace {

// Calculates the number of bins from the range and bin width.
inline int32_t CalculateActualNumBins(const float min_value,
                                      const float max_value,
                                      const float bin_width) {
  const float lower_bound = CalculateLowerBound(min_value, bin_width);
  return std::ceil((max_value - lower_bound) / bin_width);
}

TEST(CalibrationParametersTest, CalculateBinWidthSmallerThanOne) {
  float bin_width = CalculateBinWidth(/*min_value=*/0.0, /*max_value=*/25.0,
                                      /*num_bins=*/256);
  EXPECT_FLOAT_EQ(bin_width, 0.125);
  int32_t actual_num_bins =
      CalculateActualNumBins(/*min_value=*/0.0, /*max_value=*/25.0, bin_width);
  EXPECT_EQ(actual_num_bins, 200);

  // Calculate the bin width with the actual num bins.
  float raw_bin_width = 25.0 / actual_num_bins;
  EXPECT_FLOAT_EQ(bin_width, raw_bin_width);
}

TEST(CalibrationParametersTest, CalculateBinWidthLargerThanOne) {
  float bin_width = CalculateBinWidth(/*min_value=*/0.0, /*max_value=*/360.0,
                                      /*num_bins=*/256);
  EXPECT_FLOAT_EQ(bin_width, 2.0);
  int32_t actual_num_bins =
      CalculateActualNumBins(/*min_value=*/0.0, /*max_value=*/360.0, bin_width);
  EXPECT_EQ(actual_num_bins, 180);

  // Calculate the bin width with the actual num bins.
  float raw_bin_width = 360.0 / actual_num_bins;
  EXPECT_FLOAT_EQ(bin_width, raw_bin_width);
}

TEST(CalibrationParametersTest, CalculateBinWidthDivisible) {
  float bin_width = CalculateBinWidth(/*min_value=*/0.0, /*max_value=*/256.0,
                                      /*num_bins=*/256);
  EXPECT_FLOAT_EQ(bin_width, 1.0);
  int32_t actual_num_bins =
      CalculateActualNumBins(/*min_value=*/0.0, /*max_value=*/256.0, bin_width);
  EXPECT_EQ(actual_num_bins, 256);

  // Calculate the bin width with the actual num bins.
  float raw_bin_width = 256.0 / actual_num_bins;
  EXPECT_FLOAT_EQ(bin_width, raw_bin_width);
}

TEST(CalibrationParametersTest, CalculateNumBinsDivisible) {
  int32_t num_bins = CalculateActualNumBins(
      /*min_value=*/0.0, /*max_value=*/4.0, /*bin_width=*/2.0);

  // Expect 2 bins: [0, 2), [2, 4].
  EXPECT_EQ(num_bins, 2);
}

TEST(CalibrationParametersTest, CalculateNumBinsNotDivisible) {
  int32_t num_bins = CalculateActualNumBins(
      /*min_value=*/0.0, /*max_value=*/5.0, /*bin_width=*/2.0);

  // Expect 3 bins: [0, 2), [2, 4), [4, 6].
  EXPECT_EQ(num_bins, 3);
}

TEST(CalibrationParametersTest, CalculateBinIndex) {
  int32_t bin_index = CalculateBinIndexSafe(/*value=*/3.0, /*lower_bound=*/0.0,
                                            /*bin_width=*/2.0, /*num_bins=*/2);
  EXPECT_EQ(bin_index, 1);
}

TEST(CalibrationParametersTest, CalculateBinIndexMaxValue) {
  int32_t bin_index = CalculateBinIndexSafe(/*value=*/4.0, /*lower_bound=*/0.0,
                                            /*bin_width=*/2.0, /*num_bins=*/2);
  EXPECT_EQ(bin_index, 1);
}

}  // namespace
}  // namespace stablehlo::quantization
