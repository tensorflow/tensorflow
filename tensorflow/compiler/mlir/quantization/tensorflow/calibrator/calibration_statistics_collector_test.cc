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
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_average_min_max.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_histogram.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_min_max.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace calibrator {
namespace {

using ::testing::ElementsAre;

TEST(CalibrationStatisticsCollectorTest, SimpleMinMax) {
  auto collector = CalibrationStatisticsCollectorMinMax();

  collector.Collect(
      /*min=*/1.0f, /*max=*/10.f, /*histogram=*/{});
  collector.Collect(
      /*min=*/-5.0f, /*max=*/5.f, /*histogram=*/{});

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), -5.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 10.0f);
}

TEST(CalibrationStatisticsCollectorTest, SimpleAverageMinMax) {
  auto collector = CalibrationStatisticsCollectorAverageMinMax();

  collector.Collect(
      /*min=*/1.0f, /*max=*/10.f, /*histogram=*/{});
  collector.Collect(
      /*min=*/-5.0f, /*max=*/5.f, /*histogram=*/{});

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().average_min_max_statistics().min_sum(), -4.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().max_sum(), 15.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().num_samples(), 2);
}

TEST(CalibrationStatisticsCollectorTest, ClearDataAndGetResultsMinMax) {
  auto collector = CalibrationStatisticsCollectorMinMax();

  collector.Collect(
      /*min=*/1.0f, /*max=*/10.f, /*histogram=*/{});
  collector.Collect(
      /*min=*/-5.0f, /*max=*/5.f, /*histogram=*/{});

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), -5.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 10.0f);

  collector.ClearData();
  statistics = collector.GetStatistics();
  EXPECT_FALSE(statistics.has_value());

  collector.Collect(
      /*min=*/1.0f, /*max=*/10.f, /*histogram=*/{});
  collector.Collect(
      /*min=*/2.0f, /*max=*/5.f, /*histogram=*/{});

  statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), 1.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 10.0f);
}

TEST(CalibrationStatisticsCollectorTest, ClearDataAndGetResultsAverageMinMax) {
  auto collector = CalibrationStatisticsCollectorAverageMinMax();

  collector.Collect(
      /*min=*/1.0f, /*max=*/10.f, /*histogram=*/{});
  collector.Collect(
      /*min=*/-5.0f, /*max=*/5.f, /*histogram=*/{});

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().average_min_max_statistics().min_sum(), -4.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().max_sum(), 15.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().num_samples(), 2);

  collector.ClearData();
  statistics = collector.GetStatistics();
  EXPECT_FALSE(statistics.has_value());

  collector.Collect(
      /*min=*/1.0f, /*max=*/10.f, /*histogram=*/{});

  statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().average_min_max_statistics().min_sum(), 1.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().max_sum(), 10.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().num_samples(), 1);
}

TEST(HistogramStatisticsCollectorTest, SingleBatchSimple) {
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY);
  auto collector = CalibrationStatisticsCollectorHistogram();

  collector.Collect(
      /*min=*/1.f, /*max=*/16.f, /*histogram=*/{1, 0, 3, 5, 7, 6, 5, 0});

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), 0.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  // Trailing zeros should be removed.
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(1, 0, 3, 5, 7, 6, 5));
}

TEST(HistogramStatisticsCollectorTest, AggregateSameBatchSize) {
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY);
  auto collector = CalibrationStatisticsCollectorHistogram();

  collector.Collect(
      /*min=*/1.f, /*max=*/16.f, /*histogram=*/{1, 0, 3, 5, 7, 6, 5, 1});

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), 0.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(1, 0, 3, 5, 7, 6, 5, 1));

  collector.Collect(
      /*min=*/-1.f, /*max=*/12.f, /*histogram=*/{1, 0, 1, 2, 2, 1, 1, 0});

  statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), -2.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(1, 1, 1, 5, 7, 8, 7, 5, 1));
}

TEST(HistogramStatisticsCollectorTest, AggregateSmallerBatchSizeExpandLeft) {
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY);
  auto collector = CalibrationStatisticsCollectorHistogram();

  collector.Collect(
      /*min=*/1.f, /*max=*/16.f, /*histogram=*/{1, 0, 3, 5, 7, 6, 5, 1});

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), 0.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(1, 0, 3, 5, 7, 6, 5, 1));

  collector.Collect(
      /*min=*/-1.f, /*max=*/5.f, /*histogram=*/{1, 0, 1, 2, 2, 1, 1, 0});

  statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), -2.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(1, 2, 4, 5, 5, 7, 6, 5, 1));
}

TEST(HistogramStatisticsCollectorTest, AggregateSmallerBatchSizeExpandRight) {
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY);
  auto collector = CalibrationStatisticsCollectorHistogram();

  collector.Collect(
      /*min=*/1.f, /*max=*/16.f, /*histogram=*/{1, 0, 3, 5, 7, 6, 5, 1});

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), 0.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(1, 0, 3, 5, 7, 6, 5, 1));

  collector.Collect(
      /*min=*/13.f, /*max=*/19.f, /*histogram=*/{1, 0, 1, 2, 2, 1, 1, 0});

  statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), 0.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(1, 0, 3, 5, 7, 6, 6, 2, 4, 2));
}

TEST(HistogramStatisticsCollectorTest, AggregateTinyBinWidth) {
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY);
  auto collector = CalibrationStatisticsCollectorHistogram();

  collector.Collect(
      /*min=*/1.f, /*max=*/16.f, /*histogram=*/{1, 0, 3, 5, 7, 6, 5, 1});

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), 0.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(1, 0, 3, 5, 7, 6, 5, 1));

  collector.Collect(
      /*min=*/-1.f, /*max=*/-0.99998f, /*histogram=*/{1, 0, 1, 2, 2, 1, 1, 0});

  statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), -2.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(8, 1, 0, 3, 5, 7, 6, 5, 1));
}

TEST(HistogramStatisticsCollectorTest, AggregateLargerBatchSizeExpandLeft) {
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY);
  auto collector = CalibrationStatisticsCollectorHistogram();

  collector.Collect(
      /*min=*/1.f, /*max=*/16.f, /*histogram=*/{1, 0, 3, 5, 7, 6, 5, 1});

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), 0.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(1, 0, 3, 5, 7, 6, 5, 1));

  collector.Collect(
      /*min=*/-5.f, /*max=*/5.f, /*histogram=*/{1, 2, 2, 1});

  statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), -8.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(0.5, 0.5, 1, 1, 2, 1, 3.5, 5.5, 7, 6, 5, 1));
}

TEST(HistogramStatisticsCollectorTest, AggregateLargerBatchSizeExpandRight) {
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY);
  auto collector = CalibrationStatisticsCollectorHistogram();

  collector.Collect(
      /*min=*/1.f, /*max=*/16.f, /*histogram=*/{1, 0, 3, 5, 7, 6, 5, 1});

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), 0.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(1, 0, 3, 5, 7, 6, 5, 1));

  collector.Collect(
      /*min=*/10.f, /*max=*/21.f, /*histogram=*/{1, 2, 2, 1});

  statistics = collector.GetStatistics();
  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().histogram_statistics().lower_bound(), 0.f);
  EXPECT_EQ(statistics.value().histogram_statistics().bin_width(), 2.f);
  EXPECT_THAT(statistics.value().histogram_statistics().hist_freq(),
              ElementsAre(1, 0, 3, 5, 7.5, 6.5, 6, 2, 1, 1, 0.5, 0.5));
}

}  // namespace
}  // namespace calibrator
}  // namespace tensorflow
