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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_average_min_max.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_base.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_min_max.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace calibrator {
namespace {

TEST(CalibrationStatisticsCollectorTest, SimpleMinMax) {
  auto collector = CalibrationStatisticsCollectorMinMax();

  std::vector<std::vector<float>> collect_vec;

  collect_vec.push_back({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  collect_vec.push_back({1.0f, 2.0f, 3.0f, 4.0f, 10.0f});
  collect_vec.push_back({-5.0f, 2.0f, 3.0f, 4.0f, 5.0f});

  for (auto data_vec : collect_vec) {
    collector.CalibrationStatisticsCollectorBase::Collect(
        /*data_vec=*/data_vec);
  }
  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), -5.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 10.0f);
}

TEST(CalibrationStatisticsCollectorTest, SimpleAverageMinMax) {
  auto collector = CalibrationStatisticsCollectorAverageMinMax();

  std::vector<std::vector<float>> collect_vec;

  collect_vec.push_back({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});  // min=1.0f, max=5.0f
  collect_vec.push_back(
      {1.0f, 2.0f, 3.0f, 4.0f, 10.0f});  // min=1.0f, max=10.0f
  collect_vec.push_back(
      {-5.0f, 2.0f, 3.0f, 4.0f, 5.0f});  // min=-5.0f, max=5.0f

  for (auto data_vec : collect_vec) {
    collector.CalibrationStatisticsCollectorBase::Collect(
        /*data_vec=*/data_vec);
  }
  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  // 1.0f + 1.0f - 5.0f
  EXPECT_EQ(statistics.value().average_min_max_statistics().min_sum(), -3.0f);
  // 5.0f + 10.0f + 5.0f
  EXPECT_EQ(statistics.value().average_min_max_statistics().max_sum(), 20.0f);
  // collect_vec.size()
  EXPECT_EQ(statistics.value().average_min_max_statistics().num_samples(), 3);
}

TEST(CalibrationStatisticsCollectorTest, ClearDataAndGetResultsMinMax) {
  auto collector = CalibrationStatisticsCollectorMinMax();

  std::vector<std::vector<float>> collect_vec;

  collect_vec.push_back({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  collect_vec.push_back({1.0f, 2.0f, 3.0f, 4.0f, 10.0f});
  collect_vec.push_back({-5.0f, 2.0f, 3.0f, 4.0f, 5.0f});

  for (auto data_vec : collect_vec) {
    collector.CalibrationStatisticsCollectorBase::Collect(
        /*data_vec=*/data_vec);
  }

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), -5.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 10.0f);

  collector.ClearData();
  statistics = collector.GetStatistics();
  EXPECT_FALSE(statistics.has_value());

  collect_vec.pop_back();  // pop last element
  for (auto data_vec : collect_vec) {
    collector.CalibrationStatisticsCollectorBase::Collect(
        /*data_vec=*/data_vec);
  }

  statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), 1.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 10.0f);
}

TEST(CalibrationStatisticsCollectorTest, ClearDataAndGetResultsAverageMinMax) {
  auto collector = CalibrationStatisticsCollectorAverageMinMax();

  std::vector<std::vector<float>> collect_vec;

  collect_vec.push_back({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  collect_vec.push_back({1.0f, 2.0f, 3.0f, 4.0f, 20.0f});
  collect_vec.push_back({-5.0f, 2.0f, 3.0f, 4.0f, 5.0f});

  for (auto data_vec : collect_vec) {
    collector.CalibrationStatisticsCollectorBase::Collect(
        /*data_vec=*/data_vec);
  }

  std::optional<CalibrationStatistics> statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().average_min_max_statistics().min_sum(), -3.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().max_sum(), 30.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().num_samples(), 3);

  collector.ClearData();
  statistics = collector.GetStatistics();
  EXPECT_FALSE(statistics.has_value());

  collect_vec.pop_back();  // pop last element
  for (auto data_vec : collect_vec) {
    collector.CalibrationStatisticsCollectorBase::Collect(
        /*data_vec=*/data_vec);
  }

  statistics = collector.GetStatistics();

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().average_min_max_statistics().min_sum(), 2.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().max_sum(), 25.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().num_samples(), 2);
}
}  // namespace
}  // namespace calibrator
}  // namespace tensorflow
