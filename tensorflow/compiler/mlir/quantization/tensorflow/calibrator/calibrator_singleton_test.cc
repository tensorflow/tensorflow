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
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace calibrator {
namespace {

using ::stablehlo::quantization::CalibrationOptions;

TEST(CalibratorSingletonTest, SimpleMinMax) {
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_MIN_MAX);

  CalibratorSingleton::Report(/*id=*/"1", /*min=*/1.0f, /*max=*/5.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  std::optional<CalibrationStatistics> statistics =
      CalibratorSingleton::GetStatistics(/*id=*/"1");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), 1.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 5.0f);

  CalibratorSingleton::Report(/*id=*/"1", /*min=*/1.0f, /*max=*/10.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  statistics = CalibratorSingleton::GetStatistics(/*id=*/"1");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), 1.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 10.0f);

  CalibratorSingleton::Report(/*id=*/"1", /*min=*/-5.0f, /*max=*/5.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  statistics = CalibratorSingleton::GetStatistics(/*id=*/"1");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), -5.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 10.0f);
}

TEST(CalibratorSingletonTest, DifferentSessions) {
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_MIN_MAX);

  CalibratorSingleton::Report(/*id=*/"2", /*min=*/1.0f, /*max=*/5.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  std::optional<CalibrationStatistics> statistics =
      CalibratorSingleton::GetStatistics(/*id=*/"2");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), 1.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 5.0f);

  CalibratorSingleton::Report(/*id=*/"2", /*min=*/1.0f, /*max=*/10.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  statistics = CalibratorSingleton::GetStatistics(/*id=*/"2");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), 1.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 10.0f);

  CalibratorSingleton::Report(/*id=*/"3", /*min=*/-5.0f, /*max=*/5.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  statistics = CalibratorSingleton::GetStatistics(/*id=*/"3");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), -5.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 5.0f);
}

TEST(CalibratorSingletonTest, ClearAndGetEmptyResult) {
  std::vector<std::vector<float>> report_vec;
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_MIN_MAX);

  report_vec.push_back({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  report_vec.push_back({1.0f, 2.0f, 3.0f, 4.0f, 10.0f});

  CalibratorSingleton::Report(/*id=*/"4", /*min=*/1.0f, /*max=*/5.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  std::optional<CalibrationStatistics> statistics =
      CalibratorSingleton::GetStatistics(/*id=*/"4");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), 1.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 5.0f);

  CalibratorSingleton::ClearData(/*id=*/"4");
  statistics = CalibratorSingleton::GetStatistics(/*id=*/"4");

  EXPECT_FALSE(statistics.has_value());
}

TEST(CalibratorSingletonTest, ClearDataAndGetResults) {
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_MIN_MAX);

  CalibratorSingleton::Report(/*id=*/"5", /*min=*/1.0f, /*max=*/5.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  std::optional<CalibrationStatistics> statistics =
      CalibratorSingleton::GetStatistics(/*id=*/"5");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), 1.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 5.0f);

  CalibratorSingleton::Report(/*id=*/"6", /*min=*/1.0f, /*max=*/10.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  statistics = CalibratorSingleton::GetStatistics(/*id=*/"6");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), 1.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 10.0f);

  CalibratorSingleton::ClearData(/*id=*/"5");
  statistics = CalibratorSingleton::GetStatistics(/*id=*/"5");

  EXPECT_FALSE(statistics.has_value());

  CalibratorSingleton::Report(/*id=*/"6", /*min=*/1.0f, /*max=*/10.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  statistics = CalibratorSingleton::GetStatistics(/*id=*/"6");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().min_max_statistics().global_min(), 1.0f);
  EXPECT_EQ(statistics.value().min_max_statistics().global_max(), 10.0f);
}

TEST(CalibratorSingletonTest, SimpleAverageMinMax) {
  CalibrationOptions calib_opts;
  calib_opts.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_AVERAGE_MIN_MAX);

  CalibratorSingleton::Report(/*id=*/"7", /*min=*/-10.0f, /*max=*/30.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  std::optional<CalibrationStatistics> statistics =
      CalibratorSingleton::GetStatistics(/*id=*/"7");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().average_min_max_statistics().min_sum(), -10.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().max_sum(), 30.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().num_samples(), 1);

  CalibratorSingleton::Report(/*id=*/"7", /*min=*/-20.0f, /*max=*/60.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  statistics = CalibratorSingleton::GetStatistics(/*id=*/"7");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().average_min_max_statistics().min_sum(), -30.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().max_sum(), 90.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().num_samples(), 2);

  CalibratorSingleton::Report(/*id=*/"7", /*min=*/-30.0f, /*max=*/90.0f,
                              /*histogram=*/{},
                              /*calib_opts=*/calib_opts);
  statistics = CalibratorSingleton::GetStatistics(/*id=*/"7");

  EXPECT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics.value().average_min_max_statistics().min_sum(), -60.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().max_sum(), 180.0f);
  EXPECT_EQ(statistics.value().average_min_max_statistics().num_samples(), 3);
}

}  // namespace
}  // namespace calibrator
}  // namespace tensorflow
