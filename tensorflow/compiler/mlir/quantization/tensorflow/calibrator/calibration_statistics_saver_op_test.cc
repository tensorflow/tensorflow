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

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::stablehlo::quantization::CalibrationOptions;
using ::tensorflow::calibrator::CalibrationStatistics;
using ::tensorflow::calibrator::CalibrationStatisticsMap;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Key;
using ::testing::SizeIs;
using ::tsl::testing::StatusIs;

class CalibrationStatisticsSaverTest : public OpsTestBase {};

TEST_F(CalibrationStatisticsSaverTest, MissingOutputPath) {
  std::vector<std::string> ids{"1"};
  std::vector<int32_t> calibration_methods{
      CalibrationOptions::CALIBRATION_METHOD_AVERAGE_MIN_MAX};

  std::vector<NodeDefBuilder::NodeOut> inputs;
  inputs.emplace_back("min", 0, DT_FLOAT);
  inputs.emplace_back("max", 0, DT_FLOAT);

  TF_CHECK_OK(NodeDefBuilder("op", "CalibrationStatisticsSaver")
                  .Input(inputs)
                  .Attr("ids", ids)
                  .Attr("calibration_methods", calibration_methods)
                  .Finalize(node_def()));
  ASSERT_THAT(InitOp(),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("NodeDef missing attr 'output_file_path'")));
}

TEST_F(CalibrationStatisticsSaverTest, WrongNumInputs) {
  std::vector<std::string> ids{"1"};
  std::vector<int32_t> calibration_methods{
      CalibrationOptions::CALIBRATION_METHOD_AVERAGE_MIN_MAX};

  std::vector<NodeDefBuilder::NodeOut> inputs;
  inputs.emplace_back("min", 0, DT_FLOAT);
  inputs.emplace_back("max", 0, DT_FLOAT);

  TF_CHECK_OK(NodeDefBuilder("op", "CalibrationStatisticsSaver")
                  .Input(inputs)
                  .Attr("ids", ids)
                  .Attr("calibration_methods", calibration_methods)
                  .Attr("output_file_path", "/tmp/statistics.pbtxt")
                  .Finalize(node_def()));
  ASSERT_THAT(InitOp(),
              StatusIs(tsl::error::ABORTED,
                       HasSubstr("The number of inputs must be  three times "
                                 "the size of the `ids` list.")));
}

TEST_F(CalibrationStatisticsSaverTest, WrongInputTypes) {
  std::vector<std::string> ids{"1"};
  std::vector<int32_t> calibration_methods{
      CalibrationOptions::CALIBRATION_METHOD_AVERAGE_MIN_MAX};

  std::vector<NodeDefBuilder::NodeOut> inputs;
  inputs.emplace_back("min", 0, DT_FLOAT);
  inputs.emplace_back("max", 0, DT_FLOAT);
  inputs.emplace_back("histogram", 0, DT_FLOAT);

  TF_CHECK_OK(NodeDefBuilder("op", "CalibrationStatisticsSaver")
                  .Input(inputs)
                  .Attr("ids", ids)
                  .Attr("calibration_methods", calibration_methods)
                  .Attr("output_file_path", "/tmp/statistics.pbtxt")
                  .Finalize(node_def()));
  ASSERT_THAT(
      InitOp(),
      StatusIs(tsl::error::ABORTED,
               HasSubstr("The input `histogram` must have int64 type")));
}

TEST_F(CalibrationStatisticsSaverTest, SimpleMinMax) {
  std::vector<std::string> ids{"1"};
  std::vector<int32_t> calibration_methods{
      CalibrationOptions::CALIBRATION_METHOD_MIN_MAX};

  std::vector<NodeDefBuilder::NodeOut> inputs;
  inputs.emplace_back("min", 0, DT_FLOAT);
  inputs.emplace_back("max", 0, DT_FLOAT);
  inputs.emplace_back("histogram", 0, DT_INT64);

  const std::string dir = testing::TmpDir();
  const std::string output_file_path = io::JoinPath(dir, "statistics.pbtxt");

  TF_CHECK_OK(NodeDefBuilder("op", "CalibrationStatisticsSaver")
                  .Input(inputs)
                  .Attr("ids", ids)
                  .Attr("calibration_methods", calibration_methods)
                  .Attr("output_file_path", output_file_path)
                  .Finalize(node_def()));
  TF_CHECK_OK(InitOp());

  AddInputFromArray<float>(TensorShape({}), {1.f});
  AddInputFromArray<float>(TensorShape({}), {5.f});
  AddInputFromArray<int64_t>(TensorShape({0}), {});

  TF_CHECK_OK(RunOpKernel());
  kernel_.reset();

  CalibrationStatisticsMap statistics_map;
  TF_CHECK_OK(
      ReadBinaryProto(Env::Default(), output_file_path, &statistics_map));
  ASSERT_THAT(statistics_map.statistics(), SizeIs(1));
  ASSERT_THAT(statistics_map.statistics(), ElementsAre(Key("1")));

  const CalibrationStatistics& stats = statistics_map.statistics().at("1");
  ASSERT_TRUE(stats.has_min_max_statistics());
  EXPECT_FLOAT_EQ(stats.min_max_statistics().global_min(), 1.f);
  EXPECT_FLOAT_EQ(stats.min_max_statistics().global_max(), 5.f);
}

TEST_F(CalibrationStatisticsSaverTest, SimpleAverageMinMax) {
  std::vector<std::string> ids{"1"};
  std::vector<int32_t> calibration_methods{
      CalibrationOptions::CALIBRATION_METHOD_AVERAGE_MIN_MAX};

  std::vector<NodeDefBuilder::NodeOut> inputs;
  inputs.emplace_back("min", 0, DT_FLOAT);
  inputs.emplace_back("max", 0, DT_FLOAT);
  inputs.emplace_back("histogram", 0, DT_INT64);

  const std::string dir = testing::TmpDir();
  const std::string output_file_path = io::JoinPath(dir, "statistics.pbtxt");

  TF_CHECK_OK(NodeDefBuilder("op", "CalibrationStatisticsSaver")
                  .Input(inputs)
                  .Attr("ids", ids)
                  .Attr("calibration_methods", calibration_methods)
                  .Attr("output_file_path", output_file_path)
                  .Finalize(node_def()));
  TF_CHECK_OK(InitOp());

  AddInputFromArray<float>(TensorShape({}), {1.f});
  AddInputFromArray<float>(TensorShape({}), {5.f});
  AddInputFromArray<int64_t>(TensorShape({0}), {});

  TF_CHECK_OK(RunOpKernel());
  kernel_.reset();

  CalibrationStatisticsMap statistics_map;
  TF_CHECK_OK(
      ReadBinaryProto(Env::Default(), output_file_path, &statistics_map));
  ASSERT_THAT(statistics_map.statistics(), SizeIs(1));
  ASSERT_THAT(statistics_map.statistics(), ElementsAre(Key("1")));

  const CalibrationStatistics& stats = statistics_map.statistics().at("1");
  ASSERT_TRUE(stats.has_average_min_max_statistics());
  EXPECT_FLOAT_EQ(stats.average_min_max_statistics().min_sum(), 1.f);
  EXPECT_FLOAT_EQ(stats.average_min_max_statistics().max_sum(), 5.f);
  EXPECT_EQ(stats.average_min_max_statistics().num_samples(), 1);
}

TEST_F(CalibrationStatisticsSaverTest, SimpleHistogram) {
  std::vector<std::string> ids{"1"};
  std::vector<int32_t> calibration_methods{
      CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE};

  std::vector<NodeDefBuilder::NodeOut> inputs;
  inputs.emplace_back("min", 0, DT_FLOAT);
  inputs.emplace_back("max", 0, DT_FLOAT);
  inputs.emplace_back("histogram", 0, DT_INT64);

  const std::string dir = testing::TmpDir();
  const std::string output_file_path = io::JoinPath(dir, "statistics.pbtxt");

  TF_CHECK_OK(NodeDefBuilder("op", "CalibrationStatisticsSaver")
                  .Input(inputs)
                  .Attr("ids", ids)
                  .Attr("calibration_methods", calibration_methods)
                  .Attr("output_file_path", output_file_path)
                  .Finalize(node_def()));
  TF_CHECK_OK(InitOp());

  AddInputFromArray<float>(TensorShape({}), {1.f});
  AddInputFromArray<float>(TensorShape({}), {5.f});
  AddInputFromArray<int64_t>(TensorShape({8}), {1, 4, 6, 7, 3, 2, 1, 0});

  TF_CHECK_OK(RunOpKernel());
  kernel_.reset();

  CalibrationStatisticsMap statistics_map;
  TF_CHECK_OK(
      ReadBinaryProto(Env::Default(), output_file_path, &statistics_map));
  ASSERT_THAT(statistics_map.statistics(), SizeIs(1));
  ASSERT_THAT(statistics_map.statistics(), ElementsAre(Key("1")));

  const CalibrationStatistics& stats = statistics_map.statistics().at("1");
  ASSERT_TRUE(stats.has_histogram_statistics());
  EXPECT_FLOAT_EQ(stats.histogram_statistics().bin_width(), 0.5f);
  EXPECT_FLOAT_EQ(stats.histogram_statistics().lower_bound(), 1.f);
  EXPECT_THAT(stats.histogram_statistics().hist_freq(),
              ElementsAre(1, 4, 6, 7, 3, 2, 1));
}

TEST_F(CalibrationStatisticsSaverTest, MultipleStats) {
  std::vector<std::string> ids{"1", "2"};
  std::vector<int32_t> calibration_methods{
      CalibrationOptions::CALIBRATION_METHOD_AVERAGE_MIN_MAX,
      CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE};

  std::vector<NodeDefBuilder::NodeOut> inputs;
  inputs.emplace_back("min", 0, DT_FLOAT);
  inputs.emplace_back("max", 0, DT_FLOAT);
  inputs.emplace_back("histogram", 0, DT_INT64);
  inputs.emplace_back("min", 0, DT_FLOAT);
  inputs.emplace_back("max", 0, DT_FLOAT);
  inputs.emplace_back("histogram", 0, DT_INT64);

  const std::string dir = testing::TmpDir();
  const std::string output_file_path = io::JoinPath(dir, "statistics.pbtxt");

  TF_CHECK_OK(NodeDefBuilder("op", "CalibrationStatisticsSaver")
                  .Input(inputs)
                  .Attr("ids", ids)
                  .Attr("calibration_methods", calibration_methods)
                  .Attr("output_file_path", output_file_path)
                  .Finalize(node_def()));
  TF_CHECK_OK(InitOp());

  AddInputFromArray<float>(TensorShape({}), {1.f});
  AddInputFromArray<float>(TensorShape({}), {5.f});
  AddInputFromArray<int64_t>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({}), {1.f});
  AddInputFromArray<float>(TensorShape({}), {5.f});
  AddInputFromArray<int64_t>(TensorShape({8}), {1, 4, 6, 7, 3, 2, 1, 0});

  TF_CHECK_OK(RunOpKernel());
  kernel_.reset();

  CalibrationStatisticsMap statistics_map;
  TF_CHECK_OK(
      ReadBinaryProto(Env::Default(), output_file_path, &statistics_map));
  ASSERT_THAT(statistics_map.statistics(), SizeIs(2));
  ASSERT_THAT(statistics_map.statistics(), Contains(Key("1")));
  ASSERT_THAT(statistics_map.statistics(), Contains(Key("2")));

  const CalibrationStatistics& stats_1 = statistics_map.statistics().at("1");
  ASSERT_TRUE(stats_1.has_average_min_max_statistics());
  EXPECT_FLOAT_EQ(stats_1.average_min_max_statistics().min_sum(), 1.f);
  EXPECT_FLOAT_EQ(stats_1.average_min_max_statistics().max_sum(), 5.f);
  EXPECT_EQ(stats_1.average_min_max_statistics().num_samples(), 1);

  const CalibrationStatistics& stats_2 = statistics_map.statistics().at("2");
  ASSERT_TRUE(stats_2.has_histogram_statistics());
  EXPECT_FLOAT_EQ(stats_2.histogram_statistics().bin_width(), 0.5f);
  EXPECT_FLOAT_EQ(stats_2.histogram_statistics().lower_bound(), 1.f);
  EXPECT_THAT(stats_2.histogram_statistics().hist_freq(),
              ElementsAre(1, 4, 6, 7, 3, 2, 1));
}

}  // namespace
}  // namespace tensorflow
