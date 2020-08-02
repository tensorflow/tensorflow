/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/profiling/profile_summary_formatter.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace profiling {

namespace {

TEST(SummaryWriterTest, SummaryOptionStdOut) {
  ProfileSummaryDefaultFormatter writer;
  tensorflow::StatSummarizerOptions options = writer.GetStatSummarizerOptions();
  EXPECT_EQ(options.show_summary, false);
  EXPECT_EQ(options.show_memory, false);
  EXPECT_EQ(options.format_as_csv, false);
}

TEST(SummaryWriterTest, SummaryOptionCSV) {
  ProfileSummaryCSVFormatter writer;
  tensorflow::StatSummarizerOptions options = writer.GetStatSummarizerOptions();
  EXPECT_EQ(options.show_summary, false);
  EXPECT_EQ(options.show_memory, false);
  EXPECT_EQ(options.format_as_csv, true);
}
TEST(SummaryWriterTest, EmptyOutputString) {
  ProfileSummaryDefaultFormatter writer;
  std::string output = writer.GetOutputString(
      std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>(),
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()));
  EXPECT_EQ(output.size(), 0);
}

TEST(SummaryWriterTest, EmptyShortSummary) {
  ProfileSummaryDefaultFormatter writer;
  std::string output = writer.GetShortSummary(
      std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>(),
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()));
  EXPECT_EQ(output.size(), 0);
}

TEST(SummaryWriterTest, SingleSubgraphOutputString) {
  ProfileSummaryDefaultFormatter writer;
  std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>
      stats_calculator_map;
  stats_calculator_map[0] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  std::string output = writer.GetOutputString(
      stats_calculator_map,
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()));
  ASSERT_TRUE(output.find("Run Order") != std::string::npos);
  ASSERT_TRUE(output.find("Top by Computation Time") != std::string::npos);
  ASSERT_TRUE(output.find("Top by Memory Use") == std::string::npos);
  ASSERT_TRUE(output.find("Summary by node type") != std::string::npos);
  ASSERT_TRUE(output.find("nodes observed") != std::string::npos);
  ASSERT_TRUE(output.find("Primary graph") == std::string::npos);
  ASSERT_TRUE(output.find("Subgraph") == std::string::npos);
  ASSERT_TRUE(output.find("Delegate internal") == std::string::npos);
}

TEST(SummaryWriterTest, SingleSubgraphShortSummary) {
  ProfileSummaryDefaultFormatter writer;
  std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>
      stats_calculator_map;
  stats_calculator_map[0] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  std::string output = writer.GetShortSummary(
      stats_calculator_map,
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()));
  ASSERT_TRUE(output.find("Run Order") == std::string::npos);
  ASSERT_TRUE(output.find("Top by Computation Time") == std::string::npos);
  ASSERT_TRUE(output.find("Top by Memory Use") == std::string::npos);
  ASSERT_TRUE(output.find("Summary by node type") == std::string::npos);
  ASSERT_TRUE(output.find("nodes observed") != std::string::npos);
  ASSERT_TRUE(output.find("Primary graph") == std::string::npos);
  ASSERT_TRUE(output.find("Subgraph") == std::string::npos);
  ASSERT_TRUE(output.find("Delegate internal") == std::string::npos);
}

TEST(SummaryWriterTest, MultiSubgraphOutputString) {
  ProfileSummaryDefaultFormatter writer;
  std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>
      stats_calculator_map;
  stats_calculator_map[0] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  stats_calculator_map[1] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  std::string output = writer.GetOutputString(
      stats_calculator_map,
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()));
  ASSERT_TRUE(output.find("Primary graph") != std::string::npos);
  ASSERT_TRUE(output.find("Subgraph") != std::string::npos);
  ASSERT_TRUE(output.find("Delegate internal") == std::string::npos);
}

TEST(SummaryWriterTest, MultiSubgraphShortSummary) {
  ProfileSummaryDefaultFormatter writer;
  std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>
      stats_calculator_map;
  stats_calculator_map[0] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  stats_calculator_map[1] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  std::string output = writer.GetShortSummary(
      stats_calculator_map,
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()));
  ASSERT_TRUE(output.find("Primary graph") != std::string::npos);
  ASSERT_TRUE(output.find("Subgraph") != std::string::npos);
  ASSERT_TRUE(output.find("Delegate internal") == std::string::npos);
}

TEST(SummaryWriterTest, DelegationOutputString) {
  ProfileSummaryDefaultFormatter writer;
  auto delegate_stats_calculator =
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions());
  delegate_stats_calculator.UpdateRunTotalUs(1);
  std::string output = writer.GetOutputString(
      std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>(),
      delegate_stats_calculator);
  ASSERT_TRUE(output.find("Primary graph") == std::string::npos);
  ASSERT_TRUE(output.find("Subgraph") == std::string::npos);
  ASSERT_TRUE(output.find("Delegate internal") != std::string::npos);
}

TEST(SummaryWriterTest, DelegationShortSummary) {
  ProfileSummaryDefaultFormatter writer;
  auto delegate_stats_calculator =
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions());
  delegate_stats_calculator.UpdateRunTotalUs(1);
  std::string output = writer.GetShortSummary(
      std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>(),
      delegate_stats_calculator);
  ASSERT_TRUE(output.find("Primary graph") == std::string::npos);
  ASSERT_TRUE(output.find("Subgraph") == std::string::npos);
  ASSERT_TRUE(output.find("Delegate internal") != std::string::npos);
}

}  // namespace
}  // namespace profiling
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
