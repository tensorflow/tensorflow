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

#include <map>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"

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
  ASSERT_TRUE(absl::StrContains(output, "Run Order"));
  ASSERT_TRUE(absl::StrContains(output, "Top by Computation Time"));
  ASSERT_TRUE(!absl::StrContains(output, "Top by Memory Use"));
  ASSERT_TRUE(absl::StrContains(output, "Summary by node type"));
  ASSERT_TRUE(absl::StrContains(output, "nodes observed"));
  ASSERT_TRUE(!absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(!absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(!absl::StrContains(output, "Delegate internal"));
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
  ASSERT_TRUE(!absl::StrContains(output, "Run Order"));
  ASSERT_TRUE(!absl::StrContains(output, "Top by Computation Time"));
  ASSERT_TRUE(!absl::StrContains(output, "Top by Memory Use"));
  ASSERT_TRUE(!absl::StrContains(output, "Summary by node type"));
  ASSERT_TRUE(absl::StrContains(output, "nodes observed"));
  ASSERT_TRUE(!absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(!absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(!absl::StrContains(output, "Delegate internal"));
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
  ASSERT_TRUE(absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(!absl::StrContains(output, "Delegate internal"));
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
  ASSERT_TRUE(absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(!absl::StrContains(output, "Delegate internal"));
}

TEST(SummaryWriterTest, DelegationOutputString) {
  ProfileSummaryDefaultFormatter writer;
  auto delegate_stats_calculator =
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions());
  delegate_stats_calculator.UpdateRunTotalUs(1);
  std::string output = writer.GetOutputString(
      std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>(),
      delegate_stats_calculator);
  ASSERT_TRUE(!absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(!absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(absl::StrContains(output, "Delegate internal"));
}

TEST(SummaryWriterTest, DelegationShortSummary) {
  ProfileSummaryDefaultFormatter writer;
  auto delegate_stats_calculator =
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions());
  delegate_stats_calculator.UpdateRunTotalUs(1);
  std::string output = writer.GetShortSummary(
      std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>(),
      delegate_stats_calculator);
  ASSERT_TRUE(!absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(!absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(absl::StrContains(output, "Delegate internal"));
}

}  // namespace
}  // namespace profiling
}  // namespace tflite
