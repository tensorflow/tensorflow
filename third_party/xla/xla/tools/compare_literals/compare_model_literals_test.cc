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

#include "xla/tools/compare_literals/compare_model_literals.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla::compare_literals {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::SizeIs;

absl::Status WriteLiteralToFile(const Literal& literal,
                                const std::string& path) {
  LiteralProto proto = literal.ToProto();
  return tsl::WriteBinaryProto(tsl::Env::Default(), path, proto);
}

class CompareModelLiteralsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    golden_dir_ =
        tsl::io::JoinPath(testing::TempDir(), test_name, "golden_dir");
    test_dir_ = tsl::io::JoinPath(testing::TempDir(), test_name, "test_dir");
    ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(golden_dir_), IsOk());
    ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(test_dir_), IsOk());
  }

  void TearDown() override {
    int64_t undeleted_files = 0;
    int64_t undeleted_dirs = 0;
    tsl::Env::Default()
        ->DeleteRecursively(golden_dir_, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
    tsl::Env::Default()
        ->DeleteRecursively(test_dir_, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
  }

  std::string golden_dir_;
  std::string test_dir_;
};

TEST_F(CompareModelLiteralsTest, NonExistentDirectoriesReturnError) {
  EXPECT_THAT(CompareModelDirectories("/non/existent/golden", test_dir_),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Golden directory does not exist")));

  EXPECT_THAT(CompareModelDirectories(golden_dir_, "/non/existent/test"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Test directory does not exist")));
}

TEST_F(CompareModelLiteralsTest, EmptyDirectoriesReturnNoMatchingFilesError) {
  EXPECT_THAT(CompareModelDirectories(golden_dir_, test_dir_),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("No matching literal files found")));
}

TEST_F(CompareModelLiteralsTest, DisjointDirectoriesReportMissingFiles) {
  Literal lit_g = LiteralUtil::CreateR1<float>({1.0f});
  Literal lit_t = LiteralUtil::CreateR1<float>({2.0f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_g, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit_t, tsl::io::JoinPath(test_dir_, "device_0.literal_1.pb")),
              IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));
  EXPECT_THAT(result.summary.total_literals, Eq(0));
  ASSERT_THAT(result.missing_in_test, SizeIs(1));
  EXPECT_THAT(result.missing_in_test[0].literal_id, Eq(0));
  EXPECT_THAT(result.missing_in_test[0].device_id, Eq(0));
  ASSERT_THAT(result.missing_in_golden, SizeIs(1));
  EXPECT_THAT(result.missing_in_golden[0].literal_id, Eq(1));
  EXPECT_THAT(result.missing_in_golden[0].device_id, Eq(0));

  const std::string json = result.ToJson();
  EXPECT_THAT(json, HasSubstr("\"missing_in_test\""));
  EXPECT_THAT(json, HasSubstr("\"missing_in_golden\""));
}

TEST_F(CompareModelLiteralsTest, ComparesMultipleLiteralsAndDevicesCorrectly) {
  // Create literal_0 on device 0 and device 1 (Exact match)
  Literal lit0_g = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  Literal lit0_t = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit0_g,
          tsl::io::JoinPath(golden_dir_, "output.hlo_0.device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit0_t, tsl::io::JoinPath(
                              test_dir_, "output.hlo_0.device_0.literal_0.pb")),
              IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(
          lit0_g,
          tsl::io::JoinPath(golden_dir_, "output.hlo_0.device_1.literal_0.pb")),
      IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit0_t, tsl::io::JoinPath(
                              test_dir_, "output.hlo_0.device_1.literal_0.pb")),
              IsOk());

  // Create literal_1 on device 0 and device 1 (Drift on device 1)
  Literal lit1_g = LiteralUtil::CreateR1<float>({10.0f, 20.0f});
  Literal lit1_t0 = LiteralUtil::CreateR1<float>({10.0f, 20.0f});
  Literal lit1_t1 = LiteralUtil::CreateR1<float>({10.05f, 20.0f});  // 0.05 diff
  ASSERT_THAT(
      WriteLiteralToFile(
          lit1_g,
          tsl::io::JoinPath(golden_dir_, "output.hlo_0.device_0.literal_1.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(
          lit1_t0,
          tsl::io::JoinPath(test_dir_, "output.hlo_0.device_0.literal_1.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(
          lit1_g,
          tsl::io::JoinPath(golden_dir_, "output.hlo_0.device_1.literal_1.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(
          lit1_t1,
          tsl::io::JoinPath(test_dir_, "output.hlo_0.device_1.literal_1.pb")),
      IsOk());

  ModelComparisonOptions options;
  options.num_threads = 4;
  ASSERT_OK_AND_ASSIGN(
      ModelComparisonResult result,
      CompareModelDirectories(golden_dir_, test_dir_, options));

  EXPECT_THAT(result.devices, ElementsAre(0, 1));
  ASSERT_THAT(result.output_stats, SizeIs(2));

  // Verify literal_0
  const OutputLiteralStats& entry0 = result.output_stats[0];
  EXPECT_THAT(entry0.literal_index, Eq(0));
  EXPECT_THAT(entry0.literal_name, Eq("literal_0"));
  EXPECT_THAT(entry0.num_devices, Eq(2));
  EXPECT_DOUBLE_EQ(entry0.aggregated_device_stats.exact_match_pct, 100.0);
  EXPECT_DOUBLE_EQ(entry0.aggregated_device_stats.max_abs_error, 0.0);
  EXPECT_DOUBLE_EQ(entry0.aggregated_device_stats.max_rel_error, 0.0);

  // Verify literal_1
  const OutputLiteralStats& entry1 = result.output_stats[1];
  EXPECT_THAT(entry1.literal_index, Eq(1));
  EXPECT_THAT(entry1.num_devices, Eq(2));
  EXPECT_DOUBLE_EQ(entry1.aggregated_device_stats.exact_match_pct,
                   50.0);  // 1/2 match on device 1
  EXPECT_NEAR(entry1.aggregated_device_stats.max_abs_error, 0.05, 1e-4);
  ASSERT_THAT(entry1.device_stats, SizeIs(2));
  EXPECT_DOUBLE_EQ(entry1.device_stats.at(0).exact_match_pct, 100.0);
  EXPECT_DOUBLE_EQ(entry1.device_stats.at(1).exact_match_pct, 50.0);
}

TEST_F(CompareModelLiteralsTest, HandlesAsymmetricFilesGracefully) {
  Literal lit = LiteralUtil::CreateR1<float>({1.0f});
  // Both have device 0 literal 0
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
              IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(test_dir_, "device_0.literal_0.pb")),
              IsOk());
  // Only in golden: device 1 literal 0
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(golden_dir_, "device_1.literal_0.pb")),
              IsOk());
  // Only in test: device 0 literal 1
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(test_dir_, "device_0.literal_1.pb")),
              IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));

  EXPECT_THAT(result.devices, ElementsAre(0));
  ASSERT_THAT(result.output_stats, SizeIs(1));
  EXPECT_THAT(result.output_stats[0].literal_index, Eq(0));
  EXPECT_THAT(result.output_stats[0].num_devices, Eq(1));

  EXPECT_THAT(result.missing_in_test,
              ElementsAre(LiteralKey{/*literal_id=*/0, /*device_id=*/1}));
  EXPECT_THAT(result.missing_in_golden,
              ElementsAre(LiteralKey{/*literal_id=*/1, /*device_id=*/0}));
}

TEST_F(CompareModelLiteralsTest, HandlesMismatchedShapesGracefully) {
  Literal lit_g = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit_t = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_g, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit_t, tsl::io::JoinPath(test_dir_, "device_0.literal_0.pb")),
              IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));
  ASSERT_THAT(result.output_stats, SizeIs(1));
  const OutputLiteralStats& entry = result.output_stats[0];
  EXPECT_THAT(entry.failed_devices, Eq(1));
  EXPECT_DOUBLE_EQ(entry.aggregated_device_stats.exact_match_pct, 0.0);
  ASSERT_THAT(entry.device_stats, SizeIs(1));
  EXPECT_THAT(
      entry.device_stats.at(0).status,
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Shapes")));
  EXPECT_THAT(result.ToDeviceTsv(), HasSubstr("0\t0\tfalse\t"));
  EXPECT_THAT(result.ToJson(), HasSubstr("\"comparison_ok\" : false"));
  EXPECT_THAT(result.ToJson(), HasSubstr("\"error_message\" :"));
}

TEST_F(CompareModelLiteralsTest, IgnoresNonLiteralAndExtraneousFiles) {
  Literal lit = LiteralUtil::CreateR1<float>({5.0f});
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(
                           golden_dir_, "output.task_0.device_0.literal_0.pb")),
              IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(
                           test_dir_, "output.task_0.device_0.literal_0.pb")),
              IsOk());

  // Non-literal files: text logs, module protos, malformed filenames
  ASSERT_THAT(tsl::WriteStringToFile(tsl::Env::Default(),
                                     tsl::io::JoinPath(golden_dir_, "log.txt"),
                                     "execution log"),
              IsOk());
  ASSERT_THAT(tsl::WriteStringToFile(
                  tsl::Env::Default(),
                  tsl::io::JoinPath(golden_dir_, "module.pb"), "not a literal"),
              IsOk());
  ASSERT_THAT(tsl::WriteStringToFile(
                  tsl::Env::Default(),
                  tsl::io::JoinPath(golden_dir_, "literal_foo.pb"), "corrupt"),
              IsOk());
  ASSERT_THAT(
      tsl::WriteStringToFile(tsl::Env::Default(),
                             tsl::io::JoinPath(test_dir_, "test_log.txt"),
                             "execution log"),
      IsOk());
  ASSERT_THAT(
      tsl::WriteStringToFile(tsl::Env::Default(),
                             tsl::io::JoinPath(test_dir_, "test_module.pb"),
                             "not a literal"),
      IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));
  ASSERT_THAT(result.output_stats, SizeIs(1));
  EXPECT_THAT(result.output_stats[0].literal_index, Eq(0));
}

TEST_F(CompareModelLiteralsTest, AggregatesAnomaliesAcrossThreeDevices) {
  // Device 0: Exact match
  Literal lit_d0 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_d0, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(lit_d0,
                         tsl::io::JoinPath(test_dir_, "device_0.literal_0.pb")),
      IsOk());

  // Device 1: Finite drift
  Literal lit_d1_g = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit_d1_t = LiteralUtil::CreateR1<float>({1.0f, 2.1f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_d1_g, tsl::io::JoinPath(golden_dir_, "device_1.literal_0.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(lit_d1_t,
                         tsl::io::JoinPath(test_dir_, "device_1.literal_0.pb")),
      IsOk());

  // Device 2: NaN mismatch
  Literal lit_d2_g = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit_d2_t = LiteralUtil::CreateR1<float>({1.0f, std::nanf("")});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_d2_g, tsl::io::JoinPath(golden_dir_, "device_2.literal_0.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(lit_d2_t,
                         tsl::io::JoinPath(test_dir_, "device_2.literal_0.pb")),
      IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));

  EXPECT_THAT(result.devices, ElementsAre(0, 1, 2));
  ASSERT_THAT(result.output_stats, SizeIs(1));
  const OutputLiteralStats& entry = result.output_stats[0];
  EXPECT_THAT(entry.num_devices, Eq(3));
  EXPECT_THAT(entry.aggregated_device_stats.nan_mismatches, Eq(1));
  EXPECT_DOUBLE_EQ(entry.aggregated_device_stats.exact_match_pct, 50.0);
  EXPECT_TRUE(std::isinf(entry.aggregated_device_stats.max_abs_error));
  ASSERT_THAT(entry.device_stats, SizeIs(3));
  EXPECT_DOUBLE_EQ(entry.device_stats.at(0).max_abs_error, 0.0);
  EXPECT_DOUBLE_EQ(entry.device_stats.at(0).exact_match_pct, 100.0);
  EXPECT_NEAR(entry.device_stats.at(1).max_abs_error, 0.1, 1e-4);
  EXPECT_TRUE(std::isinf(entry.device_stats.at(2).max_abs_error));
  EXPECT_THAT(entry.device_stats.at(2).nan_mismatches, Eq(1));
  EXPECT_THAT(result.summary.nan_inf_mismatch_literals, Eq(1));
  EXPECT_THAT(result.summary.differing_literals, Eq(1));
}

TEST_F(CompareModelLiteralsTest, TargetDevicesFilterWorks) {
  Literal lit = LiteralUtil::CreateR1<float>({1.0f});
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
              IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(test_dir_, "device_0.literal_0.pb")),
              IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(golden_dir_, "device_1.literal_0.pb")),
              IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(test_dir_, "device_1.literal_0.pb")),
              IsOk());

  ModelComparisonOptions options;
  options.target_devices = {0};

  ASSERT_OK_AND_ASSIGN(
      ModelComparisonResult result,
      CompareModelDirectories(golden_dir_, test_dir_, options));

  EXPECT_THAT(result.devices, ElementsAre(0));
  ASSERT_THAT(result.output_stats, SizeIs(1));
  EXPECT_THAT(result.output_stats[0].num_devices, Eq(1));
}

TEST_F(CompareModelLiteralsTest, JsonAndTsvFormattingAndWriting) {
  Literal lit_g = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit_t = LiteralUtil::CreateR1<float>({1.0f, 2.1f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_g, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit_t, tsl::io::JoinPath(test_dir_, "device_0.literal_0.pb")),
              IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));

  const std::string json = result.ToJson();
  EXPECT_THAT(json, HasSubstr("\"golden_dir\" :"));
  EXPECT_THAT(json, HasSubstr("\"total_literals\" : 1"));
  EXPECT_THAT(json, HasSubstr("\"literal_0\""));
  EXPECT_THAT(json, HasSubstr("\"devices\" :"));
  EXPECT_THAT(json, HasSubstr("\"summary\" :"));
  EXPECT_THAT(json, HasSubstr("\"exact_match_literals\" : 0"));
  EXPECT_THAT(json, HasSubstr("\"differing_literals\" : 1"));

  const std::string tsv = result.ToTsv();
  EXPECT_THAT(tsv, HasSubstr("literal\tshape\ttype\telements\tdevices"));
  EXPECT_THAT(tsv, HasSubstr("0\tf32[2]\tf32\t2\t1"));

  const std::string dev_tsv = result.ToDeviceTsv();
  EXPECT_THAT(dev_tsv, HasSubstr("literal\tdevice\tcomparison_ok\terror_"
                                 "message\tshape\ttype\telements"));
  EXPECT_THAT(dev_tsv, HasSubstr("0\t0\ttrue\t-\tf32[2]\tf32\t2"));

  const std::string json_path =
      tsl::io::JoinPath(test_dir_, "nested", "out.json");
  const std::string tsv_path =
      tsl::io::JoinPath(test_dir_, "nested", "out.tsv");
  const std::string dev_tsv_path =
      tsl::io::JoinPath(test_dir_, "nested", "out_dev.tsv");

  ASSERT_THAT(
      WriteModelComparisonOutputs(result, json_path, tsv_path, dev_tsv_path),
      IsOk());

  std::string read_json;
  ASSERT_THAT(tsl::ReadFileToString(tsl::Env::Default(), json_path, &read_json),
              IsOk());
  EXPECT_THAT(read_json, Eq(json));

  std::string read_tsv;
  ASSERT_THAT(tsl::ReadFileToString(tsl::Env::Default(), tsv_path, &read_tsv),
              IsOk());
  EXPECT_THAT(read_tsv, Eq(tsv));

  std::string read_dev_tsv;
  ASSERT_THAT(
      tsl::ReadFileToString(tsl::Env::Default(), dev_tsv_path, &read_dev_tsv),
      IsOk());
  EXPECT_THAT(read_dev_tsv, Eq(dev_tsv));
}

TEST_F(CompareModelLiteralsTest, SummaryToStringOutputIsValid) {
  Literal lit_g = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit_t = LiteralUtil::CreateR1<float>({1.0f, 2.25f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_g, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit_t, tsl::io::JoinPath(test_dir_, "device_0.literal_0.pb")),
              IsOk());

  // literal_1: exact match
  Literal lit1 = LiteralUtil::CreateR1<float>({3.0f, 4.0f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit1, tsl::io::JoinPath(golden_dir_, "device_0.literal_1.pb")),
      IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit1, tsl::io::JoinPath(test_dir_, "device_0.literal_1.pb")),
              IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));

  EXPECT_THAT(result.summary.total_literals, Eq(2));
  EXPECT_THAT(result.summary.exact_match_literals, Eq(1));
  EXPECT_THAT(result.summary.differing_literals, Eq(1));
  EXPECT_DOUBLE_EQ(result.summary.worst_abs_error, 0.25);
  EXPECT_THAT(result.summary.worst_abs_literal, Eq(0));

  const std::string summary = result.SummaryToString();
  EXPECT_THAT(summary, HasSubstr("Model Comparison Summary:"));
  EXPECT_THAT(summary, HasSubstr("Total Literals: 2 across 1 device(s)"));
  EXPECT_THAT(summary, HasSubstr("Exact Match Literals: 1 (50.00%)"));
  EXPECT_THAT(summary, HasSubstr("Differing Literals:   1 (50.00%)"));
  EXPECT_THAT(summary,
              HasSubstr("Worst Absolute Error: 2.500000e-01 (literal_0)"));
  EXPECT_THAT(summary,
              HasSubstr("Worst Relative Error: 1.250000e-01 (literal_0)"));
}

TEST_F(CompareModelLiteralsTest, SummaryIncludesWithinToleranceCount) {
  Literal lit_g = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit_t = LiteralUtil::CreateR1<float>({1.0f, 2.25f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_g, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit_t, tsl::io::JoinPath(test_dir_, "device_0.literal_0.pb")),
              IsOk());

  Literal lit1 = LiteralUtil::CreateR1<float>({3.0f, 4.0f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit1, tsl::io::JoinPath(golden_dir_, "device_0.literal_1.pb")),
      IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit1, tsl::io::JoinPath(test_dir_, "device_0.literal_1.pb")),
              IsOk());

  ModelComparisonOptions options;
  options.comparison_options.abs_error_bound = 0.5;
  ASSERT_OK_AND_ASSIGN(
      ModelComparisonResult result,
      CompareModelDirectories(golden_dir_, test_dir_, options));

  EXPECT_THAT(result.summary.total_literals, Eq(2));
  EXPECT_THAT(result.summary.exact_match_literals, Eq(1));
  EXPECT_THAT(result.summary.within_tolerance_literals, Eq(1));
  EXPECT_THAT(result.summary.differing_literals, Eq(0));

  const std::string summary = result.SummaryToString();
  EXPECT_THAT(summary, HasSubstr("Exact Match Literals: 1 (50.00%)"));
  EXPECT_THAT(summary, HasSubstr("Within Tolerance:     1 (50.00%)"));
  EXPECT_THAT(summary, HasSubstr("Differing Literals:   0 (0.00%)"));
}

}  // namespace
}  // namespace xla::compare_literals
