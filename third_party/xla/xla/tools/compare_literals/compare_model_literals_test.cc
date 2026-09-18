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
#include <limits>
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
using ::testing::IsEmpty;
using ::testing::Not;
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
  EXPECT_NEAR(entry1.aggregated_device_stats.max_rel_error, 0.005, 1e-4);
  EXPECT_NEAR(entry1.aggregated_device_stats.mean_rel_error, 0.00125, 1e-6);
  ASSERT_THAT(entry1.device_stats, SizeIs(2));
  EXPECT_DOUBLE_EQ(entry1.device_stats.at(0).exact_match_pct, 100.0);
  EXPECT_DOUBLE_EQ(entry1.device_stats.at(0).mean_rel_error, 0.0);
  EXPECT_DOUBLE_EQ(entry1.device_stats.at(1).exact_match_pct, 50.0);
  EXPECT_NEAR(entry1.device_stats.at(1).mean_rel_error, 0.0025, 1e-6);
  EXPECT_NEAR(entry1.device_stats.at(1).max_abs_error, 0.05, 1e-4);
  EXPECT_NEAR(entry1.device_stats.at(1).max_rel_error, 0.005, 1e-4);
  EXPECT_THAT(entry1.device_stats.at(1).element_count, Eq(2));
  EXPECT_THAT(entry1.device_stats.at(1).exact_matches, Eq(1));
  EXPECT_THAT(entry1.device_stats.at(1).mismatches, Eq(1));
  EXPECT_THAT(entry1.device_stats.at(1).nan_mismatches, Eq(0));
  EXPECT_THAT(entry1.device_stats.at(1).inf_mismatches, Eq(0));
  EXPECT_GT(entry1.device_stats.at(1).suggested_abs_error, 0.0);
  EXPECT_GT(entry1.device_stats.at(1).suggested_rel_error, 0.0);
  EXPECT_GT(entry1.aggregated_device_stats.suggested_abs_error, 0.0);
  EXPECT_GT(entry1.aggregated_device_stats.suggested_rel_error, 0.0);
  EXPECT_THAT(result.summary.nan_inf_mismatch_literals, Eq(0));
  EXPECT_THAT(result.SummaryToString(),
              Not(HasSubstr("NaN/Inf Mismatch Literals:")));
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
  EXPECT_THAT(
      result.summary.issues,
      ElementsAre(
          HasSubstr("Literal 0, device 1: missing in test directory"),
          HasSubstr("Literal 1, device 0: missing in golden directory")));
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Issues (2):"));
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
  EXPECT_TRUE(entry.shape_mismatch);
  EXPECT_THAT(entry.failed_devices, Eq(1));
  EXPECT_DOUBLE_EQ(entry.aggregated_device_stats.exact_match_pct, 0.0);
  EXPECT_DOUBLE_EQ(entry.aggregated_device_stats.mean_rel_error, 0.0);
  ASSERT_THAT(entry.device_stats, SizeIs(1));
  EXPECT_THAT(
      entry.device_stats.at(0).status,
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Shapes")));
  EXPECT_THAT(result.summary.shape_mismatch_literals, Eq(1));
  EXPECT_THAT(result.summary.failed_device_comparisons, Eq(1));
  EXPECT_THAT(result.summary.differing_literals, Eq(1));
  EXPECT_THAT(
      result.summary.issues,
      ElementsAre(HasSubstr("comparison failed: Shapes must be equal")));
  EXPECT_THAT(result.SummaryToString(),
              HasSubstr("Shape Mismatch Literals:    1"));
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Issues (1):"));
  EXPECT_THAT(result.ToDeviceTsv(), HasSubstr("0\t0\tfalse\t"));
  EXPECT_THAT(entry.device_stats.at(0).shape_str, Eq("-"));
  EXPECT_THAT(entry.device_stats.at(0).element_type, Eq("-"));
  EXPECT_THAT(entry.device_stats.at(0).element_count, Eq(0));
  EXPECT_THAT(result.ToDeviceTsv(), HasSubstr("\t-\t-\t0\t0\t0.0000\t"));
}

TEST_F(CompareModelLiteralsTest,
       OneDeviceMatchesAndAnotherDeviceFailsSetsExactMatchPctToZero) {
  // Device 0 matches exactly, device 1 has shape mismatch (fails).
  // Aggregate exact_match_pct must be 0.0, not 100.0.
  Literal lit_g0 = LiteralUtil::CreateR1<float>({1.0f});
  Literal lit_t0 = LiteralUtil::CreateR1<float>({1.0f});
  Literal lit_g1 = LiteralUtil::CreateR1<float>({1.0f});
  Literal lit_t1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});

  ASSERT_THAT(
      WriteLiteralToFile(
          lit_g0, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(lit_t0,
                         tsl::io::JoinPath(test_dir_, "device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_g1, tsl::io::JoinPath(golden_dir_, "device_1.literal_0.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(lit_t1,
                         tsl::io::JoinPath(test_dir_, "device_1.literal_0.pb")),
      IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));

  ASSERT_THAT(result.output_stats, SizeIs(1));
  const OutputLiteralStats& entry = result.output_stats[0];
  EXPECT_THAT(entry.failed_devices, Eq(1));
  EXPECT_DOUBLE_EQ(entry.aggregated_device_stats.exact_match_pct, 0.0);
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

TEST_F(CompareModelLiteralsTest, InfMismatchIncrementsNanInfMismatchLiterals) {
  // Golden has finite float, test has infinity (pure Inf mismatch, no NaN)
  Literal lit_g = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit_t = LiteralUtil::CreateR1<float>(
      {1.0f, std::numeric_limits<float>::infinity()});
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
  EXPECT_THAT(entry.aggregated_device_stats.nan_mismatches, Eq(0));
  EXPECT_THAT(entry.aggregated_device_stats.inf_mismatches, Eq(1));
  ASSERT_THAT(entry.device_stats, SizeIs(1));
  EXPECT_THAT(entry.device_stats.at(0).inf_mismatches, Eq(1));
  EXPECT_THAT(entry.device_stats.at(0).nan_mismatches, Eq(0));
  EXPECT_THAT(result.summary.nan_inf_mismatch_literals, Eq(1));
  EXPECT_THAT(result.summary.differing_literals, Eq(1));
  EXPECT_THAT(result.SummaryToString(),
              HasSubstr("NaN/Inf Mismatch Literals:  1"));
}

TEST_F(CompareModelLiteralsTest,
       NanInfMismatchNotWithinToleranceEvenWithLargeBound) {
  // Even with huge error bound, NaN mismatch is differing, not within tolerance
  Literal lit_g = LiteralUtil::CreateR1<float>({1.0f});
  Literal lit_t = LiteralUtil::CreateR1<float>({std::nanf("")});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_g, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit_t, tsl::io::JoinPath(test_dir_, "device_0.literal_0.pb")),
              IsOk());

  ModelComparisonOptions options;
  options.comparison_options.abs_error_bound = 1e9;
  options.comparison_options.rel_error_bound = 1e9;
  ASSERT_OK_AND_ASSIGN(
      ModelComparisonResult result,
      CompareModelDirectories(golden_dir_, test_dir_, options));

  EXPECT_THAT(result.summary.exact_match_literals, Eq(0));
  EXPECT_THAT(result.summary.within_tolerance_literals, Eq(0));
  EXPECT_THAT(result.summary.differing_literals, Eq(1));
  EXPECT_THAT(result.summary.nan_inf_mismatch_literals, Eq(1));
}

TEST_F(CompareModelLiteralsTest,
       DiscoveredDevicesSortedNumericallyNotLexicographically) {
  // Put device 10 on literal 0 and device 2 on literal 1 so golden_map
  // encounters device 10 first. Devices must be sorted numerically: {2, 10}.
  Literal lit = LiteralUtil::CreateR1<float>({1.0f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit, tsl::io::JoinPath(golden_dir_, "device_10.literal_0.pb")),
      IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(test_dir_, "device_10.literal_0.pb")),
              IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(golden_dir_, "device_2.literal_1.pb")),
              IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(test_dir_, "device_2.literal_1.pb")),
              IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));

  EXPECT_THAT(result.devices, ElementsAre(2, 10));
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

TEST_F(CompareModelLiteralsTest, TsvFormattingAndWriting) {
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

  const std::string tsv = result.ToTsv();
  EXPECT_THAT(tsv, HasSubstr("literal\tshape\ttype\telements\tdevices"));
  EXPECT_THAT(tsv, HasSubstr("0\tf32[2]\tf32\t2\t1"));

  const std::string dev_tsv = result.ToDeviceTsv();
  EXPECT_THAT(dev_tsv, HasSubstr("literal\tdevice\tcomparison_ok\terror_"
                                 "message\tshape\ttype\telements"));
  EXPECT_THAT(dev_tsv, HasSubstr("0\t0\ttrue\t-\tf32[2]\tf32\t2"));

  const std::string tsv_path =
      tsl::io::JoinPath(test_dir_, "nested", "out.tsv");
  const std::string dev_tsv_path =
      tsl::io::JoinPath(test_dir_, "nested", "out_dev.tsv");

  ASSERT_THAT(WriteModelComparisonOutputs(result, tsv_path, dev_tsv_path),
              IsOk());

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
  EXPECT_THAT(result.summary.nan_inf_mismatch_literals, Eq(0));

  const std::string summary = result.SummaryToString();
  EXPECT_THAT(summary, HasSubstr("Exact Match Literals: 1 (50.00%)"));
  EXPECT_THAT(summary, HasSubstr("Within Tolerance:     1 (50.00%)"));
  EXPECT_THAT(summary, HasSubstr("Differing Literals:   0 (0.00%)"));
  EXPECT_THAT(summary, Not(HasSubstr("NaN/Inf Mismatch Literals:")));
}

TEST_F(CompareModelLiteralsTest, ReportsCrossDeviceShapeMismatchInSummary) {
  // Device 0 has shape f32[2] on both golden and test (exact match)
  Literal lit_d0 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_d0, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(lit_d0,
                         tsl::io::JoinPath(test_dir_, "device_0.literal_0.pb")),
      IsOk());

  // Device 1 has shape f32[4] on both golden and test (exact match on device 1,
  // but cross-device shape mismatch against device 0)
  Literal lit_d1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_d1, tsl::io::JoinPath(golden_dir_, "device_1.literal_0.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(lit_d1,
                         tsl::io::JoinPath(test_dir_, "device_1.literal_0.pb")),
      IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));

  ASSERT_THAT(result.output_stats, SizeIs(1));
  const OutputLiteralStats& entry = result.output_stats[0];
  EXPECT_TRUE(entry.shape_mismatch);
  EXPECT_THAT(result.summary.shape_mismatch_literals, Eq(1));
  EXPECT_THAT(result.summary.exact_match_literals, Eq(0));
  EXPECT_THAT(result.summary.differing_literals, Eq(1));
  EXPECT_THAT(
      result.summary.issues,
      ElementsAre(HasSubstr(
          "Literal 0: shape mismatch across devices (device 1 has f32[4] vs "
          "device 0 has f32[2])")));

  const std::string summary = result.SummaryToString();
  EXPECT_THAT(summary, HasSubstr("Shape Mismatch Literals:    1"));
  EXPECT_THAT(summary, HasSubstr("Differing Literals:   1 (100.00%)"));
  EXPECT_THAT(summary, HasSubstr("Issues (1):"));
  EXPECT_THAT(summary, HasSubstr("shape mismatch across devices"));

  const std::string dev_tsv = result.ToDeviceTsv();
  EXPECT_THAT(dev_tsv, HasSubstr("0\t0\ttrue\t-\tf32[2]\tf32\t2"));
  EXPECT_THAT(dev_tsv, HasSubstr("0\t1\ttrue\t-\tf32[4]\tf32\t4"));
}

TEST_F(CompareModelLiteralsTest,
       ReportsCrossDeviceElementTypeMismatchInSummary) {
  // Device 0 has shape f32[2]
  Literal lit_f32 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_f32, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(lit_f32,
                         tsl::io::JoinPath(test_dir_, "device_0.literal_0.pb")),
      IsOk());

  // Device 1 has shape s32[2]
  Literal lit_s32 = LiteralUtil::CreateR1<int32_t>({1, 2});
  ASSERT_THAT(
      WriteLiteralToFile(
          lit_s32, tsl::io::JoinPath(golden_dir_, "device_1.literal_0.pb")),
      IsOk());
  ASSERT_THAT(
      WriteLiteralToFile(lit_s32,
                         tsl::io::JoinPath(test_dir_, "device_1.literal_0.pb")),
      IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));

  ASSERT_THAT(result.output_stats, SizeIs(1));
  const OutputLiteralStats& entry = result.output_stats[0];
  EXPECT_TRUE(entry.shape_mismatch);
  EXPECT_THAT(result.summary.shape_mismatch_literals, Eq(1));
  EXPECT_THAT(result.summary.differing_literals, Eq(1));
  EXPECT_THAT(
      result.summary.issues,
      ElementsAre(HasSubstr(
          "Literal 0: element type mismatch across devices (device 1 has s32 "
          "vs device 0 has f32)")));

  const std::string summary = result.SummaryToString();
  EXPECT_THAT(summary, HasSubstr("Shape Mismatch Literals:    1"));
  EXPECT_THAT(summary, HasSubstr("Issues (1):"));
  EXPECT_THAT(summary, HasSubstr("element type mismatch across devices"));
}

TEST_F(CompareModelLiteralsTest,
       OneDirectoryEmptyReturnsModelComparisonResultWithMissingFiles) {
  // Test case 1: Golden has a literal file, test directory is completely empty.
  Literal lit = LiteralUtil::CreateR1<float>({42.0f});
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
              IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));
  EXPECT_THAT(result.summary.total_literals, Eq(0));
  EXPECT_THAT(result.output_stats, IsEmpty());
  EXPECT_THAT(result.missing_in_test,
              ElementsAre(LiteralKey{/*literal_id=*/0, /*device_id=*/0}));
  EXPECT_THAT(result.missing_in_golden, IsEmpty());

  // Test case 2: Golden directory is completely empty, test has a literal file.
  ASSERT_THAT(tsl::Env::Default()->DeleteFile(
                  tsl::io::JoinPath(golden_dir_, "device_0.literal_0.pb")),
              IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(test_dir_, "device_0.literal_1.pb")),
              IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result2,
                       CompareModelDirectories(golden_dir_, test_dir_));
  EXPECT_THAT(result2.summary.total_literals, Eq(0));
  EXPECT_THAT(result2.output_stats, IsEmpty());
  EXPECT_THAT(result2.missing_in_golden,
              ElementsAre(LiteralKey{/*literal_id=*/1, /*device_id=*/0}));
  EXPECT_THAT(result2.missing_in_test, IsEmpty());
}

TEST_F(CompareModelLiteralsTest, DuplicateLiteralKeepsOneEntryAndLogsWarning) {
  Literal lit = LiteralUtil::CreateR1<float>({1.0f});

  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(golden_dir_, "dup1.literal_0.pb")),
              IsOk());
  ASSERT_THAT(WriteLiteralToFile(
                  lit, tsl::io::JoinPath(golden_dir_, "dup2.literal_0.pb")),
              IsOk());

  ASSERT_THAT(
      WriteLiteralToFile(lit, tsl::io::JoinPath(test_dir_, "literal_0.pb")),
      IsOk());

  ASSERT_OK_AND_ASSIGN(ModelComparisonResult result,
                       CompareModelDirectories(golden_dir_, test_dir_));
  ASSERT_THAT(result.output_stats, SizeIs(1));
  EXPECT_DOUBLE_EQ(
      result.output_stats[0].aggregated_device_stats.exact_match_pct, 100.0);
}

TEST_F(CompareModelLiteralsTest, WriteModelComparisonOutputsPartialPaths) {
  ModelComparisonResult result;
  result.golden_dir = golden_dir_;
  result.test_dir = test_dir_;

  const std::string tsv_path = tsl::io::JoinPath(test_dir_, "only_out.tsv");
  // Only TSV, empty device_tsv
  EXPECT_THAT(WriteModelComparisonOutputs(result, tsv_path, ""), IsOk());
  EXPECT_TRUE(tsl::Env::Default()->FileExists(tsv_path).ok());

  const std::string dev_tsv_path =
      tsl::io::JoinPath(test_dir_, "only_dev_out.tsv");
  // Only device TSV, empty tsv
  EXPECT_THAT(WriteModelComparisonOutputs(result, "", dev_tsv_path), IsOk());
  EXPECT_TRUE(tsl::Env::Default()->FileExists(dev_tsv_path).ok());

  // Both empty
  EXPECT_THAT(WriteModelComparisonOutputs(result, "", ""), IsOk());
}

TEST(ParseLiteralFilenameTest, MatchesVariousPatterns) {
  // Standard dot delimiter
  auto p1 = ParseLiteralFilename("device_0.literal_1.pb");
  ASSERT_TRUE(p1.has_value());
  EXPECT_EQ(p1->literal_id, 1);
  EXPECT_EQ(p1->device_id, 0);

  // Underscore delimiter
  auto p2 = ParseLiteralFilename("device_2_literal_3.pb");
  ASSERT_TRUE(p2.has_value());
  EXPECT_EQ(p2->literal_id, 3);
  EXPECT_EQ(p2->device_id, 2);

  // Multi-part prefix with task and hlo identifiers
  auto p3 = ParseLiteralFilename("output.hlo_0.task_0.device_7.literal_27.pb");
  ASSERT_TRUE(p3.has_value());
  EXPECT_EQ(p3->literal_id, 27);
  EXPECT_EQ(p3->device_id, 7);

  // Literal without device (defaults to device 0)
  auto p4 = ParseLiteralFilename("literal_5.pb");
  ASSERT_TRUE(p4.has_value());
  EXPECT_EQ(p4->literal_id, 5);
  EXPECT_EQ(p4->device_id, 0);

  // Model prefix without device
  auto p5 = ParseLiteralFilename("model_eval.literal_42.pb");
  ASSERT_TRUE(p5.has_value());
  EXPECT_EQ(p5->literal_id, 42);
  EXPECT_EQ(p5->device_id, 0);

  // Non-device tokens containing "device"
  auto p6 = ParseLiteralFilename("my_device_mesh_literal_5.pb");
  ASSERT_TRUE(p6.has_value());
  EXPECT_EQ(p6->literal_id, 5);
  EXPECT_EQ(p6->device_id, 0);

  // Non-matching filenames
  EXPECT_FALSE(ParseLiteralFilename("literal_1.txt").has_value());
  EXPECT_FALSE(ParseLiteralFilename("literal_abc.pb").has_value());
  EXPECT_FALSE(ParseLiteralFilename("checkpoint.pb").has_value());
  EXPECT_FALSE(ParseLiteralFilename("device_0.pb").has_value());
  EXPECT_FALSE(ParseLiteralFilename("").has_value());
}

}  // namespace
}  // namespace xla::compare_literals
