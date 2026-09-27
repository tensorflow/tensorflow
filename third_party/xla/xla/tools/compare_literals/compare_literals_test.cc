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

#include "xla/tools/compare_literals/compare_literals.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tools/compare_literals/element_comparator.h"
#include "xla/tsl/platform/env.h"
#include "xla/types.h"
#include "tsl/platform/path.h"

namespace xla::compare_literals {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Not;

TEST(CompareLiteralsTest, ExactMatch) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.element_type, "f32");
  EXPECT_EQ(result.shape_str, "f32[4]");
  EXPECT_EQ(result.total_elements, 4);
  EXPECT_EQ(result.exact_matches, 4);
  EXPECT_EQ(result.mismatches, 0);
  EXPECT_DOUBLE_EQ(result.max_abs_error, 0.0);
  EXPECT_DOUBLE_EQ(result.max_rel_error, 0.0);
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Element Type: f32"));
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Shape: f32[4]"));
}

TEST(CompareLiteralsTest, WithinTolerance) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 10.0f, 100.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0001f, 10.001f, 100.01f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.total_elements, 3);
  EXPECT_EQ(result.mismatches, 0);
  EXPECT_NEAR(result.max_abs_error, 0.01, 1e-4);
  EXPECT_NEAR(result.max_rel_error, 1e-4, 1e-5);
}

TEST(CompareLiteralsTest, ExceedsTolerance) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, 2.5f, 3.0f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.total_elements, 3);
  EXPECT_EQ(result.exact_matches, 2);
  EXPECT_EQ(result.mismatches, 1);
  EXPECT_NEAR(result.max_abs_error, 0.5, 1e-5);
  EXPECT_NEAR(result.max_rel_error, 0.25, 1e-5);
  ASSERT_EQ(result.top_mismatches.size(), 1);
  EXPECT_EQ(result.top_mismatches[0].linear_index, 1);
  EXPECT_EQ(result.top_mismatches[0].clean_str, "2");
  EXPECT_EQ(result.top_mismatches[0].dirty_str, "2.5");
}

TEST(CompareLiteralsTest, NaNHandling) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, kNaN});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, kNaN});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.exact_matches, 2);
}

TEST(CompareLiteralsTest, NaNMismatch) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, kNaN});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.nan_mismatches, 1);
  EXPECT_EQ(result.mismatches, 1);
  EXPECT_TRUE(std::isinf(result.max_abs_error));
  EXPECT_TRUE(std::isinf(result.max_rel_error));
}

TEST(CompareLiteralsTest, ShapeMismatch) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});

  ComparisonOptions options;
  EXPECT_THAT(CompareLiterals(lit1, lit2, options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Shapes must be equal")));
}

TEST(CompareLiteralsTest, HistogramAndHeatmapOutput) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 4.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.01f, 2.02f, 4.04f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  std::string hist_str = result.histogram.ToString();
  EXPECT_THAT(hist_str, HasSubstr("Summary: min ="));

  std::string heatmap_str = result.heatmap.ToString(/*use_color=*/false);
  EXPECT_THAT(heatmap_str, HasSubstr("2D Error Heatmap"));
  EXPECT_THAT(heatmap_str, HasSubstr("Legend:"));
}

TEST(ElementComparatorTest, RecordIndividualElements) {
  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;

  ElementComparator<float> comparator(options, /*total_elements=*/3);
  comparator.RecordElement(0, 1.0f, 1.0f);      // exact match
  comparator.RecordElement(1, 10.0f, 10.005f);  // within tolerance
  comparator.RecordElement(2, 2.0f, 2.5f);      // mismatch

  ComparisonResult result = comparator.Finalize();
  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.total_elements, 3);
  EXPECT_EQ(result.exact_matches, 1);
  EXPECT_EQ(result.mismatches, 1);
  EXPECT_NEAR(result.max_abs_error, 0.5, 1e-5);
  EXPECT_NEAR(result.max_rel_error, 0.25, 1e-5);
  ASSERT_EQ(result.top_mismatches.size(), 1);
  EXPECT_EQ(result.top_mismatches[0].clean_str, "2");
  EXPECT_EQ(result.top_mismatches[0].dirty_str, "2.5");
}

TEST(ElementComparatorTest, ComplexNumbers) {
  ComparisonOptions options;
  options.abs_error_bound = 1e-2;
  options.rel_error_bound = 1e-2;

  ElementComparator<std::complex<float>> comparator(options,
                                                    /*total_elements=*/2);
  comparator.RecordElement(0, {1.0f, 2.0f}, {1.0f, 2.0f});
  comparator.RecordElement(1, {1.0f, 0.0f}, {2.0f, 0.0f});

  ComparisonResult result = comparator.Finalize();
  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.exact_matches, 1);
  EXPECT_EQ(result.mismatches, 1);
  EXPECT_DOUBLE_EQ(result.max_abs_error, 1.0);
}

TEST(ElementComparatorTest, Int8AndUint8FormattingAndDiff) {
  ComparisonOptions options;
  options.abs_error_bound = 0.0;
  options.rel_error_bound = 0.0;

  ElementComparator<int8_t> int8_comp(options, /*total_elements=*/1);
  int8_comp.RecordElement(0, static_cast<int8_t>(65), static_cast<int8_t>(66));
  ComparisonResult int8_result = int8_comp.Finalize();
  EXPECT_FALSE(int8_result.passed);
  ASSERT_EQ(int8_result.top_mismatches.size(), 1);
  // Must format as number "65", not ASCII 'A'.
  EXPECT_EQ(int8_result.top_mismatches[0].clean_str, "65");
  EXPECT_EQ(int8_result.top_mismatches[0].dirty_str, "66");

  ElementComparator<uint8_t> uint8_comp(options, /*total_elements=*/1);
  uint8_comp.RecordElement(0, static_cast<uint8_t>(48),
                           static_cast<uint8_t>(49));
  ComparisonResult uint8_result = uint8_comp.Finalize();
  EXPECT_FALSE(uint8_result.passed);
  ASSERT_EQ(uint8_result.top_mismatches.size(), 1);
  // Must format as number "48", not ASCII '0'.
  EXPECT_EQ(uint8_result.top_mismatches[0].clean_str, "48");
  EXPECT_EQ(uint8_result.top_mismatches[0].dirty_str, "49");
}

TEST(CompareLiteralsTest, NegativeValues) {
  Literal lit1 = LiteralUtil::CreateR1<float>({-10.0f, -100.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({-10.005f, -99.95f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-1;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_TRUE(result.passed);
  EXPECT_NEAR(result.max_abs_error, 0.05, 1e-5);
  EXPECT_NEAR(result.max_rel_error, 5e-4, 1e-6);

  // Exceeding tolerance on negative numbers
  Literal lit3 = LiteralUtil::CreateR1<float>({-2.0f});
  Literal lit4 = LiteralUtil::CreateR1<float>({-2.5f});
  ASSERT_OK_AND_ASSIGN(ComparisonResult result2,
                       CompareLiterals(lit3, lit4, options));
  EXPECT_FALSE(result2.passed);
  EXPECT_NEAR(result2.max_abs_error, 0.5, 1e-5);
  EXPECT_NEAR(result2.max_rel_error, 0.25, 1e-5);
}

TEST(CompareLiteralsTest, ExactMatchesIncludedInWelford) {
  // 3 exact matches, 1 element with 0.04 relative error.
  Literal lit1 = LiteralUtil::CreateR1<float>({10.0f, 10.0f, 10.0f, 10.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({10.0f, 10.0f, 10.0f, 10.4f});

  ComparisonOptions options;
  options.abs_error_bound = 1.0;
  options.rel_error_bound = 0.1;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_TRUE(result.passed);
  // Mean relative error across 4 elements: (0 + 0 + 0 + 0.04) / 4 = 0.01.
  EXPECT_NEAR(result.histogram.mean_rel_error, 0.01, 1e-5);
}

TEST(CompareLiteralsTest, LargeInt64Comparison) {
  // Values above 2^53 that differ by 1.
  constexpr int64_t kBase = 9007199254740992LL;  // 2^53
  Literal lit1 = LiteralUtil::CreateR1<int64_t>({kBase, kBase + 1});
  Literal lit2 = LiteralUtil::CreateR1<int64_t>({kBase, kBase});

  ComparisonOptions options;
  options.abs_error_bound = 0.0;
  options.rel_error_bound = 0.0;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.exact_matches, 1);
  EXPECT_EQ(result.mismatches, 1);
  EXPECT_DOUBLE_EQ(result.max_abs_error, 1.0);
}

TEST(CompareLiteralsTest, SingleElementMedian) {
  Literal lit1 = LiteralUtil::CreateR1<float>({100.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({102.0f});  // +2% rel error

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  ASSERT_GE(result.histogram.median_bin_index, 0);
  EXPECT_EQ(result.histogram.bins[result.histogram.median_bin_index].count, 1);
}

TEST(CompareLiteralsTest, SuggestedErrorSpecFailingComparison) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, 2.5f, 3.0f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_FALSE(result.passed);
  ASSERT_TRUE(result.suggested_error_spec.has_value());
  const auto& spec = *result.suggested_error_spec;

  EXPECT_GE(spec.pure_abs_bound, 0.5);
  EXPECT_GE(spec.pure_rel_bound, 0.25);
  EXPECT_GE(spec.margin_abs_bound, spec.abs_bound);
  EXPECT_GE(spec.margin_rel_bound, spec.rel_bound);

  // Crucial verification: running CompareLiterals with suggested balanced
  // bounds MUST pass!
  ComparisonOptions passing_options;
  passing_options.abs_error_bound = spec.abs_bound;
  passing_options.rel_error_bound = spec.rel_bound;
  ASSERT_OK_AND_ASSIGN(ComparisonResult passing_result,
                       CompareLiterals(lit1, lit2, passing_options));
  EXPECT_TRUE(passing_result.passed);
  EXPECT_EQ(passing_result.mismatches, 0);

  // Pure absolute bound verification (with rel = 0)
  ComparisonOptions pure_abs_options;
  pure_abs_options.abs_error_bound = spec.pure_abs_bound;
  pure_abs_options.rel_error_bound = 0.0;
  ASSERT_OK_AND_ASSIGN(ComparisonResult pure_abs_result,
                       CompareLiterals(lit1, lit2, pure_abs_options));
  EXPECT_TRUE(pure_abs_result.passed);

  // Pure relative bound verification (with abs = 0)
  ComparisonOptions pure_rel_options;
  pure_rel_options.abs_error_bound = 0.0;
  pure_rel_options.rel_error_bound = spec.pure_rel_bound;
  ASSERT_OK_AND_ASSIGN(ComparisonResult pure_rel_result,
                       CompareLiterals(lit1, lit2, pure_rel_options));
  EXPECT_TRUE(pure_rel_result.passed);

  // Output formatting verification
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Suggested ErrorSpec"));
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Balanced:"));
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Pure Absolute:"));
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Pure Relative:"));
}

TEST(CompareLiteralsTest, SuggestedErrorSpecIncludedOnPassingComparison) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, 2.01f, 3.0f});

  ComparisonOptions options;
  options.abs_error_bound = 0.1;
  options.rel_error_bound = 0.1;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_TRUE(result.passed);
  ASSERT_TRUE(result.suggested_error_spec.has_value());

  // Verify SummaryToString defaults to including Suggested ErrorSpec on PASS
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Suggested ErrorSpec"));
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Balanced:"));

  // Verify SummaryToString with use_color = false emits plain text without ANSI
  EXPECT_THAT(result.SummaryToString(/*use_color=*/false),
              HasSubstr("PASS (MATCH)"));
}

TEST(CompareLiteralsTest, SuggestedErrorSpecNulloptOnNanMismatches) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, kNaN});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.nan_mismatches, 1);
  EXPECT_FALSE(result.suggested_error_spec.has_value());
}

TEST(CompareLiteralsTest, InfinityMatchesAndMismatches) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  Literal clean_inf = LiteralUtil::CreateR1<float>({kInf, -kInf});
  Literal dirty_inf = LiteralUtil::CreateR1<float>({kInf, -kInf});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult match_result,
                       CompareLiterals(clean_inf, dirty_inf, options));
  EXPECT_TRUE(match_result.passed);
  EXPECT_EQ(match_result.exact_matches, 2);
  EXPECT_EQ(match_result.inf_mismatches, 0);

  Literal clean_mix = LiteralUtil::CreateR1<float>({1.0f, -kInf});
  Literal dirty_mix = LiteralUtil::CreateR1<float>({kInf, -kInf});
  ASSERT_OK_AND_ASSIGN(ComparisonResult mismatch_result,
                       CompareLiterals(clean_mix, dirty_mix, options));
  EXPECT_FALSE(mismatch_result.passed);
  EXPECT_EQ(mismatch_result.inf_mismatches, 1);
  EXPECT_EQ(mismatch_result.mismatches, 1);
  EXPECT_EQ(mismatch_result.max_abs_error,
            std::numeric_limits<double>::infinity());
  EXPECT_FALSE(mismatch_result.suggested_error_spec.has_value());
  ASSERT_FALSE(mismatch_result.top_mismatches.empty());
  EXPECT_EQ(mismatch_result.top_mismatches[0].linear_index, 0);
}

TEST(CompareLiteralsTest, MismatchedLayoutFallback) {
  Literal lit_row = LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}});
  Literal lit_col = lit_row.Relayout(LayoutUtil::MakeLayout({0, 1}));
  ASSERT_FALSE(
      LayoutUtil::Equal(lit_row.shape().layout(), lit_col.shape().layout()));

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit_row, lit_col, options));
  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.exact_matches, 4);
  EXPECT_EQ(result.mismatches, 0);
}

TEST(CompareLiteralsTest, SummaryToMarkdownReport) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, 2.5f});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  std::string md = result.SummaryToMarkdown();
  EXPECT_THAT(md, HasSubstr("# Comparison Report"));
  EXPECT_THAT(md, HasSubstr("## Summary Statistics"));
  EXPECT_THAT(md, HasSubstr("## 1D Signed Relative Error Distribution"));
  EXPECT_THAT(
      md, HasSubstr("| Markers | Range | Count | Percent | Distribution |"));
  EXPECT_THAT(md, HasSubstr("## 2D Error Heatmap"));
  EXPECT_THAT(md, HasSubstr("| Rel \\ Abs |"));
  EXPECT_THAT(md, HasSubstr("🟩 0.0%"));
  EXPECT_THAT(md, HasSubstr("Legend: 🟩 0.0% failures"));
  EXPECT_THAT(md, HasSubstr("## First Mismatches"));
  EXPECT_THAT(md, HasSubstr("## Suggested ErrorSpec"));
  // Heatmap and histogram must be native Markdown tables without code fences.
  EXPECT_THAT(md, Not(HasSubstr("```")));
}

TEST(CompareLiteralsTest, HeatmapPercentageAndYellowThresholdConfigurable) {
  // 100 elements: 98 matching, 2 differing by 0.5 (abs_diff = 0.5, rel_diff =
  // 0.25). Total elements = 100, failures at tight tolerances = 2 (2.0%).
  std::vector<float> clean_vals(100, 2.0f);
  std::vector<float> dirty_vals = clean_vals;
  dirty_vals[0] = 2.5f;
  dirty_vals[1] = 2.5f;

  Literal lit1 = LiteralUtil::CreateR1<float>(clean_vals);
  Literal lit2 = LiteralUtil::CreateR1<float>(dirty_vals);

  // Run with default yellow threshold (0.5%): 2.0% > 0.5% so cell should be RED
  // (🟥).
  ComparisonOptions options_default;
  options_default.abs_error_bound = 1e-3;
  options_default.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result_default,
                       CompareLiterals(lit1, lit2, options_default));

  std::string md_default = result_default.heatmap.ToMarkdown();
  EXPECT_THAT(md_default, HasSubstr("2.0%"));
  // At target tolerance (1e-3, 1e-3), failure rate is 2.0% > 0.5%, so it is
  // red:
  EXPECT_THAT(md_default, HasSubstr("🟥 **[2.0%]**"));

  // Run with custom yellow threshold (5.0%): 2.0% <= 5.0% so cell should be
  // YELLOW (🟨).
  ComparisonOptions options_custom;
  options_custom.abs_error_bound = 1e-3;
  options_custom.rel_error_bound = 1e-3;
  options_custom.heatmap_yellow_pct = 5.0;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result_custom,
                       CompareLiterals(lit1, lit2, options_custom));

  std::string md_custom = result_custom.heatmap.ToMarkdown();
  EXPECT_THAT(md_custom, HasSubstr("🟨 **[2.0%]**"));
  EXPECT_THAT(md_custom,
              HasSubstr("Legend: 🟩 0.0% failures (100% pass), 🟨 <= 5.0% "
                        "failures, 🟥 > 5.0% failures"));

  // Console output should also display percentages and yellow threshold:
  std::string console = result_custom.heatmap.ToString(/*use_color=*/false);
  EXPECT_THAT(console, HasSubstr("2.0%"));
  EXPECT_THAT(console, HasSubstr("Yellow <= 5.0%"));
}

TEST(CompareLiteralsTest, CompareLiteralProtosAndFiles) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});

  LiteralProto proto1 = lit1.ToProto();
  LiteralProto proto2 = lit2.ToProto();

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult proto_result,
                       CompareLiteralProtos(proto1, proto2, options));
  EXPECT_TRUE(proto_result.passed);

  std::string clean_path = tsl::io::JoinPath(testing::TempDir(), "clean.pb");
  std::string dirty_path = tsl::io::JoinPath(testing::TempDir(), "dirty.pb");

  ASSERT_OK(tsl::WriteBinaryProto(tsl::Env::Default(), clean_path, proto1));
  ASSERT_OK(tsl::WriteBinaryProto(tsl::Env::Default(), dirty_path, proto2));

  ASSERT_OK_AND_ASSIGN(ComparisonResult file_result,
                       CompareLiteralFiles(clean_path, dirty_path, options));
  EXPECT_TRUE(file_result.passed);

  EXPECT_THAT(
      CompareLiteralFiles("/nonexistent/path/clean.pb", dirty_path, options),
      StatusIs(absl::StatusCode::kNotFound,
               HasSubstr("Failed to read clean literal file")));
  EXPECT_THAT(
      CompareLiteralFiles(clean_path, "/nonexistent/path/dirty.pb", options),
      StatusIs(absl::StatusCode::kNotFound,
               HasSubstr("Failed to read dirty literal file")));
}

TEST(CompareLiteralsTest, AllZerosCleanAndDirty) {
  Literal clean = LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});
  Literal dirty = LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(clean, dirty, options));

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.total_elements, 4);
  EXPECT_EQ(result.exact_matches, 4);
  EXPECT_EQ(result.mismatches, 0);
  EXPECT_DOUBLE_EQ(result.max_abs_error, 0.0);
  EXPECT_DOUBLE_EQ(result.max_rel_error, 0.0);
  EXPECT_TRUE(
      result.histogram.bins[result.histogram.median_bin_index].is_exact_zero);
  EXPECT_EQ(result.histogram.bins[result.histogram.median_bin_index].count, 4);
}

TEST(CompareLiteralsTest, SignedZerosEquivalent) {
  Literal clean = LiteralUtil::CreateR1<float>({+0.0f, -0.0f});
  Literal dirty = LiteralUtil::CreateR1<float>({-0.0f, +0.0f});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(clean, dirty, options));

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.exact_matches, 2);
  EXPECT_EQ(result.mismatches, 0);
}

TEST(CompareLiteralsTest, ZeroReferenceWithNonZeroDirty) {
  Literal clean = LiteralUtil::CreateR1<float>({0.0f, 0.0f});
  Literal dirty = LiteralUtil::CreateR1<float>({1e-4f, 0.5f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(clean, dirty, options));

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.exact_matches, 0);
  EXPECT_EQ(result.mismatches, 1);
  EXPECT_NEAR(result.max_abs_error, 0.5, 1e-5);
  EXPECT_DOUBLE_EQ(result.max_rel_error, 0.0);
  ASSERT_TRUE(result.suggested_error_spec.has_value());
  EXPECT_GE(result.suggested_error_spec->pure_abs_bound, 0.5);
  EXPECT_TRUE(std::isinf(result.suggested_error_spec->pure_rel_bound));
  EXPECT_TRUE(std::isinf(result.suggested_error_spec->margin_pure_rel_bound));
  EXPECT_GE(result.suggested_error_spec->abs_bound, 0.5);
  EXPECT_DOUBLE_EQ(result.suggested_error_spec->rel_bound, 0.0);
}

TEST(CompareLiteralsTest, Rank0ScalarLiterals) {
  Literal clean_scalar = LiteralUtil::CreateR0<float>(42.0f);
  Literal dirty_scalar = LiteralUtil::CreateR0<float>(42.0f);

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult match_result,
                       CompareLiterals(clean_scalar, dirty_scalar, options));
  EXPECT_TRUE(match_result.passed);
  EXPECT_EQ(match_result.shape_str, "f32[]");
  EXPECT_EQ(match_result.total_elements, 1);
  EXPECT_EQ(match_result.exact_matches, 1);

  Literal mismatch_scalar = LiteralUtil::CreateR0<float>(43.0f);
  ASSERT_OK_AND_ASSIGN(ComparisonResult mismatch_result,
                       CompareLiterals(clean_scalar, mismatch_scalar, options));
  EXPECT_FALSE(mismatch_result.passed);
  EXPECT_EQ(mismatch_result.mismatches, 1);
  EXPECT_DOUBLE_EQ(mismatch_result.max_abs_error, 1.0);
}

TEST(CompareLiteralsTest, Rank3TensorLiterals) {
  Literal clean = LiteralUtil::CreateR3<float>(
      {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}});
  Literal dirty = LiteralUtil::CreateR3<float>(
      {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.5f}}});

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(clean, dirty, options));

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.shape_str, "f32[2,2,2]");
  EXPECT_EQ(result.total_elements, 8);
  EXPECT_EQ(result.exact_matches, 7);
  EXPECT_EQ(result.mismatches, 1);
  EXPECT_NEAR(result.max_abs_error, 0.5, 1e-5);
  ASSERT_EQ(result.top_mismatches.size(), 1);
  EXPECT_EQ(result.top_mismatches[0].clean_str, "8");
  EXPECT_EQ(result.top_mismatches[0].dirty_str, "8.5");
}

TEST(CompareLiteralsTest, EmptyLiteralZeroElements) {
  Literal clean = LiteralUtil::CreateR1<float>({});
  Literal dirty = LiteralUtil::CreateR1<float>({});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(clean, dirty, options));

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.total_elements, 0);
  EXPECT_EQ(result.exact_matches, 0);
  EXPECT_EQ(result.mismatches, 0);
  EXPECT_TRUE(result.suggested_error_spec.has_value());
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Total Elements: 0"));
}

TEST(CompareLiteralsTest, HeatmapIncludesAllZeroColumnAndRow) {
  Literal clean = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  Literal dirty = LiteralUtil::CreateR1<float>({1.0f, 2.05f, 3.0f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-4;
  options.rel_error_bound = 1e-4;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(clean, dirty, options));

  EXPECT_FALSE(result.passed);
  std::string md = result.heatmap.ToMarkdown();
  std::string console = result.heatmap.ToString(/*use_color=*/false);

  // The true max absolute error is 0.05. The next grid threshold is 5e-2.
  // The table must include the 5e-2 column (or higher), which has all 0.0%.
  EXPECT_THAT(md, HasSubstr("5e-2"));
  EXPECT_THAT(console, HasSubstr("5e-2"));
  // Target tolerance *1e-4 must also be visible:
  EXPECT_THAT(md, HasSubstr("1e-4"));
  EXPECT_THAT(console, HasSubstr("1e-4"));
}

TEST(CompareLiteralsTest, ExactToleranceBoundaryPasses) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.125f});
  ComparisonOptions options;
  options.abs_error_bound = 0.125;
  options.rel_error_bound = 0.125;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));
  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.mismatches, 0);
}

TEST(CompareLiteralsTest, StandaloneMathHelpers) {
  std::vector<double> thresholds = BuildThresholds(0.035);
  EXPECT_THAT(thresholds, ::testing::Contains(0.035));
  EXPECT_TRUE(std::is_sorted(thresholds.begin(), thresholds.end()));

  // Verify non-positive targets are not added (kills Mutant 1)
  std::vector<double> zero_thresholds = BuildThresholds(0.0);
  EXPECT_THAT(zero_thresholds, ::testing::Not(::testing::Contains(0.0)));
  std::vector<double> neg_thresholds = BuildThresholds(-1.0);
  EXPECT_THAT(neg_thresholds, ::testing::Not(::testing::Contains(-1.0)));

  std::vector<RelErrorBin> bins = CreateDefaultRelBins();
  EXPECT_GE(bins.size(), 30);

  // Verify 1.0 multiplier boundaries for e > -6 exist (kills Mutant 2)
  bool has_one_milli = false;
  bool has_one = false;
  for (const auto& b : bins) {
    if (std::abs(b.lower - 1e-3) < 1e-9 || std::abs(b.upper - 1e-3) < 1e-9) {
      has_one_milli = true;
    }
    if (std::abs(b.lower - 1.0) < 1e-9 || std::abs(b.upper - 1.0) < 1e-9) {
      has_one = true;
    }
  }
  EXPECT_TRUE(has_one_milli);
  EXPECT_TRUE(has_one);

  int zero_idx = FindRelBin(0.0, bins);
  EXPECT_TRUE(bins[zero_idx].is_exact_zero);
  int pos_idx = FindRelBin(0.05, bins);
  EXPECT_GE(pos_idx, zero_idx);
  int neg_idx = FindRelBin(-0.05, bins);
  EXPECT_LE(neg_idx, zero_idx);

  constexpr double kInf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(FindRelBin(-kInf, bins), 0);
  EXPECT_EQ(FindRelBin(kInf, bins), static_cast<int>(bins.size() - 1));

  // Verify asymmetric bins where exact_zero is NOT at mid (kills Mutant 3)
  std::vector<RelErrorBin> zero_at_start = {
      {0.0, 0.0, 0, true},
      {0.0, 1.0, 0, false},
      {1.0, 2.0, 0, false},
  };
  EXPECT_EQ(FindRelBin(0.0, zero_at_start), 0);

  // Verify bins where rel_err >= lower is strictly required (kills Mutant 25)
  std::vector<RelErrorBin> reverse_bins = {
      {0.5, 1.0, 0, false},
      {0.0, 0.5, 0, false},
  };
  EXPECT_EQ(FindRelBin(0.2, reverse_bins), 1);

  std::vector<std::vector<int64_t>> hist_2d = {
      {1, 2, 0},
      {3, 4, 0},
      {0, 0, 0},
  };
  auto mismatch_counts = ComputeHeatmapMismatchCounts(hist_2d, 2, 2);
  EXPECT_EQ(mismatch_counts[0][0], 4);

  // Verify SuggestedErrorSpec when max_abs_error > 0 but max_rel_error == 0
  // (kills Mutant 26)
  ErrorHeatmap dummy_heatmap;
  dummy_heatmap.abs_thresholds = {1e-3, 1e-2, 1e-1};
  dummy_heatmap.rel_thresholds = {1e-3, 1e-2, 1e-1};
  dummy_heatmap.mismatch_counts = {
      {0, 0, 0},
      {0, 0, 0},
      {0, 0, 0},
  };
  auto spec_abs_only = ComputeSuggestedErrorSpec(dummy_heatmap, 0.05, 0.0);
  ASSERT_TRUE(spec_abs_only.has_value());
  EXPECT_GE(spec_abs_only->pure_abs_bound, 0.05);
  EXPECT_TRUE(std::isinf(spec_abs_only->pure_rel_bound));
  EXPECT_TRUE(std::isinf(spec_abs_only->margin_pure_rel_bound));
  EXPECT_GE(spec_abs_only->abs_bound, 0.05);
  EXPECT_DOUBLE_EQ(spec_abs_only->rel_bound, 0.0);
}

TEST(CompareLiteralsTest, MultiPointParetoKneeSelection) {
  Literal clean = LiteralUtil::CreateR1<float>({1000.0f, 0.01f});
  Literal dirty = LiteralUtil::CreateR1<float>({1005.0f, 0.012f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-4;
  options.rel_error_bound = 1e-4;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(clean, dirty, options));

  EXPECT_FALSE(result.passed);
  ASSERT_TRUE(result.suggested_error_spec.has_value());
  const auto& spec = *result.suggested_error_spec;

  ComparisonOptions balanced_options;
  balanced_options.abs_error_bound = spec.abs_bound;
  balanced_options.rel_error_bound = spec.rel_bound;
  ASSERT_OK_AND_ASSIGN(ComparisonResult balanced_result,
                       CompareLiterals(clean, dirty, balanced_options));
  EXPECT_TRUE(balanced_result.passed);
}

TEST(CompareLiteralsTest, Bfloat16AndHalfSupport) {
  Literal clean_bf16 = LiteralUtil::CreateR1<bfloat16>(
      {bfloat16(1.0f), bfloat16(2.0f), bfloat16(3.0f)});
  Literal dirty_bf16 = LiteralUtil::CreateR1<bfloat16>(
      {bfloat16(1.0f), bfloat16(2.5f), bfloat16(3.0f)});

  ComparisonOptions options;
  options.abs_error_bound = 0.05;
  options.rel_error_bound = 0.05;
  ASSERT_OK_AND_ASSIGN(ComparisonResult bf16_result,
                       CompareLiterals(clean_bf16, dirty_bf16, options));
  EXPECT_FALSE(bf16_result.passed);
  EXPECT_EQ(bf16_result.mismatches, 1);

  Literal clean_f16 =
      LiteralUtil::CreateR1<half>({half(1.0f), half(2.0f), half(3.0f)});
  Literal dirty_f16 =
      LiteralUtil::CreateR1<half>({half(1.0f), half(2.0f), half(3.0f)});
  ASSERT_OK_AND_ASSIGN(ComparisonResult f16_result,
                       CompareLiterals(clean_f16, dirty_f16, options));
  EXPECT_TRUE(f16_result.passed);
}

TEST(CompareLiteralsTest, DoubleAndInt32AndBoolSupport) {
  Literal clean_f64 = LiteralUtil::CreateR1<double>({1.0, 2.0});
  Literal dirty_f64 = LiteralUtil::CreateR1<double>({1.0, 2.0});
  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult f64_result,
                       CompareLiterals(clean_f64, dirty_f64, options));
  EXPECT_TRUE(f64_result.passed);

  Literal clean_s32 = LiteralUtil::CreateR1<int32_t>({10, 20});
  Literal dirty_s32 = LiteralUtil::CreateR1<int32_t>({10, 25});
  ASSERT_OK_AND_ASSIGN(ComparisonResult s32_result,
                       CompareLiterals(clean_s32, dirty_s32, options));
  EXPECT_FALSE(s32_result.passed);
  EXPECT_EQ(s32_result.mismatches, 1);

  Literal clean_bool = LiteralUtil::CreateR1<bool>({true, false});
  Literal dirty_bool = LiteralUtil::CreateR1<bool>({true, false});
  ASSERT_OK_AND_ASSIGN(ComparisonResult bool_result,
                       CompareLiterals(clean_bool, dirty_bool, options));
  EXPECT_TRUE(bool_result.passed);
}

TEST(CompareLiteralsTest, ComplexLiteralsEndToEnd) {
  Literal clean = LiteralUtil::CreateR1<complex64>(
      {complex64(1.0f, 2.0f), complex64(3.0f, 4.0f)});
  Literal dirty = LiteralUtil::CreateR1<complex64>(
      {complex64(1.0f, 2.0f), complex64(4.0f, 4.0f)});

  ComparisonOptions options;
  options.abs_error_bound = 0.1;
  options.rel_error_bound = 0.1;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(clean, dirty, options));
  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.exact_matches, 1);
  EXPECT_EQ(result.mismatches, 1);
}

TEST(CompareLiteralsTest, NonArrayTupleLiteralRejected) {
  Literal lit1 = LiteralUtil::MakeTupleFromSlices({});
  Literal lit2 = LiteralUtil::MakeTupleFromSlices({});

  ComparisonOptions options;
  auto result = CompareLiterals(lit1, lit2, options);
  EXPECT_THAT(result.status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Only array literals are supported")));
}

TEST(CompareLiteralsTest, MaxMismatchesCapRespected) {
  std::vector<float> clean_vals(20, 1.0f);
  std::vector<float> dirty_vals(20, 2.0f);
  Literal clean = LiteralUtil::CreateR1<float>(clean_vals);
  Literal dirty = LiteralUtil::CreateR1<float>(dirty_vals);

  ComparisonOptions options;
  options.max_mismatches_to_record = 5;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(clean, dirty, options));
  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.mismatches, 20);
  EXPECT_EQ(result.top_mismatches.size(), 5);
}

}  // namespace
}  // namespace xla::compare_literals
