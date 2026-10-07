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

#ifndef XLA_TOOLS_COMPARE_LITERALS_COMPARE_LITERALS_H_
#define XLA_TOOLS_COMPARE_LITERALS_COMPARE_LITERALS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/xla_data.pb.h"

namespace xla::compare_literals {

struct ComparisonOptions {
  double abs_error_bound = 1e-3;
  double rel_error_bound = 1e-3;
  int max_mismatches_to_record = 10;
  double heatmap_yellow_pct = 0.5;
};

// Represents a bin in the 1D relative error distribution.
struct RelErrorBin {
  double lower = 0.0;
  double upper = 0.0;
  int64_t count = 0;
  bool is_exact_zero = false;
};

// 1D Relative Error Histogram with ASCII formatting.
struct RelErrorHistogram {
  std::vector<RelErrorBin> bins;
  int64_t total_samples = 0;
  double min_rel_error = 0.0;
  double max_rel_error = 0.0;
  double mean_rel_error = 0.0;
  double std_dev_rel_error = 0.0;
  int median_bin_index = -1;

  // Formats as an ASCII bar chart similar to dot_algorithms_test.cc.
  std::string ToString(int max_bar_width = 40) const;
  // Formats natively as a Markdown table.
  std::string ToMarkdown(int max_bar_width = 30) const;
};

// 2D Heatmap of element mismatches for pairs of (abs_threshold, rel_threshold).
struct ErrorHeatmap {
  // Sorted threshold boundaries.
  std::vector<double> abs_thresholds;
  std::vector<double> rel_thresholds;

  // 2D grid: mismatch_counts[rel_idx][abs_idx] is the number of elements
  // having abs_diff > abs_thresholds[abs_idx] AND
  // rel_diff > rel_thresholds[rel_idx].
  std::vector<std::vector<int64_t>> mismatch_counts;

  // The user's target tolerance parameters.
  double target_abs = 0.0;
  double target_rel = 0.0;
  int target_abs_idx = -1;
  int target_rel_idx = -1;
  int64_t total_elements = 0;
  double yellow_threshold_pct = 0.5;

  // Formats the 2D matrix into a terminal-friendly table with ANSI colors.
  std::string ToString(bool use_color = true) const;
  // Formats natively as a Markdown table.
  std::string ToMarkdown() const;
};

// Detailed info for an individual element mismatch.
struct MismatchDetail {
  int64_t linear_index = 0;
  std::string clean_str;
  std::string dirty_str;
  double abs_diff = 0.0;
  double rel_diff = 0.0;
};

// Suggested error specification (abs and rel bounds) to make comparison pass.
struct SuggestedErrorSpec {
  // Balanced point on Pareto frontier (knee in log-log space).
  double abs_bound = 0.0;
  double rel_bound = 0.0;
  double margin_abs_bound = 0.0;
  double margin_rel_bound = 0.0;

  // Pure absolute bound (ignoring relative error).
  double pure_abs_bound = 0.0;
  double margin_pure_abs_bound = 0.0;

  // Pure relative bound (ignoring absolute error).
  double pure_rel_bound = 0.0;
  double margin_pure_rel_bound = 0.0;

  std::string ToString() const;
};

// Result of comparing two literals.
struct ComparisonResult {
  bool passed = false;
  std::string element_type;
  std::string shape_str;
  int64_t total_elements = 0;
  int64_t exact_matches = 0;
  int64_t mismatches = 0;
  int64_t nan_mismatches = 0;
  int64_t inf_mismatches = 0;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;

  std::vector<MismatchDetail> top_mismatches;
  RelErrorHistogram histogram;
  ErrorHeatmap heatmap;
  std::optional<SuggestedErrorSpec> suggested_error_spec;

  std::string SummaryToString(bool use_color = true) const;
  std::string SummaryToMarkdown() const;
};

// Standard decade multipliers (1-2-5 sequence) used for threshold grids and
// histogram binning.
inline constexpr double kDefaultMultipliers[] = {1.0, 2.0, 5.0};

// Builds a wide grid of thresholds covering 10^-7 to 10^2 with {1, 2, 5}
// steps per decade, ensuring the target tolerance is always included.
std::vector<double> BuildThresholds(double target);

// Subdivided boundaries with {1, 2, 5} steps per decade.
std::vector<RelErrorBin> CreateDefaultRelBins();

// Finds the 1D histogram bin index for a given relative error.
int FindRelBin(double rel_err, absl::Span<const RelErrorBin> bins);

// Computes the 2D suffix sum matrix from the raw 2D histogram.
std::vector<std::vector<int64_t>> ComputeHeatmapMismatchCounts(
    const std::vector<std::vector<int64_t>>& hist_2d, int num_rel_thresh,
    int num_abs_thresh);

// Computes suggested ErrorSpec (balanced on Pareto frontier, pure abs, pure
// rel) based on the 2D heatmap suffix sums and max recorded errors.
std::optional<SuggestedErrorSpec> ComputeSuggestedErrorSpec(
    const ErrorHeatmap& heatmap, double max_abs_error, double max_rel_error);

// Compares two LiteralSlice objects and computes statistics, 1D histogram, and
// 2D heatmap in a single pass.
absl::StatusOr<ComparisonResult> CompareLiterals(
    const LiteralSlice& clean, const LiteralSlice& dirty,
    const ComparisonOptions& options);

// Compares two LiteralProto objects.
absl::StatusOr<ComparisonResult> CompareLiteralProtos(
    const LiteralProto& clean_proto, const LiteralProto& dirty_proto,
    const ComparisonOptions& options);

// Reads two LiteralProto binary files from disk and compares them.
absl::StatusOr<ComparisonResult> CompareLiteralFiles(
    absl::string_view clean_file, absl::string_view dirty_file,
    const ComparisonOptions& options);

}  // namespace xla::compare_literals

#endif  // XLA_TOOLS_COMPARE_LITERALS_COMPARE_LITERALS_H_
