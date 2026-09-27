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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/tools/compare_literals/compare_literals.h"

namespace xla::compare_literals {
namespace {

std::string FormatCompactSci(double v) {
  if (std::isnan(v)) {
    return "nan";
  }
  if (std::isinf(v)) {
    return v < 0 ? "-inf" : "inf";
  }
  if (v == 0.0) {
    return "0";
  }
  bool negative = v < 0.0;
  if (negative) {
    v = -v;
  }

  int exp = static_cast<int>(std::floor(std::log10(v) + 1e-9));
  exp = std::clamp(exp, std::numeric_limits<double>::min_exponent10,
                   std::numeric_limits<double>::max_exponent10);
  double divisor = std::pow(10.0, exp);
  double mantissa =
      (divisor > 0.0) ? std::round((v / divisor) * 1e6) / 1e6 : 0.0;
  if (mantissa == 0.0) {
    return "0";
  }
  if (mantissa >= 10.0) {
    mantissa /= 10.0;
    exp += 1;
  }

  return absl::StrFormat("%s%ge%d", negative ? "-" : "", mantissa, exp);
}

std::string FormatBar(int64_t count, int64_t max_count, int max_bar_width) {
  int bar_len = (max_count > 0)
                    ? static_cast<int>(std::round(static_cast<double>(count) *
                                                  max_bar_width / max_count))
                    : 0;
  if (count > 0 && bar_len == 0) {
    bar_len = 1;
  }
  return std::string(bar_len, '#');
}

struct HeatmapWindow {
  int col_start;
  int col_end;
  int row_start;
  int row_end;
};

HeatmapWindow ComputeHeatmapWindow(
    const std::vector<std::vector<int64_t>>& mismatch_counts,
    int num_cols_total, int num_rows_total, int target_abs_idx,
    int target_rel_idx, int max_cols, int max_rows) {
  CHECK_GT(num_cols_total, 0);
  CHECK_GT(num_rows_total, 0);
  const int target_abs = std::clamp(target_abs_idx >= 0 ? target_abs_idx : 0, 0,
                                    num_cols_total - 1);
  const int target_rel = std::clamp(target_rel_idx >= 0 ? target_rel_idx : 0, 0,
                                    num_rows_total - 1);

  // Find first column where all rows are zero.
  // Because row 0 is the strictest relative threshold, if
  // mismatch_counts[0][a] == 0, then mismatch_counts[r][a] == 0 for all r >= 0.
  int col_zero = num_cols_total - 1;
  for (int a = 0; a < num_cols_total; ++a) {
    if (mismatch_counts[0][a] == 0) {
      col_zero = a;
      break;
    }
  }

  // Find first row where all columns are zero.
  // Because col 0 is the strictest absolute threshold, if
  // mismatch_counts[r][0] == 0, then mismatch_counts[r][a] == 0 for all a >= 0.
  int row_zero = num_rows_total - 1;
  for (int r = 0; r < num_rows_total; ++r) {
    if (mismatch_counts[r][0] == 0) {
      row_zero = r;
      break;
    }
  }

  int col_end = std::min(num_cols_total - 1, std::max(target_abs, col_zero));
  if (col_end == 0) {
    col_end = std::min(num_cols_total - 1, max_cols - 1);
  }
  int col_start = std::max(0, col_end - max_cols + 1);
  if (col_start > target_abs) {
    col_start = target_abs;
    col_end = std::min(num_cols_total - 1, col_start + max_cols - 1);
  }

  int row_end = std::min(num_rows_total - 1, std::max(target_rel, row_zero));
  if (row_end == 0) {
    row_end = std::min(num_rows_total - 1, max_rows - 1);
  }
  int row_start = std::max(0, row_end - max_rows + 1);
  if (row_start > target_rel) {
    row_start = target_rel;
    row_end = std::min(num_rows_total - 1, row_start + max_rows - 1);
  }

  return {col_start, col_end, row_start, row_end};
}

}  // namespace

std::string RelErrorHistogram::ToString(int max_bar_width) const {
  std::string out =
      "1D Signed Relative Error Distribution ((actual - expected) / "
      "|expected|):\n";
  int64_t max_bin_count = 0;
  for (const auto& b : bins) {
    max_bin_count = std::max(max_bin_count, b.count);
  }

  for (size_t i = 0; i < bins.size(); ++i) {
    const auto& b = bins[i];
    if (b.count == 0 && static_cast<int>(i) != median_bin_index &&
        !b.is_exact_zero) {
      continue;
    }

    std::string bar = FormatBar(b.count, max_bin_count, max_bar_width);

    std::string markers;
    if (static_cast<int>(i) == median_bin_index) {
      markers += " <--- median";
    }
    if (b.is_exact_zero) {
      markers += " <--- exact match (zero)";
    } else if (mean_rel_error >= b.lower && mean_rel_error < b.upper) {
      markers += " <--- mean";
    }

    std::string range_str;
    if (b.is_exact_zero) {
      range_str = "     [  0  ]    ";
    } else {
      char left_bracket = std::isinf(b.lower) ? '(' : '[';
      char right_bracket = ')';
      std::string lower_str =
          std::isinf(b.lower) ? "-inf" : FormatCompactSci(b.lower);
      std::string upper_str =
          std::isinf(b.upper) ? "+inf" : FormatCompactSci(b.upper);
      range_str = absl::StrFormat("%c%6s, %6s%c", left_bracket, lower_str,
                                  upper_str, right_bracket);
    }

    double pct = total_samples > 0
                     ? (100.0 * static_cast<double>(b.count) / total_samples)
                     : 0.0;
    std::string pct_str;
    if (b.count > 0 && pct < 0.05) {
      pct_str = " <0.1%";
    } else {
      pct_str = absl::StrFormat("%5.1f%%", pct);
    }

    absl::StrAppendFormat(&out, "  %2d: %s %8lld (%s) %s%s\n", i, range_str,
                          b.count, pct_str, bar, markers);
  }

  absl::StrAppendFormat(
      &out,
      "  Summary: min = %1.3e | max = %1.3e | mean = %1.3e | std_dev = %1.3e\n",
      min_rel_error, max_rel_error, mean_rel_error, std_dev_rel_error);
  return out;
}

std::string RelErrorHistogram::ToMarkdown(int max_bar_width) const {
  if (bins.empty() || total_samples == 0) {
    return "";
  }

  int64_t max_bin_count = 0;
  for (const auto& b : bins) {
    max_bin_count = std::max(max_bin_count, b.count);
  }

  std::string out;
  absl::StrAppend(&out,
                  "| Markers | Range | Count | Percent | Distribution |\n");
  absl::StrAppend(&out, "| :--- | :--- | :---: | :---: | :--- |\n");

  for (size_t i = 0; i < bins.size(); ++i) {
    const auto& b = bins[i];
    if (b.count == 0 && static_cast<int>(i) != median_bin_index &&
        !b.is_exact_zero) {
      continue;
    }

    double pct = 100.0 * static_cast<double>(b.count) / total_samples;
    std::string pct_str =
        (b.count > 0 && pct < 0.05) ? "<0.1%" : absl::StrFormat("%.1f%%", pct);

    std::vector<std::string> marker_labels;
    if (static_cast<int>(i) == median_bin_index) {
      marker_labels.push_back("**Median**");
    }
    if (b.is_exact_zero) {
      marker_labels.push_back("**Zero**");
    }
    if (!b.is_exact_zero && mean_rel_error >= b.lower &&
        mean_rel_error < b.upper) {
      marker_labels.push_back("**Mean**");
    }
    std::string marker_str = absl::StrJoin(marker_labels, ", ");

    std::string range_str;
    if (b.is_exact_zero) {
      range_str = "`[0]`";
    } else if (std::isinf(b.lower)) {
      range_str = absl::StrFormat("`(-inf, %s)`", FormatCompactSci(b.upper));
    } else if (std::isinf(b.upper)) {
      range_str = absl::StrFormat("`(%s, +inf)`", FormatCompactSci(b.lower));
    } else {
      range_str = absl::StrFormat("`[%s, %s)`", FormatCompactSci(b.lower),
                                  FormatCompactSci(b.upper));
    }

    std::string bar = FormatBar(b.count, max_bin_count, max_bar_width);
    absl::StrAppendFormat(&out, "| %s | %s | %lld | %s | `%s` |\n", marker_str,
                          range_str, b.count, pct_str, bar);
  }

  return out;
}

std::string ErrorHeatmap::ToString(bool use_color) const {
  if (abs_thresholds.empty() || rel_thresholds.empty()) {
    return "";
  }

  const int num_cols_total = static_cast<int>(abs_thresholds.size());
  const int num_rows_total = static_cast<int>(rel_thresholds.size());
  constexpr int kMaxCols = 13;
  constexpr int kMaxRows = 25;

  HeatmapWindow win =
      ComputeHeatmapWindow(mismatch_counts, num_cols_total, num_rows_total,
                           target_abs_idx, target_rel_idx, kMaxCols, kMaxRows);
  const int col_start = win.col_start;
  const int col_end = win.col_end;
  const int row_start = win.row_start;
  const int row_end = win.row_end;

  std::string out =
      "2D Error Heatmap (Percentage of elements failing: abs_diff > X AND "
      "rel_diff > Y):\n";

  // Table header (from largest abs_threshold down to smallest)
  absl::StrAppend(&out, "    Rel \\ Abs  |");
  for (int a = col_end; a >= col_start; --a) {
    std::string col_hdr = FormatCompactSci(abs_thresholds[a]);
    if (a == target_abs_idx) {
      col_hdr = absl::StrCat("*", col_hdr);
    }
    if (col_hdr.size() > 8) {
      col_hdr = col_hdr.substr(0, 8);
    }
    absl::StrAppendFormat(&out, " %8s |", col_hdr);
  }
  absl::StrAppend(&out, "\n    -----------+");
  for (int a = col_end; a >= col_start; --a) {
    absl::StrAppend(&out, "----------+");
  }
  absl::StrAppend(&out, "\n");

  // Rows from smallest rel_threshold up to largest
  for (int r = row_start; r <= row_end; ++r) {
    std::string row_hdr = FormatCompactSci(rel_thresholds[r]);
    if (r == target_rel_idx) {
      row_hdr = absl::StrCat("*", row_hdr);
    }
    if (row_hdr.size() > 10) {
      row_hdr = row_hdr.substr(0, 10);
    }
    absl::StrAppendFormat(&out, "    %10s |", row_hdr);

    for (int a = col_end; a >= col_start; --a) {
      int64_t count = mismatch_counts[r][a];
      double pct = total_elements > 0
                       ? (100.0 * static_cast<double>(count) / total_elements)
                       : 0.0;
      bool is_target = (r == target_rel_idx && a == target_abs_idx);

      std::string val_str;
      if (count == 0) {
        val_str = "0.0%";
      } else if (pct < 0.05) {
        val_str = "<0.1%";
      } else {
        val_str = absl::StrFormat("%.1f%%", pct);
      }

      std::string cell_str;
      if (is_target) {
        cell_str = absl::StrFormat("[%s]", val_str);
        if (cell_str.size() < 8) {
          int left = (8 - static_cast<int>(cell_str.size())) / 2;
          int right = 8 - static_cast<int>(cell_str.size()) - left;
          cell_str = absl::StrCat(std::string(left, ' '), cell_str,
                                  std::string(right, ' '));
        }
      } else {
        int pad = 8 - static_cast<int>(val_str.size());
        int left = pad / 2;
        int right = pad - left;
        cell_str = absl::StrCat(std::string(left, ' '), val_str,
                                std::string(right, ' '));
      }

      if (use_color) {
        if (count == 0) {
          cell_str = absl::StrCat("\033[32m", cell_str, "\033[0m");
        } else if (pct <= yellow_threshold_pct) {
          cell_str = absl::StrCat("\033[33m", cell_str, "\033[0m");
        } else {
          cell_str = absl::StrCat("\033[31m", cell_str, "\033[0m");
        }
        if (is_target) {
          cell_str = absl::StrCat("\033[1m", cell_str);
        }
      }
      absl::StrAppendFormat(&out, " %s |", cell_str);
    }
    absl::StrAppend(&out, "\n");
  }

  absl::StrAppend(&out, "    -----------+");
  for (int a = col_end; a >= col_start; --a) {
    absl::StrAppend(&out, "----------+");
  }
  absl::StrAppend(&out, "\n");
  absl::StrAppendFormat(
      &out,
      "    Legend: [*] Target Tolerance (abs, rel)  |  Green = 0.0%% (100%% "
      "pass), Yellow <= %.1f%%, Red > %.1f%% failures\n",
      yellow_threshold_pct, yellow_threshold_pct);

  return out;
}

std::string ErrorHeatmap::ToMarkdown() const {
  if (abs_thresholds.empty() || rel_thresholds.empty()) {
    return "";
  }

  const int num_cols_total = static_cast<int>(abs_thresholds.size());
  const int num_rows_total = static_cast<int>(rel_thresholds.size());
  constexpr int kMaxCols = 13;
  constexpr int kMaxRows = 25;

  HeatmapWindow win =
      ComputeHeatmapWindow(mismatch_counts, num_cols_total, num_rows_total,
                           target_abs_idx, target_rel_idx, kMaxCols, kMaxRows);
  const int col_start = win.col_start;
  const int col_end = win.col_end;
  const int row_start = win.row_start;
  const int row_end = win.row_end;

  std::string out;
  absl::StrAppend(&out, "| Rel \\ Abs |");
  for (int a = col_end; a >= col_start; --a) {
    std::string col_hdr = FormatCompactSci(abs_thresholds[a]);
    if (a == target_abs_idx) {
      absl::StrAppendFormat(&out, " **%s** *(Target)* |", col_hdr);
    } else {
      absl::StrAppendFormat(&out, " %s |", col_hdr);
    }
  }
  absl::StrAppend(&out, "\n| :--- |");
  for (int a = col_end; a >= col_start; --a) {
    absl::StrAppend(&out, " :---: |");
  }
  absl::StrAppend(&out, "\n");

  for (int r = row_start; r <= row_end; ++r) {
    std::string row_hdr = FormatCompactSci(rel_thresholds[r]);
    if (r == target_rel_idx) {
      absl::StrAppendFormat(&out, "| **%s** *(Target)* |", row_hdr);
    } else {
      absl::StrAppendFormat(&out, "| %s |", row_hdr);
    }

    for (int a = col_end; a >= col_start; --a) {
      int64_t count = mismatch_counts[r][a];
      double pct = total_elements > 0
                       ? (100.0 * static_cast<double>(count) / total_elements)
                       : 0.0;
      bool is_target = (r == target_rel_idx && a == target_abs_idx);

      std::string val_str;
      const char* tile = "🟩";
      if (count == 0) {
        tile = "🟩";
        val_str = "0.0%";
      } else {
        val_str = (pct < 0.05) ? "<0.1%" : absl::StrFormat("%.1f%%", pct);
        if (pct <= yellow_threshold_pct) {
          tile = "🟨";
        } else {
          tile = "🟥";
        }
      }

      if (is_target) {
        absl::StrAppendFormat(&out, " %s **[%s]** |", tile, val_str);
      } else {
        absl::StrAppendFormat(&out, " %s %s |", tile, val_str);
      }
    }
    absl::StrAppend(&out, "\n");
  }

  absl::StrAppendFormat(
      &out,
      "\n*Legend: 🟩 0.0%% failures (100%% pass), 🟨 <= %.1f%% failures, "
      "🟥 > %.1f%% failures. Cells indicate percentage of elements failing: "
      "`abs_diff > Abs` AND `rel_diff > Rel`. Target tolerance `(abs, rel)` "
      "is highlighted with `[ ]`.*\n",
      yellow_threshold_pct, yellow_threshold_pct);

  return out;
}

std::string ComparisonResult::SummaryToString(bool use_color) const {
  std::string out;
  std::string pass_str =
      use_color ? "\033[32mPASS (MATCH)\033[0m" : "PASS (MATCH)";
  std::string fail_str =
      use_color ? "\033[31mFAIL (MISMATCH)\033[0m" : "FAIL (MISMATCH)";
  absl::StrAppendFormat(&out, "Verdict: %s\n", passed ? pass_str : fail_str);
  absl::StrAppendFormat(&out, "  Element Type: %s\n", element_type);
  absl::StrAppendFormat(&out, "  Shape: %s\n", shape_str);
  absl::StrAppendFormat(&out, "  Total Elements: %lld\n", total_elements);
  double match_pct =
      total_elements > 0 ? (100.0 * exact_matches / total_elements) : 0.0;
  absl::StrAppendFormat(&out, "  Exact Matches: %lld (%1.2f%%)\n",
                        exact_matches, match_pct);
  double mismatch_pct =
      total_elements > 0 ? (100.0 * mismatches / total_elements) : 0.0;
  absl::StrAppendFormat(&out,
                        "  Mismatches (exceeding tolerance): %lld (%1.4f%%)\n",
                        mismatches, mismatch_pct);
  absl::StrAppendFormat(&out, "  NaN Mismatches: %lld\n", nan_mismatches);
  absl::StrAppendFormat(&out, "  Inf Mismatches: %lld\n", inf_mismatches);
  absl::StrAppendFormat(&out, "  Max Absolute Error: %1.4e\n", max_abs_error);
  absl::StrAppendFormat(&out, "  Max Relative Error: %1.4e\n", max_rel_error);

  if (!top_mismatches.empty()) {
    absl::StrAppend(&out, "\nFirst Mismatches (up to 10):\n");
    for (const auto& m : top_mismatches) {
      absl::StrAppendFormat(
          &out,
          "  Index %lld: clean = %s, dirty = %s (abs = %.4e, rel = %.4e)\n",
          m.linear_index, m.clean_str, m.dirty_str, m.abs_diff, m.rel_diff);
    }
  }

  if (suggested_error_spec.has_value()) {
    absl::StrAppend(&out, "\n", suggested_error_spec->ToString(), "\n");
  }

  return out;
}

std::string ComparisonResult::SummaryToMarkdown() const {
  std::string out;
  absl::StrAppend(&out, "# Comparison Report\n\n");
  absl::StrAppendFormat(&out, "**Verdict**: %s\n\n",
                        passed ? "✅ **PASS**" : "❌ **FAIL**");

  absl::StrAppend(&out, "## Summary Statistics\n\n");
  absl::StrAppend(&out, "| Metric | Value |\n");
  absl::StrAppend(&out, "| :--- | :--- |\n");
  absl::StrAppendFormat(&out, "| Element Type | `%s` |\n", element_type);
  absl::StrAppendFormat(&out, "| Shape | `%s` |\n", shape_str);
  absl::StrAppendFormat(&out, "| Total Elements | %lld |\n", total_elements);
  double match_pct =
      total_elements > 0 ? (100.0 * exact_matches / total_elements) : 0.0;
  absl::StrAppendFormat(&out, "| Exact Matches | %lld (%.2f%%) |\n",
                        exact_matches, match_pct);
  double mismatch_pct =
      total_elements > 0 ? (100.0 * mismatches / total_elements) : 0.0;
  absl::StrAppendFormat(
      &out, "| Mismatches (Exceeding Tolerance) | %lld (%.4f%%) |\n",
      mismatches, mismatch_pct);
  absl::StrAppendFormat(&out, "| NaN Mismatches | %lld |\n", nan_mismatches);
  absl::StrAppendFormat(&out, "| Inf Mismatches | %lld |\n", inf_mismatches);
  absl::StrAppendFormat(&out, "| Max Absolute Error | `%.4e` |\n",
                        max_abs_error);
  absl::StrAppendFormat(&out, "| Max Relative Error | `%.4e` |\n",
                        max_rel_error);

  if (histogram.total_samples > 0) {
    absl::StrAppend(&out, "\n## 1D Signed Relative Error Distribution\n\n");
    absl::StrAppend(&out, histogram.ToMarkdown());
    absl::StrAppendFormat(
        &out,
        "\n*Summary: min = `%.3e`, max = `%.3e`, mean = `%.3e`, std_dev = "
        "`%.3e`*\n",
        histogram.min_rel_error, histogram.max_rel_error,
        histogram.mean_rel_error, histogram.std_dev_rel_error);
  }

  if (!heatmap.abs_thresholds.empty()) {
    absl::StrAppend(&out, "\n## 2D Error Heatmap\n\n");
    absl::StrAppend(&out, heatmap.ToMarkdown());
  }

  if (!top_mismatches.empty()) {
    absl::StrAppend(&out, "\n## First Mismatches\n\n");
    absl::StrAppend(&out, "| Index | Clean | Dirty | Abs Diff | Rel Diff |\n");
    absl::StrAppend(&out, "| :---: | :--- | :--- | :---: | :---: |\n");
    for (const auto& m : top_mismatches) {
      absl::StrAppendFormat(&out, "| %lld | `%s` | `%s` | `%.4e` | `%.4e` |\n",
                            m.linear_index, m.clean_str, m.dirty_str,
                            m.abs_diff, m.rel_diff);
    }
  }

  if (suggested_error_spec.has_value()) {
    const auto& spec = *suggested_error_spec;
    absl::StrAppend(&out, "\n## Suggested ErrorSpec\n\n");
    absl::StrAppendFormat(
        &out,
        "- **Balanced**: abs = `%s`, rel = `%s` (with 2x margin: abs = `%s`, "
        "rel = `%s`)\n",
        FormatCompactSci(spec.abs_bound), FormatCompactSci(spec.rel_bound),
        FormatCompactSci(spec.margin_abs_bound),
        FormatCompactSci(spec.margin_rel_bound));
    absl::StrAppendFormat(
        &out, "- **Pure Absolute**: abs = `%s` (with 2x margin: abs = `%s`)\n",
        FormatCompactSci(spec.pure_abs_bound),
        FormatCompactSci(spec.margin_pure_abs_bound));
    absl::StrAppendFormat(
        &out, "- **Pure Relative**: rel = `%s` (with 2x margin: rel = `%s`)\n",
        FormatCompactSci(spec.pure_rel_bound),
        FormatCompactSci(spec.margin_pure_rel_bound));
  }

  return out;
}

std::string SuggestedErrorSpec::ToString() const {
  std::string s;
  absl::StrAppend(&s, "Suggested ErrorSpec (to pass all elements):\n");
  absl::StrAppendFormat(
      &s,
      "  Balanced:       abs = %s, rel = %s  (with 2x margin: abs = %s, rel = "
      "%s)\n",
      FormatCompactSci(abs_bound), FormatCompactSci(rel_bound),
      FormatCompactSci(margin_abs_bound), FormatCompactSci(margin_rel_bound));
  absl::StrAppendFormat(
      &s,
      "  Pure Absolute:  abs = %s              (with 2x margin: abs = %s)\n",
      FormatCompactSci(pure_abs_bound),
      FormatCompactSci(margin_pure_abs_bound));
  absl::StrAppendFormat(
      &s, "  Pure Relative:  rel = %s              (with 2x margin: rel = %s)",
      FormatCompactSci(pure_rel_bound),
      FormatCompactSci(margin_pure_rel_bound));
  return s;
}

}  // namespace xla::compare_literals
