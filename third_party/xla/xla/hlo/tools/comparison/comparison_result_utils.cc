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

#include "xla/hlo/tools/comparison/comparison_result_utils.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"

namespace xla::numerics::comparison {

std::vector<ColorThreshold> GetScoreThresholds() {
  return {
      {-1.0, "#d3d3d3"},                                         // Grey
      {0.0, "#99ff99"},                                          // Green
      {1.0, "#c0f580"},  {5.0, "#e0ee40"},  {10.0, "#eeee00"},   // Yellow
      {30.0, "#ffc000"}, {60.0, "#ff8000"}, {100.0, "#ff1717"},  // Red
  };
}

std::string GetColorForScore(double score) {
  auto thresholds = GetScoreThresholds();
  if (score < thresholds[1].value) {  // Handles -1.0 as well
    return thresholds[0].background_color;
  }
  if (score >= thresholds.back().value) {
    return thresholds.back().background_color;
  }

  for (int i = 1; i < thresholds.size() - 1; ++i) {
    if (score >= thresholds[i].value && score < thresholds[i + 1].value) {
      int r1, g1, b1;
      sscanf(thresholds[i].background_color.c_str(), "#%02x%02x%02x", &r1, &g1,
             &b1);
      int r2, g2, b2;
      sscanf(thresholds[i + 1].background_color.c_str(), "#%02x%02x%02x", &r2,
             &g2, &b2);

      double ratio = (score - thresholds[i].value) /
                     (thresholds[i + 1].value - thresholds[i].value);
      int r = static_cast<int>(r1 + ratio * (r2 - r1));
      int g = static_cast<int>(g1 + ratio * (g2 - g1));
      int b = static_cast<int>(b1 + ratio * (b2 - b1));
      return absl::StrFormat("#%02x%02x%02x", r, g, b);
    }
  }
  return "#ffffff";
}

// We use p=4 for generalized mean to compute the final score from block
// differences. Generalized mean is defined as M_p(x_1,...,x_n) = (1/n *
// sum(x_i^p))^(1/p).
// When p=1, M_p is arithmetic mean, which tends to average out local
// differences.
// When p=inf, M_p is max, which is sensitive to single block difference but
// may not be as comparable across tensors with different number of blocks.
// We choose p=4 as a compromise between p=1 and p=inf: it gives a much
// larger weight to blocks with large differences compared to arithmetic mean,
// thus being able to capture local changes, but it is less sensitive to
// number of blocks than max.
// For scores across different data sets (data parallelism due to manual
// sharding), we take the max score.
static constexpr int kGeneralizedMeanPower = 4;

double ComputeDiffScore(const ComparisonResultProto& result) {
  if (result.baseline_tensor_summaries_size() !=
      result.target_tensor_summaries_size()) {
    return -1;
  }
  if (result.baseline_tensor_summaries_size() == 0) {
    return -1.0;
  }

  double max_score = 0.0;

  for (int summary_idx = 0;
       summary_idx < result.baseline_tensor_summaries_size(); ++summary_idx) {
    const auto& baseline_summary =
        result.baseline_tensor_summaries(summary_idx);
    const auto& target_summary = result.target_tensor_summaries(summary_idx);

    if (baseline_summary.block_summaries_size() !=
        target_summary.block_summaries_size()) {
      return -1;
    }

    if (baseline_summary.block_summaries_size() == 0) {
      continue;
    }

    std::vector<double> block_diffs;
    constexpr double kEpsilon = 1e-15;
    // kAbsoluteTolerance is used to suppress large diff scores when both
    // values are small. There is no one-size-fits-all value for this
    // tolerance. This value is chosen based on experience. Users can
    // compute a custom score using raw summary data if this default is
    // not suitable.
    constexpr double kAbsoluteTolerance = 1e-2;

    for (int i = 0; i < baseline_summary.block_summaries_size(); ++i) {
      const auto& baseline_block = baseline_summary.block_summaries(i);
      const auto& target_block = target_summary.block_summaries(i);

      if (baseline_block.count() != target_block.count()) {
        return -1;
      }

      auto rel_diff = [&](float a, float b) -> double {
        if (std::isnan(a)) {
          return std::isnan(b) ? 0.0 : 1.0;
        }
        if (std::isnan(b)) {
          return 1.0;
        }
        if (a == b) {
          return 0.0;  // Also handles inf.
        }
        if (std::isinf(a) || std::isinf(b)) {
          return 1.0;
        }
        double denom =
            kEpsilon + kAbsoluteTolerance + std::abs(a) + std::abs(b);
        return std::abs(a - b) / denom;
      };

      double diff_mean = rel_diff(baseline_block.mean(), target_block.mean());
      double diff_stddev =
          rel_diff(baseline_block.stddev(), target_block.stddev());
      double diff_min = rel_diff(baseline_block.min(), target_block.min());
      double diff_max = rel_diff(baseline_block.max(), target_block.max());
      double diff_nan_count =
          rel_diff(baseline_block.nan_count(), target_block.nan_count());
      double diff_pos_inf_count = rel_diff(baseline_block.pos_inf_count(),
                                           target_block.pos_inf_count());
      double diff_neg_inf_count = rel_diff(baseline_block.neg_inf_count(),
                                           target_block.neg_inf_count());
      double diff_zero_count =
          rel_diff(baseline_block.zero_count(), target_block.zero_count());

      double block_sum_pow4 =
          std::pow(diff_mean, kGeneralizedMeanPower) +
          std::pow(diff_stddev, kGeneralizedMeanPower) +
          (std::pow(diff_min, kGeneralizedMeanPower) +
           std::pow(diff_max, kGeneralizedMeanPower)) /
              2 +
          (std::pow(diff_nan_count, kGeneralizedMeanPower) +
           std::pow(diff_pos_inf_count, kGeneralizedMeanPower) +
           std::pow(diff_neg_inf_count, kGeneralizedMeanPower) +
           std::pow(diff_zero_count, kGeneralizedMeanPower)) /
              4;
      block_diffs.push_back(
          std::pow(block_sum_pow4 / 8.0, 1.0 / kGeneralizedMeanPower));
    }

    double sum_pow4 = 0.0;
    for (double diff : block_diffs) {
      sum_pow4 += std::pow(diff, kGeneralizedMeanPower);
    }
    double current_score =
        std::pow(sum_pow4 / block_diffs.size(), 1.0 / kGeneralizedMeanPower);
    if (current_score > max_score) {
      max_score = current_score;
    }
  }

  return max_score * 100.0;
}

}  // namespace xla::numerics::comparison
