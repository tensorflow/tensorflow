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
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/tools/compare_literals/compare_literals.h"

namespace xla::compare_literals {

std::vector<double> BuildThresholds(double target) {
  std::vector<double> thresholds;
  thresholds.reserve(35);

  for (int e = -7; e <= 2; ++e) {
    double base = std::pow(10.0, e);
    for (double m : kDefaultMultipliers) {
      thresholds.push_back(m * base);
    }
  }
  thresholds.push_back(1e3);

  if (target > 0.0) {
    thresholds.push_back(target);
  }
  absl::c_sort(thresholds);
  thresholds.erase(std::unique(thresholds.begin(), thresholds.end()),
                   thresholds.end());
  return thresholds;
}

std::vector<RelErrorBin> CreateDefaultRelBins() {
  constexpr double kInfinity = std::numeric_limits<double>::infinity();

  // Positive boundaries: 0.0, 2e-6, 5e-6, 1e-5, ..., 10.0, +inf
  std::vector<double> pos_bounds = {0.0};
  for (int e = -6; e <= 0; ++e) {
    double base = std::pow(10.0, e);
    for (double m : kDefaultMultipliers) {
      if (e == -6 && m == 1.0) {
        continue;
      }
      pos_bounds.push_back(m * base);
    }
  }
  pos_bounds.push_back(10.0);
  pos_bounds.push_back(kInfinity);

  std::vector<RelErrorBin> bins;
  bins.reserve(2 * pos_bounds.size() - 1);

  // Negative bins: ordered ascending from -inf to 0
  for (int i = static_cast<int>(pos_bounds.size()) - 1; i >= 1; --i) {
    bins.push_back({-pos_bounds[i], -pos_bounds[i - 1], 0, false});
  }

  // Exact zero bin
  bins.push_back({0.0, 0.0, 0, true});

  // Positive bins: ordered ascending from 0 to +inf
  for (size_t i = 1; i < pos_bounds.size(); ++i) {
    bins.push_back({pos_bounds[i - 1], pos_bounds[i], 0, false});
  }

  CHECK(std::isinf(bins.begin()->lower));
  CHECK(std::isinf(bins.rbegin()->upper));
  return bins;
}

int FindRelBin(double rel_err, absl::Span<const RelErrorBin> bins) {
  if (rel_err == 0.0) {
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].is_exact_zero) {
        return static_cast<int>(i);
      }
    }
  }
  for (size_t i = 0; i < bins.size(); ++i) {
    if (bins[i].is_exact_zero) {
      continue;
    }
    if (rel_err >= bins[i].lower && rel_err < bins[i].upper) {
      return static_cast<int>(i);
    }
  }
  if (rel_err < bins.front().upper) {
    return 0;
  }
  return static_cast<int>(bins.size() - 1);
}

std::vector<std::vector<int64_t>> ComputeHeatmapMismatchCounts(
    const std::vector<std::vector<int64_t>>& hist_2d, int num_rel_thresh,
    int num_abs_thresh) {
  std::vector<std::vector<int64_t>> mismatch_counts(
      num_rel_thresh, std::vector<int64_t>(num_abs_thresh, 0));

  std::vector<std::vector<int64_t>> suffix(
      num_rel_thresh + 2, std::vector<int64_t>(num_abs_thresh + 2, 0));

  for (int r = num_rel_thresh; r >= 0; --r) {
    for (int a = num_abs_thresh; a >= 0; --a) {
      suffix[r][a] = hist_2d[r][a] + suffix[r + 1][a] + suffix[r][a + 1] -
                     suffix[r + 1][a + 1];
    }
  }

  for (int r = 0; r < num_rel_thresh; ++r) {
    for (int a = 0; a < num_abs_thresh; ++a) {
      mismatch_counts[r][a] = suffix[r + 1][a + 1];
    }
  }

  return mismatch_counts;
}

std::optional<SuggestedErrorSpec> ComputeSuggestedErrorSpec(
    const ErrorHeatmap& heatmap, double max_abs_error, double max_rel_error) {
  if (!std::isfinite(max_abs_error) || !std::isfinite(max_rel_error)) {
    return std::nullopt;
  }
  if (heatmap.abs_thresholds.empty() || heatmap.rel_thresholds.empty() ||
      heatmap.mismatch_counts.empty()) {
    return std::nullopt;
  }
  if (max_abs_error == 0.0 && max_rel_error == 0.0) {
    return SuggestedErrorSpec{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }

  SuggestedErrorSpec spec;

  // Pure absolute bound: smallest threshold >= max_abs_error
  auto abs_pure_it = absl::c_lower_bound(heatmap.abs_thresholds, max_abs_error);
  spec.pure_abs_bound = (abs_pure_it != heatmap.abs_thresholds.end())
                            ? *abs_pure_it
                            : max_abs_error;
  auto m_abs_pure_it =
      absl::c_lower_bound(heatmap.abs_thresholds, 2.0 * spec.pure_abs_bound);
  spec.margin_pure_abs_bound = (m_abs_pure_it != heatmap.abs_thresholds.end())
                                   ? *m_abs_pure_it
                                   : 2.0 * spec.pure_abs_bound;

  // When mismatches exist solely against reference values of 0.0, max_rel_error
  // remains 0.0. A relative tolerance cannot satisfy differences against
  // reference zeros, so a pure relative bound is impossible (+inf) and no
  // (a, r) trade-off exists.
  if (max_abs_error > 0.0 && max_rel_error == 0.0) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    spec.abs_bound = spec.pure_abs_bound;
    spec.rel_bound = 0.0;
    spec.margin_abs_bound = spec.margin_pure_abs_bound;
    spec.margin_rel_bound = 0.0;
    spec.pure_rel_bound = kInf;
    spec.margin_pure_rel_bound = kInf;
    return spec;
  }

  // Pure relative bound: smallest threshold >= max_rel_error
  auto rel_pure_it = absl::c_lower_bound(heatmap.rel_thresholds, max_rel_error);
  spec.pure_rel_bound = (rel_pure_it != heatmap.rel_thresholds.end())
                            ? *rel_pure_it
                            : max_rel_error;
  auto m_rel_pure_it =
      absl::c_lower_bound(heatmap.rel_thresholds, 2.0 * spec.pure_rel_bound);
  spec.margin_pure_rel_bound = (m_rel_pure_it != heatmap.rel_thresholds.end())
                                   ? *m_rel_pure_it
                                   : 2.0 * spec.pure_rel_bound;

  // Pareto frontier for balanced (a, r):
  // For each abs threshold a, find smallest rel threshold r where
  // mismatch_counts[r][a] == 0.
  const int num_abs = heatmap.abs_thresholds.size();
  const int num_rel = heatmap.rel_thresholds.size();

  struct Candidate {
    double a_val;
    double r_val;
  };
  std::vector<Candidate> frontier;
  int prev_r = num_rel;
  for (int a = 0; a < num_abs; ++a) {
    double a_val = heatmap.abs_thresholds[a];
    if (a_val >= spec.pure_abs_bound) {
      break;
    }

    int found_r = -1;
    for (int r = 0; r < num_rel; ++r) {
      if (heatmap.mismatch_counts[r][a] == 0) {
        found_r = r;
        break;
      }
    }
    if (found_r != -1 && found_r < prev_r) {
      double r_val = heatmap.rel_thresholds[found_r];
      if (r_val < spec.pure_rel_bound) {
        frontier.push_back({a_val, r_val});
        prev_r = found_r;
      }
    }
  }

  if (frontier.empty()) {
    spec.abs_bound = spec.pure_abs_bound;
    spec.rel_bound = spec.pure_rel_bound;
    spec.margin_abs_bound = spec.margin_pure_abs_bound;
    spec.margin_rel_bound = spec.margin_pure_rel_bound;
    return spec;
  }

  if (frontier.size() == 1) {
    spec.abs_bound = frontier[0].a_val;
    spec.rel_bound = frontier[0].r_val;
  } else {
    // Find knee in normalized log-space.
    // In frontier, a_val increases from front to back, and r_val decreases from
    // front to back.
    double min_u = std::log10(std::max(1e-15, frontier.front().a_val));
    double max_u = std::log10(std::max(1e-15, frontier.back().a_val));
    double min_v = std::log10(std::max(1e-15, frontier.back().r_val));
    double max_v = std::log10(std::max(1e-15, frontier.front().r_val));

    double span_u = max_u - min_u;
    double span_v = max_v - min_v;

    double best_dist = std::numeric_limits<double>::infinity();
    int best_idx = 0;

    for (size_t i = 0; i < frontier.size(); ++i) {
      double u = std::log10(std::max(1e-15, frontier[i].a_val));
      double v = std::log10(std::max(1e-15, frontier[i].r_val));
      double norm_u = span_u > 1e-9 ? (u - min_u) / span_u : 0.0;
      double norm_v = span_v > 1e-9 ? (v - min_v) / span_v : 0.0;
      double dist = norm_u * norm_u + norm_v * norm_v;
      if (dist < best_dist) {
        best_dist = dist;
        best_idx = i;
      }
    }

    spec.abs_bound = frontier[best_idx].a_val;
    spec.rel_bound = frontier[best_idx].r_val;
  }

  auto m_abs_it =
      absl::c_lower_bound(heatmap.abs_thresholds, 2.0 * spec.abs_bound);
  spec.margin_abs_bound = (m_abs_it != heatmap.abs_thresholds.end())
                              ? *m_abs_it
                              : 2.0 * spec.abs_bound;

  auto m_rel_it =
      absl::c_lower_bound(heatmap.rel_thresholds, 2.0 * spec.rel_bound);
  spec.margin_rel_bound = (m_rel_it != heatmap.rel_thresholds.end())
                              ? *m_rel_it
                              : 2.0 * spec.rel_bound;

  return spec;
}

}  // namespace xla::compare_literals
