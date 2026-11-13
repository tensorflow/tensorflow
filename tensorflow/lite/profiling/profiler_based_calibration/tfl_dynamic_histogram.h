/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_DYNAMIC_HISTOGRAM_H_
#define TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_DYNAMIC_HISTOGRAM_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"

namespace odml {

inline constexpr int kDefaultMaxBins = 2048;

// A 1D dynamic float histogram that automatically expands its range to
// accommodate new data points while keeping its bin count bounded by `max_bins`
// via bin-width doubling and pairwise compaction.
//
// Mirrors Python `_DynamicHistogram1D` in `histogram_utils.py`.
class DynamicHistogram {
 public:
  // Constructs an uninitialized histogram bounded to at most `max_bins` bins.
  // `max_bins` must be at least 2.
  explicit DynamicHistogram(int max_bins = kDefaultMaxBins)
      : max_bins_(max_bins) {
    ABSL_DCHECK_GE(max_bins, 2);
  }

  // Adds elements from `data` of length `n` that fall within
  // `[min_val, max_val]`, reusing the caller's precomputed range so `data` is
  // only scanned once for binning. Requires `data != nullptr`, `n > 0`, and
  // finite `min_val <= max_val`.
  void Add(const float* data, int64_t n, float min_val, float max_val);

  const std::vector<int64_t>& counts() const& { return counts_; }
  std::vector<int64_t> counts() && { return std::move(counts_); }
  double lower_bound() const { return lower_bound_; }
  double bin_width() const { return bin_width_; }

 private:
  // Initializes bin edges and counts from the first observed `[d_min, d_max]`
  // range, padded by 10% on each side (or 1e-4 when `d_min == d_max`).
  void Initialize(double d_min, double d_max);

  // Expands the histogram left and/or right so it covers `[d_min, d_max]`,
  // doubling bin width and compacting whenever expansion would exceed
  // `max_bins_`.
  void ExpandToFit(double d_min, double d_max);

  // Doubles `bin_width_` and merges adjacent bin pairs to halve bin count.
  void DoubleBinWidthAndCompact();

  int max_bins_;
  double bin_width_ = 0.0;
  double lower_bound_ = 0.0;
  std::vector<int64_t> counts_;
};

}  // namespace odml

#endif  // TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_DYNAMIC_HISTOGRAM_H_
