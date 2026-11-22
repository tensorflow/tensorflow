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

#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_dynamic_histogram.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "absl/log/absl_check.h"

namespace odml {

void DynamicHistogram::Add(const float* data, int64_t n, float min_val,
                           float max_val) {
  ABSL_DCHECK(data != nullptr);
  ABSL_DCHECK_GT(n, 0);
  ABSL_DCHECK(std::isfinite(min_val));
  ABSL_DCHECK(std::isfinite(max_val));
  ABSL_DCHECK_LE(min_val, max_val);

  if (counts_.empty()) {
    Initialize(min_val, max_val);
  } else {
    ExpandToFit(min_val, max_val);
  }

  // Bin each element in `[min_val, max_val]` (excluding NaN and +-inf),
  // clamping in double before casting to avoid float-to-integer overflow UB on
  // extreme ranges.
  const double max_idx = static_cast<double>(counts_.size() - 1);
  for (int64_t i = 0; i < n; ++i) {
    const float val = data[i];
    if (val >= min_val && val <= max_val) {
      const double raw_idx =
          std::floor((static_cast<double>(val) - lower_bound_) / bin_width_);
      const auto idx = static_cast<size_t>(std::clamp(raw_idx, 0.0, max_idx));
      ++counts_[idx];
    }
  }
}

void DynamicHistogram::Initialize(double d_min, double d_max) {
  // Pad the first observed range by 10% on each side (or 1e-4 for a constant
  // tensor) and spread it across all `max_bins_` bins.
  const double range_val = d_max - d_min;
  const double pad = (range_val > 0.0) ? range_val * 0.1 : 1e-4;
  lower_bound_ = d_min - pad;
  bin_width_ = std::max((range_val + 2.0 * pad) / max_bins_, 1e-5);
  counts_.assign(max_bins_, 0);
}

void DynamicHistogram::ExpandToFit(double d_min, double d_max) {
  // Expand left when `d_min` falls below `lower_bound_`, keeping `needed_left`
  // in double until compaction brings it within `max_bins_` to avoid integer
  // cast overflow.
  double needed_left = std::ceil((lower_bound_ - d_min) / bin_width_);
  if (needed_left > 0.0) {
    while (static_cast<double>(counts_.size()) + needed_left > max_bins_) {
      DoubleBinWidthAndCompact();
      needed_left = std::ceil((lower_bound_ - d_min) / bin_width_);
    }
    const auto num_new_left = static_cast<size_t>(needed_left);
    counts_.insert(counts_.begin(), num_new_left, 0);
    lower_bound_ -= static_cast<double>(num_new_left) * bin_width_;
  }

  // Expand right when `d_max` exceeds the current upper bound.
  auto bins_needed_right = [&]() {
    const double upper_bound =
        lower_bound_ + static_cast<double>(counts_.size()) * bin_width_;
    return std::ceil((d_max - upper_bound) / bin_width_);
  };
  if (double needed_right = bins_needed_right(); needed_right > 0.0) {
    while (static_cast<double>(counts_.size()) + needed_right > max_bins_) {
      DoubleBinWidthAndCompact();
      needed_right = bins_needed_right();
    }
    const auto num_new_right = static_cast<size_t>(needed_right);
    counts_.insert(counts_.end(), num_new_right, 0);
  }
}

void DynamicHistogram::DoubleBinWidthAndCompact() {
  // Pad to an even bin count before summing adjacent bin pairs.
  // NOMUTANTS -- Pushing an extra 0 when `counts_.size()` is already even
  // leaves `counts_.size() / 2` and all summed pairs unchanged.
  if (counts_.size() % 2 != 0) {
    counts_.push_back(0);
  }
  const size_t half_size = counts_.size() / 2;
  for (size_t i = 0; i < half_size; ++i) {
    counts_[i] = counts_[2 * i] + counts_[2 * i + 1];
  }
  counts_.resize(half_size);
  bin_width_ *= 2.0;
}

}  // namespace odml
