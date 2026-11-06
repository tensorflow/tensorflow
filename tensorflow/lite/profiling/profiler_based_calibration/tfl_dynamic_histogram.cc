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

namespace odml {

void DynamicHistogram::Initialize(double d_min, double d_max) {
  // Pad the first observed range by 10% on each side (or 1e-4 for a constant
  // tensor) and spread it across all `max_bins_` bins.
  const double range_val = d_max - d_min;
  const double pad = (range_val > 0.0) ? range_val * 0.1 : 1e-4;
  lower_bound_ = d_min - pad;
  bin_width_ = std::max((range_val + 2.0 * pad) / max_bins_, 1e-5);
  counts_.assign(max_bins_, 0);
  initialized_ = true;
}

void DynamicHistogram::ExpandToFit(double d_min, double d_max) {
  auto bins_needed_left = [&]() {
    return static_cast<int64_t>(std::ceil((lower_bound_ - d_min) / bin_width_));
  };
  if (int64_t num_new_left = bins_needed_left(); num_new_left > 0) {
    while (static_cast<int64_t>(counts_.size()) + num_new_left > max_bins_) {
      DoubleBinWidthAndCompact();
      num_new_left = bins_needed_left();
    }
    counts_.insert(counts_.begin(), num_new_left, 0);
    lower_bound_ -= num_new_left * bin_width_;
  }

  auto bins_needed_right = [&]() {
    const double upper_bound =
        lower_bound_ + static_cast<double>(counts_.size()) * bin_width_;
    return static_cast<int64_t>(std::ceil((d_max - upper_bound) / bin_width_));
  };
  if (int64_t num_new_right = bins_needed_right(); num_new_right > 0) {
    while (static_cast<int64_t>(counts_.size()) + num_new_right > max_bins_) {
      DoubleBinWidthAndCompact();
      num_new_right = bins_needed_right();
    }
    counts_.insert(counts_.end(), num_new_right, 0);
  }
}

void DynamicHistogram::DoubleBinWidthAndCompact() {
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
