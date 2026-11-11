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
#include <optional>
#include <vector>

namespace odml {

DynamicHistogram::DynamicHistogram(int max_bins,
                                   std::optional<double> initial_bin_width)
    : max_bins_(max_bins),
      initial_bin_width_(initial_bin_width),
      counts_(1, 0) {}

void DynamicHistogram::Initialize(double d_min, double d_max) {
  int64_t num_bins = max_bins_;
  if (!initial_bin_width_.has_value()) {
    const double range_val = d_max - d_min;
    double d_min_padded;
    double d_max_padded;
    if (range_val > 0.0) {
      const double pad = range_val * 0.1;
      d_min_padded = d_min - pad;
      d_max_padded = d_max + pad;
    } else {
      d_min_padded = d_min - 1e-4;
      d_max_padded = d_max + 1e-4;
    }
    lower_bound_ = d_min_padded;
    bin_width_ = std::max((d_max_padded - d_min_padded) / max_bins_, 1e-5);
  } else {
    lower_bound_ = d_min;
    bin_width_ = *initial_bin_width_;
    num_bins = std::max<int64_t>(
        1, static_cast<int64_t>(std::ceil((d_max - d_min) / bin_width_)));
  }

  counts_.assign(num_bins, 0);
  initialized_ = true;
}

void DynamicHistogram::ExpandToFit(double d_min, double d_max) {
  // 1. Expand Left.
  if (d_min < lower_bound_) {
    int64_t num_new_left =
        static_cast<int64_t>(std::ceil((lower_bound_ - d_min) / bin_width_));
    while (static_cast<int64_t>(counts_.size()) + num_new_left > max_bins_) {
      DoubleBinWidthAndCompact();
      num_new_left =
          static_cast<int64_t>(std::ceil((lower_bound_ - d_min) / bin_width_));
    }
    counts_.insert(counts_.begin(), num_new_left, 0);
    lower_bound_ -= num_new_left * bin_width_;
  }

  // 2. Expand Right.
  double upper_bound =
      lower_bound_ + static_cast<double>(counts_.size()) * bin_width_;
  if (d_max > upper_bound) {
    int64_t num_new_right =
        static_cast<int64_t>(std::ceil((d_max - upper_bound) / bin_width_));
    while (static_cast<int64_t>(counts_.size()) + num_new_right > max_bins_) {
      DoubleBinWidthAndCompact();
      upper_bound =
          lower_bound_ + static_cast<double>(counts_.size()) * bin_width_;
      num_new_right =
          static_cast<int64_t>(std::ceil((d_max - upper_bound) / bin_width_));
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
