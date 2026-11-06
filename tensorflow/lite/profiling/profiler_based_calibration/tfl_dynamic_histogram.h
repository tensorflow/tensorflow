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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace odml {

inline constexpr int kDefaultMaxBins = 2048;
// Exclusive bounds for valid floating-point values, matching the Python
// `naive_min_max_quantize.DEFAULT_VALID_RANGE`.
inline constexpr float kValidFloatMin = -3e38f;
inline constexpr float kValidFloatMax = 3e38f;

// A 1D dynamic histogram that automatically expands its range to accommodate
// new data points while keeping its bin count bounded by `max_bins` via
// bin-width doubling and pairwise compaction.
//
// Mirrors Python `_DynamicHistogram1D` in `histogram_utils.py`.
class DynamicHistogram {
 public:
  explicit DynamicHistogram(int max_bins = kDefaultMaxBins)
      : max_bins_(max_bins) {}

  // Adds elements from `data` of length `n` into the histogram.
  // For floating-point types, only finite values strictly within
  // `(kValidFloatMin, kValidFloatMax)` are added. For integral types, all
  // values are added.
  template <typename T>
  void Add(const T* data, int64_t n);

  const std::vector<int64_t>& counts() const { return counts_; }
  double lower_bound() const { return lower_bound_; }
  double bin_width() const { return bin_width_; }
  float global_min() const { return global_min_; }
  float global_max() const { return global_max_; }
  bool initialized() const { return initialized_; }

 private:
  void Initialize(double d_min, double d_max);
  void ExpandToFit(double d_min, double d_max);
  void DoubleBinWidthAndCompact();

  int max_bins_;
  double bin_width_ = 0.0;
  double lower_bound_ = 0.0;
  std::vector<int64_t> counts_;
  bool initialized_ = false;
  float global_min_ = std::numeric_limits<float>::infinity();
  float global_max_ = -std::numeric_limits<float>::infinity();
};

template <typename T>
void DynamicHistogram::Add(const T* data, int64_t n) {
  if (data == nullptr || n <= 0) {
    return;
  }

  auto is_valid = [](float val) {
    if constexpr (std::is_integral_v<T>) {
      return true;
    } else {
      return std::isfinite(val) && val > kValidFloatMin && val < kValidFloatMax;
    }
  };

  float d_min = std::numeric_limits<float>::infinity();
  float d_max = -std::numeric_limits<float>::infinity();
  bool has_valid = false;

  for (int64_t i = 0; i < n; ++i) {
    const float val = static_cast<float>(data[i]);
    if (!is_valid(val)) {
      continue;
    }
    has_valid = true;
    d_min = std::min(d_min, val);
    d_max = std::max(d_max, val);
  }

  if (!has_valid) {
    return;
  }

  global_min_ = std::min(global_min_, d_min);
  global_max_ = std::max(global_max_, d_max);

  if (!initialized_) {
    Initialize(d_min, d_max);
  } else {
    ExpandToFit(d_min, d_max);
  }

  const int64_t max_idx = static_cast<int64_t>(counts_.size()) - 1;
  for (int64_t i = 0; i < n; ++i) {
    const float val = static_cast<float>(data[i]);
    if (!is_valid(val)) {
      continue;
    }
    int64_t idx = static_cast<int64_t>(
        std::floor((static_cast<double>(val) - lower_bound_) / bin_width_));
    idx = std::clamp<int64_t>(idx, 0, max_idx);
    ++counts_[idx];
  }
}

}  // namespace odml

#endif  // TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_DYNAMIC_HISTOGRAM_H_
