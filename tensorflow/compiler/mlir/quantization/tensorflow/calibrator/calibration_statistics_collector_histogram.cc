/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_histogram.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>

#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/calibration_parameters.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace tensorflow {
namespace calibrator {
namespace {

using ::stablehlo::quantization::CalculateBinIndex;
using ::stablehlo::quantization::CalculateBinWidth;
using ::stablehlo::quantization::CalculateLowerBound;

// Gets the histogram frequencies for the given range.
float GetRangeFrequencies(absl::Span<const int64_t> histogram,
                          const float bin_width, const float lower_bound,
                          const float range_start, const float range_end) {
  float freq_sum = 0.f;
  for (float range = std::max(range_start, lower_bound); range < range_end;
       range += bin_width) {
    const int32_t idx = CalculateBinIndex(range, lower_bound, bin_width);
    if (idx >= histogram.size()) break;

    //  If the range is smaller than bin width, add the proportional value of
    //  that bin.
    const float proportion = std::min(range_end - range, bin_width) / bin_width;
    freq_sum += histogram[idx] * proportion;
  }
  return freq_sum;
}

}  // namespace

void CalibrationStatisticsCollectorHistogram::ClearData() {
  hist_freq_.clear();
}

void CalibrationStatisticsCollectorHistogram::Collect(
    const float min, const float max, absl::Span<const int64_t> histogram) {
  if (histogram.empty()) return;

  // Reconstruct the bin width, lower and upper bound from the collected data.
  const float collected_bin_width =
      CalculateBinWidth(min, max, histogram.size());
  const float collected_lower_bound =
      CalculateLowerBound(min, collected_bin_width);
  const float collected_upper_bound =
      std::ceil(max / collected_bin_width) * collected_bin_width;

  // When histogram is not initialized.
  if (hist_freq_.empty()) {
    bin_width_ = collected_bin_width;
    lower_bound_ = collected_lower_bound;
  }

  const auto [lower_idx, upper_idx] =
      ExpandHistogramIfNeeded(collected_lower_bound, collected_upper_bound);
  for (int32_t idx = lower_idx; idx <= upper_idx; ++idx) {
    // Calculate the range covered by this index then add with the collected
    // frequency associated to that range.
    const float range_start = lower_bound_ + idx * bin_width_;
    hist_freq_[idx] += GetRangeFrequencies(histogram, collected_bin_width,
                                           collected_lower_bound, range_start,
                                           range_start + bin_width_);
  }
}

std::optional<CalibrationStatistics>
CalibrationStatisticsCollectorHistogram::GetStatistics() const {
  if (hist_freq_.empty()) return std::nullopt;

  CalibrationStatistics::HistogramStatistics hist_stats;

  // Skip trailing zeros in the histogram.
  int32_t real_size = hist_freq_.size();
  for (; real_size > 0; --real_size) {
    if (hist_freq_[real_size - 1] != 0) break;
  }

  hist_stats.set_lower_bound(lower_bound_);
  hist_stats.set_bin_width(bin_width_);
  hist_stats.mutable_hist_freq()->Assign(hist_freq_.begin(),
                                         hist_freq_.begin() + real_size);

  CalibrationStatistics statistics;
  statistics.mutable_histogram_statistics()->CopyFrom(hist_stats);

  return statistics;
}

std::pair<int32_t, int32_t>
CalibrationStatisticsCollectorHistogram::ExpandHistogramIfNeeded(
    const float lower_bound, const float upper_bound) {
  int32_t lower_idx = CalculateBinIndex(lower_bound, lower_bound_, bin_width_);
  // If lower_idx < 0, then expand the histogram to the left.
  if (lower_idx < 0) {
    hist_freq_.insert(hist_freq_.begin(), -lower_idx, 0);
    lower_bound_ -= bin_width_ * (-lower_idx);
    lower_idx = 0;
  }

  int32_t upper_idx = CalculateBinIndex(upper_bound, lower_bound_, bin_width_);
  // If upper_idx >= hist_freq_.size(), then expand the histogram to the right.
  if (upper_idx >= hist_freq_.size()) {
    hist_freq_.resize(upper_idx + 1, 0);
  }
  return std::make_pair(lower_idx, upper_idx);
}

}  // namespace calibrator
}  // namespace tensorflow
