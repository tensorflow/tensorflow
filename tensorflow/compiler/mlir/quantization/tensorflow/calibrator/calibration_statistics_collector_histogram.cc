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
#include <deque>
#include <optional>

#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace tensorflow {
namespace calibrator {

void CalibrationStatisticsCollectorHistogram::ClearData() {
  num_bins_ = 256;
  bin_width_ = 0;
  hist_freq_.resize(num_bins_, 0);
}

void CalibrationStatisticsCollectorHistogram::Collect(const float *data,
                                                      const unsigned int N) {
  if (N == 0) return;

  // When histogram is not initialized.
  if (bin_width_ == 0) {
    hist_freq_.resize(num_bins_, 0);
    auto minmax = std::minmax_element(data, data + N);

    // The min and max of the first data will be the range of the histogram.
    float min_value = std::floor(*minmax.first);
    float max_value = std::ceil(*minmax.second);

    // The bin width is (max - min) divided by num_bins.
    bin_width_ = (max_value - min_value) / num_bins_;

    // The lower bound is min value of data.
    lower_bound_ = min_value;

    // This is the worst case of first initialization, so it returns
    // instantly. 1e-9 is threshold.
    if (std::abs(bin_width_) < 1e-9) return;
  }

  for (int i = 0; i < N; ++i) {
    int idx = GetHistogramIndex(data[i]);
    hist_freq_[idx]++;
  }
}

std::optional<CalibrationStatistics>
CalibrationStatisticsCollectorHistogram::GetStatistics() const {
  if (bin_width_ == 0) return std::nullopt;

  CalibrationStatistics::HistogramStatistics hist_stats;

  hist_stats.set_lower_bound(lower_bound_);
  hist_stats.set_bin_width(bin_width_);
  hist_stats.mutable_hist_freq()->Assign(hist_freq_.begin(), hist_freq_.end());

  CalibrationStatistics statistics;
  statistics.mutable_histogram_statistics()->CopyFrom(hist_stats);

  return statistics;
}

int CalibrationStatisticsCollectorHistogram::ExpandHistogramIfNeeded(int idx) {
  // If idx < 0, then expand the histogram to the left.
  if (idx < 0) {
    hist_freq_.insert(hist_freq_.begin(), -idx, 0);
    lower_bound_ -= bin_width_ * (-idx);
    idx = 0;
  }

  // If idx >= hist_freq_.size(), then expand the histogram to the left.
  if (idx >= hist_freq_.size()) {
    hist_freq_.resize(idx + 1, 0);
  }

  return idx;
}

int CalibrationStatisticsCollectorHistogram::GetHistogramIndex(
    const float value) {
  // Calculate index of histogram
  int idx = (value - lower_bound_) / bin_width_;

  return ExpandHistogramIfNeeded(idx);
}

}  // namespace calibrator
}  // namespace tensorflow
