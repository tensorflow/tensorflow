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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_CALIBRATION_STATISTICS_COLLECTOR_HISTOGRAM_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_CALIBRATION_STATISTICS_COLLECTOR_HISTOGRAM_H_

#include <cstdint>
#include <deque>
#include <optional>

#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_base.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace tensorflow {
namespace calibrator {

using ::stablehlo::quantization::CalibrationOptions;

class CalibrationStatisticsCollectorHistogram
    : public CalibrationStatisticsCollectorBase {
 public:
  explicit CalibrationStatisticsCollectorHistogram(
      const CalibrationOptions& calib_opts) {
    ClearData();
    num_bins_ = calib_opts.calibration_parameters().initial_num_bins();
  }

  void ClearData() override;

  void Collect(const float* data, unsigned int N) override;

  std::optional<CalibrationStatistics> GetStatistics() const override;

 private:
  // Returns expanded histogram's index. If idx < 0, then expand the histogram
  // to the left. If idx >= hist_freq_.size(), then expand the histogram to the
  // right.
  int ExpandHistogramIfNeeded(int idx);

  // Calculate the histogram index of value and if index of value is exceeds the
  // range of histogram, then this function extends hist_freq_ and updates
  // lower_bound_. This function returns the expanded histogram's index.
  int GetHistogramIndex(float value);

  // hist_freq_[i] saves frequency of range [bins[i], bins[i + 1]).
  // bins[i]     = lower_bound_ + bin_width_ * i
  // bins[i + 1] = lower_bound_ + bin_width_ * (i + 1)
  std::deque<int64_t> hist_freq_;

  // The number of bins when histogram is initialized. It can be increased
  // because histogram is dynamically expanded by sample inputs.
  int num_bins_;

  // Width of bin
  float bin_width_;

  // The first bin's left value. [left, right)
  float lower_bound_;
};

}  // namespace calibrator
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_CALIBRATION_STATISTICS_COLLECTOR_HISTOGRAM_H_
