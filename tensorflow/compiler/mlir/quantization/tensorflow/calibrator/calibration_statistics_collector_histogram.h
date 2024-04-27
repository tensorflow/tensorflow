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
#include <utility>

#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_base.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace tensorflow {
namespace calibrator {


class CalibrationStatisticsCollectorHistogram
    : public CalibrationStatisticsCollectorBase {
 public:
  explicit CalibrationStatisticsCollectorHistogram() { ClearData(); }

  void ClearData() override;

  void Collect(float min, float max,
               absl::Span<const int64_t> histogram) override;

  std::optional<CalibrationStatistics> GetStatistics() const override;

 private:
  // Expands the histogram so the lower_bound and upper_bound can fit in the
  // histogram. Returns the indexes associated to those values.
  std::pair<int32_t, int32_t> ExpandHistogramIfNeeded(float lower_bound,
                                                      float upper_bound);

  // hist_freq_[i] saves frequency of range [bins[i], bins[i + 1]).
  // bins[i]     = lower_bound_ + bin_width_ * i
  // bins[i + 1] = lower_bound_ + bin_width_ * (i + 1)
  std::deque<float> hist_freq_;

  // Width of bin
  float bin_width_;

  // The first bin's left value. [left, right)
  float lower_bound_;
};

}  // namespace calibrator
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_CALIBRATION_STATISTICS_COLLECTOR_HISTOGRAM_H_
