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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_CALIBRATION_STATISTICS_COLLECTOR_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_CALIBRATION_STATISTICS_COLLECTOR_H_

#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace calibrator {

class CalibrationStatisticsCollector {
 public:
  explicit CalibrationStatisticsCollector() { ClearData(); }

  // Collect data for calibration using float vector.
  // It internally calls private method Collect(float*, unsigned int).
  void Collect(const std::vector<float> &data_vec);

  // Collect data for calibration using absl::Span<float>.
  // It internally calls private method Collect(float*, unsigned int).
  void Collect(absl::Span<float> data_span);

  /// Collect data for calibration using Tensor
  // It internally calls private method Collect(float*, unsigned int).
  void Collect(const Tensor &data_tensor);

  // proto message CalibrationStatistics is returned.
  std::optional<CalibrationStatistics> GetStatistics();

  // Clear collected data.
  void ClearData();

 private:
  // Collect data for calibration using float pointer and data size N.
  // if N == 0, function terminates immediately.
  void Collect(const float *data, unsigned int N);

  CalibrationStatistics::MinMaxStatistics min_max_statistics_;
  CalibrationStatistics::AverageMinMaxStatistics average_min_max_statistics_;
};

}  // namespace calibrator
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_CALIBRATION_STATISTICS_COLLECTOR_H_
