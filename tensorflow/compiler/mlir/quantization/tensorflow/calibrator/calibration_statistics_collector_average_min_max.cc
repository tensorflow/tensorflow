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
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_average_min_max.h"

#include <algorithm>
#include <limits>
#include <optional>

#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"

namespace tensorflow {
namespace calibrator {

void CalibrationStatisticsCollectorAverageMinMax::ClearData() {
  average_min_max_statistics_.set_min_sum(0.0);
  average_min_max_statistics_.set_max_sum(0.0);
  average_min_max_statistics_.set_num_samples(0);
}

void CalibrationStatisticsCollectorAverageMinMax::Collect(
    const float *data, const unsigned int N) {
  float input_min = std::numeric_limits<float>::max(),
        input_max = std::numeric_limits<float>::lowest();

  for (int i = 0; i < N; ++i) {
    input_min = std::min(input_min, data[i]);
    input_max = std::max(input_max, data[i]);
  }

  float current_min_sum = average_min_max_statistics_.min_sum();
  float current_max_sum = average_min_max_statistics_.max_sum();
  int current_num_samples = average_min_max_statistics_.num_samples();

  average_min_max_statistics_.set_min_sum(current_min_sum + input_min);
  average_min_max_statistics_.set_max_sum(current_max_sum + input_max);
  average_min_max_statistics_.set_num_samples(current_num_samples + 1);
}

std::optional<CalibrationStatistics>
CalibrationStatisticsCollectorAverageMinMax::GetStatistics() const {
  if (average_min_max_statistics_.num_samples() == 0) return std::nullopt;

  CalibrationStatistics statistics;
  statistics.mutable_average_min_max_statistics()->CopyFrom(
      average_min_max_statistics_);

  return statistics;
}

}  // namespace calibrator
}  // namespace tensorflow
