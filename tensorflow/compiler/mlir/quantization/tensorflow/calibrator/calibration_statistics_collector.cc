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
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace calibrator {

void CalibrationStatisticsCollector::ClearData() {
  // global_min will be updated by std::min(global_min, input_value) so
  // it is initialized with the value numeric_limits<float>::max().
  min_max_statistics_.set_global_min(std::numeric_limits<float>::max());

  // global_max will be updated by std::max(global_max, input_value) so it
  // is initialized with the value numeric_limits<float>::lowest().
  min_max_statistics_.set_global_max(std::numeric_limits<float>::lowest());

  average_min_max_statistics_.set_min_sum(0.0);
  average_min_max_statistics_.set_max_sum(0.0);
  average_min_max_statistics_.set_num_samples(0);
}

void CalibrationStatisticsCollector::Collect(
    const std::vector<float>& data_vec) {
  Collect(data_vec.data(), data_vec.size());
}

void CalibrationStatisticsCollector::Collect(absl::Span<float> data_span) {
  Collect(data_span.data(), data_span.size());
}

void CalibrationStatisticsCollector::Collect(const Tensor& data_tensor) {
  auto data_flat = data_tensor.flat<float>();
  Collect(data_flat.data(), data_flat.size());
}

// TODO - b/295125836 : Collects statistics required to calibration method
void CalibrationStatisticsCollector::Collect(const float* data,
                                             const unsigned int N) {
  if (N == 0) return;

  float input_min = std::numeric_limits<float>::max(),
        input_max = std::numeric_limits<float>::lowest();

  for (int i = 0; i < N; i++) {
    input_min = std::min(input_min, data[i]);
    input_max = std::max(input_max, data[i]);
  }

  float current_global_min = min_max_statistics_.global_min();
  float current_global_max = min_max_statistics_.global_max();

  min_max_statistics_.set_global_min(std::min(current_global_min, input_min));
  min_max_statistics_.set_global_max(std::max(current_global_max, input_max));

  float current_min_sum = average_min_max_statistics_.min_sum();
  float current_max_sum = average_min_max_statistics_.max_sum();
  float current_num_samples = average_min_max_statistics_.num_samples();

  average_min_max_statistics_.set_min_sum(current_min_sum + input_min);
  average_min_max_statistics_.set_max_sum(current_max_sum + input_max);
  average_min_max_statistics_.set_num_samples(current_num_samples + 1);
}

std::optional<CalibrationStatistics>
CalibrationStatisticsCollector::GetStatistics() {
  if (average_min_max_statistics_.num_samples() == 0) return std::nullopt;

  CalibrationStatistics statistics;

  *(statistics.mutable_min_max_statistics()) = min_max_statistics_;

  *(statistics.mutable_average_min_max_statistics()) =
      average_min_max_statistics_;

  return statistics;
}

}  // namespace calibrator
}  // namespace tensorflow
