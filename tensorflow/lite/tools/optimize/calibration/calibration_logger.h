/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_LOGGER_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_LOGGER_H_

#include <limits>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace optimize {
namespace calibration {

class MinMax {
 public:
  TfLiteStatus Update(const float* values, size_t tensor_size,
                      ErrorReporter* error_reporter);

  bool HasValues() const { return has_values_; }

  TfLiteStatus Get(float* min_val, float* max_val) const {
    if (!has_values_) return kTfLiteError;
    *min_val = min_;
    *max_val = max_;
    return kTfLiteOk;
  }

 private:
  bool has_values_ = false;
  float min_ = std::numeric_limits<float>::max();
  float max_ = std::numeric_limits<float>::min();
};

// Captures min max values for tensors.
class Logger {
 public:
  // Log the value for tensor at |tensor_index| which has |tensor_values|
  TfLiteStatus LogTensorValue(int subgraph_index, int tensor_index,
                              const float* tensor_values, size_t tensor_size,
                              ErrorReporter* error_reporter) {
    std::tuple<int, int> key{subgraph_index, tensor_index};
    return tensor_id_to_stats_map_[key].Update(tensor_values, tensor_size,
                                               error_reporter);
  }

  // Returns a map from tensor_index -> observed min max values.
  const absl::flat_hash_map<std::tuple<int, int>, MinMax>&
  GetCalibrationValues() const {
    return tensor_id_to_stats_map_;
  }

 private:
  absl::flat_hash_map<std::tuple<int, int>, MinMax> tensor_id_to_stats_map_;
};

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_LOGGER_H_
