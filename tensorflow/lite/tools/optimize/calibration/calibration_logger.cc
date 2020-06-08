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
#include "tensorflow/lite/tools/optimize/calibration/calibration_logger.h"

#include <algorithm>
#include <cmath>

#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace optimize {
namespace calibration {

TfLiteStatus MinMax::Update(const float* values, size_t tensor_size,
                            ErrorReporter* error_reporter) {
  if (tensor_size <= 0) return kTfLiteOk;

  // TODO(shashishekhar): Make it possible to use weighted/moving average.
  for (size_t i = 0; i < tensor_size; ++i) {
    if (std::isnan(values[i])) {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "Model resulted in Nan value during calibration. Please "
          "make sure model results in all real-values during "
          "inference with provided dataset.");
      return kTfLiteError;
    }
  }
  // We are only logging absolute min/max here.
  const auto minmax = std::minmax_element(values, values + tensor_size);
  min_ = std::min<float>(min_, *minmax.first);
  max_ = std::max<float>(max_, *minmax.second);

  if (!has_values_) has_values_ = true;
  return kTfLiteOk;
}

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
