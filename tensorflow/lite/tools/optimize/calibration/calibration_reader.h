/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_READER_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_READER_H_

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_logger.h"

namespace tflite {
namespace optimize {
namespace calibration {
// Warning: This is not a public API and subject to change.
//
// Reads calibrator data collected by running the interpreter through
// a calibration set.
class CalibrationReader {
 public:
  struct CalibrationStats {
    float min;
    float max;
  };
  explicit CalibrationReader(const Logger* logger) : logger_(logger) {}

  // Gets a map from tensor index to recorded calibration values.
  virtual TfLiteStatus GetTensorStatsAsMap(
      absl::flat_hash_map<std::tuple<int, int>, CalibrationStats>*
          tensor_id_to_stats_map) const;

  // Annotates the tensors in the given model with statistics captured during
  // calibration.
  // "update" is a flag: when set to true, the min/max are updated, instead of
  // being overwritten.
  virtual TfLiteStatus AddCalibrationToModel(ModelT* model, bool update) const;

  virtual ~CalibrationReader() {}

 private:
  const Logger* logger_;
};
}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_READER_H_
