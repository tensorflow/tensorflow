/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_CALIBRATOR_SINGLETON_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_CALIBRATOR_SINGLETON_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_base.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace calibrator {

using tensorflow::quantization::CalibrationOptions;

class CalibratorSingleton {
 public:
  // Clears the collected information.
  static void ClearCollectedInformation();

  // Clears the collected data of the given node id.
  static void ClearData(absl::string_view id);

  // Reports data to singleton using float vector.
  // Only calculates the required statistics from CalibrationMethod based
  // on CalibrationOptions.
  static void Report(absl::string_view id, const std::vector<float>& data_vec,
                     const CalibrationOptions& calib_opts);

  // Reports data to singleton using absl::Span
  // Only calculates the required statistics from CalibrationMethod based
  // on CalibrationOptions.
  static void Report(absl::string_view id, absl::Span<float> data_span,
                     const CalibrationOptions& calib_opts);

  // Reports data to singleton using absl::Span
  // Only calculates the required statistics from CalibrationMethod based
  // on CalibrationOptions.
  static void Report(absl::string_view id, const Tensor& data_tensor,
                     const CalibrationOptions& calib_opts);

  // Returns the calibration statistics of the given id.
  static std::optional<CalibrationStatistics> GetStatistics(
      absl::string_view id);

 private:
  static CalibratorSingleton& GetInstance();
  static absl::Mutex lock_;
  static void AssignIfNotExists(std::string id_str,
                                const CalibrationOptions& calib_opts);

  absl::flat_hash_map<std::string,
                      std::unique_ptr<CalibrationStatisticsCollectorBase>>
      id_to_collector_;

  CalibratorSingleton() = default;
  ~CalibratorSingleton() = default;
};

}  // namespace calibrator
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_CALIBRATOR_SINGLETON_H_
