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

#include <map>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"

namespace tensorflow {
namespace calibrator {

class CalibratorSingleton {
 public:
  // Clears the collected information.
  static void ClearCollectedInformation();

  // Clears the collected data of the given node id.
  static void ClearData(absl::string_view id);

  // Collects min and max values from the TensorFlow operator executions.
  static void ReportMinMax(absl::string_view id, float min, float max);

  // Returns the min and max values of the given id.
  static std::optional<std::pair<float, float>> GetMinMax(absl::string_view id);

 private:
  static CalibratorSingleton& GetInstance();
  static absl::Mutex lock_;

  std::map<absl::string_view, float> id_to_min_;
  std::map<absl::string_view, float> id_to_max_;

  CalibratorSingleton() = default;
  ~CalibratorSingleton() = default;
};

}  // namespace calibrator
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_CALIBRATOR_SINGLETON_H_
