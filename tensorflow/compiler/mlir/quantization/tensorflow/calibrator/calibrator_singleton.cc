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
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace calibrator {

ABSL_CONST_INIT absl::Mutex CalibratorSingleton::lock_(absl::kConstInit);

CalibratorSingleton& CalibratorSingleton::GetInstance() {
  static CalibratorSingleton* calibrator = new CalibratorSingleton();
  return *calibrator;
}

void CalibratorSingleton::ClearCollectedInformation() {
  absl::MutexLock lock(&lock_);

  CalibratorSingleton& instance = GetInstance();
  instance.id_to_collector_.clear();
}

void CalibratorSingleton::ClearData(absl::string_view id) {
  absl::MutexLock lock(&lock_);

  CalibratorSingleton& instance = GetInstance();

  const std::string id_str{id};
  instance.id_to_collector_[id_str].ClearData();
}

void CalibratorSingleton::Report(absl::string_view id,
                                 absl::Span<float> data_span) {
  absl::MutexLock lock(&lock_);

  CalibratorSingleton& instance = GetInstance();

  const std::string id_str{id};
  instance.id_to_collector_[id_str].Collect(data_span);
}

void CalibratorSingleton::Report(absl::string_view id,
                                 const std::vector<float>& data_vec) {
  absl::MutexLock lock(&lock_);

  CalibratorSingleton& instance = GetInstance();

  const std::string id_str{id};
  instance.id_to_collector_[id_str].Collect(data_vec);
}

void CalibratorSingleton::Report(absl::string_view id,
                                 const Tensor& data_tensor) {
  absl::MutexLock lock(&lock_);

  CalibratorSingleton& instance = GetInstance();

  const std::string id_str{id};
  instance.id_to_collector_[id_str].Collect(data_tensor);
}

std::optional<CalibrationStatistics> CalibratorSingleton::GetStatistics(
    absl::string_view id) {
  absl::MutexLock lock(&lock_);

  CalibratorSingleton& instance = GetInstance();

  const std::string id_str{id};
  return instance.id_to_collector_[id_str].GetStatistics();
}

}  // namespace calibrator
}  // namespace tensorflow
