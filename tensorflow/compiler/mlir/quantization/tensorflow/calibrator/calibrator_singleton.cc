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
  instance.id_to_min_.clear();
  instance.id_to_max_.clear();
}

void CalibratorSingleton::ClearData(absl::string_view id) {
  absl::MutexLock lock(&lock_);

  CalibratorSingleton& instance = GetInstance();
  instance.id_to_min_.erase(id);
  instance.id_to_max_.erase(id);
}

void CalibratorSingleton::ReportMinMax(absl::string_view id, float min,
                                       float max) {
  absl::MutexLock lock(&lock_);

  CalibratorSingleton& instance = GetInstance();
  // Update min value.
  if (instance.id_to_min_.count(id) != 0) {
    float cur_min = instance.id_to_min_[id];
    if (cur_min > min) instance.id_to_min_[id] = min;
  } else {
    instance.id_to_min_[id] = min;
  }
  // Update max value.
  if (instance.id_to_max_.count(id) != 0) {
    float cur_max = instance.id_to_max_[id];
    if (cur_max < max) instance.id_to_max_[id] = max;
  } else {
    instance.id_to_max_[id] = max;
  }
}

std::optional<std::pair<float, float>> CalibratorSingleton::GetMinMax(
    absl::string_view id) {
  absl::MutexLock lock(&lock_);

  CalibratorSingleton& instance = GetInstance();

  if (instance.id_to_min_.count(id) == 0 ||
      instance.id_to_max_.count(id) == 0) {
    return std::nullopt;
  }

  return std::pair<float, float>(instance.id_to_min_[id],
                                 instance.id_to_max_[id]);
}

}  // namespace calibrator
}  // namespace tensorflow
