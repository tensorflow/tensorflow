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

#include <algorithm>
#include <string>

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

  const std::string id_str{id};
  instance.id_to_min_.erase(id_str);
  instance.id_to_max_.erase(id_str);
}

void CalibratorSingleton::ReportMinMax(absl::string_view id,
                                       const float min_val,
                                       const float max_val) {
  absl::MutexLock lock(&lock_);

  CalibratorSingleton& instance = GetInstance();

  const std::string id_str{id};

  // Update the min value.
  if (auto min_itr = instance.id_to_min_.find(id_str);
      min_itr != instance.id_to_min_.end()) {
    min_itr->second = std::min(min_val, min_itr->second);
  } else {
    instance.id_to_min_[id_str] = min_val;
  }

  // Update the max values.
  if (auto max_itr = instance.id_to_max_.find(id_str);
      max_itr != instance.id_to_max_.end()) {
    max_itr->second = std::max(max_val, max_itr->second);
  } else {
    instance.id_to_max_[id_str] = max_val;
  }
}

std::optional<std::pair<float, float>> CalibratorSingleton::GetMinMax(
    absl::string_view id) {
  absl::MutexLock lock(&lock_);

  CalibratorSingleton& instance = GetInstance();

  const std::string id_str{id};
  const auto min_itr = instance.id_to_min_.find(id_str);
  const auto max_itr = instance.id_to_max_.find(id_str);
  if (min_itr == instance.id_to_min_.end() ||
      max_itr == instance.id_to_max_.end()) {
    return std::nullopt;
  }

  return std::pair<float, float>{min_itr->second, max_itr->second};
}

}  // namespace calibrator
}  // namespace tensorflow
