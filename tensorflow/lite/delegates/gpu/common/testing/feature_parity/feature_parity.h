/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_FEATURE_PARITY_FEATURE_PARITY_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_FEATURE_PARITY_FEATURE_PARITY_H_

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/testing/feature_parity/generators/add.h"
#include "tensorflow/lite/delegates/gpu/common/testing/feature_parity/utils.h"

namespace tflite {

// Specifies the common ops feature parity for all gpu delegates. NameModel
// structure stores the model name and model itself.
inline std::vector<TestParams> GetFeatureParity() {
  return {Add2SameShapeTensors(), AddBroadcast()};
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_FEATURE_PARITY_FEATURE_PARITY_H_
