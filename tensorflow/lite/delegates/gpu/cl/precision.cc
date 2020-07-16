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

#include "tensorflow/lite/delegates/gpu/cl/precision.h"

namespace tflite {
namespace gpu {
namespace cl {

std::string ToString(CalculationsPrecision precision) {
  switch (precision) {
    case CalculationsPrecision::F32_F16:
      return "CalculationsPrecision::F32_F16";
    case CalculationsPrecision::F32:
      return "CalculationsPrecision::F32";
    case CalculationsPrecision::F16:
      return "CalculationsPrecision::F16";
  }
}

DataType DeduceDataTypeFromPrecision(CalculationsPrecision precision) {
  if (precision == CalculationsPrecision::F32) {
    return DataType::FLOAT32;
  } else {
    return DataType::FLOAT16;
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
