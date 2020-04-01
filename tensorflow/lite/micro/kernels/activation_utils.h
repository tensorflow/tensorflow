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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_ACTIVATION_UTILS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_ACTIVATION_UTILS_H_

#include <algorithm>
#include <cmath>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"

namespace tflite {
namespace ops {
namespace micro {

// Returns the floating point value for a fused activation:
inline float ActivationValFloat(TfLiteFusedActivation act, float a) {
  switch (act) {
    case kTfLiteActNone:
      return a;
    case kTfLiteActRelu:
      return std::max(0.0f, a);
    case kTfLiteActRelu1:
      return std::max(-1.0f, std::min(a, 1.0f));
    case kTfLiteActRelu6:
      return std::max(0.0f, std::min(a, 6.0f));
    case kTfLiteActTanh:
      return std::tanh(a);
    case kTfLiteActSignBit:
      return std::signbit(a);
    case kTfLiteActSigmoid:
      return 1.0f / (1.0f + std::exp(-a));
  }
  return 0.0f;  // To indicate an unsupported activation (i.e. when a new fused
                // activation is added to the enum and not handled here).
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_ACTIVATION_UTILS_H_
