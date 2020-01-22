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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_ACTIVATION_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_ACTIVATION_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflite {
namespace ops {
namespace micro {

// Returns the floating point value for a fused activation:
inline float ActivationValFloat(TfLiteFusedActivation act, float a) {
  switch (act) {
    case kTfLiteActNone:
      return a;
    case kTfLiteActRelu:
      return a < 0.f ? 0.f : a;
    default:
      // TODO(kreeger): Implement more activations.
      exit(1);
  }
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_ACTIVATION_UTILS_H_
