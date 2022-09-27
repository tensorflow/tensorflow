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
#ifndef TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_UTIL_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace delegates {
namespace coreml {

// Checks if Binary ops have supported broadcastable shapes.
// Core ml arithmetic ops - Add and Mul support broadcasts among
// [B, 1, 1, 1], [B, C, 1, 1], [B, 1, H, W], [B, C, H, W].
// other shapes should be rejected. Unless it is a constant tensor of size 1,
// which will be added as data.

bool IsBinaryOpSupported(const TfLiteRegistration* registration,
                         const TfLiteNode* node, TfLiteContext* context);

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_UTIL_H_
