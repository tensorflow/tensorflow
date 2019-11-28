/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Forward declares registrations for specific FC layer implementations. Do not
// include this header if you are fine with any FC implementation, include
// builtin_op_kernels.h instead. This implementation-specific registration is
// only available for FC, as these versions are explicitly tested and supported.

#ifndef TENSORFLOW_LITE_KERNELS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_FULLY_CONNECTED_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace ops {
namespace builtin {
TfLiteRegistration* Register_FULLY_CONNECTED_REF();
TfLiteRegistration* Register_FULLY_CONNECTED_GENERIC_OPT();
TfLiteRegistration* Register_FULLY_CONNECTED_PIE();
}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_FULLY_CONNECTED_H_
