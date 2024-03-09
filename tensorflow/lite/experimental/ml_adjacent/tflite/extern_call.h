/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_TFLITE_EXTERN_CALL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_TFLITE_EXTERN_CALL_H_

#include "tensorflow/lite/core/c/common.h"

namespace tflite::extern_call {

// Compile time options passed to this kernel at runtime.
struct ExternCallOptions {
  // A single custom op is used to represent a call to an arbitrary function
  // in the library. The function that is called is encoded in `func_id`.
  // Because of compiler op def, these will be encded at compile time
  // as `char[]` and will serialize `uint8_t`, so we match this type.
  uint8_t func_id;
};

TfLiteRegistration* Register_EXTERN_CALL();

}  // namespace tflite::extern_call

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_TFLITE_EXTERN_CALL_H_
