/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/c_api_experimental.h"

#include <stdint.h>

#include <memory>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/mutable_op_resolver.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

TfLiteStatus TfLiteInterpreterResetVariableTensors(
    TfLiteInterpreter* interpreter) {
  return interpreter->impl->ResetVariableTensors();
}

void TfLiteInterpreterOptionsAddBuiltinOp(
    TfLiteInterpreterOptions* options, TfLiteBuiltinOperator op,
    const TfLiteRegistration* registration, int32_t min_version,
    int32_t max_version) {
  options->op_resolver.AddBuiltin(static_cast<tflite::BuiltinOperator>(op),
                                  registration, min_version, max_version);
}

void TfLiteInterpreterOptionsAddCustomOp(TfLiteInterpreterOptions* options,
                                         const char* name,
                                         const TfLiteRegistration* registration,
                                         int min_version, int max_version) {
  options->op_resolver.AddCustom(name, registration, min_version, max_version);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
