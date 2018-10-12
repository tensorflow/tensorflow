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

#include "tensorflow/contrib/lite/experimental/c/c_api_experimental.h"

#include "tensorflow/contrib/lite/experimental/c/c_api_internal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

TFL_Status TFL_InterpreterResetVariableTensors(TFL_Interpreter* interpreter) {
  return interpreter->impl->ResetVariableTensors();
}

void TFL_InterpreterOptionsAddBuiltinOp(TFL_InterpreterOptions* options,
                                        TFL_BuiltinOperator op,
                                        const TFL_Registration* registration,
                                        int32_t min_version,
                                        int32_t max_version) {
  options->op_resolver.AddBuiltin(static_cast<tflite::BuiltinOperator>(op),
                                  registration, min_version, max_version);
}

void TFL_InterpreterOptionsAddCustomOp(TFL_InterpreterOptions* options,
                                       const char* name,
                                       const TFL_Registration* registration,
                                       int min_version, int max_version) {
  options->op_resolver.AddCustom(name, registration, min_version, max_version);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
