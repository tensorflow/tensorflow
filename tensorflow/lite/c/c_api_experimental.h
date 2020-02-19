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
#ifndef TENSORFLOW_LITE_C_C_API_EXPERIMENTAL_H_
#define TENSORFLOW_LITE_C_C_API_EXPERIMENTAL_H_

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Resets all variable tensors to zero.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterResetVariableTensors(
    TfLiteInterpreter* interpreter);

// Adds an op registration for a builtin operator.
//
// NOTE: The interpreter will make a copy of `registration` internally, so the
// caller should ensure that its contents (function pointers, etc...) remain
// valid for the duration of the interpreter's lifetime. A common practice is
// making the provided TfLiteRegistration instance static.
TFL_CAPI_EXPORT void TfLiteInterpreterOptionsAddBuiltinOp(
    TfLiteInterpreterOptions* options, TfLiteBuiltinOperator op,
    const TfLiteRegistration* registration, int min_version, int max_version);

// Adds an op registration for a custom operator.
//
// NOTE: The interpreter will make a copy of `registration` internally, so the
// caller should ensure that its contents (function pointers, etc...) remain
// valid for the duration of any created interpreter's lifetime. A common
// practice is making the provided TfLiteRegistration instance static.
TFL_CAPI_EXPORT void TfLiteInterpreterOptionsAddCustomOp(
    TfLiteInterpreterOptions* options, const char* name,
    const TfLiteRegistration* registration, int min_version, int max_version);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_C_C_API_EXPERIMENTAL_H_
