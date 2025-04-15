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

#include "tensorflow/lite/core/c/c_api_experimental.h"

#include <stdint.h>

#include <memory>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/profiling/telemetry/profiler.h"
#include "tensorflow/lite/signature_runner.h"

extern "C" {

TfLiteStatus TfLiteInterpreterResetVariableTensors(
    TfLiteInterpreter* interpreter) {
  return interpreter->impl->ResetVariableTensors();
}

int32_t TfLiteInterpreterGetVariableTensorCount(
    const TfLiteInterpreter* interpreter) {
  return static_cast<int32_t>(interpreter->impl->variables().size());
}

TfLiteTensor* TfLiteInterpreterGetVariableTensor(
    const TfLiteInterpreter* interpreter, int32_t input_index) {
  return interpreter->impl->tensor(interpreter->impl->variables()[input_index]);
}

void TfLiteInterpreterOptionsAddBuiltinOp(
    TfLiteInterpreterOptions* options, TfLiteBuiltinOperator op,
    const TfLiteRegistration* registration, int32_t min_version,
    int32_t max_version) {
  options->mutable_op_resolver.AddBuiltin(
      static_cast<tflite::BuiltinOperator>(op), registration, min_version,
      max_version);
}

TfLiteInterpreter* TfLiteInterpreterCreateWithSelectedOps(
    const TfLiteModel* model,
    const TfLiteInterpreterOptions* optional_options) {
  tflite::MutableOpResolver resolver;
  return tflite::internal::InterpreterCreateWithOpResolver(
      model, optional_options, &resolver);
}

void TfLiteInterpreterOptionsAddCustomOp(TfLiteInterpreterOptions* options,
                                         const char* name,
                                         const TfLiteRegistration* registration,
                                         int32_t min_version,
                                         int32_t max_version) {
  options->mutable_op_resolver.AddCustom(name, registration, min_version,
                                         max_version);
}

void TfLiteInterpreterOptionsSetOpResolverExternal(
    TfLiteInterpreterOptions* options,
    const TfLiteOperator* (*find_builtin_op)(void* user_data, int op,
                                             int version),
    const TfLiteOperator* (*find_custom_op)(void* user_data,
                                            const char* custom_op, int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op_external = find_builtin_op;
  options->op_resolver_callbacks.find_custom_op_external = find_custom_op;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetOpResolverExternalWithFallback(
    TfLiteInterpreterOptions* options,
    const TfLiteOperator* (*find_builtin_op_external)(void* user_data, int op,
                                                      int version),
    const TfLiteOperator* (*find_custom_op_external)(void* user_data,
                                                     const char* custom_op,
                                                     int version),
    const TfLiteRegistration* (*find_builtin_op)(void* user_data,
                                                 TfLiteBuiltinOperator op,
                                                 int version),
    const TfLiteRegistration* (*find_custom_op)(void* user_data, const char* op,
                                                int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op_external =
      find_builtin_op_external;
  options->op_resolver_callbacks.find_custom_op_external =
      find_custom_op_external;
  options->op_resolver_callbacks.find_builtin_op = find_builtin_op;
  options->op_resolver_callbacks.find_custom_op = find_custom_op;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetOpResolver(
    TfLiteInterpreterOptions* options,
    const TfLiteRegistration* (*find_builtin_op)(void* user_data,
                                                 TfLiteBuiltinOperator op,
                                                 int version),
    const TfLiteRegistration* (*find_custom_op)(void* user_data, const char* op,
                                                int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op = find_builtin_op;
  options->op_resolver_callbacks.find_custom_op = find_custom_op;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetOpResolverV1(
    TfLiteInterpreterOptions* options,
    const TfLiteRegistration_V1* (*find_builtin_op_v1)(void* user_data,
                                                       TfLiteBuiltinOperator op,
                                                       int version),
    const TfLiteRegistration_V1* (*find_custom_op_v1)(void* user_data,
                                                      const char* op,
                                                      int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op_v1 = find_builtin_op_v1;
  options->op_resolver_callbacks.find_custom_op_v1 = find_custom_op_v1;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetOpResolverV3(
    TfLiteInterpreterOptions* options,
    const TfLiteRegistration_V3* (*find_builtin_op_v3)(void* user_data,
                                                       TfLiteBuiltinOperator op,
                                                       int version),
    const TfLiteRegistration_V3* (*find_custom_op_v3)(void* user_data,
                                                      const char* op,
                                                      int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op_v3 = find_builtin_op_v3;
  options->op_resolver_callbacks.find_custom_op_v3 = find_custom_op_v3;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetOpResolverV2(
    TfLiteInterpreterOptions* options,
    const TfLiteRegistration_V2* (*find_builtin_op_v2)(void* user_data,
                                                       TfLiteBuiltinOperator op,
                                                       int version),
    const TfLiteRegistration_V2* (*find_custom_op_v2)(void* user_data,
                                                      const char* op,
                                                      int version),
    void* op_resolver_user_data) {
  options->op_resolver_callbacks = {};  // Sets all fields to null.
  options->op_resolver_callbacks.find_builtin_op_v2 = find_builtin_op_v2;
  options->op_resolver_callbacks.find_custom_op_v2 = find_custom_op_v2;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetUseNNAPI(TfLiteInterpreterOptions* options,
                                         bool enable) {
  options->use_nnapi = enable;
}

void TfLiteInterpreterOptionsSetEnableDelegateFallback(
    TfLiteInterpreterOptions* options, bool enable) {
  options->enable_delegate_fallback = enable;
}

TfLiteStatus TfLiteInterpreterModifyGraphWithDelegate(
    const TfLiteInterpreter* interpreter, TfLiteDelegate* delegate) {
  return interpreter->impl->ModifyGraphWithDelegate(delegate);
}

int32_t TfLiteInterpreterGetInputTensorIndex(
    const TfLiteInterpreter* interpreter, int32_t input_index) {
  return interpreter->impl->inputs()[input_index];
}

int32_t TfLiteInterpreterGetOutputTensorIndex(
    const TfLiteInterpreter* interpreter, int32_t output_index) {
  return interpreter->impl->outputs()[output_index];
}

TfLiteStatus TfLiteInterpreterSetBufferHandle(TfLiteInterpreter* interpreter,
                                              TfLiteTensor* tensor,
                                              TfLiteBufferHandle buffer_handle,
                                              TfLiteOpaqueDelegate* delegate) {
  return interpreter->impl->SetBufferHandle(tensor, buffer_handle, delegate);
}

TfLiteStatus TfLiteInterpreterGetBufferHandle(TfLiteInterpreter* interpreter,
                                              int tensor_index,
                                              TfLiteBufferHandle* buffer_handle,
                                              TfLiteOpaqueDelegate** delegate) {
  return interpreter->impl->GetBufferHandle(tensor_index, buffer_handle,
                                            delegate);
}

void TfLiteSetAllowBufferHandleOutput(const TfLiteInterpreter* interpreter,
                                      bool allow_buffer_handle_output) {
  interpreter->impl->SetAllowBufferHandleOutput(allow_buffer_handle_output);
}

TfLiteStatus TfLiteInterpreterSetCustomAllocationForTensor(
    TfLiteInterpreter* interpreter, int tensor_index,
    const TfLiteCustomAllocation* allocation, int64_t flags) {
  if (allocation == nullptr) {
    return kTfLiteError;
  }
  return interpreter->impl->SetCustomAllocationForTensor(tensor_index,
                                                         *allocation, flags);
}

TfLiteStatus TfLiteInterpreterEnsureTensorDataIsReadable(
    TfLiteInterpreter* interpreter, int tensor_index) {
  return interpreter->impl->EnsureTensorDataIsReadable(tensor_index);
}

void TfLiteInterpreterOptionsSetTelemetryProfiler(
    TfLiteInterpreterOptions* options,
    TfLiteTelemetryProfilerStruct* profiler) {
  options->telemetry_profiler = profiler;
}

}  // extern "C"
