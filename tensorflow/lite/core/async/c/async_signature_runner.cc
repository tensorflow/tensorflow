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
#include "tensorflow/lite/core/async/c/async_signature_runner.h"

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/async/async_signature_runner.h"
#include "tensorflow/lite/core/async/c/internal.h"
#include "tensorflow/lite/core/c/c_api_types.h"

TfLiteAsyncSignatureRunner* TfLiteInterpreterGetAsyncSignatureRunner(
    const TfLiteInterpreter* interpreter, const char* signature_key) {
  if (!interpreter) return nullptr;
  tflite::async::AsyncSignatureRunner* runner =
      interpreter->impl->GetAsyncSignatureRunner(signature_key);
  if (!runner) return nullptr;
  return new TfLiteAsyncSignatureRunner{runner};
}

TfLiteStatus TfLiteAsyncSignatureRunnerRegisterBuffer(
    TfLiteAsyncSignatureRunner* async_signature_runner, TfLiteIoType io_type,
    const TfLiteBackendBuffer* buffer, const TfLiteAttributeMap* attrs,
    TfLiteBufferHandle* handle) {
  if (!async_signature_runner) return kTfLiteError;
  return async_signature_runner->impl->RegisterBuffer(io_type, buffer, attrs,
                                                      handle);
}

TfLiteStatus TfLiteAsyncSignatureRunnerRegisterBufferSlice(
    TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteBufferHandle buffer_pool, const TfLiteAttributeMap* attrs,
    TfLiteBufferHandle* handle) {
  if (!async_signature_runner) return kTfLiteError;
  return async_signature_runner->impl->RegisterBufferSlice(buffer_pool, attrs,
                                                           handle);
}

TfLiteStatus TfLiteAsyncSignatureRunnerUnregisterBuffer(
    TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteBufferHandle handle) {
  if (!async_signature_runner) return kTfLiteError;
  return async_signature_runner->impl->UnregisterBuffer(handle);
}

TfLiteStatus TfLiteAsyncSignatureRunnerGetSupportedBufferTypes(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteIoType io_type, const char* const** types, size_t* num_types) {
  if (async_signature_runner == nullptr || types == nullptr ||
      num_types == nullptr)
    return kTfLiteError;
  const auto& buffer_types =
      async_signature_runner->impl->SupportedBufferTypes(io_type);
  *types = buffer_types.data();
  *num_types = buffer_types.size();
  return kTfLiteOk;
}

TfLiteStatus TfLiteAsyncSignatureRunnerGetSupportedSynchronizationTypes(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteIoType io_type, const char* const** types, size_t* num_types) {
  if (async_signature_runner == nullptr || types == nullptr ||
      num_types == nullptr)
    return kTfLiteError;
  const auto& synchronization_types =
      async_signature_runner->impl->SupportedSynchronizations(io_type);
  *types = synchronization_types.data();
  *num_types = synchronization_types.size();
  return kTfLiteOk;
}

bool TfLiteAsyncSignatureRunnerReconcileRestrictions(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteIoType io_type, const char* name,
    const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict) {
  if (!async_signature_runner) return false;
  return async_signature_runner->impl->ReconcileRestrictions(
      io_type, name, user_provided_attributes, merged, conflict);
}

bool TfLiteAsyncSignatureRunnerReconcileRestrictionsByIndex(
    const TfLiteAsyncSignatureRunner* async_signature_runner, int tensor_index,
    const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict) {
  if (!async_signature_runner) return false;
  return async_signature_runner->impl->ReconcileRestrictions(
      tensor_index, user_provided_attributes, merged, conflict);
}

TfLiteStatus TfLiteAsyncSignatureRunnerSetAttributes(
    TfLiteAsyncSignatureRunner* async_signature_runner, TfLiteIoType io_type,
    const char* name, const TfLiteAttributeMap* attrs) {
  if (!async_signature_runner) return kTfLiteError;
  return async_signature_runner->impl->SetAttributes(io_type, name, attrs);
}

TfLiteStatus TfLiteAsyncSignatureRunnerSetAttributesByIndex(
    TfLiteAsyncSignatureRunner* async_signature_runner, int tensor_index,
    const TfLiteAttributeMap* attrs) {
  if (!async_signature_runner) return kTfLiteError;
  return async_signature_runner->impl->SetAttributes(tensor_index, attrs);
}

TfLiteStatus TfLiteAsyncSignatureRunnerPrepareBackends(
    TfLiteAsyncSignatureRunner* async_signature_runner) {
  if (!async_signature_runner) return kTfLiteError;
  return async_signature_runner->impl->PrepareBackends();
}

TfLiteExecutionTask* TfLiteAsyncSignatureRunnerCreateTask(
    TfLiteAsyncSignatureRunner* async_signature_runner) {
  if (!async_signature_runner) return nullptr;
  return async_signature_runner->impl->CreateTask();
}

TfLiteStatus TfLiteAsyncSignatureRunnerInvokeAsync(
    TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteExecutionTask* task) {
  if (!async_signature_runner) return kTfLiteError;
  return async_signature_runner->impl->InvokeAsync(task);
}

TfLiteStatus TfLiteAsyncSignatureRunnerWait(
    TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteExecutionTask* task) {
  if (!async_signature_runner) return kTfLiteError;
  return async_signature_runner->impl->Wait(task);
}

TfLiteStatus TfLiteAsyncSignatureRunnerFinish(
    TfLiteAsyncSignatureRunner* async_signature_runner,
    TfLiteExecutionTask* task) {
  if (!async_signature_runner) return kTfLiteError;
  return async_signature_runner->impl->Finish(task);
}

size_t TfLiteAsyncSignatureRunnerGetInputCount(
    const TfLiteAsyncSignatureRunner* async_signature_runner) {
  if (!async_signature_runner) return 0;
  return async_signature_runner->impl->input_size();
}

const char* TfLiteAsyncSignatureRunnerGetInputName(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    int32_t input_index) {
  if (!async_signature_runner) return nullptr;
  size_t count =
      TfLiteAsyncSignatureRunnerGetInputCount(async_signature_runner);
  if (input_index < 0 || input_index >= count) {
    return nullptr;
  }
  const auto& input_names = async_signature_runner->impl->input_names();
  if (input_index >= input_names.size()) {
    return nullptr;
  }
  return input_names[input_index];
}

size_t TfLiteAsyncSignatureRunnerGetOutputCount(
    const TfLiteAsyncSignatureRunner* async_signature_runner) {
  if (!async_signature_runner) return 0;
  return async_signature_runner->impl->output_size();
}

const char* TfLiteAsyncSignatureRunnerGetOutputName(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    int32_t output_index) {
  if (!async_signature_runner) return nullptr;
  size_t count =
      TfLiteAsyncSignatureRunnerGetOutputCount(async_signature_runner);
  if (output_index < 0 || output_index >= count) {
    return nullptr;
  }
  const auto& output_names = async_signature_runner->impl->output_names();
  if (output_index >= output_names.size()) {
    return nullptr;
  }
  return async_signature_runner->impl->output_names()[output_index];
}

const TfLiteOpaqueTensor* TfLiteAsyncSignatureRunnerGetInputTensor(
    TfLiteAsyncSignatureRunner* async_signature_runner,
    const char* input_name) {
  if (!async_signature_runner) return nullptr;
  return async_signature_runner->impl->input_tensor(input_name);
}

const TfLiteOpaqueTensor* TfLiteAsyncSignatureRunnerGetOutputTensor(
    const TfLiteAsyncSignatureRunner* async_signature_runner,
    const char* output_name) {
  if (!async_signature_runner) return nullptr;
  return async_signature_runner->impl->output_tensor(output_name);
}

void TfLiteAsyncSignatureRunnerDelete(
    TfLiteAsyncSignatureRunner* signature_runner) {
  delete signature_runner;
}

const int* TfLiteAsyncSignatureRunnerInputTensorIndices(
    const TfLiteAsyncSignatureRunner* async_signature_runner) {
  if (!async_signature_runner) return nullptr;
  return async_signature_runner->impl->inputs().data();
}

const int* TfLiteAsyncSignatureRunnerOutputTensorIndices(
    const TfLiteAsyncSignatureRunner* async_signature_runner) {
  if (!async_signature_runner) return nullptr;
  return async_signature_runner->impl->outputs().data();
}

const TfLiteOpaqueTensor* TfLiteAsyncSignatureRunnerGetTensor(
    const TfLiteAsyncSignatureRunner* async_signature_runner, int index) {
  if (!async_signature_runner) return nullptr;
  return async_signature_runner->impl->tensor(index);
}
