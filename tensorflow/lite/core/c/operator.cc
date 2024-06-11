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
#include "tensorflow/lite/core/c/operator.h"

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/common_internal.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/c/c_api_types.h"

TfLiteOperator* TfLiteOperatorCreate(TfLiteBuiltinOperator builtin_code,
                                     const char* custom_name, int version) {
  return TfLiteOperatorCreateWithData(builtin_code, custom_name, version,
                                      /*user_data=*/nullptr);
}

TfLiteOperator* TfLiteOperatorCreateWithData(TfLiteBuiltinOperator builtin_code,
                                             const char* custom_name,
                                             int version, void* user_data) {
  return new TfLiteOperator{.custom_name = custom_name,
                            .version = version,
                            .init = nullptr,
                            .free = nullptr,
                            .prepare = nullptr,
                            .invoke = nullptr,
                            .async_kernel = nullptr,
                            .builtin_code = builtin_code,
                            .node_index = -1,
                            .inplace_operator = kTfLiteInplaceOpNone,
                            .user_data = user_data};
}

void TfLiteOperatorDelete(TfLiteOperator* reg) { delete reg; }

void TfLiteOperatorSetInit(TfLiteOperator* registration,
                           void* (*init)(TfLiteOpaqueContext* context,
                                         const char* buffer, size_t length)) {
  registration->init = init;
}

TfLiteStatus TfLiteOperatorSetInitWithData(
    TfLiteOperator* registration,
    void* (*init)(void* user_data, TfLiteOpaqueContext* context,
                  const char* buffer, size_t length)) {
  registration->init_with_data = init;
  return kTfLiteOk;
}

void TfLiteOperatorSetFree(TfLiteOperator* registration,
                           void (*free)(TfLiteOpaqueContext* context,
                                        void* data)) {
  registration->free = free;
}

TfLiteStatus TfLiteOperatorSetFreeWithData(
    TfLiteOperator* registration,
    void (*free)(void* user_data, TfLiteOpaqueContext* context, void* data)) {
  registration->free_with_data = free;
  return kTfLiteOk;
}

void TfLiteOperatorSetPrepare(
    TfLiteOperator* registration,
    TfLiteStatus (*prepare)(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node)) {
  registration->prepare = prepare;
}

TfLiteStatus TfLiteOperatorSetPrepareWithData(
    TfLiteOperator* registration,
    TfLiteStatus (*prepare)(void* user_data, TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node)) {
  registration->prepare_with_data = prepare;
  return kTfLiteOk;
}

void TfLiteOperatorSetInvoke(
    TfLiteOperator* registration,
    TfLiteStatus (*invoke)(TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node)) {
  registration->invoke = invoke;
}

TfLiteStatus TfLiteOperatorSetInvokeWithData(
    TfLiteOperator* registration,
    TfLiteStatus (*invoke)(void* user_data, TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node)) {
  registration->invoke_with_data = invoke;
  return kTfLiteOk;
}

void TfLiteOperatorSetAsyncKernel(
    TfLiteOperator* registration,
    TfLiteAsyncKernel* (*async_kernel)(TfLiteOpaqueContext* context,
                                       TfLiteOpaqueNode* node)) {
  registration->async_kernel = async_kernel;
}

TfLiteStatus TfLiteOperatorSetAsyncKernelWithData(
    TfLiteOperator* registration,
    TfLiteAsyncKernel* (*async_kernel)(void* user_data,
                                       TfLiteOpaqueContext* context,
                                       TfLiteOpaqueNode* node)) {
  registration->async_kernel_with_data = async_kernel;
  return kTfLiteOk;
}

void TfLiteOperatorSetInplaceOperator(TfLiteOperator* registration,
                                      uint64_t inplace_operator) {
  registration->inplace_operator = inplace_operator;
}

TfLiteBuiltinOperator TfLiteOperatorGetBuiltInCode(
    const TfLiteOperator* registration) {
  return static_cast<TfLiteBuiltinOperator>(registration->builtin_code);
}

const char* TfLiteOperatorGetCustomName(const TfLiteOperator* registration) {
  return registration->custom_name;
}

int TfLiteOperatorGetVersion(const TfLiteOperator* registration) {
  if (!registration) {
    return -1;
  }
  return registration->version;
}

void* TfLiteOperatorGetUserData(const TfLiteOperator* registration) {
  if (!registration) {
    return nullptr;
  }
  return registration->user_data;
}
