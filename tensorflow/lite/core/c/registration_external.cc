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
#include "tensorflow/lite/core/c/registration_external.h"

#include "tensorflow/lite/c/common_internal.h"

TfLiteRegistrationExternal* TfLiteRegistrationExternalCreate(
    TfLiteBuiltinOperator builtin_code, const char* custom_name, int version) {
  return new TfLiteRegistrationExternal{/*.custom_name =*/custom_name,
                                        /*.version =*/version,
                                        /*.init =*/nullptr,
                                        /*.free =*/nullptr,
                                        /*.prepare =*/nullptr,
                                        /*.invoke =*/nullptr,
                                        /*.async_kernel =*/nullptr,
                                        /*.builtin_code =*/builtin_code,
                                        /*.node_index =*/-1};
}

void TfLiteRegistrationExternalDelete(TfLiteRegistrationExternal* reg) {
  delete reg;
}

void TfLiteRegistrationExternalSetInit(
    TfLiteRegistrationExternal* registration,
    void* (*init)(TfLiteOpaqueContext* context, const char* buffer,
                  size_t length)) {
  registration->init = init;
}

void TfLiteRegistrationExternalSetFree(
    TfLiteRegistrationExternal* registration,
    void (*free)(TfLiteOpaqueContext* context, void* data)) {
  registration->free = free;
}

void TfLiteRegistrationExternalSetPrepare(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*prepare)(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node)) {
  registration->prepare = prepare;
}

void TfLiteRegistrationExternalSetInvoke(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*invoke)(TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node)) {
  registration->invoke = invoke;
}

void TfLiteRegistrationExternalSetAsyncKernel(
    TfLiteRegistrationExternal* registration,
    TfLiteAsyncKernel* (*async_kernel)(TfLiteOpaqueContext* context,
                                       TfLiteOpaqueNode* node)) {
  registration->async_kernel = async_kernel;
}

void TfLiteRegistrationExternalSetInplaceOperator(
    TfLiteRegistrationExternal* registration, uint64_t inplace_operator) {
  registration->inplace_operator = inplace_operator;
}

TfLiteBuiltinOperator TfLiteRegistrationExternalGetBuiltInCode(
    const TfLiteRegistrationExternal* registration) {
  return static_cast<TfLiteBuiltinOperator>(registration->builtin_code);
}

const char* TfLiteRegistrationExternalGetCustomName(
    const TfLiteRegistrationExternal* registration) {
  return registration->custom_name;
}

int TfLiteRegistrationExternalGetVersion(
    const TfLiteRegistrationExternal* registration) {
  if (!registration) {
    return -1;
  }
  return registration->version;
}
