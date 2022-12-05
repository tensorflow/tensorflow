/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/c_api_internal.h"

#include "tensorflow/lite/c/common_internal.h"

namespace tflite {
namespace internal {

void TfLiteRegistrationExternalSetInitWithData(
    TfLiteRegistrationExternal* registration, void* data,
    void* (*init)(void* data, TfLiteOpaqueContext* context, const char* buffer,
                  size_t length)) {
  // Note, we expect the caller of 'registration->init' to supply as 'data'
  // what we store in 'registration->init_data'.
  registration->init = init;
  registration->init_data = data;
}

void TfLiteRegistrationExternalSetPrepareWithData(
    TfLiteRegistrationExternal* registration, void* data,
    TfLiteStatus (*prepare)(void* data, TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node)) {
  // Note, we expect the caller of 'registration->prepare' to supply as 'data'
  // what we store in 'registration->prepare_data'.
  registration->prepare = prepare;
  registration->prepare_data = data;
}

void TfLiteRegistrationExternalSetInvokeWithData(
    TfLiteRegistrationExternal* registration, void* data,
    TfLiteStatus (*invoke)(void* data, TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node)) {
  // Note, we expect the caller of 'registration->invoke' to supply as 'data'
  // what we store in 'registration->invoke_data'.
  registration->invoke = invoke;
  registration->invoke_data = data;
}

void TfLiteRegistrationExternalSetFreeWithData(
    TfLiteRegistrationExternal* registration, void* data,
    void (*free)(void* data, TfLiteOpaqueContext* context, void* buffer)) {
  // Note, we expect the caller of 'registration->free' to supply as 'data'
  // what we store in 'registration->free_data'.
  registration->free = free;
  registration->free_data = data;
}

}  // namespace internal
}  // namespace tflite
