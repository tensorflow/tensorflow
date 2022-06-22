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
#ifndef TENSORFLOW_LITE_C_C_API_COMMON_INTERNAL_H_
#define TENSORFLOW_LITE_C_C_API_COMMON_INTERNAL_H_

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"

// Internal structures and subroutines used by the C API. These are likely to
// change and should not be depended on directly by any C API clients.
//
// NOTE: This header does not follow C conventions and does not define a C API.
// It is effectively an (internal) implementation detail of the C API.

// `TfLiteRegistrationExternal` is an external version of `TfLiteRegistration`
// for C API which doesn't use internal types (such as `TfLiteContext`) but only
// uses stable API types (such as `TfLiteOpaqueContext`). The purpose of each
// field is the exactly the same as with `TfLiteRegistration`.
typedef struct TfLiteRegistrationExternal {
  // Custom op name.
  const char* custom_name;

  // The version of the op. The version should be higher than 0.
  int version;

  // Initializes the op from serialized data.
  void* (*init)(TfLiteOpaqueContext* context, const char* buffer,
                size_t length);

  // The pointer `buffer` is the data previously returned by an init invocation.
  void (*free)(TfLiteOpaqueContext* context, void* buffer);

  // Called when the inputs that this node depends on have been resized.
  TfLiteStatus (*prepare)(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node);

  // Called when the node is executed. (should read node->inputs and output to
  // node->outputs).
  TfLiteStatus (*invoke)(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node);
} TfLiteRegistrationExternal;

#endif  // TENSORFLOW_LITE_C_C_API_INTERNAL_H_
