/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_TENSORHANDLE_LIST_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_TENSORHANDLE_LIST_H_

#include <stddef.h>

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/eager/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An opaque type that is acts like a list of TF_ConcreteFunction pointers.
typedef struct TF_TensorHandleList TF_TensorHandleList;

// Returns the size of `list`.
TF_CAPI_EXPORT extern size_t TF_TensorHandleListSize(
    const TF_TensorHandleList* list);

// Returns the `i`th TFE_TensorHandle in the list.
TF_CAPI_EXPORT extern TFE_TensorHandle* TF_TensorHandleListGet(
    const TF_TensorHandleList* list, int i);

// Deletes `list`.
TF_CAPI_EXPORT extern void TF_DeleteTensorHandleList(
    const TF_TensorHandleList* list);

#ifdef __cplusplus
}  // end extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_TENSORHANDLE_LIST_H_
