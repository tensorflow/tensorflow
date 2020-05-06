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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_CONCRETE_FUNCTION_LIST_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_CONCRETE_FUNCTION_LIST_H_

#include <stddef.h>

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/experimental/saved_model/public/concrete_function.h"

// An opaque type that is acts like a list of TF_ConcreteFunction pointers.
typedef struct TF_ConcreteFunctionList TF_ConcreteFunctionList;

// Returns the size of `list`.
TF_CAPI_EXPORT size_t
TF_ConcreteFunctionListSize(TF_ConcreteFunctionList* list);

// Returns the `i`th TF_ConcreteFunction in the list.
TF_CAPI_EXPORT TF_ConcreteFunction* TF_ConcreteFunctionListGet(
    TF_ConcreteFunctionList* list, int i);

// Deletes `list`.
TF_CAPI_EXPORT void TF_DeleteConcreteFunctionList(
    TF_ConcreteFunctionList* list);

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_CONCRETE_FUNCTION_LIST_H_
