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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_CONCRETE_FUNCTION_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_CONCRETE_FUNCTION_H_

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/experimental/saved_model/public/function_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An opaque type that corresponds to a Function loaded from a SavedModel.
// TODO(bmzhao): Work together w/srbs@ to make sure this composes w/the
// C++ Unified Eager/Graph API's AbstractFunction
typedef struct TF_ConcreteFunction TF_ConcreteFunction;

// Returns FunctionMetadata associated with `func`. Metadata's lifetime is
// bound to `func`, which is bound to the TF_SavedModel it was loaded from.
TF_CAPI_EXPORT extern TF_FunctionMetadata* TF_ConcreteFunctionGetMetadata(
    TF_ConcreteFunction* func);

// Returns a TFE_Op suitable for executing this function. Caller must provide
// all function inputs in `inputs`, and must not add any additional inputs on
// the returned op. (i.e. don't call TFE_OpAddInput or TFE_OpAddInputList).
// The caller is responsible for deleting the returned TFE_Op. If op
// construction fails, `status` will be non-OK and the returned pointer will be
// null.
// TODO(bmzhao): Remove this function in a subsequent change; Design + implement
// a Function Execution interface for ConcreteFunction that accepts a tagged
// union of types (tensorflow::Value). This effectively requires moving much of
// the implementation of function.py/def_function.py to C++, and exposing a
// high-level API here. A strawman for what this interface could look like:
// TF_Value* TF_ExecuteFunction(TFE_Context*, TF_ConcreteFunction*, TF_Value*
// inputs, int num_inputs, TF_Status* status);
TF_CAPI_EXPORT extern TFE_Op* TF_ConcreteFunctionMakeCallOp(
    TF_ConcreteFunction* func, TFE_TensorHandle** inputs, int num_inputs,
    TF_Status* status);

#ifdef __cplusplus
}  // end extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_CONCRETE_FUNCTION_H_
