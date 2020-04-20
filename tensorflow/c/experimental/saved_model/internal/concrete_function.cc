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

#include "tensorflow/c/experimental/saved_model/public/concrete_function.h"

#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/experimental/saved_model/core/concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/function_metadata.h"
#include "tensorflow/c/experimental/saved_model/internal/concrete_function_type.h"
#include "tensorflow/c/experimental/saved_model/internal/function_metadata_type.h"

extern "C" {

TF_FunctionMetadata* TF_ConcreteFunctionGetMetadata(TF_ConcreteFunction* func) {
  return tensorflow::wrap(&tensorflow::unwrap(func)->GetFunctionMetadata());
}

TF_OutputList* TF_ConcreteFunctionGetCaptures(TF_ConcreteFunction* func) {
  // TODO(bmzhao): Refactor TF_OutputList struct definition into a separate
  // internal header, and implement this function.
  return nullptr;
}

TFE_Op* TF_ConcreteFunctionGetCallOp(TF_ConcreteFunction* func) {
  return new TFE_Op{tensorflow::unwrap(func)->GetCallOp()};
}

}  // end extern "C"
