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

#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/experimental/saved_model/core/concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/function_metadata.h"
#include "tensorflow/c/experimental/saved_model/internal/concrete_function_type.h"
#include "tensorflow/c/experimental/saved_model/internal/function_metadata_type.h"
#include "tensorflow/c/experimental/saved_model/internal/tensorhandle_list_type.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/platform/status.h"

extern "C" {

TF_FunctionMetadata* TF_ConcreteFunctionGetMetadata(TF_ConcreteFunction* func) {
  return tensorflow::wrap(const_cast<tensorflow::FunctionMetadata*>(
      &tensorflow::unwrap(func)->GetFunctionMetadata()));
}

const TF_TensorHandleList* TF_ConcreteFunctionGetCaptures(
    TF_ConcreteFunction* func) {
  return tensorflow::wrap(&tensorflow::unwrap(func)->GetCaptures());
}

TFE_Op* TF_ConcreteFunctionGetCallOp(TF_ConcreteFunction* func,
                                     TF_Status* status) {
  tensorflow::ImmediateOpPtr call_op(nullptr);
  status->status = tensorflow::unwrap(func)->GetCallOp(&call_op);
  return tensorflow::wrap(call_op.release());
}

}  // end extern "C"
