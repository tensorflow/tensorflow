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

#include "tensorflow/c/experimental/saved_model/public/signature_def_function.h"

#include <cstddef>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/experimental/saved_model/core/signature_def_function.h"
#include "tensorflow/c/experimental/saved_model/core/signature_def_function_metadata.h"
#include "tensorflow/c/experimental/saved_model/internal/signature_def_function_metadata_type.h"
#include "tensorflow/c/experimental/saved_model/internal/signature_def_function_type.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/platform/status.h"

extern "C" {

TF_SignatureDefFunctionMetadata* TF_SignatureDefFunctionGetMetadata(
    TF_SignatureDefFunction* func) {
  return tensorflow::wrap(const_cast<tensorflow::SignatureDefFunctionMetadata*>(
      &tensorflow::unwrap(func)->GetFunctionMetadata()));
}

TFE_Op* TF_SignatureDefFunctionMakeCallOp(TF_SignatureDefFunction* func,
                                          TFE_TensorHandle** inputs,
                                          int num_inputs, TF_Status* status) {
  tensorflow::ImmediateOpPtr call_op;
  absl::Span<tensorflow::AbstractTensorHandle* const> input_span(
      reinterpret_cast<tensorflow::AbstractTensorHandle**>(
          tensorflow::unwrap(inputs)),
      static_cast<size_t>(num_inputs));
  status->status = tensorflow::unwrap(func)->MakeCallOp(input_span, &call_op);
  if (!status->status.ok()) {
    return nullptr;
  }
  return tensorflow::wrap(call_op.release());
}

}  // end extern "C"
