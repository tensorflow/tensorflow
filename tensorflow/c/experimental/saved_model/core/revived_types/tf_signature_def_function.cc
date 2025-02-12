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

#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_signature_def_function.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/flat_tensor_function.h"
#include "tensorflow/c/experimental/saved_model/core/signature_def_function_metadata.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

TFSignatureDefFunction::TFSignatureDefFunction(
    std::unique_ptr<FlatTensorFunction> func,
    SignatureDefFunctionMetadata metadata)
    : func_(std::move(func)), metadata_(std::move(metadata)) {}

absl::Status TFSignatureDefFunction::Create(
    const FunctionDef* function_def,
    std::vector<ImmediateExecutionTensorHandle*> captures,
    SignatureDefFunctionMetadata metadata, ImmediateExecutionContext* ctx,
    std::unique_ptr<TFSignatureDefFunction>* out) {
  std::unique_ptr<FlatTensorFunction> func;
  TF_RETURN_IF_ERROR(FlatTensorFunction::Create(
      function_def, std::move(captures), ctx, &func));

  out->reset(new TFSignatureDefFunction(std::move(func), std::move(metadata)));
  return absl::Status();
}

const SignatureDefFunctionMetadata&
TFSignatureDefFunction::GetFunctionMetadata() const {
  return metadata_;
}

absl::Status TFSignatureDefFunction::MakeCallOp(
    absl::Span<AbstractTensorHandle* const> inputs, ImmediateOpPtr* out) const {
  return func_->MakeCallOp(inputs, out);
}

}  // namespace tensorflow
