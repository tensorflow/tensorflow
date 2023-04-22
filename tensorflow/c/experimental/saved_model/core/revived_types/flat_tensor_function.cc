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

#include "tensorflow/c/experimental/saved_model/core/revived_types/flat_tensor_function.h"

#include <memory>
#include <string>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {

FlatTensorFunction::FlatTensorFunction(
    const std::string& name, std::vector<ImmediateTensorHandlePtr> captures,
    ImmediateExecutionContext* ctx)
    : name_(name), captures_(std::move(captures)), ctx_(ctx) {}

FlatTensorFunction::~FlatTensorFunction() {
  Status status = ctx_->RemoveFunction(name_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to remove functiondef " << name_ << ". "
               << status.error_message();
  }
}

Status FlatTensorFunction::Create(
    const FunctionDef* function_def,
    std::vector<ImmediateExecutionTensorHandle*> captures,
    ImmediateExecutionContext* ctx, std::unique_ptr<FlatTensorFunction>* out) {
  TF_RETURN_IF_ERROR(ctx->AddFunctionDef(*function_def));
  std::vector<ImmediateTensorHandlePtr> owned_captures;
  owned_captures.reserve(captures.size());
  for (ImmediateExecutionTensorHandle* capture : captures) {
    capture->Ref();
    owned_captures.push_back(ImmediateTensorHandlePtr(capture));
  }

  out->reset(new FlatTensorFunction(function_def->signature().name(),
                                    std::move(owned_captures), ctx));
  return Status();
}

Status FlatTensorFunction::MakeCallOp(
    absl::Span<AbstractTensorHandle* const> inputs, ImmediateOpPtr* out) const {
  out->reset(ctx_->CreateOperation());
  // In eager mode, TF2 python executes functions by constructing an op with
  // the name of the functiondef:
  // https://github.com/tensorflow/tensorflow/blob/66668ec0ca432e2f38a575b814f45b6d299d01ed/tensorflow/python/eager/function.py#L545
  // In graph mode, we create a PartitionedCallOp instead:
  // https://github.com/tensorflow/tensorflow/blob/66668ec0ca432e2f38a575b814f45b6d299d01ed/tensorflow/python/eager/function.py#L573

  // TODO(bmzhao): After discussing with Allen, we should execute this via a
  // PartitionedCallOp for compatibility with "tooling that assumes functions in
  // graphs are PartitionedCallOps".
  TF_RETURN_IF_ERROR((*out)->Reset(name_.c_str(), nullptr));

  // Adding the user-provided inputs to the function.
  TF_RETURN_IF_ERROR((*out)->AddInputList(inputs));

  absl::Span<AbstractTensorHandle* const> captures(
      reinterpret_cast<AbstractTensorHandle* const*>(captures_.data()),
      captures_.size());

  // Adding the captures of the function.
  TF_RETURN_IF_ERROR((*out)->AddInputList(captures));
  return Status();
}

}  // namespace tensorflow
