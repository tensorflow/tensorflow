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

#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"

#include <memory>
#include <string>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/flat_tensor_function.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {

TFConcreteFunction::TFConcreteFunction(std::unique_ptr<FlatTensorFunction> func,
                                       FunctionMetadata metadata)
    : func_(std::move(func)), metadata_(std::move(metadata)) {}

Status TFConcreteFunction::Create(
    const FunctionDef* function_def,
    std::vector<ImmediateExecutionTensorHandle*> captures,
    FunctionMetadata metadata, ImmediateExecutionContext* ctx,
    std::unique_ptr<TFConcreteFunction>* out) {
  std::unique_ptr<FlatTensorFunction> func;
  TF_RETURN_IF_ERROR(FlatTensorFunction::Create(
      function_def, std::move(captures), ctx, &func));

  out->reset(new TFConcreteFunction(std::move(func), std::move(metadata)));
  return Status();
}

const FunctionMetadata& TFConcreteFunction::GetFunctionMetadata() const {
  return metadata_;
}

Status TFConcreteFunction::MakeCallOp(
    absl::Span<AbstractTensorHandle* const> inputs, ImmediateOpPtr* out) const {
  return func_->MakeCallOp(inputs, out);
}

}  // namespace tensorflow
