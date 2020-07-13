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

#include "tensorflow/c/experimental/saved_model/core/revived_types/variable.h"

#include <memory>

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/ops/variable_ops.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

Variable::Variable(ImmediateExecutionContext* ctx, DataType dtype,
                   TensorShape shape, absl::optional<std::string> name,
                   ImmediateTensorHandlePtr handle)
    : TensorHandleConvertible(std::move(handle)),
      name_(name.has_value() ? *name : "Variable"),
      dtype_(dtype),
      shape_(shape),
      ctx_(ctx) {}

Variable::~Variable() {
  // If the handle is null (perhaps because variable was std::moved from), then
  // we don't have to do anything.
  if (handle_ == nullptr) {
    return;
  }

  Status status = internal::DestroyResource(ctx_, handle_.get());
  if (!status.ok()) {
    LOG(ERROR) << "Error destroying variable: " << name_
               << "due to: " << status;
  }
}

DataType Variable::dtype() { return dtype_; }

TensorShape Variable::shape() { return shape_; }

Status Variable::Assign(ImmediateExecutionTensorHandle* handle) {
  return internal::AssignVariable(ctx_, handle_.get(), dtype_, handle);
}

Status Variable::ReadValue(ImmediateTensorHandlePtr* out) {
  return internal::ReadVariable(ctx_, handle_.get(), dtype_, out);
}

Status Variable::CreateUninitialized(ImmediateExecutionContext* ctx,
                                     DataType dtype, TensorShape shape,
                                     absl::optional<std::string> name,
                                     std::unique_ptr<Variable>* output) {
  ImmediateTensorHandlePtr handle;
  TF_RETURN_IF_ERROR(internal::CreateUninitializedResourceVariable(
      ctx, dtype, shape, &handle));

  output->reset(
      new Variable(ctx, dtype, shape, std::move(name), std::move(handle)));
  return Status();
}

}  // namespace tensorflow
