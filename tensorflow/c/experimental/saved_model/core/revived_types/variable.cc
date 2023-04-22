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
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
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

Status Variable::CreateUninitialized(
    ImmediateExecutionContext* ctx, DataType dtype, TensorShape shape,
    absl::optional<std::string> name, const char* raw_device_name,
    const std::vector<std::string>& component_devices,
    std::unique_ptr<Variable>* output) {
  ImmediateTensorHandlePtr handle;

  if (component_devices.empty()) {
    TF_RETURN_IF_ERROR(internal::CreateUninitializedResourceVariable(
        ctx, dtype, shape, raw_device_name, &handle));
    output->reset(
        new Variable(ctx, dtype, shape, std::move(name), std::move(handle)));
    return Status();
  }

  if (!tensorflow::isa<EagerContext>(ctx)) {
    return errors::InvalidArgument(
        "Can only load distributed variables with EagerContext.");
  }

  EagerContext* eager_ctx = reinterpret_cast<EagerContext*>(ctx);

  std::vector<TensorHandle*> handles;
  for (const auto& device : component_devices) {
    ImmediateTensorHandlePtr handlePtr;
    TF_RETURN_IF_ERROR(internal::CreateUninitializedResourceVariable(
        ctx, dtype, shape, device.empty() ? nullptr : device.c_str(),
        &handlePtr));
    if (!tensorflow::isa<TensorHandle>(handlePtr.get())) {
      return errors::Internal("Returned replica handle has unsupported type.");
    }
    handles.push_back(reinterpret_cast<TensorHandle*>(handlePtr.release()));
  }
  TensorHandle* packed_handle;
  TF_RETURN_IF_ERROR(TensorHandle::CreatePackedHandle(
      std::move(handles), eager_ctx, &packed_handle));
  // The call to `CreatePackedHandle` incremented the handles' reference count,
  // which we must now decrement to make the packed handle the owner of those
  // handles. We can't loop through the `handles` vector because it was
  // `std::move`d in the call above.
  for (int i = 0; i != packed_handle->NumPackedHandles(); ++i) {
    TensorHandle* component;
    TF_RETURN_IF_ERROR(packed_handle->ExtractPackedHandle(i, &component));
    component->Unref();
  }

  handle.reset(packed_handle);
  output->reset(
      new Variable(ctx, dtype, shape, std::move(name), std::move(handle)));
  return Status();
}

}  // namespace tensorflow
