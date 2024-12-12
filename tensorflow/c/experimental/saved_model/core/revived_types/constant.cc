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

#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

Constant::Constant(ImmediateTensorHandlePtr handle)
    : TensorHandleConvertible(std::move(handle)) {}

absl::Status Constant::Create(ImmediateExecutionContext* ctx,
                              AbstractTensorInterface* tensor,
                              std::unique_ptr<Constant>* output) {
  ImmediateExecutionTensorHandle* handle = ctx->CreateLocalHandle(tensor);
  if (handle == nullptr) {
    return errors::Internal("Failed to convert tensor to tensorhandle");
  }
  output->reset(new Constant(ImmediateTensorHandlePtr(handle)));
  return absl::Status();
}

}  // namespace tensorflow
