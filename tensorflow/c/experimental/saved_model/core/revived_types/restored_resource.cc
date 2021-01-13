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

#include "tensorflow/c/experimental/saved_model/core/revived_types/restored_resource.h"

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

Status ExecuteNoArgDummyReturnFunction(TFConcreteFunction* func) {
  ImmediateOpPtr function_op;
  TF_RETURN_IF_ERROR(func->MakeCallOp({}, &function_op));

  AbstractTensorHandle* dummy_output = nullptr;
  int num_retvals = 1;
  TF_RETURN_IF_ERROR(function_op->Execute(
      absl::MakeSpan(&dummy_output, num_retvals), &num_retvals));
  AbstractTensorHandlePtr owned_dummy_output(dummy_output);
  return Status();
}

}  // namespace

RestoredResource::RestoredResource(const std::string& device,
                                   TFConcreteFunction* create_resource,
                                   TFConcreteFunction* initialize,
                                   TFConcreteFunction* destroy_resource,
                                   ImmediateTensorHandlePtr resource_handle)
    : TensorHandleConvertible(std::move(resource_handle)),
      device_(device),
      create_resource_(create_resource),
      initialize_(initialize),
      destroy_resource_(destroy_resource) {}

Status RestoredResource::Initialize() const {
  return ExecuteNoArgDummyReturnFunction(initialize_);
}

RestoredResource::~RestoredResource() {
  // Note(bmzhao): SavedModels saved before
  // https://github.com/tensorflow/tensorflow/commit/3c806101f57768e479f8646e7518bbdff1632ca3
  // did not have their destroy_resource function saved, meaning they will
  // leak resources.
  if (destroy_resource_ != nullptr) {
    Status status = ExecuteNoArgDummyReturnFunction(destroy_resource_);
    if (!status.ok()) {
      LOG(WARNING)
          << "Failed executing destroy_resource function for RestoredResource: "
          << status.error_message();
    }
  }
}

}  // namespace tensorflow
