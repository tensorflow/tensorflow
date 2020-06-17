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

#include "tensorflow/c/experimental/saved_model/core/ops/variable_ops.h"

#include "absl/types/span.h"
#include "tensorflow/c/eager/context_interface.h"
#include "tensorflow/c/experimental/saved_model/core/ops/owned_eager_op.h"
#include "tensorflow/c/experimental/saved_model/core/ops/owned_tensor_handle.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace internal {

static const char kNoSharingResourceID[] =
    "cd2c89b7-88b7-44c8-ad83-06c2a9158347";

Status CreateUninitializedResourceVariable(AbstractContextInterface* ctx,
                                           DataType dtype, TensorShape shape,
                                           AbstractTensorHandlePtr* handle) {
  AbstractOpPtr varhandle_op = AbstractOpPtr(ctx->CreateOperation());

  TF_RETURN_IF_ERROR(varhandle_op->Reset("VarHandleOp", nullptr));
  TF_RETURN_IF_ERROR(varhandle_op->SetAttrType("dtype", dtype));

  // Note that if shape is unknown rank, shape.dim_sizes() will be empty, and
  // shape.dims() will be -1.
  gtl::InlinedVector<int64, 4> dim_sizes = shape.dim_sizes();
  TF_RETURN_IF_ERROR(varhandle_op->SetAttrShape(
      "shape", reinterpret_cast<const int64_t*>(dim_sizes.data()),
      shape.dims()));
  TF_RETURN_IF_ERROR(varhandle_op->SetAttrString("container", "", 0));
  TF_RETURN_IF_ERROR(varhandle_op->SetAttrString(
      "shared_name", kNoSharingResourceID, strlen(kNoSharingResourceID)));

  AbstractTensorHandleInterface* var_handle = nullptr;
  int num_retvals = 1;
  TF_RETURN_IF_ERROR(varhandle_op->Execute(
      absl::MakeSpan(&var_handle, num_retvals), &num_retvals));
  handle->reset(var_handle);
  return Status();
}

Status AssignVariable(AbstractContextInterface* ctx,
                      AbstractTensorHandleInterface* variable_handle,
                      DataType dtype, AbstractTensorHandleInterface* value) {
  AbstractOpPtr assign_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(assign_op->Reset("AssignVariableOp", nullptr));
  TF_RETURN_IF_ERROR(assign_op->SetAttrType("dtype", dtype));
  TF_RETURN_IF_ERROR(assign_op->AddInput(variable_handle));
  TF_RETURN_IF_ERROR(assign_op->AddInput(value));

  int num_retvals = 0;
  TF_RETURN_IF_ERROR(assign_op->Execute({}, &num_retvals));
  return Status();
}

Status ReadVariable(AbstractContextInterface* ctx,
                    AbstractTensorHandleInterface* variable_handle,
                    DataType dtype, AbstractTensorHandlePtr* output) {
  AbstractOpPtr read_op = AbstractOpPtr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(read_op->Reset("ReadVariableOp", nullptr));
  TF_RETURN_IF_ERROR(read_op->SetAttrType("dtype", dtype));
  TF_RETURN_IF_ERROR(read_op->AddInput(variable_handle));

  AbstractTensorHandleInterface* value = nullptr;
  int num_retvals = 1;
  TF_RETURN_IF_ERROR(
      read_op->Execute(absl::MakeSpan(&value, num_retvals), &num_retvals));
  output->reset(value);
  return Status();
}

Status DestroyResource(AbstractContextInterface* ctx,
                       AbstractTensorHandleInterface* handle) {
  AbstractOpPtr destroy_op = AbstractOpPtr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(destroy_op->Reset("DestroyResourceOp", nullptr));
  TF_RETURN_IF_ERROR(destroy_op->SetAttrBool("ignore_lookup_error", true));
  TF_RETURN_IF_ERROR(destroy_op->AddInput(handle));

  int num_retvals = 0;
  TF_RETURN_IF_ERROR(destroy_op->Execute({}, &num_retvals));
  return Status();
}

}  // namespace internal
}  // namespace tensorflow
