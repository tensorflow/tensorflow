/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/c/experimental/ops/resource_variable_ops.h"

#include <cstring>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tracing_utils.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::tracing::MaybeSetOpName;

namespace tensorflow {
namespace ops {

// Op: VarHandleOp()
// Summary: Creates a handle to a Variable resource.
//
// Description:
absl::Status VarHandleOp(AbstractContext* ctx, AbstractTensorHandle** resource,
                         DataType dtype, const PartialTensorShape shape,
                         const char* container, const char* shared_name,
                         absl::Span<string const> allowed_devices,
                         const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("VarHandleOp", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrString("container", container, strlen(container)));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrString("shared_name", shared_name, strlen(shared_name)));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrType("dtype", dtype));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrShape("shape", shape));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrStringList("allowed_devices", allowed_devices));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(resource, 1), &num_retvals);
}

// Op: ReadVariableOp()
// Summary: Reads the value of a variable.
//
// Description:
//   The tensor returned by this operation is immutable.
//
//   The value returned by this operation is guaranteed to be influenced by all
//   the writes on which this operation depends directly or indirectly, and to
//   not be influenced by any of the writes which depend directly or indirectly
//   on this operation.
absl::Status ReadVariableOp(AbstractContext* ctx,
                            AbstractTensorHandle* const resource,
                            AbstractTensorHandle** value, DataType dtype,
                            const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("ReadVariableOp", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(resource));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrType("dtype", dtype));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(value, 1), &num_retvals);
}

// Op: AssignVariableOp()
// Summary: Assigns a new value to a variable.
//
// Description:
//   Any ReadVariableOp with a control dependency on this op is guaranteed to
//   return this value or a subsequent newer value of the variable.
absl::Status AssignVariableOp(AbstractContext* ctx,
                              AbstractTensorHandle* const resource,
                              AbstractTensorHandle* const value,
                              bool validate_shape, const char* name,
                              const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("AssignVariableOp", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(resource));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(value));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrBool("validate_shape", validate_shape));
  int num_retvals = 0;
  std::vector<AbstractTensorHandle*> dummy_outputs;
  return op_ptr->Execute(absl::MakeSpan(dummy_outputs), &num_retvals);
}

// Op: DestroyResourceOp()
// Summary: Deletes the resource specified by the handle.
//
// Description:
//   All subsequent operations using the resource will result in a NotFound
//   error status.
absl::Status DestroyResourceOp(AbstractContext* ctx,
                               AbstractTensorHandle* const resource,
                               bool ignore_lookup_error, const char* name,
                               const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("DestroyResourceOp", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(resource));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrBool("ignore_lookup_error", ignore_lookup_error));
  int num_retvals = 0;
  std::vector<AbstractTensorHandle*> dummy_outputs;
  return op_ptr->Execute(absl::MakeSpan(dummy_outputs), &num_retvals);
}

}  // namespace ops
}  // namespace tensorflow
