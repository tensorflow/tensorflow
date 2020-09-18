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
#include "tensorflow/c/experimental/ops/array_ops.h"

#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace ops {

Status Identity(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr identity_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      identity_op->Reset("Identity", /*raw_device_name=*/nullptr));
  if (isa<tensorflow::tracing::TracingOperation>(identity_op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(identity_op.get())
                           ->SetOpName(name));
  }
  TF_RETURN_IF_ERROR(identity_op->AddInput(inputs[0]));
  int num_retvals = 1;
  return identity_op->Execute(outputs, &num_retvals);
}

Status ZerosLike(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> inputs,
                 absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr z_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(z_op->Reset("ZerosLike", /*raw_device_name=*/nullptr));
  if (isa<tensorflow::tracing::TracingOperation>(z_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(z_op.get())->SetOpName(name));
  }
  TF_RETURN_IF_ERROR(z_op->AddInput(inputs[0]));
  int num_retvals = 1;
  return z_op->Execute(outputs, &num_retvals);
}

Status Shape(AbstractContext* ctx,
             absl::Span<AbstractTensorHandle* const> inputs,
             absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr shape_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(shape_op->Reset("Shape", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(shape_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(shape_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(shape_op->AddInput(inputs[0]));  // input
  int num_retvals = 1;
  TF_RETURN_IF_ERROR(shape_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

Status ExpandDims(AbstractContext* ctx,
                  absl::Span<AbstractTensorHandle* const> inputs,
                  absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op->Reset("ExpandDims", /*raw_device_name=*/nullptr));
  if (isa<tensorflow::tracing::TracingOperation>(op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(op.get())->SetOpName(name));
  }
  TF_RETURN_IF_ERROR(op->AddInput(inputs[0]));
  TF_RETURN_IF_ERROR(op->AddInput(inputs[1]));
  int num_retvals = 1;
  return op->Execute(outputs, &num_retvals);
}

}  // namespace ops
}  // namespace tensorflow
