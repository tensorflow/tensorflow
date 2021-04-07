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
#include "tensorflow/c/experimental/ops/math_ops.h"

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tracing_utils.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"

using tensorflow::tracing::MaybeSetOpName;

namespace tensorflow {
namespace ops {

Status Mul(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, absl::Span<AbstractTensorHandle*> z,
           const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Mul", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  int num_retvals = 1;
  return op_ptr->Execute(z, &num_retvals);
}

Status Conj(AbstractContext* ctx, AbstractTensorHandle* const input,
            absl::Span<AbstractTensorHandle*> output, const char* name) {
  auto dtype = input->DataType();
  if (DataTypeIsFloating(BaseType(dtype)) ||
      DataTypeIsInteger(BaseType(dtype))) {
    TF_RETURN_IF_ERROR(Identity(ctx, input, output, name));
  } else if (DataTypeIsComplex(BaseType(dtype)) ||
             BaseType(dtype) == DT_VARIANT) {
    AbstractOperationPtr op_ptr(ctx->CreateOperation());
    TF_RETURN_IF_ERROR(op_ptr->Reset("Conj", /*raw_device_name=*/nullptr));
    TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
    TF_RETURN_IF_ERROR(op_ptr->AddInput(input));

    int num_retvals = 1;
    TF_RETURN_IF_ERROR(op_ptr->Execute(output, &num_retvals));
  } else {
    return errors::InvalidArgument(
        "Expected numeric or variant tensor, got dtype ", dtype);
  }
  return Status::OK();
}

Status Add(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, absl::Span<AbstractTensorHandle*> z,
           const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("AddV2", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(op_ptr->Execute(z, &num_retvals));
  return Status::OK();
}

Status MatMul(AbstractContext* ctx, AbstractTensorHandle* const a,
              AbstractTensorHandle* const b,
              absl::Span<AbstractTensorHandle*> product, const char* name,
              bool transpose_a, bool transpose_b) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("MatMul", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(a));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(b));

  TF_RETURN_IF_ERROR(op_ptr->SetAttrBool("transpose_a", transpose_a));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrBool("transpose_b", transpose_b));

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(op_ptr->Execute(product, &num_retvals));
  return Status::OK();
}

Status Neg(AbstractContext* ctx, AbstractTensorHandle* const x,
           absl::Span<AbstractTensorHandle*> y, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Neg", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));

  int num_retvals = 1;
  return op_ptr->Execute(y, &num_retvals);
}

Status Sum(AbstractContext* ctx, AbstractTensorHandle* const input,
           AbstractTensorHandle* const reduction_indices,
           absl::Span<AbstractTensorHandle*> output, const char* name,
           bool keep_dims) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Sum", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(input));              // input_vals
  TF_RETURN_IF_ERROR(op_ptr->AddInput(reduction_indices));  // reduction_indices

  TF_RETURN_IF_ERROR(op_ptr->SetAttrBool("keep_dims", keep_dims));

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(op_ptr->Execute(output, &num_retvals));
  return Status::OK();
}

Status Sub(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, absl::Span<AbstractTensorHandle*> z,
           const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Sub", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(op_ptr->Execute(z, &num_retvals));
  return Status::OK();
}

Status Div(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, absl::Span<AbstractTensorHandle*> z,
           const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Div", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));  // x
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));  // y

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(op_ptr->Execute(z, &num_retvals));  // z = x / y
  return Status::OK();
}

Status DivNoNan(AbstractContext* ctx, AbstractTensorHandle* const x,
                AbstractTensorHandle* const y,
                absl::Span<AbstractTensorHandle*> z, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("DivNoNan", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));  // x
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));  // y

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(
      op_ptr->Execute(z, &num_retvals));  // z = x / y, (z_i = 0 if y_i = 0)
  return Status::OK();
}

Status Exp(AbstractContext* ctx, AbstractTensorHandle* const x,
           absl::Span<AbstractTensorHandle*> y, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Exp", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));

  int num_retvals = 1;
  return op_ptr->Execute(y, &num_retvals);
}

Status Sqrt(AbstractContext* ctx, AbstractTensorHandle* const x,
            absl::Span<AbstractTensorHandle*> y, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Sqrt", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));

  int num_retvals = 1;
  Status s = op_ptr->Execute(y, &num_retvals);
  return s;
}

Status SqrtGrad(AbstractContext* ctx, AbstractTensorHandle* const y,
                AbstractTensorHandle* const dy,
                absl::Span<AbstractTensorHandle*> z, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("SqrtGrad", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(dy));

  int num_retvals = 1;
  Status s = op_ptr->Execute(z, &num_retvals);
  return s;
}

Status Log1p(AbstractContext* ctx, AbstractTensorHandle* const x,
             absl::Span<AbstractTensorHandle*> y, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Log1p", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));

  int num_retvals = 1;
  Status s = op_ptr->Execute(y, &num_retvals);
  return s;
}

}  // namespace ops
}  // namespace tensorflow
