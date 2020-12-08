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

Status Mul(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr mul_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(mul_op->Reset("Mul", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(mul_op.get(), name));
  TF_RETURN_IF_ERROR(mul_op->AddInput(inputs[0]));
  TF_RETURN_IF_ERROR(mul_op->AddInput(inputs[1]));
  int num_retvals = 1;
  return mul_op->Execute(outputs, &num_retvals);
}

Status Conj(AbstractContext* ctx,
            absl::Span<AbstractTensorHandle* const> inputs,
            absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  auto dtype = inputs[0]->DataType();
  if (DataTypeIsFloating(BaseType(dtype)) ||
      DataTypeIsInteger(BaseType(dtype))) {
    TF_RETURN_IF_ERROR(Identity(ctx, inputs, outputs, name));
  } else if (DataTypeIsComplex(BaseType(dtype)) ||
             BaseType(dtype) == DT_VARIANT) {
    AbstractOperationPtr conj_op(ctx->CreateOperation());
    TF_RETURN_IF_ERROR(conj_op->Reset("Conj", /*raw_device_name=*/nullptr));
    TF_RETURN_IF_ERROR(MaybeSetOpName(conj_op.get(), name));
    TF_RETURN_IF_ERROR(conj_op->AddInput(inputs[0]));

    int num_retvals = 1;
    TF_RETURN_IF_ERROR(conj_op->Execute(outputs, &num_retvals));
  } else {
    return errors::InvalidArgument(
        "Expected numeric or variant tensor, got dtype ", dtype);
  }
  return Status::OK();
}

Status Add(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr add_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(add_op->Reset("AddV2", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(add_op.get(), name));
  TF_RETURN_IF_ERROR(add_op->AddInput(inputs[0]));
  TF_RETURN_IF_ERROR(add_op->AddInput(inputs[1]));

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(add_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

Status Sub(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr sub_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(sub_op->Reset("Sub", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(sub_op.get(), name));
  TF_RETURN_IF_ERROR(sub_op->AddInput(inputs[0]));
  TF_RETURN_IF_ERROR(sub_op->AddInput(inputs[1]));

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(sub_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

Status MatMul(AbstractContext* ctx,
              absl::Span<AbstractTensorHandle* const> inputs,
              absl::Span<AbstractTensorHandle*> outputs, const char* name,
              bool transpose_a = false, bool transpose_b = false) {
  AbstractOperationPtr matmul_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(matmul_op->Reset("MatMul", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(matmul_op.get(), name));
  TF_RETURN_IF_ERROR(matmul_op->AddInput(inputs[0]));
  TF_RETURN_IF_ERROR(matmul_op->AddInput(inputs[1]));

  TF_RETURN_IF_ERROR(matmul_op->SetAttrBool("transpose_a", transpose_a));
  TF_RETURN_IF_ERROR(matmul_op->SetAttrBool("transpose_b", transpose_b));

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(matmul_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

Status Neg(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr neg_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(neg_op->Reset("Neg", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(neg_op.get(), name));
  TF_RETURN_IF_ERROR(neg_op->AddInput(inputs[0]));

  int num_retvals = 1;
  return neg_op->Execute(outputs, &num_retvals);
}

Status Sum(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr sum_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(sum_op->Reset("Sum", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(sum_op.get(), name));
  TF_RETURN_IF_ERROR(sum_op->AddInput(inputs[0]));  // input_vals
  TF_RETURN_IF_ERROR(sum_op->AddInput(inputs[1]));  // reduction_indices

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(sum_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

Status Div(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr div_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(div_op->Reset("Div", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(div_op.get(), name));
  TF_RETURN_IF_ERROR(div_op->AddInput(inputs[0]));  // x
  TF_RETURN_IF_ERROR(div_op->AddInput(inputs[1]));  // y

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(div_op->Execute(outputs, &num_retvals));  // z = x / y
  return Status::OK();
}

Status DivNoNan(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr div_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(div_op->Reset("DivNoNan", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(div_op.get(), name));
  TF_RETURN_IF_ERROR(div_op->AddInput(inputs[0]));  // x
  TF_RETURN_IF_ERROR(div_op->AddInput(inputs[1]));  // y

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(div_op->Execute(
      outputs, &num_retvals));  // z = x / y, (z_i = 0 if y_i = 0)
  return Status::OK();
}

Status Exp(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr exp_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(exp_op->Reset("Exp", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(exp_op.get(), name));
  TF_RETURN_IF_ERROR(exp_op->AddInput(inputs[0]));

  int num_retvals = 1;
  return exp_op->Execute(outputs, &num_retvals);
}

Status Sqrt(AbstractContext* ctx,
            absl::Span<AbstractTensorHandle* const> inputs,
            absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr sqrt_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(sqrt_op->Reset("Sqrt", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(sqrt_op.get(), name));
  TF_RETURN_IF_ERROR(sqrt_op->AddInput(inputs[0]));

  int num_retvals = 1;
  Status s = sqrt_op->Execute(outputs, &num_retvals);
  return s;
}

Status SqrtGrad(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr sqrt_grad_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      sqrt_grad_op->Reset("SqrtGrad", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(sqrt_grad_op.get(), name));
  TF_RETURN_IF_ERROR(sqrt_grad_op->AddInput(inputs[0]));
  TF_RETURN_IF_ERROR(sqrt_grad_op->AddInput(inputs[1]));

  int num_retvals = 1;
  Status s = sqrt_grad_op->Execute(outputs, &num_retvals);
  return s;
}

Status Log1p(AbstractContext* ctx,
             absl::Span<AbstractTensorHandle* const> inputs,
             absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr log1p_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(log1p_op->Reset("Log1p", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(log1p_op.get(), name));
  TF_RETURN_IF_ERROR(log1p_op->AddInput(inputs[0]));

  int num_retvals = 1;
  Status s = log1p_op->Execute(outputs, &num_retvals);
  return s;
}

}  // namespace ops
}  // namespace tensorflow
