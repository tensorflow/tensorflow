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

// ============== Ops used for Gradient Computation =============================

// Creates an Identity op.
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

Status MatMul(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, const char* name,
                bool transpose_a, bool transpose_b) {
  
  AbstractOperationPtr matmul_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      matmul_op->Reset("MatMul", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(matmul_op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(matmul_op.get())
                           ->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(matmul_op->AddInput(inputs[0]));
  TF_RETURN_IF_ERROR(matmul_op->AddInput(inputs[1]));

  TF_RETURN_IF_ERROR(matmul_op->SetAttrBool("transpose_a", transpose_a));
  TF_RETURN_IF_ERROR(matmul_op->SetAttrBool("transpose_b", transpose_b));
  
  int num_retvals = 1;
  TF_RETURN_IF_ERROR(matmul_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

// Softmax Loss given scores and labels, used by the SoftMaxLossGradient
Status SparseSoftmaxCrossEntropyLoss(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  
  AbstractOperationPtr sm_loss_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      sm_loss_op->Reset("SparseSoftmaxCrossEntropyWithLogits", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(sm_loss_op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(sm_loss_op.get())
                           ->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(sm_loss_op->AddInput(inputs[0])); // input scores
  TF_RETURN_IF_ERROR(sm_loss_op->AddInput(inputs[1])); // labels

  // Outputs will contain: [loss_vals, gradients]. 
  int num_retvals = 2;
  TF_RETURN_IF_ERROR(sm_loss_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

// Computes Relu gradient given input features
Status ReluGrad(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, 
                const char* name) {
  
  AbstractOperationPtr relugrad_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      relugrad_op->Reset("ReluGrad", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(relugrad_op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(relugrad_op.get())
                           ->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(relugrad_op->AddInput(inputs[0])); //upstream grads
  TF_RETURN_IF_ERROR(relugrad_op->AddInput(inputs[1])); //relu inputs

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(relugrad_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

// ======================= Operations for Tape =====================

// Computes `inputs[0] + inputs[1]` and records it on the tape.
// Status Add(AbstractContext* ctx, Tape* tape,
//            absl::Span<AbstractTensorHandle* const> inputs,
//            absl::Span<AbstractTensorHandle*> outputs,
//            const GradientRegistry& registry) {
  
//   AbstractOperationPtr add_op(ctx->CreateOperation());
//   ForwardOperation forward_op;
//   forward_op.ctx = ctx;
//   TF_RETURN_IF_ERROR(
//       Reset(add_op.get(), "Add", /*raw_device_name=*/nullptr, &forward_op));
//   if (isa<tracing::TracingOperation>(add_op.get())) {
//     TF_RETURN_IF_ERROR(
//         dyn_cast<tracing::TracingOperation>(add_op.get())->SetOpName("my_add"));
//   }
//   TF_RETURN_IF_ERROR(AddInput(add_op.get(), inputs[0], &forward_op));
//   TF_RETURN_IF_ERROR(AddInput(add_op.get(), inputs[1], &forward_op));
//   int num_retvals = 1;
//   return Execute(add_op.get(), ctx, outputs, &num_retvals, &forward_op, tape,
//                  registry);
// }

// // Computes `inputs[0] * inputs[1]` for matrices and records it on the tape.
// Status MatMul(AbstractContext* ctx, Tape* tape,
//            absl::Span<AbstractTensorHandle* const> inputs,
//            absl::Span<AbstractTensorHandle*> outputs, const char* name,
//            bool transpose_a, bool transpose_b,
//            const GradientRegistry& registry) {
  
//   AbstractOperationPtr matmul_op(ctx->CreateOperation());
//   ForwardOperation forward_op;
//   forward_op.ctx = ctx;
//   TF_RETURN_IF_ERROR(
//       Reset(matmul_op.get(), "MatMul", /*raw_device_name=*/nullptr, &forward_op));
//   if (isa<tracing::TracingOperation>(matmul_op.get())) {
//     TF_RETURN_IF_ERROR(
//         dyn_cast<tracing::TracingOperation>(matmul_op.get())->SetOpName(name));
//   }

//   TF_RETURN_IF_ERROR(AddInput(matmul_op.get(), inputs[0], &forward_op));
//   TF_RETURN_IF_ERROR(AddInput(matmul_op.get(), inputs[1], &forward_op));
//   matmul_op->SetAttrBool("transpose_a",transpose_a);
//   matmul_op->SetAttrBool("transpose_b",transpose_b);

//   int num_retvals = 1;
//   return Execute(matmul_op.get(), ctx, outputs, &num_retvals, &forward_op, tape,
//                  registry);
// }

// // Computes `Relu(inputs[0])` and records it on the tape.
// Status Relu(AbstractContext* ctx, Tape* tape,
//            absl::Span<AbstractTensorHandle* const> inputs,
//            absl::Span<AbstractTensorHandle*> outputs, const char* name,
//            const GradientRegistry& registry) {
  
//   AbstractOperationPtr relu_op(ctx->CreateOperation());
//   ForwardOperation forward_op;
//   forward_op.ctx = ctx;
//   TF_RETURN_IF_ERROR(
//       Reset(relu_op.get(), "Relu", /*raw_device_name=*/nullptr, &forward_op));
//   if (isa<tracing::TracingOperation>(relu_op.get())) {
//     TF_RETURN_IF_ERROR(
//         dyn_cast<tracing::TracingOperation>(relu_op.get())->SetOpName(name));
//   }
//   TF_RETURN_IF_ERROR(AddInput(relu_op.get(), inputs[0], &forward_op));
//   int num_retvals = 1;
//   return Execute(relu_op.get(), ctx, outputs, &num_retvals, &forward_op, tape,
//                  registry);
// }

// // Computes `SoftmaxLoss(scores, labels)` for matrices and records it on the tape.
// Status SparseSoftmaxCrossEntropyLoss(AbstractContext* ctx, Tape* tape,
//            absl::Span<AbstractTensorHandle* const> inputs,
//            absl::Span<AbstractTensorHandle*> outputs, const char* name,
//            const GradientRegistry& registry) {
  
//   AbstractTensorHandle* scores = inputs[0];
//   AbstractTensorHandle* labels = inputs[1];

//   AbstractOperationPtr sm_op(ctx->CreateOperation());
//   ForwardOperation forward_op;
//   forward_op.ctx = ctx;
//   TF_RETURN_IF_ERROR(
//       Reset(sm_op.get(), "SparseSoftmaxCrossEntropyWithLogits", /*raw_device_name=*/nullptr, &forward_op));
//   if (isa<tracing::TracingOperation>(sm_op.get())) {
//     TF_RETURN_IF_ERROR(
//         dyn_cast<tracing::TracingOperation>(sm_op.get())->SetOpName(name));
//   }

//   TF_RETURN_IF_ERROR(AddInput(sm_op.get(), scores, &forward_op));
//   TF_RETURN_IF_ERROR(AddInput(sm_op.get(), labels, &forward_op));

//   int num_retvals = 2; // returns loss values and backprop
//   return Execute(sm_op.get(), ctx, outputs, &num_retvals, &forward_op, tape,
//                  registry);
// }



}  // namespace ops
}  // namespace tensorflow
