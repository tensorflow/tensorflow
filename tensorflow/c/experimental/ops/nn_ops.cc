
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
#include "tensorflow/c/experimental/ops/nn_ops.h"

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace ops {

// Softmax Loss given scores and labels, used by the SoftMaxLossGradient
Status SparseSoftmaxCrossEntropyLoss(
    AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
    absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr sm_loss_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(sm_loss_op->Reset("SparseSoftmaxCrossEntropyWithLogits",
                                       /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(sm_loss_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(sm_loss_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(sm_loss_op->AddInput(inputs[0]));  // input scores
  TF_RETURN_IF_ERROR(sm_loss_op->AddInput(inputs[1]));  // labels

  // Outputs will contain: [loss_vals, gradients].
  int num_retvals = 2;
  TF_RETURN_IF_ERROR(sm_loss_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

// Computes Relu gradient given input features
Status ReluGrad(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, const char* name) {
  AbstractOperationPtr relugrad_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      relugrad_op->Reset("ReluGrad", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(relugrad_op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(relugrad_op.get())
                           ->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(relugrad_op->AddInput(inputs[0]));  // upstream grads
  TF_RETURN_IF_ERROR(relugrad_op->AddInput(inputs[1]));  // relu inputs

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(relugrad_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

// Computes Bias Add gradient given upstream grads
Status BiasAddGrad(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs,
                   const char* data_format, const char* name) {
  AbstractOperationPtr bias_add_grad_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      bias_add_grad_op->Reset("BiasAddGrad", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(bias_add_grad_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(bias_add_grad_op.get())
            ->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(bias_add_grad_op->SetAttrString("data_format", data_format,
                                                     strlen(data_format)));
  TF_RETURN_IF_ERROR(bias_add_grad_op->AddInput(inputs[0]));  // upstream grads

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(bias_add_grad_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

// Computes Batchnorm gradients w.r.t to input x, scale, and offset
Status FusedBatchNormV3(AbstractContext* ctx,
                        absl::Span<AbstractTensorHandle* const> inputs,
                        absl::Span<AbstractTensorHandle*> outputs,
                        const char* name) {
  AbstractOperationPtr fbn_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      fbn_op->Reset("FusedBatchNormV3", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(fbn_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(fbn_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(fbn_op->SetAttrBool("is_training", true));

  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[0]));  // x = input to BN
  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[1]));  // scale
  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[2]));  // offset
  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[3]));  // means (optional)
  TF_RETURN_IF_ERROR(fbn_op->AddInput(
      inputs[4]));  // vars (optional)
                    // Returns [y, batch_mean, batch_var, reserve_space_1,
                    // reserve_space_2, reserve_space_3]
  int num_retvals = 6;
  TF_RETURN_IF_ERROR(fbn_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

// Computes Batchnorm gradients w.r.t to input x, scale, and offset
Status FusedBatchNormGradV3(AbstractContext* ctx,
                            absl::Span<AbstractTensorHandle* const> inputs,
                            absl::Span<AbstractTensorHandle*> outputs,
                            const char* name) {
  AbstractOperationPtr fbn_grad_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      fbn_grad_op->Reset("FusedBatchNormGradV3", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(fbn_grad_op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(fbn_grad_op.get())
                           ->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(fbn_grad_op->SetAttrBool("is_training", true));

  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[0]));  // upstream grads
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[1]));  // x = input to BN
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[2]));  // scale input to BN
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[3]));  // reserve_space_1
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[4]));  // reserve_space_2
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[5]));  // reserve_space_3

  // Returns [x_grad, scale_grad, offset_grad, reserve_space_4, reserve_space_5]
  // Last 2 outputs not used.

  int num_retvals = 5;
  TF_RETURN_IF_ERROR(fbn_grad_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

// Computes Batchnorm gradients w.r.t to input x, scale, and offset
Status FusedBatchNormV2(AbstractContext* ctx,
                        absl::Span<AbstractTensorHandle* const> inputs,
                        absl::Span<AbstractTensorHandle*> outputs,
                        const char* name) {
  AbstractOperationPtr fbn_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      fbn_op->Reset("FusedBatchNormV2", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(fbn_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(fbn_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(fbn_op->SetAttrBool("is_training", true));

  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[0]));  // x = input to BN
  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[1]));  // scale
  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[2]));  // offset
  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[3]));  // means (optional)
  TF_RETURN_IF_ERROR(
      fbn_op->AddInput(inputs[4]));  // vars (optional)
                                     // Returns [y, batch_mean, batch_var,
                                     // reserve_space_1, reserve_space_2]
  int num_retvals = 6;
  TF_RETURN_IF_ERROR(fbn_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

Status FusedBatchNormGradV2(AbstractContext* ctx,
                            absl::Span<AbstractTensorHandle* const> inputs,
                            absl::Span<AbstractTensorHandle*> outputs,
                            const char* name) {
  AbstractOperationPtr fbn_grad_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      fbn_grad_op->Reset("FusedBatchNormGradV2", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(fbn_grad_op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(fbn_grad_op.get())
                           ->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[0]));  // upstream grads
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[1]));  // x = input to BN
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[2]));  // scale input to BN
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[3]));  // reserve_space_1
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[4]));  // reserve_space_2
  // Returns [x_grad, scale_grad, offset_grad, reserve_space_3, reserve_space_4]
  // Last 2 outputs not used.

  int num_retvals = 5;
  TF_RETURN_IF_ERROR(fbn_grad_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

// Computes Batchnorm gradients w.r.t to input x, scale, and offset
Status FusedBatchNorm(AbstractContext* ctx,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs,
                      const char* name) {
  AbstractOperationPtr fbn_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      fbn_op->Reset("FusedBatchNorm", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(fbn_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(fbn_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(fbn_op->SetAttrBool("is_training", true));

  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[0]));  // x = input to BN
  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[1]));  // scale
  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[2]));  // offset
  TF_RETURN_IF_ERROR(fbn_op->AddInput(inputs[3]));  // means (optional)
  TF_RETURN_IF_ERROR(
      fbn_op->AddInput(inputs[4]));  // vars (optional)
                                     // Returns [y, batch_mean, batch_var,
                                     // reserve_space_1, reserve_space_2]
  int num_retvals = 5;
  TF_RETURN_IF_ERROR(fbn_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

Status FusedBatchNormGrad(AbstractContext* ctx,
                          absl::Span<AbstractTensorHandle* const> inputs,
                          absl::Span<AbstractTensorHandle*> outputs,
                          const char* name) {
  AbstractOperationPtr fbn_grad_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      fbn_grad_op->Reset("FusedBatchNormGrad", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(fbn_grad_op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(fbn_grad_op.get())
                           ->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[0]));  // upstream grads
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[1]));  // x = input to BN
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[2]));  // scale input to BN
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[3]));  // reserve_space_1
  TF_RETURN_IF_ERROR(fbn_grad_op->AddInput(inputs[4]));  // reserve_space_2
  // Returns [x_grad, scale_grad, offset_grad, reserve_space_3, reserve_space_4]
  // Last 2 outputs not used.

  int num_retvals = 5;
  TF_RETURN_IF_ERROR(fbn_grad_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

Status Conv2D(AbstractContext* ctx,
              absl::Span<AbstractTensorHandle* const> inputs,
              absl::Span<AbstractTensorHandle*> outputs, int64_t* strides,
              int num_dims, const char* padding, const char* name) {
  AbstractOperationPtr conv_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(conv_op->Reset("Conv2D", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(conv_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(conv_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(conv_op->SetAttrIntList("strides", strides, num_dims));
  TF_RETURN_IF_ERROR(
      conv_op->SetAttrString("padding", padding, strlen(padding)));

  TF_RETURN_IF_ERROR(conv_op->AddInput(inputs[0]));  // input
  TF_RETURN_IF_ERROR(conv_op->AddInput(inputs[1]));  // filter

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(conv_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

Status Conv2DBackpropInput(AbstractContext* ctx,
                           absl::Span<AbstractTensorHandle* const> inputs,
                           absl::Span<AbstractTensorHandle*> outputs,
                           int64_t* strides, int num_dims, const char* padding,
                           const char* name) {
  AbstractOperationPtr conv_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      conv_op->Reset("Conv2DBackpropInput", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(conv_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(conv_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(conv_op->SetAttrIntList("strides", strides, num_dims));
  TF_RETURN_IF_ERROR(
      conv_op->SetAttrString("padding", padding, strlen(padding)));

  TF_RETURN_IF_ERROR(conv_op->AddInput(inputs[0]));  // input sizes
  TF_RETURN_IF_ERROR(conv_op->AddInput(inputs[1]));  // filter
  TF_RETURN_IF_ERROR(conv_op->AddInput(inputs[2]));  // upstream_grad

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(conv_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

Status Conv2DBackpropFilter(AbstractContext* ctx,
                            absl::Span<AbstractTensorHandle* const> inputs,
                            absl::Span<AbstractTensorHandle*> outputs,
                            int64_t* strides, int num_dims, const char* padding,
                            const char* name) {
  AbstractOperationPtr conv_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      conv_op->Reset("Conv2DBackpropFilter", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(conv_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(conv_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(conv_op->SetAttrIntList("strides", strides, num_dims));
  TF_RETURN_IF_ERROR(
      conv_op->SetAttrString("padding", padding, strlen(padding)));

  TF_RETURN_IF_ERROR(conv_op->AddInput(inputs[0]));  // input
  TF_RETURN_IF_ERROR(conv_op->AddInput(inputs[1]));  // filter sizes
  TF_RETURN_IF_ERROR(conv_op->AddInput(inputs[2]));  // upstream_grad

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(conv_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

Status MaxPoolGrad(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs, int num_dims,
                   int64_t* ksize, int64_t* strides, const char* padding,
                   const char* data_format, const char* name) {
  AbstractOperationPtr mp_grad_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      mp_grad_op->Reset("MaxPoolGrad", /*raw_device_name=*/nullptr));

  if (isa<tracing::TracingOperation>(mp_grad_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tracing::TracingOperation>(mp_grad_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(mp_grad_op->SetAttrIntList("ksize", ksize, num_dims));
  TF_RETURN_IF_ERROR(mp_grad_op->SetAttrIntList("strides", strides, num_dims));
  TF_RETURN_IF_ERROR(
      mp_grad_op->SetAttrString("padding", padding, strlen(padding)));
  TF_RETURN_IF_ERROR(mp_grad_op->SetAttrString("data_format", data_format,
                                               strlen(data_format)));

  TF_RETURN_IF_ERROR(mp_grad_op->AddInput(inputs[0]));  // orig_input
  TF_RETURN_IF_ERROR(mp_grad_op->AddInput(inputs[1]));  // orig_output
  TF_RETURN_IF_ERROR(mp_grad_op->AddInput(inputs[2]));  // upstream_grad

  int num_retvals = 1;
  TF_RETURN_IF_ERROR(mp_grad_op->Execute(outputs, &num_retvals));
  return Status::OK();
}

}  // namespace ops
}  // namespace tensorflow
