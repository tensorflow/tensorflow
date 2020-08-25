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
#include "tensorflow/c/eager/resnet_gradients_testutil.h"
#include "tensorflow/c/eager/mnist_gradients_testutil.h"

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"

using std::vector;
using tracing::TracingOperation;

// ========================== Tape Ops ==============================

// Computes Sum = A + bias
Status BiasAdd(AbstractContext* ctx, Tape* tape,
   absl::Span<AbstractTensorHandle* const> inputs,
   absl::Span<AbstractTensorHandle*> outputs, const char* name,
   const GradientRegistry& registry) {
       
 AbstractTensorHandle* value = inputs[0];
 AbstractTensorHandle* bias = inputs[1];
 
 AbstractOperationPtr biasAdd_op(ctx->CreateOperation());
 ForwardOperation forward_op;
 forward_op.ctx = ctx;
 TF_RETURN_IF_ERROR(Reset(biasAdd_op.get(), "BiasAdd",
                          /*raw_device_name=*/nullptr, &forward_op));
 if (isa<TracingOperation>(biasAdd_op.get())) {
   TF_RETURN_IF_ERROR(
       dyn_cast<tracing::TracingOperation>(biasAdd_op.get())->SetOpName(name));
 }
 
 TF_RETURN_IF_ERROR(AddInput(biasAdd_op.get(), value, &forward_op));
 TF_RETURN_IF_ERROR(AddInput(biasAdd_op.get(), bias, &forward_op));
 
 int num_retvals = 1;
 return Execute(biasAdd_op.get(), ctx, outputs, &num_retvals, &forward_op, tape,
                registry);
}

Status Conv2D(AbstractContext* ctx, Tape* tape,
   absl::Span<AbstractTensorHandle* const> inputs,
   absl::Span<AbstractTensorHandle*> outputs,
   int64_t* strides, int num_dims, const char* padding,
   const char* name, const GradientRegistry& registry) {
 
 AbstractTensorHandle* x = inputs[0];
 AbstractTensorHandle* filter = inputs[1];
 
 AbstractOperationPtr conv_op(ctx->CreateOperation());
 ForwardOperation forward_op;
 forward_op.ctx = ctx;
 TF_RETURN_IF_ERROR(Reset(conv_op.get(), "Conv2D",
                          /*raw_device_name=*/nullptr, &forward_op));
 if (isa<TracingOperation>(conv_op.get())) {
   TF_RETURN_IF_ERROR(
       dyn_cast<tracing::TracingOperation>(conv_op.get())->SetOpName(name));
 }
 
// Status SetAttrString(AbstractOperation* op_, const char* attr_name,
//                      const char* data, size_t length,
//                      ForwardOperation* forward_op_)

 bool gpu = true;
 TF_RETURN_IF_ERROR(tensorflow::gradients::internal::SetAttrIntList(
      conv_op.get(), "strides", strides, num_dims, &forward_op));
 TF_RETURN_IF_ERROR(tensorflow::gradients::internal::SetAttrString(
      conv_op.get(), "padding", padding, strlen(padding), &forward_op));
 TF_RETURN_IF_ERROR(tensorflow::gradients::internal::SetAttrBool(
      conv_op.get(), "use_cudnn_on_gpu", gpu, &forward_op));


 TF_RETURN_IF_ERROR(AddInput(conv_op.get(), x, &forward_op));
 TF_RETURN_IF_ERROR(AddInput(conv_op.get(), filter, &forward_op));
 
 int num_retvals = 1;
 Status s = Execute(conv_op.get(), ctx, outputs, &num_retvals, &forward_op, tape,
                    registry);
 return s;
}

Status Log1p(AbstractContext* ctx, Tape* tape,
           absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name,
           const GradientRegistry& registry) {
  AbstractOperationPtr log_op(ctx->CreateOperation());
  ForwardOperation forward_op;
  forward_op.ctx = ctx;
  TF_RETURN_IF_ERROR(
      Reset(log_op.get(), "Log1p", /*raw_device_name=*/nullptr, &forward_op));
  if (isa<TracingOperation>(log_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<TracingOperation>(log_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(AddInput(log_op.get(), inputs[0], &forward_op));

  int num_retvals = 1;
  return Execute(log_op.get(), ctx, outputs, &num_retvals, &forward_op, tape,
                 registry);
}

Status Sub(AbstractContext* ctx, Tape* tape,
           absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name,
           const GradientRegistry& registry) {
  AbstractOperationPtr sub_op(ctx->CreateOperation());
  ForwardOperation forward_op;
  forward_op.ctx = ctx;
  TF_RETURN_IF_ERROR(
      Reset(sub_op.get(), "Sub", /*raw_device_name=*/nullptr, &forward_op));
  if (isa<TracingOperation>(sub_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<TracingOperation>(sub_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(AddInput(sub_op.get(), inputs[0], &forward_op));
  TF_RETURN_IF_ERROR(AddInput(sub_op.get(), inputs[1], &forward_op));

  int num_retvals = 1;
  return Execute(sub_op.get(), ctx, outputs, &num_retvals, &forward_op, tape,
                 registry);
}

Status Neg(AbstractContext* ctx, Tape* tape,
           absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name,
           const GradientRegistry& registry) {
  AbstractOperationPtr neg_op(ctx->CreateOperation());
  ForwardOperation forward_op;
  forward_op.ctx = ctx;
  TF_RETURN_IF_ERROR(
      Reset(neg_op.get(), "Neg", /*raw_device_name=*/nullptr, &forward_op));
  if (isa<TracingOperation>(neg_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<TracingOperation>(neg_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(AddInput(neg_op.get(), inputs[0], &forward_op));

  int num_retvals = 1;
  return Execute(neg_op.get(), ctx, outputs, &num_retvals, &forward_op, tape,
                 registry);
}

Status DivNoNan(AbstractContext* ctx, Tape* tape,
           absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name,
           const GradientRegistry& registry) {
  AbstractOperationPtr div_op(ctx->CreateOperation());
  ForwardOperation forward_op;
  forward_op.ctx = ctx;
  TF_RETURN_IF_ERROR(
      Reset(div_op.get(), "DivNoNan", /*raw_device_name=*/nullptr, &forward_op));
  if (isa<TracingOperation>(div_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<TracingOperation>(div_op.get())->SetOpName(name));
  }

  TF_RETURN_IF_ERROR(AddInput(div_op.get(), inputs[0], &forward_op)); // x
  TF_RETURN_IF_ERROR(AddInput(div_op.get(), inputs[1], &forward_op)); // y

  int num_retvals = 1; // z = x / y
  return Execute(div_op.get(), ctx, outputs, &num_retvals, &forward_op, tape,
                 registry);
}

//===================== Test Models to run =========================

Status BiasAddGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
 TapeVSpace vspace(ctx);
 auto tape = new Tape(/*persistent=*/false);
 tape->Watch(ToId(inputs[0]));  // Watch A
 tape->Watch(ToId(inputs[1]));  // Watch bias
 std::vector<AbstractTensorHandle*> ba_outputs(1);
 TF_RETURN_IF_ERROR(BiasAdd(ctx, tape, inputs, absl::MakeSpan(ba_outputs),
                         "bias_add_test0", registry)); 
 
 std::unordered_map<tensorflow::int64, TapeTensor>
     source_tensors_that_are_targets;
 
 std::vector<AbstractTensorHandle*> out_grads;
 TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(ba_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
 
 for (auto ba_output : ba_outputs) {
   ba_output->Unref();
 }
 
 outputs[0] = out_grads[0];
 outputs[1] = out_grads[1];
 delete tape;
 return Status::OK();
}

Status Conv2DGradModel(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs,
                   const GradientRegistry& registry) {
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch input.
  tape->Watch(ToId(inputs[1]));  // Watch filter.
  std::vector<AbstractTensorHandle*> conv_outputs(1);
  int64_t strides[] = {1,1,1,1};
  TF_RETURN_IF_ERROR(Conv2D(ctx, tape, inputs, absl::MakeSpan(conv_outputs),
                      strides, /*num_dims=*/4, /*padding=*/"SAME", "conv_2d_test", registry));
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;
  
  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(conv_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
  
  outputs[0] = conv_outputs[0];
  outputs[1] = out_grads[0];
  outputs[2] = out_grads[1];
  
  delete tape;
  return Status::OK();
}

Status SubGradModel(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs,
                   const GradientRegistry& registry) {
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch A
  tape->Watch(ToId(inputs[1]));  // Watch B
  std::vector<AbstractTensorHandle*> sub_outputs(1);
  
  TF_RETURN_IF_ERROR(Sub(ctx, tape, inputs, absl::MakeSpan(sub_outputs),
                           "sub_test", registry));
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;
  
  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(sub_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
  
  outputs[0] = out_grads[0];
  outputs[1] = out_grads[1];
  
  delete tape;
  return Status::OK();
}

Status MulGradModel(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs,
                   const GradientRegistry& registry) {
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch A
  tape->Watch(ToId(inputs[1]));  // Watch B
  std::vector<AbstractTensorHandle*> mul_outputs(1);
  
  TF_RETURN_IF_ERROR(Mul(ctx, tape, inputs, absl::MakeSpan(mul_outputs),
                           "mul_test", registry));
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;
  
  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(mul_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
  
  outputs[0] = out_grads[0];
  outputs[1] = out_grads[1];
  
  delete tape;
  return Status::OK();
}

Status NegGradModel(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs,
                   const GradientRegistry& registry) {
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch A
  std::vector<AbstractTensorHandle*> neg_outputs(1);
  
  TF_RETURN_IF_ERROR(Neg(ctx, tape, inputs, absl::MakeSpan(neg_outputs),
                           "neg_test", registry));
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;
  
  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(neg_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
  
  outputs[0] = out_grads[0];
  
  delete tape;
  return Status::OK();
}

Status DivGradModel(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs,
                   const GradientRegistry& registry) {
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch X
  tape->Watch(ToId(inputs[1]));  // Watch Y
  std::vector<AbstractTensorHandle*> div_outputs(1);
  
  TF_RETURN_IF_ERROR(DivNoNan(ctx, tape, inputs, absl::MakeSpan(div_outputs),
                           "div_test", registry));
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;
  
  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(div_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
  
  outputs[0] = out_grads[0];
  outputs[1] = out_grads[1];
  
  delete tape;
  return Status::OK();
}

Status Log1pGradModel(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs,
                   const GradientRegistry& registry) {
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch A
  std::vector<AbstractTensorHandle*> log_outputs(1);
  
  TF_RETURN_IF_ERROR(Log1p(ctx, tape, inputs, absl::MakeSpan(log_outputs),
                           "log1p_test", registry));
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;
  
  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(log_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
  
  outputs[0] = out_grads[0];
  delete tape;
  return Status::OK();
}

// ====================== End Models ================================


