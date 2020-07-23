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
#include "tensorflow/c/eager/mnist_gradients_util.h"
#include "tensorflow/c/eager/gradients.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"

// Computes `inputs[0] + inputs[1]` and records it on the tape.
Status Add(AbstractContext* ctx, Tape* tape,
           absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs,
           const GradientRegistry& registry) {
  
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

//===================== Test Models to run =========================

// Computes
// y = inputs[0] + inputs[1]
// return grad(y, {inputs[0], inputs[1]})
Status AddGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
  
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch x.
  tape->Watch(ToId(inputs[1]));  // Watch y.
  std::vector<AbstractTensorHandle*> add_outputs(1);
  TF_RETURN_IF_ERROR(Add(ctx, tape, inputs, absl::MakeSpan(add_outputs),
                         registry));  // Compute x+y.
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;

  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(add_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads));
  for (auto add_output : add_outputs) {
    add_output->Release();
  }
  outputs[0] = out_grads[0];
  outputs[1] = out_grads[1];
  delete tape;
  return Status::OK();
}

// Computes
// y = inputs[0] * inputs[1]
// return grad(y, {inputs[0], inputs[1]})
Status MatMulGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
  
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch x.
  tape->Watch(ToId(inputs[1]));  // Watch y.
  std::vector<AbstractTensorHandle*> mm_outputs(1);
  TF_RETURN_IF_ERROR(MatMul(ctx, tape, inputs, absl::MakeSpan(mm_outputs), 
      "matmul0", /*transpose_a=*/false, /*transpose_b=*/false, registry));  // Compute x*y.
  
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;

  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(mm_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads));
  for (auto mm_output : mm_outputs) {
    mm_output->Release();
  }
  outputs[0] = out_grads[0];
  outputs[1] = out_grads[1];
  delete tape;
  return Status::OK();
}

// Model to run 2-layer net
Status MNISTForwardModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
  /**
    * We will trace a 2-layer fully connected network for an MNIST model:
    *
    *   def mnist_forward(X, W1, W2, y_labels):
    *     mm_out_1 = tf.matmul(X,W1)
    *     hidden_layer = tf.ReLu(mm_out_1)
    *     scores = tf.matmul(hidden_layer,W2)
    *     softmax = tf.softmaxLoss(scores,y_labels)
    *     return scores, softmax 
    *
    * Use this convention for inputs:
    * 
    *   inputs = [X, W1, W2, y_labels]
    *
    */ 
  AbstractTensorHandle* X = inputs[0];
  AbstractTensorHandle* W1 = inputs[1];
  AbstractTensorHandle* W2 = inputs[2];
  AbstractTensorHandle* y_labels = inputs[3];

  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(W1));  // Watch W1.
  tape->Watch(ToId(W2));  // Watch W2.
  std::vector<AbstractTensorHandle*> temp_outputs(1);

  TF_RETURN_IF_ERROR(MatMul(ctx, tape, {X, W1}, absl::MakeSpan(temp_outputs),
                     "matmul0",/*transpose_a=*/false,/*transpose_b=*/false, registry));  // Compute X*W1

  TF_RETURN_IF_ERROR(Relu(ctx, tape, {temp_outputs[0]}, absl::MakeSpan(temp_outputs),
                     "relu", registry)); // Compute Relu(X*W1)

  TF_RETURN_IF_ERROR(MatMul(ctx, tape, {temp_outputs[0], W2}, absl::MakeSpan(temp_outputs),
                     "matmul1",/*transpose_a=*/false,/*transpose_b=*/false, registry));  // Compute W2*Relu(X*W1)
  
  AbstractTensorHandle* scores = temp_outputs[0];

  TF_RETURN_IF_ERROR(SparseSoftmaxCrossEntropyLoss(ctx, tape, {scores, y_labels}, absl::MakeSpan(temp_outputs),
                     "softmax_loss", registry));  // Compute Softmax(Scores,labels)
  
  AbstractTensorHandle* loss_vals  = temp_outputs[0];
  
  outputs[0] = scores;
  outputs[1] = loss_vals;
  delete tape;
  return Status::OK();
}

Status MatMulTransposeModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
  
  AbstractTensorHandle* X = inputs[0];
  AbstractTensorHandle* W1 = inputs[1];
 
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(X));
  tape->Watch(ToId(W1));
  std::vector<AbstractTensorHandle*> temp_outputs(1);

  TF_RETURN_IF_ERROR(MatMul(ctx, tape, {X, W1}, absl::MakeSpan(temp_outputs),
                     "matmul0",/*transpose_a=*/true,/*transpose_b=*/false, registry));  // Compute X*W1

  outputs[0] =  temp_outputs[0];

  delete tape;
  return Status::OK();
}


Status ReluGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
 
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch X
  std::vector<AbstractTensorHandle*> relu_outputs(1);
  TF_RETURN_IF_ERROR(Relu(ctx, tape, inputs, absl::MakeSpan(relu_outputs), 
      "relu0", registry));  // Relu(X)
  
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;

  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(relu_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads));
  
  for (auto relu_output : relu_outputs) {
    relu_output->Release();
  }

  outputs[0] = out_grads[0];
  delete tape;
  return Status::OK();
}

Status SoftmaxLossGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
 
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch scores.
  tape->Watch(ToId(inputs[1]));  // Watch labels.
  std::vector<AbstractTensorHandle*> sm_outputs(2);
  TF_RETURN_IF_ERROR(SparseSoftmaxCrossEntropyLoss(ctx, tape, inputs,
                    absl::MakeSpan(sm_outputs), "softmax0", registry));  // Compute x*y.
  
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;

  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(sm_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads));
  
  for (auto sm_output : sm_outputs) {
    sm_output->Release();
  }
  outputs[0] = out_grads[0];
  outputs[1] = out_grads[1];
  delete tape;
  return Status::OK();

}

Status MNISTGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
  
  AbstractTensorHandle* X = inputs[0];
  AbstractTensorHandle* W1 = inputs[1];
  AbstractTensorHandle* W2 = inputs[2];
  AbstractTensorHandle* y_labels = inputs[3];                    

  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/true);
  tape->Watch(ToId(X));  // Watch X.
  tape->Watch(ToId(W1));  // Watch W1.
  tape->Watch(ToId(W2));  // Watch W1.
  std::vector<AbstractTensorHandle*> temp_outputs(1);
  TF_RETURN_IF_ERROR(MatMul(ctx, tape, {X, W1}, absl::MakeSpan(temp_outputs), 
      "matmul0", /*transpose_a=*/false, /*transpose_b=*/false, registry));  // Compute X*W1
  
  AbstractTensorHandle* mm = temp_outputs[0];

  TF_RETURN_IF_ERROR(Relu(ctx, tape, {mm}, absl::MakeSpan(temp_outputs),  // Relu(X*W1)
      "relu0", registry)); 

  AbstractTensorHandle* hidden = temp_outputs[0];
  
  TF_RETURN_IF_ERROR(MatMul(ctx, tape, {hidden, W2}, absl::MakeSpan(temp_outputs), 
      "matmul1", /*transpose_a=*/false, /*transpose_b=*/false, registry));  // W2*Relu(X*W1)
  
  AbstractTensorHandle* scores = temp_outputs[0];
  
  temp_outputs.resize(2);
  TF_RETURN_IF_ERROR(SparseSoftmaxCrossEntropyLoss(ctx, tape, {scores, y_labels}, absl::MakeSpan(temp_outputs), 
      "softmaxloss", registry));  // W2*Relu(X*W1)

  AbstractTensorHandle* loss = temp_outputs[0];

  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;
  
  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(loss)},
      /*source_tensor_ids=*/{ToId(W1), ToId(W2)},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads));
  
  for (auto temp_output : temp_outputs) {
    temp_output->Release();
  }

  outputs[0] = out_grads[0];  // dW1
  outputs[1] = out_grads[1];  // dW2
  delete tape;
  return Status::OK();
}

// ============================= End Models ================================

AbstractContext* BuildFunction(const char* fn_name) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_ExecutionContext* graph_ctx = TF_CreateFunction(fn_name, status.get());
  return unwrap(graph_ctx);
}

Status CreateParamsForInputs(AbstractContext* ctx,
                             absl::Span<AbstractTensorHandle* const> inputs,
                             std::vector<AbstractTensorHandle*>* params) {
  
  tracing::TracingTensorHandle* handle = nullptr;
  for (auto input : inputs) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingContext>(ctx)->AddParameter(
        input->DataType(), &handle));
    params->emplace_back(handle);
  }
  return Status::OK();
}

// Runs `model` maybe wrapped in a function.
Status RunModel(Model model, AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, bool use_function,
                const GradientRegistry& registry) {
  
  if (use_function) {
    const char* fn_name = "test_fn";
    std::unique_ptr<AbstractFunction> scoped_func;
    {
      AbstractContextPtr func_ctx(BuildFunction(fn_name));
      std::vector<AbstractTensorHandle*> func_inputs;
      func_inputs.reserve(inputs.size());
      TF_RETURN_IF_ERROR(
          CreateParamsForInputs(func_ctx.get(), inputs, &func_inputs));
      OutputList output_list;
      output_list.expected_num_outputs = outputs.size();
      output_list.outputs.resize(outputs.size());
      TF_RETURN_IF_ERROR(model(func_ctx.get(), absl::MakeSpan(func_inputs),
                               absl::MakeSpan(output_list.outputs), registry));
      
      for (auto func_input : func_inputs) {
        func_input->Release();
      }
      AbstractFunction* func = nullptr;
      TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingContext>(func_ctx.get())
                             ->Finalize(&output_list, &func));
      scoped_func.reset(func);

      for (auto output : output_list.outputs) {
        output->Release();
      }
      
      TF_RETURN_IF_ERROR(ctx->RegisterFunction(func));
    }

    AbstractOperationPtr fn_op(ctx->CreateOperation());
    TF_RETURN_IF_ERROR(fn_op->Reset(fn_name, /*raw_device_name=*/nullptr));
    for (auto input : inputs) {
      TF_RETURN_IF_ERROR(fn_op->AddInput(input));
    }
    int retvals = outputs.size();
    TF_RETURN_IF_ERROR(fn_op->Execute(outputs, &retvals));
    TF_RETURN_IF_ERROR(ctx->RemoveFunction(fn_name));
    return Status::OK();
  } else {
    return model(ctx, inputs, outputs, registry);
  }
}

Status BuildImmediateExecutionContext(bool use_tfrt, AbstractContext** ctx) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetTfrt(opts, use_tfrt);
  *ctx = unwrap(TF_NewEagerExecutionContext(opts, status.get()));
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_DeleteContextOptions(opts);
  return Status::OK();
}


// }  // namespace
// }  // namespace internal
// }  // namespace gradients
// }  // namespace tensorflow
