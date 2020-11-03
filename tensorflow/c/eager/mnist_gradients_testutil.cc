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
#include "tensorflow/c/eager/gradients_util.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"


namespace tensorflow {
namespace gradients {
namespace internal {

using std::vector;

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
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(
      ops::Add(tape_ctx.get(), inputs, absl::MakeSpan(add_outputs), "Add"));
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;

  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(add_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
  for (auto add_output : add_outputs) {
    add_output->Unref();
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
  vector<AbstractTensorHandle*> mm_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::MatMul(tape_ctx.get(), inputs,
                                 absl::MakeSpan(mm_outputs), "matmul0",
                                 /*transpose_a=*/false,
                                 /*transpose_b=*/false));  // Compute x*y.

  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;

  vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(mm_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
  for (auto mm_output : mm_outputs) {
    mm_output->Unref();
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
   *     hidden_layer = tf.nn.relu(mm_out_1)
   *     scores = tf.matmul(hidden_layer,W2)
   *     softmax =
   *        tf.nn.sparse_softmax_cross_entropy_with_logits(scores,
   *                                                       y_labels)
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
  vector<AbstractTensorHandle*> temp_outputs(1);

  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::MatMul(tape_ctx.get(), {X, W1},
                                 absl::MakeSpan(temp_outputs), "matmul0",
                                 /*transpose_a=*/false,
                                 /*transpose_b=*/false));  // Compute X*W1

  TF_RETURN_IF_ERROR(ops::Relu(tape_ctx.get(), {temp_outputs[0]},
                               absl::MakeSpan(temp_outputs),
                               "relu"));  // Compute Relu(X*W1)

  TF_RETURN_IF_ERROR(ops::MatMul(
      tape_ctx.get(), {temp_outputs[0], W2}, absl::MakeSpan(temp_outputs),
      "matmul1",
      /*transpose_a=*/false, /*transpose_b=*/false));  // Compute W2*Relu(X*W1)

  AbstractTensorHandle* scores = temp_outputs[0];

  temp_outputs.resize(2);
  TF_RETURN_IF_ERROR(ops::SparseSoftmaxCrossEntropyWithLogits(
      tape_ctx.get(), {scores, y_labels}, absl::MakeSpan(temp_outputs),
      "softmax_loss"));  // Compute Softmax(Scores,labels)

  AbstractTensorHandle* loss_vals = temp_outputs[0];

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
  vector<AbstractTensorHandle*> temp_outputs(1);

  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::MatMul(tape_ctx.get(), {X, W1},
                                 absl::MakeSpan(temp_outputs), "matmul0",
                                 /*transpose_a=*/true,
                                 /*transpose_b=*/false));  // Compute X*W1

  outputs[0] = temp_outputs[0];

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
  vector<AbstractTensorHandle*> relu_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::Relu(tape_ctx.get(), inputs,
                               absl::MakeSpan(relu_outputs),
                               "relu0"));  // Relu(X)

  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;

  vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(relu_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0])}, source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));

  for (auto relu_output : relu_outputs) {
    relu_output->Unref();
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
  vector<AbstractTensorHandle*> sm_outputs(2);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::SparseSoftmaxCrossEntropyWithLogits(
      tape_ctx.get(), inputs, absl::MakeSpan(sm_outputs), "softmax0"));

  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;

  vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(sm_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));

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
  tape->Watch(ToId(X));   // Watch X.
  tape->Watch(ToId(W1));  // Watch W1.
  tape->Watch(ToId(W2));  // Watch W1.
  vector<AbstractTensorHandle*> temp_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::MatMul(tape_ctx.get(), {X, W1},
                                 absl::MakeSpan(temp_outputs), "matmul0",
                                 /*transpose_a=*/false,
                                 /*transpose_b=*/false));  // Compute X*W1

  AbstractTensorHandle* mm = temp_outputs[0];

  TF_RETURN_IF_ERROR(ops::Relu(tape_ctx.get(), {mm},
                               absl::MakeSpan(temp_outputs),  // Relu(X*W1)
                               "relu0"));

  AbstractTensorHandle* hidden = temp_outputs[0];

  TF_RETURN_IF_ERROR(ops::MatMul(
      tape_ctx.get(), {hidden, W2}, absl::MakeSpan(temp_outputs), "matmul1",
      /*transpose_a=*/false, /*transpose_b=*/false));  // W2*Relu(X*W1)

  AbstractTensorHandle* scores = temp_outputs[0];

  temp_outputs.resize(2);
  TF_RETURN_IF_ERROR(ops::SparseSoftmaxCrossEntropyWithLogits(
      tape_ctx.get(), {scores, y_labels}, absl::MakeSpan(temp_outputs),
      "softmaxloss"));  // W2*Relu(X*W1)

  AbstractTensorHandle* loss = temp_outputs[0];

  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;

  vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(
      tape->ComputeGradient(vspace, /*target_tensor_ids=*/{ToId(loss)},
                            /*source_tensor_ids=*/{ToId(W1), ToId(W2)},
                            source_tensors_that_are_targets,
                            /*output_gradients=*/{}, &out_grads,
                            /*build_default_zeros_grads=*/false));

  // Only release 2nd temp output as first holds loss values.
  temp_outputs[1]->Unref();

  outputs[0] = out_grads[0];  // dW1
  outputs[1] = out_grads[1];  // dW2
  outputs[2] = loss;

  delete tape;
  return Status::OK();
}

Status ScalarMulModel(AbstractContext* ctx,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs,
                      const GradientRegistry& registry) {
  AbstractTensorHandle* eta = inputs[0];
  AbstractTensorHandle* A = inputs[1];

  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  vector<AbstractTensorHandle*> temp_outputs(1);

  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::Mul(tape_ctx.get(), {eta, A},
                              absl::MakeSpan(temp_outputs),
                              "scalarMul0"));  // Compute eta*A

  outputs[0] = temp_outputs[0];

  delete tape;
  return Status::OK();
}

Status MatMulModel(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs,
                   const GradientRegistry& registry) {
  AbstractTensorHandle* X = inputs[0];
  AbstractTensorHandle* W1 = inputs[1];

  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  std::vector<AbstractTensorHandle*> temp_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::MatMul(tape_ctx.get(), {X, W1},
                                 absl::MakeSpan(temp_outputs), "matmul0",
                                 /*transpose_a=*/false,
                                 /*transpose_b=*/false));  // Compute X*W1

  outputs[0] = temp_outputs[0];
  delete tape;
  return Status::OK();
}

Status MulModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs,
                const GradientRegistry& registry) {
  AbstractTensorHandle* x = inputs[0];
  AbstractTensorHandle* y = inputs[1];

  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  std::vector<AbstractTensorHandle*> temp_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::Mul(tape_ctx.get(), {x, y},
                              absl::MakeSpan(temp_outputs),
                              "mul0"));  // Compute x*y

  outputs[0] = temp_outputs[0];
  delete tape;
  return Status::OK();
}

Status SoftmaxModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
  AbstractTensorHandle* x = inputs[0];
  AbstractTensorHandle* labels = inputs[1];

  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  std::vector<AbstractTensorHandle*> temp_outputs(2);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::SparseSoftmaxCrossEntropyWithLogits(
      tape_ctx.get(), {x, labels}, absl::MakeSpan(temp_outputs), "sm_loss"));

  outputs[0] = temp_outputs[0];  // loss values

  delete tape;
  return Status::OK();
}

// ============================= End Models ================================

}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
