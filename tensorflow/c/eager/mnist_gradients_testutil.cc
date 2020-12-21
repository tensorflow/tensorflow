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
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(inputs[0]);  // Watch x.
  tape->Watch(inputs[1]);  // Watch y.
  std::vector<AbstractTensorHandle*> add_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(
      ops::Add(tape_ctx.get(), inputs, absl::MakeSpan(add_outputs), "Add"));
  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/add_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{}, outputs));
  for (auto add_output : add_outputs) {
    add_output->Unref();
  }
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
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(inputs[0]);  // Watch x.
  tape->Watch(inputs[1]);  // Watch y.
  vector<AbstractTensorHandle*> mm_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::MatMul(tape_ctx.get(), inputs,
                                 absl::MakeSpan(mm_outputs), "matmul0",
                                 /*transpose_a=*/false,
                                 /*transpose_b=*/false));  // Compute x*y.

  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/mm_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{}, outputs));
  for (auto mm_output : mm_outputs) {
    mm_output->Unref();
  }
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

  vector<AbstractTensorHandle*> temp_outputs(1);

  TF_RETURN_IF_ERROR(ops::MatMul(ctx, {X, W1}, absl::MakeSpan(temp_outputs),
                                 "matmul0",
                                 /*transpose_a=*/false,
                                 /*transpose_b=*/false));  // Compute X*W1

  TF_RETURN_IF_ERROR(ops::Relu(ctx, {temp_outputs[0]},
                               absl::MakeSpan(temp_outputs),
                               "relu"));  // Compute Relu(X*W1)

  TF_RETURN_IF_ERROR(ops::MatMul(
      ctx, {temp_outputs[0], W2}, absl::MakeSpan(temp_outputs), "matmul1",
      /*transpose_a=*/false, /*transpose_b=*/false));  // Compute W2*Relu(X*W1)

  AbstractTensorHandle* scores = temp_outputs[0];

  temp_outputs.resize(2);
  TF_RETURN_IF_ERROR(ops::SparseSoftmaxCrossEntropyWithLogits(
      ctx, {scores, y_labels}, absl::MakeSpan(temp_outputs),
      "softmax_loss"));  // Compute Softmax(Scores,labels)

  AbstractTensorHandle* loss_vals = temp_outputs[0];

  outputs[0] = scores;
  outputs[1] = loss_vals;
  return Status::OK();
}

Status MatMulTransposeModel(AbstractContext* ctx,
                            absl::Span<AbstractTensorHandle* const> inputs,
                            absl::Span<AbstractTensorHandle*> outputs,
                            const GradientRegistry& registry) {
  AbstractTensorHandle* X = inputs[0];
  AbstractTensorHandle* W1 = inputs[1];

  TF_RETURN_IF_ERROR(ops::MatMul(ctx, {X, W1}, outputs, "matmul0",
                                 /*transpose_a=*/true,
                                 /*transpose_b=*/false));  // Compute X*W1
  return Status::OK();
}

Status ReluGradModel(AbstractContext* ctx,
                     absl::Span<AbstractTensorHandle* const> inputs,
                     absl::Span<AbstractTensorHandle*> outputs,
                     const GradientRegistry& registry) {
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(inputs[0]);  // Watch X
  vector<AbstractTensorHandle*> relu_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::Relu(tape_ctx.get(), inputs,
                               absl::MakeSpan(relu_outputs),
                               "relu0"));  // Relu(X)

  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/relu_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{}, outputs));

  for (auto relu_output : relu_outputs) {
    relu_output->Unref();
  }

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

  auto tape = new Tape(/*persistent=*/true);
  tape->Watch(X);   // Watch X.
  tape->Watch(W1);  // Watch W1.
  tape->Watch(W2);  // Watch W1.
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

  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/{loss},
                                           /*sources=*/{W1, W2},
                                           /*output_gradients=*/{},
                                           outputs.subspan(0, 2)));

  // Only release 2nd temp output as first holds loss values.
  temp_outputs[1]->Unref();
  outputs[2] = loss;
  delete tape;
  return Status::OK();
}

Status ScalarMulModel(AbstractContext* ctx,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs,
                      const GradientRegistry& registry) {
  return ops::Mul(ctx, inputs, outputs,
                  "scalarMul0");  // Compute eta*A
}

Status MatMulModel(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs,
                   const GradientRegistry& registry) {
  return ops::MatMul(ctx, inputs, outputs, "matmul0",
                     /*transpose_a=*/false,
                     /*transpose_b=*/false);  // Compute X*W1
}

Status MulModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs,
                const GradientRegistry& registry) {
  return ops::Mul(ctx, inputs, outputs,
                  "mul0");  // Compute x*y
}

// ============================= End Models ================================

}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
