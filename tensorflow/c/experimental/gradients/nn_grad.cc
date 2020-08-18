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
#include "tensorflow/c/experimental/gradients/nn_grad.h"

#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"

using std::vector;
using tensorflow::ops::Conj;
using tensorflow::ops::Identity;
using tensorflow::ops::Mul;
using tensorflow::ops::ReluGrad;
using tensorflow::ops::SparseSoftmaxCrossEntropyLoss;
using tensorflow::ops::ZerosLike;

namespace tensorflow {
namespace gradients {
namespace {

class ReluGradientFunction : public GradientFunction {
 public:
  explicit ReluGradientFunction(vector<AbstractTensorHandle*> f_outputs)
      : forward_outputs(f_outputs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    AbstractTensorHandle* activations = forward_outputs[0];
    grad_outputs->resize(1);
    vector<AbstractTensorHandle*> relugrad_outputs(1);

    // Calculate Grad
    std::string name = "relu_grad";

    TF_RETURN_IF_ERROR(ReluGrad(ctx->ctx, {upstream_grad, activations},
                                absl::MakeSpan(relugrad_outputs),
                                name.c_str()));
    (*grad_outputs)[0] = relugrad_outputs[0];

    return Status::OK();
  }
  ~ReluGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_outputs;
};

class SparseSoftmaxCrossEntropyLossGradientFunction : public GradientFunction {
 public:
  explicit SparseSoftmaxCrossEntropyLossGradientFunction(
      vector<AbstractTensorHandle*> f_outputs)
      : forward_outputs(f_outputs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(2);

    // Grad for Softmax Input
    std::string name = "Mul_Softmax_Grad";
    vector<AbstractTensorHandle*> mul_outputs(1);
    TF_RETURN_IF_ERROR(
        ops::Mul(ctx->ctx, {grad_inputs[0], forward_outputs[1]},
                 absl::MakeSpan(mul_outputs),
                 name.c_str()));  // upstream_grad * local softmax grad
    (*grad_outputs)[0] = mul_outputs[0];

    // Grad for labels is null
    (*grad_outputs)[1] = nullptr;

    return Status::OK();
  }
  ~SparseSoftmaxCrossEntropyLossGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_outputs;
};

}  // namespace

BackwardFunction* ReluRegisterer(const ForwardOperation& op) {
  auto gradient_function = new ReluGradientFunction(op.outputs);
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* SparseSoftmaxCrossEntropyLossRegisterer(
    const ForwardOperation& op) {
  auto gradient_function =
      new SparseSoftmaxCrossEntropyLossGradientFunction(op.outputs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

}  // namespace gradients
}  // namespace tensorflow
