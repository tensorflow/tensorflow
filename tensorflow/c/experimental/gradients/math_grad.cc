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
#include "tensorflow/c/experimental/gradients/math_grad.h"

#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"

using std::vector;
using tensorflow::ops::Conj;
using tensorflow::ops::Identity;
using tensorflow::ops::Mul;

namespace tensorflow {
namespace gradients {
namespace {

class AddGradientFunction : public GradientFunction {
 public:
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(2);
    vector<AbstractTensorHandle*> identity_outputs(1);
    // TODO(b/145674566): Handle name unification in tracing code.
    // TODO(b/161805092): Support broadcasting.

    std::string name = "Identity_A_" + std::to_string(counter);
    TF_RETURN_IF_ERROR(ops::Identity(ctx->ctx, {grad_inputs[0]},
                                     absl::MakeSpan(identity_outputs),
                                     name.c_str()));
    (*grad_outputs)[0] = identity_outputs[0];

    name = "Identity_B_" + std::to_string(counter);
    TF_RETURN_IF_ERROR(ops::Identity(ctx->ctx, {grad_inputs[0]},
                                     absl::MakeSpan(identity_outputs),
                                     name.c_str()));
    (*grad_outputs)[1] = identity_outputs[0];

    counter += 1;
    return Status::OK();
  }
  ~AddGradientFunction() override {}

 private:
  long counter;
};

class ExpGradientFunction : public GradientFunction {
 public:
  explicit ExpGradientFunction(AbstractTensorHandle* exp) : exp_(exp) {
    exp->Ref();
  }
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    vector<AbstractTensorHandle*> conj_outputs(1);
    TF_RETURN_IF_ERROR(
        Conj(ctx->ctx, {exp_.get()}, absl::MakeSpan(conj_outputs), "ExpConj"));
    AbstractTensorHandlePtr conj_output_releaser(conj_outputs[0]);
    grad_outputs->resize(1);
    TF_RETURN_IF_ERROR(Mul(ctx->ctx, {conj_outputs[0], grad_inputs[0]},
                           absl::MakeSpan(*grad_outputs), "ExpGradMul"));
    return Status::OK();
  }
  ~ExpGradientFunction() override {}

 private:
  AbstractTensorHandlePtr exp_;
};

class MatMulGradientFunction : public GradientFunction {
 public:
  explicit MatMulGradientFunction(AbstractContext* ctx, std::vector<AbstractTensorHandle*> f_inputs) : 
            ctx_(ctx), forward_inputs(f_inputs) {}
  
  Status Compute(absl::Span<AbstractTensorHandle* const> grad_inputs,
                 std::vector<AbstractTensorHandle*>* grad_outputs) override {
    /* Given upstream grad U and a matmul op A*B, the gradients are:
     *
     *    dA = U * B.T
     *    dB = A.T * U
     *
     *    where A.T means `transpose(A)`
     */

    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    grad_outputs->resize(2);
    std::vector<AbstractTensorHandle*> matmul_outputs(1);

    // Gradient for A
    TF_RETURN_IF_ERROR(MatMul(ctx_, {upstream_grad, forward_inputs[1]},
                              absl::MakeSpan(matmul_outputs), "mm0",  
                              /*transpose_a = */false, /*transpose_b = */true));

    (*grad_outputs)[0] = matmul_outputs[0];

    // Gradient for B
    TF_RETURN_IF_ERROR(MatMul(ctx_, {forward_inputs[0], upstream_grad},
                              absl::MakeSpan(matmul_outputs), "mm1", 
                              /*transpose_a = */true, /*transpose_b = */false));

    (*grad_outputs)[1] = matmul_outputs[0];

    counter += 1;  // update counter for names
    return Status::OK();
  }
  ~MatMulGradientFunction() override {}

 private:
  AbstractContext* ctx_;
  std::vector<AbstractTensorHandle*> forward_inputs;
  long counter;
  std::vector<AbstractTensorHandle*> forward_inputs;
};

class ReluGradientFunction : public GradientFunction {
 public:
  explicit ReluGradientFunction(AbstractContext* ctx, std::vector<AbstractTensorHandle*> f_inputs) : 
            ctx_(ctx), forward_inputs(f_inputs) {}
  
  Status Compute(absl::Span<AbstractTensorHandle* const> grad_inputs,
                 std::vector<AbstractTensorHandle*>* grad_outputs) override {
    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    AbstractTensorHandle* input_features = forward_inputs[0];
    grad_outputs->resize(1);
    std::vector<AbstractTensorHandle*> relugrad_outputs(1);

    // Calculate Grad
    TF_RETURN_IF_ERROR(ReluGrad(ctx_, {upstream_grad, input_features},
                              absl::MakeSpan(relugrad_outputs), "relu_grad"));

    (*grad_outputs)[0] = relugrad_outputs[0];

    counter += 1;
    return Status::OK();
  }
  ~ReluGradientFunction() override {}

 private:
  AbstractContext* ctx_;
  std::vector<AbstractTensorHandle*> forward_inputs;
};

class SparseSoftmaxCrossEntropyLossGradientFunction : public GradientFunction {
 public:
  explicit SparseSoftmaxCrossEntropyLossGradientFunction(AbstractContext* ctx, 
            std::vector<AbstractTensorHandle*> f_inputs, std::vector<AbstractTensorHandle*> f_outputs) : 
            ctx_(ctx), forward_inputs(f_inputs), forward_outputs(f_outputs)  {}
  
  Status Compute(absl::Span<AbstractTensorHandle* const> grad_inputs,
                 std::vector<AbstractTensorHandle*>* grad_outputs) override {
    // Forward Inputs : [scores, labels]

    grad_outputs->resize(2);
    std::vector<AbstractTensorHandle*> sm_outputs(2);
    
    // Calculate Grad
    TF_RETURN_IF_ERROR(SparseSoftmaxCrossEntropyLoss(ctx_, {forward_inputs[0], forward_inputs[1]},
                              absl::MakeSpan(sm_outputs), "softmax_loss"));

    // Calculate Grad
    std::string name = "sm_loss" + std::to_string(counter);

    TF_RETURN_IF_ERROR(SparseSoftmaxCrossEntropyLoss(
        ctx->ctx, {forward_inputs[0], forward_inputs[1]},
        absl::MakeSpan(sm_outputs), name.c_str()));

    // TODO(amturati): fix error where we have to return the softmax loss as the
    // 2nd grad for the labels to avoid mangled stack trace. Also avoid running
    // forward operation again, check to see if forward_outputs are being
    // passed.

    // SparseSoftmaxCrossEntropyLoss returns [loss_vals, grads], so return 2nd
    // output.
    (*grad_outputs)[0] = sm_outputs[1];  // return backprop for scores
    (*grad_outputs)[1] = sm_outputs[0];  // nullptr causes Mangled Stack Trace

    counter += 1;
    return Status::OK();
  }
  ~SparseSoftmaxCrossEntropyLossGradientFunction() override {}

 private:
  AbstractContext* ctx_;
  std::vector<AbstractTensorHandle*> forward_inputs;
  std::vector<AbstractTensorHandle*> forward_outputs;
};

}  // namespace

BackwardFunction* AddRegisterer(const ForwardOperation& op) {
  auto gradient_function = new AddGradientFunction;
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* ExpRegisterer(const ForwardOperation& op) {
  auto gradient_function = new ExpGradientFunction(op.outputs[0]);
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

}  // namespace gradients
}  // namespace tensorflow
