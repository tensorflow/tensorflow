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
#include "tensorflow/c/experimental/ops/nn_ops.h"

using std::vector;
using tensorflow::ops::Conj;
using tensorflow::ops::Identity;
using tensorflow::ops::Mul;
using tensorflow::ops::MatMul;
using tensorflow::ops::ReluGrad;
using tensorflow::ops::SparseSoftmaxCrossEntropyLoss;
using tensorflow::ops::ZerosLike;

namespace tensorflow {
namespace gradients {
namespace {

class AddGradientFunction : public GradientFunction {
 public:
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(2);
    std::vector<AbstractTensorHandle*> identity_outputs(1);
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
  int64_t counter;
};

class ExpGradientFunction : public GradientFunction {
 public:
  explicit ExpGradientFunction(AbstractTensorHandle* exp) : exp_(exp) {
    exp->Ref();
  }
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    std::vector<AbstractTensorHandle*> conj_outputs(1);
    std::string name = "Conj_Exp_Grad_" + std::to_string(counter);
    TF_RETURN_IF_ERROR(
        Conj(ctx->ctx, {exp_.get()}, absl::MakeSpan(conj_outputs), name.c_str()));
    AbstractTensorHandlePtr conj_output_releaser(conj_outputs[0]);
    grad_outputs->resize(1);

    name = "Mul_Exp_Grad_" + std::to_string(counter);
    TF_RETURN_IF_ERROR(Mul(ctx->ctx, {conj_outputs[0], grad_inputs[0]},
                           absl::MakeSpan(*grad_outputs), name.c_str()));
    return Status::OK();
  }
  ~ExpGradientFunction() override {}

 private:
  int64_t counter;
  AbstractTensorHandlePtr exp_;
};

class MatMulGradientFunction : public GradientFunction {
 public:
  explicit MatMulGradientFunction(std::vector<AbstractTensorHandle*> f_inputs, AttrBuilder f_attrs)
      : forward_inputs(f_inputs), attrs(f_attrs) {}

  Status Compute(Context* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_inputs,
                 std::vector<AbstractTensorHandle*>* grad_outputs) override {
    /* Given upstream grad U and a matmul op A*B, the gradients are:
     *
     *    dA = U * B.T
     *    dB = A.T * U
     *
     *    where A.T means `transpose(A)`
     */

    // TODO(amturati): figure why adding attrs to the function breaks the counter

    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    grad_outputs->resize(2);
    
    // Get transpose attrs
    bool t_a;
    attrs.Get("transpose_a", &t_a);

    bool t_b;
    attrs.Get("transpose_b", &t_b);
   
    // Conj each input
    std::vector<AbstractTensorHandle*> conj_outputs(1);
    std::string name = "Conj_A_MatMul_Grad_" + std::to_string(counter);
    TF_RETURN_IF_ERROR(
        Conj(ctx->ctx, {forward_inputs[0]}, absl::MakeSpan(conj_outputs), name.c_str()));
    
    AbstractTensorHandle* A = conj_outputs[0];

    name = "Conj_B_MatMul_Grad_" + std::to_string(counter);
    TF_RETURN_IF_ERROR(
        Conj(ctx->ctx, {forward_inputs[1]}, absl::MakeSpan(conj_outputs), name.c_str()));
    
    AbstractTensorHandle* B = conj_outputs[0];

    // Calc Grad
    std::vector<AbstractTensorHandle*> matmul_A_outputs(1);
    std::vector<AbstractTensorHandle*> matmul_B_outputs(1);
    std::string name_grad_A = "MatMul_Grad_A_" + std::to_string(counter);
    std::string name_grad_B = "MatMul_Grad_B_" + std::to_string(counter);
    if(!t_a && !t_b) {
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {upstream_grad, B},
                              absl::MakeSpan(matmul_A_outputs), name_grad_A.c_str(),
                              /*transpose_a = */ false,
                              /*transpose_b = */ true));
  
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {A, upstream_grad},
                              absl::MakeSpan(matmul_B_outputs), name_grad_B.c_str(),
                              /*transpose_a = */ true,
                              /*transpose_b = */ false));
    }
    else if(!t_a && t_b) {
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {upstream_grad, B},
                                absl::MakeSpan(matmul_A_outputs), name_grad_A.c_str(),
                                /*transpose_a = */ false,
                                /*transpose_b = */ false));
    
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {upstream_grad, A},
                                absl::MakeSpan(matmul_B_outputs), name_grad_B.c_str(),
                                /*transpose_a = */ true,
                                /*transpose_b = */ false));

    }
    else if(t_a && !t_b)  {
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {B, upstream_grad},
                                absl::MakeSpan(matmul_A_outputs), name_grad_A.c_str(),
                                /*transpose_a = */ false,
                                /*transpose_b = */ true));
    
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {A, upstream_grad},
                                absl::MakeSpan(matmul_B_outputs), name_grad_B.c_str(),
                                /*transpose_a = */ false,
                                /*transpose_b = */ false));
    }
    else { // t_a && t_b 
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {B, upstream_grad},
                                absl::MakeSpan(matmul_A_outputs), name_grad_A.c_str(),
                                /*transpose_a = */ true,
                                /*transpose_b = */ true));
    
      TF_RETURN_IF_ERROR(MatMul(ctx->ctx, {upstream_grad, A},
                                absl::MakeSpan(matmul_B_outputs), name_grad_B.c_str(),
                                /*transpose_a = */ true,
                                /*transpose_b = */ true));
    }

    // Gradient for A
    (*grad_outputs)[0] = matmul_A_outputs[0];

    // Gradient for B
    (*grad_outputs)[1] = matmul_B_outputs[0];

    counter += 1;  // update counter for names
    return Status::OK();
  }
  ~MatMulGradientFunction() override {}

 private:
  int64_t counter;
  std::vector<AbstractTensorHandle*> forward_inputs;
  AttrBuilder attrs;
};

class ReluGradientFunction : public GradientFunction {
 public:
  explicit ReluGradientFunction(std::vector<AbstractTensorHandle*> f_outputs)
      : forward_outputs(f_outputs) {}

  Status Compute(Context* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_inputs,
                 std::vector<AbstractTensorHandle*>* grad_outputs) override {
    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    AbstractTensorHandle* activations = forward_outputs[0];
    grad_outputs->resize(1);
    std::vector<AbstractTensorHandle*> relugrad_outputs(1);

    // Calculate Grad
    std::string name = "relu_grad" + std::to_string(counter);

    TF_RETURN_IF_ERROR(ReluGrad(ctx->ctx, {upstream_grad, activations},
                                absl::MakeSpan(relugrad_outputs),
                                name.c_str()));

    (*grad_outputs)[0] = relugrad_outputs[0];

    counter += 1;
    return Status::OK();
  }
  ~ReluGradientFunction() override {}

 private:
  int64_t counter;
  std::vector<AbstractTensorHandle*> forward_outputs;
};


class SparseSoftmaxCrossEntropyLossGradientFunction : public GradientFunction {
 public:
  explicit SparseSoftmaxCrossEntropyLossGradientFunction(
      std::vector<AbstractTensorHandle*> f_inputs,
      std::vector<AbstractTensorHandle*> f_outputs)
      : forward_inputs(f_inputs), forward_outputs(f_outputs) {}

  Status Compute(Context* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_inputs,
                 std::vector<AbstractTensorHandle*>* grad_outputs) override {
  
    grad_outputs->resize(2);

    // Grad for Softmax Input 
    std::string name = "Mul_Softmax_Grad_" + std::to_string(counter);
    std::vector<AbstractTensorHandle*> mul_outputs(1);
    TF_RETURN_IF_ERROR(ops::Mul(ctx->ctx, {grad_inputs[0], forward_outputs[1]},
                                     absl::MakeSpan(mul_outputs),
                                     name.c_str())); // upstream_grad * local softmax grad
    (*grad_outputs)[0] = mul_outputs[0];

    // Grad for labels
    // TODO(amturati): check to see if ZerosLike is ok instead of nullptr
    name = "Zeros_Softmax_Grad_" + std::to_string(counter);
    std::vector<AbstractTensorHandle*> z_outputs(1);
    TF_RETURN_IF_ERROR(ops::ZerosLike(ctx->ctx, {forward_inputs[1]},
                                     absl::MakeSpan(z_outputs),
                                     name.c_str()));
    (*grad_outputs)[1] = z_outputs[0];  // nullptr causes Mangled Stack Trace

    counter += 1;
    return Status::OK();
  }
  ~SparseSoftmaxCrossEntropyLossGradientFunction() override {}

 private:
  int64_t counter;
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

GradientFunction* MatMulRegisterer(const ForwardOperation& op) {
  return new MatMulGradientFunction(op.inputs, op.attrs);
}

GradientFunction* ReluRegisterer(const ForwardOperation& op) {
  return new ReluGradientFunction(op.outputs);
}

GradientFunction* SparseSoftmaxCrossEntropyLossRegisterer(
    const ForwardOperation& op) {
  return new SparseSoftmaxCrossEntropyLossGradientFunction(op.inputs,
                                                           op.outputs);
}


}  // namespace gradients
}  // namespace tensorflow
