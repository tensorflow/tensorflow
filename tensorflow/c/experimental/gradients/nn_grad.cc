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

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"

using std::vector;
using tensorflow::ops::BiasAddGrad;
using tensorflow::ops::Mul;
using tensorflow::ops::ReluGrad;

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

Status BroadcastMul(AbstractContext* ctx, AbstractTensorHandle* vec,
                    AbstractTensorHandle* mat,
                    absl::Span<AbstractTensorHandle*> outputs) {
  if (!isa<ImmediateExecutionContext>(ctx)) {
    // TODO(b/168850692): Fix this.
    return errors::Unimplemented(
        "BroadcastMul is not supported in tracing mode yet.");
  }
  auto imm_ctx = dyn_cast<ImmediateExecutionContext>(ctx);
  AbstractTensorPtr minus_1(imm_ctx->CreateInt32Scalar(-1));
  ImmediateTensorHandlePtr dim(imm_ctx->CreateLocalHandle(minus_1.get()));
  vector<AbstractTensorHandle*> expand_dims_outputs(1);
  TF_RETURN_IF_ERROR(ops::ExpandDims(ctx, {vec, dim.get()},
                                     absl::MakeSpan(expand_dims_outputs),
                                     "ExpandDims"));
  TF_RETURN_IF_ERROR(
      ops::Mul(ctx, {expand_dims_outputs[0], mat}, outputs, "Mul"));
  expand_dims_outputs[0]->Unref();
  return Status::OK();
}

class SparseSoftmaxCrossEntropyWithLogitsGradientFunction
    : public GradientFunction {
 public:
  explicit SparseSoftmaxCrossEntropyWithLogitsGradientFunction(
      vector<AbstractTensorHandle*> f_outputs)
      : forward_outputs(f_outputs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(2);

    // Grad for Softmax Input
    vector<AbstractTensorHandle*> mul_outputs(1);
    TF_RETURN_IF_ERROR(BroadcastMul(
        ctx->ctx, grad_inputs[0], forward_outputs[1],
        absl::MakeSpan(mul_outputs)));  // upstream_grad * local softmax grad
    (*grad_outputs)[0] = mul_outputs[0];

    // Grad for labels is null
    (*grad_outputs)[1] = nullptr;

    return Status::OK();
  }
  ~SparseSoftmaxCrossEntropyWithLogitsGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_outputs;
};

// TODO(vnvo2409): Add python test
class BiasAddGradientFunction : public GradientFunction {
 public:
  explicit BiasAddGradientFunction(AttrBuilder f_attrs)
      : forward_attrs(f_attrs) {}

  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    /* Given upstream grad U and a BiasAdd: A + bias, the gradients are:
     *
     *    dA = U
     *    dbias = reduceSum(U, dims = channel_dim)
     */

    AbstractTensorHandle* upstream_grad = grad_inputs[0];
    DCHECK(upstream_grad);
    grad_outputs->resize(2);

    // Recover data format from forward pass for gradient.
    std::string data_format;
    forward_attrs.Get("data_format", &data_format);

    // Grad for A
    (*grad_outputs)[0] = upstream_grad;
    (*grad_outputs)[0]->Ref();

    // Grad for bias
    vector<AbstractTensorHandle*> bias_add_grad_outputs(1);
    std::string name = "bias_add_grad";
    TF_RETURN_IF_ERROR(BiasAddGrad(ctx->ctx, {upstream_grad},
                                   absl::MakeSpan(bias_add_grad_outputs),
                                   data_format.c_str(), name.c_str()));

    (*grad_outputs)[1] = bias_add_grad_outputs[0];
    return Status::OK();
  }
  ~BiasAddGradientFunction() override {}

 private:
  AttrBuilder forward_attrs;
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

BackwardFunction* SparseSoftmaxCrossEntropyWithLogitsRegisterer(
    const ForwardOperation& op) {
  auto gradient_function =
      new SparseSoftmaxCrossEntropyWithLogitsGradientFunction(op.outputs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

BackwardFunction* BiasAddRegisterer(const ForwardOperation& op) {
  // For ops with a single output, the gradient function is not called if there
  // is no incoming gradient. So we do not need to worry about creating zeros
  // grads in this case.
  auto gradient_function = new BiasAddGradientFunction(op.attrs);
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

}  // namespace gradients
}  // namespace tensorflow
