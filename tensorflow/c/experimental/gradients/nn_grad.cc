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
      : forward_outputs_(f_outputs) {
    for (auto output : forward_outputs_) {
      if (output) {
        output->Ref();
      }
    }
  }

  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
    AbstractTensorHandle* upstream_grad = grad_outputs[0];
    AbstractTensorHandle* activations = forward_outputs_[0];

    // Calculate Grad
    std::string name = "relu_grad";
    TF_RETURN_IF_ERROR(ReluGrad(ctx, upstream_grad, activations,
                                &grad_inputs[0], name.c_str()));
    return Status::OK();
  }
  ~ReluGradientFunction() override {
    for (auto output : forward_outputs_) {
      if (output) {
        output->Unref();
      }
    }
  }

 private:
  // TODO(b/174778737): Only hold needed outputs.
  vector<AbstractTensorHandle*> forward_outputs_;
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
  AbstractTensorHandle* expand_dims_outputs;
  TF_RETURN_IF_ERROR(
      ops::ExpandDims(ctx, vec, dim.get(), &expand_dims_outputs, "ExpandDims"));
  TF_RETURN_IF_ERROR(
      ops::Mul(ctx, expand_dims_outputs, mat, &outputs[0], "Mul"));
  expand_dims_outputs->Unref();
  return Status::OK();
}

class SparseSoftmaxCrossEntropyWithLogitsGradientFunction
    : public GradientFunction {
 public:
  explicit SparseSoftmaxCrossEntropyWithLogitsGradientFunction(
      vector<AbstractTensorHandle*> f_outputs)
      : forward_outputs_(f_outputs) {}

  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
    // Grad for Softmax Input
    TF_RETURN_IF_ERROR(BroadcastMul(
        ctx, grad_outputs[0], forward_outputs_[1],
        grad_inputs.subspan(0, 1)));  // upstream_grad * local softmax grad

    // Grad for labels is null
    grad_inputs[1] = nullptr;
    return Status::OK();
  }
  ~SparseSoftmaxCrossEntropyWithLogitsGradientFunction() override {}

 private:
  vector<AbstractTensorHandle*> forward_outputs_;
};

// TODO(vnvo2409): Add python test
class BiasAddGradientFunction : public GradientFunction {
 public:
  explicit BiasAddGradientFunction(AttrBuilder f_attrs)
      : forward_attrs_(f_attrs) {}

  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
    /* Given upstream grad U and a BiasAdd: A + bias, the gradients are:
     *
     *    dA = U
     *    dbias = reduceSum(U, dims = channel_dim)
     */

    AbstractTensorHandle* upstream_grad = grad_outputs[0];
    DCHECK(upstream_grad);

    // Recover data format from forward pass for gradient.
    std::string data_format;
    TF_RETURN_IF_ERROR(forward_attrs_.Get("data_format", &data_format));

    // Grad for A
    grad_inputs[0] = upstream_grad;
    grad_inputs[0]->Ref();

    // Grad for bias
    std::string name = "bias_add_grad";
    TF_RETURN_IF_ERROR(BiasAddGrad(ctx, upstream_grad, &grad_inputs[1],
                                   data_format.c_str(), name.c_str()));

    return Status::OK();
  }
  ~BiasAddGradientFunction() override {}

 private:
  AttrBuilder forward_attrs_;
};

}  // namespace

GradientFunction* ReluRegisterer(const ForwardOperation& op) {
  return new ReluGradientFunction(op.outputs);
}

GradientFunction* SparseSoftmaxCrossEntropyWithLogitsRegisterer(
    const ForwardOperation& op) {
  return new SparseSoftmaxCrossEntropyWithLogitsGradientFunction(op.outputs);
}

GradientFunction* BiasAddRegisterer(const ForwardOperation& op) {
  return new BiasAddGradientFunction(op.attrs);
}

}  // namespace gradients
}  // namespace tensorflow
