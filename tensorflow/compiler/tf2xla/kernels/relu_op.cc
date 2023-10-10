/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Native XLA implementations of XLA Relu Ops

#include "tensorflow/compiler/tf2xla/kernels/relu_op.h"

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/literal.h"

namespace xla {
XlaOp Relu(XlaOp x) { return Max(ScalarLike(x, 0), x); }

XlaOp Relu6(XlaOp x) {
  auto zero = ScalarLike(x, 0);
  auto six = ScalarLike(x, 6);
  return Clamp(zero, x, six);
}
}  // namespace xla

namespace tensorflow {
namespace {

REGISTER_XLA_OP(Name("Relu"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Relu6"), MlirXlaOpKernel);

class LeakyReluOp : public XlaOpKernel {
 public:
  explicit LeakyReluOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    auto features = ctx->Input("features");
    auto prod_with_alpha = features * xla::ScalarLike(features, alpha_);
    auto gt_zero = xla::Gt(features, xla::ScalarLike(features, 0));
    auto output = xla::Select(gt_zero, features, prod_with_alpha);
    ctx->SetOutput(0, output);
  }
  float alpha_;
};
REGISTER_XLA_OP(Name("LeakyRelu"), LeakyReluOp);

REGISTER_XLA_OP(Name("ReluGrad"), MlirXlaOpKernel);

class Relu6GradOp : public XlaOpKernel {
 public:
  explicit Relu6GradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  // Return the lhs (incoming gradient) if the rhs (input feature) > 0,
  // otherwise return 0.
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    const TensorShape shape = ctx->InputShape(0);
    const auto zero =
        xla::Broadcast(XlaHelpers::Zero(b, input_type(0)), shape.dim_sizes());
    const auto six = xla::Broadcast(
        XlaHelpers::IntegerLiteral(b, input_type(0), 6), shape.dim_sizes());
    auto out = xla::Select(
        xla::And(xla::Lt(ctx->Input(1), six), xla::Gt(ctx->Input(1), zero)),
        ctx->Input(0), zero);
    ctx->SetOutput(0, out);
  }
};
REGISTER_XLA_OP(Name("Relu6Grad"), Relu6GradOp);

class LeakyReluGradOp : public XlaOpKernel {
 public:
  explicit LeakyReluGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    auto gradients = ctx->Input("gradients");
    auto features = ctx->Input("features");
    auto output =
        xla::Select(xla::Gt(features, xla::ScalarLike(features, 0)), gradients,
                    gradients * xla::ScalarLike(gradients, alpha_));
    ctx->SetOutput(0, output);
  }
  float alpha_;
};
REGISTER_XLA_OP(Name("LeakyReluGrad"), LeakyReluGradOp);

}  // namespace
}  // namespace tensorflow
