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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"

namespace tensorflow {
namespace {

class ReluOp : public XlaOpKernel {
 public:
  explicit ReluOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  // Computes the max of the scalar input x and 0.
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();
    auto zero = XlaHelpers::Zero(builder, input_type(0));
    ctx->SetOutput(0, xla::Max(zero, ctx->Input(0)));
  }
};
REGISTER_XLA_OP(Name("Relu"), ReluOp);

class Relu6Op : public XlaOpKernel {
 public:
  explicit Relu6Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  // Clamp the scalar input between 0 and 6.
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();
    auto zero = XlaHelpers::Zero(builder, input_type(0));
    auto six = XlaHelpers::IntegerLiteral(builder, input_type(0), 6);
    ctx->SetOutput(0, xla::Clamp(zero, ctx->Input(0), six));
  }
};
REGISTER_XLA_OP(Name("Relu6"), Relu6Op);

class LeakyReluOp : public XlaOpKernel {
 public:
  explicit LeakyReluOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    auto features = ctx->Input("features");
    auto output =
        xla::Max(features, features * xla::ScalarLike(features, alpha_));
    ctx->SetOutput(0, output);
  }
  float alpha_;
};
REGISTER_XLA_OP(Name("LeakyRelu"), LeakyReluOp);

class ReluGradOp : public XlaOpKernel {
 public:
  explicit ReluGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  // Return the lhs (incoming gradient) if the rhs (input feature) > 0,
  // otherwise return 0.
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    const TensorShape shape = ctx->InputShape(0);
    const auto zero =
        xla::Broadcast(XlaHelpers::Zero(b, input_type(0)), shape.dim_sizes());
    const auto pred = xla::Gt(ctx->Input(1), zero);
    ctx->SetOutput(0, xla::Select(pred, ctx->Input(0), zero));
  }
};
REGISTER_XLA_OP(Name("ReluGrad"), ReluGradOp);

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
