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

// XLA implementations of Random ops
// TODO(misard,phawkins): handle random number generator seeds/states correctly.
// TODO(misard,phawkins): add tests.

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class RandomUniformOp : public XlaOpKernel {
 public:
  explicit RandomUniformOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    const DataType dtype = output_type(0);
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::ComputationBuilder* b = ctx->builder();
    xla::ComputationDataHandle result = b->RngUniform(
        XlaHelpers::Zero(b, dtype), XlaHelpers::One(b, dtype), xla_shape);

    ctx->SetOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomUniformOp);
};

REGISTER_XLA_OP(Name("RandomUniform"), RandomUniformOp);

class RandomUniformIntOp : public XlaOpKernel {
 public:
  explicit RandomUniformIntOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeToXLAShape(input_type(1), shape, &xla_shape));

    const TensorShape minval_shape = ctx->InputShape(1);
    const TensorShape maxval_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval_shape),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval_shape),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval_shape.DebugString()));

    auto minval = ctx->Input(1);
    auto maxval = ctx->Input(2);
    ctx->SetOutput(0, ctx->builder()->RngUniform(minval, maxval, xla_shape));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomUniformIntOp);
};

REGISTER_XLA_OP(Name("RandomUniformInt"), RandomUniformIntOp);

class RandomStandardNormalOp : public XlaOpKernel {
 public:
  explicit RandomStandardNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::ComputationBuilder* b = ctx->builder();

    // Normal distribution with a mean of 0 and a standard deviation of 1:
    xla::ComputationDataHandle result = b->RngNormal(
        XlaHelpers::Zero(b, dtype), XlaHelpers::One(b, dtype), xla_shape);

    ctx->SetOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomStandardNormalOp);
};

REGISTER_XLA_OP(Name("RandomStandardNormal"), RandomStandardNormalOp);

class TruncatedNormalOp : public XlaOpKernel {
 public:
  explicit TruncatedNormalOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));
    xla::Shape xla_element_shape =
        xla::ShapeUtil::MakeShape(xla_shape.element_type(), {});

    xla::ComputationBuilder* b = ctx->builder();
    xla::ComputationDataHandle mean = XlaHelpers::Zero(b, dtype);
    xla::ComputationDataHandle stddev = XlaHelpers::One(b, dtype);
    xla::ComputationDataHandle candidate =
        b->RngNormal(mean, stddev, xla_shape);

    auto two_sd = [dtype](bool negate, xla::ComputationBuilder* b) {
      return XlaHelpers::FloatLiteral(b, dtype, negate ? -2.0 : 2.0);
    };
    auto out_of_range_mask = [two_sd](xla::ComputationDataHandle candidate,
                                      xla::ComputationBuilder* b) {
      xla::ComputationDataHandle too_large = b->Gt(candidate, two_sd(false, b));
      xla::ComputationDataHandle too_small = b->Lt(candidate, two_sd(true, b));
      return b->LogicalOr(too_large, too_small);
    };

    // The algorithm we're using is roughly:
    //
    // while (any(candidate < mean-2*sd || candidate > mean+2*sd)) {
    //   out_of_range_mask := candidate < mean-2*sd || candidate > mean+2*sd
    //   candidate = select(out_of_range_mask, rng_normal(), candidate)
    // }
    std::unique_ptr<xla::ComputationBuilder> test_builder =
        b->CreateSubBuilder("truncated_normal_test");
    {
      auto* b = test_builder.get();
      xla::ComputationDataHandle candidate =
          b->Parameter(0, xla_shape, "candidate");
      xla::ComputationDataHandle oor_mask = out_of_range_mask(candidate, b);
      OP_REQUIRES_OK(ctx, Any(out_of_range_mask(candidate, b), b).status());
    }

    std::unique_ptr<xla::ComputationBuilder> body_builder =
        b->CreateSubBuilder("truncated_normal_body");
    {
      auto* b = body_builder.get();
      xla::ComputationDataHandle candidate =
          b->Parameter(0, xla_shape, "candidate");
      xla::ComputationDataHandle to_resample = out_of_range_mask(candidate, b);
      xla::ComputationDataHandle mean = XlaHelpers::Zero(b, dtype);
      xla::ComputationDataHandle stddev = XlaHelpers::One(b, dtype);
      b->Select(to_resample, b->RngNormal(mean, stddev, xla_shape), candidate);
    }

    xla::StatusOr<xla::Computation> test_computation = test_builder->Build();
    OP_REQUIRES_OK(ctx, test_computation.status());
    xla::StatusOr<xla::Computation> body_computation = body_builder->Build();
    OP_REQUIRES_OK(ctx, body_computation.status());
    xla::ComputationDataHandle result =
        b->While(test_computation.ValueOrDie(), body_computation.ValueOrDie(),
                 candidate);

    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("TruncatedNormal"), TruncatedNormalOp);

}  // anonymous namespace
}  // namespace tensorflow
