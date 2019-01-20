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

#include <cmath>

#include "tensorflow/compiler/tf2xla/lib/random.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace tensorflow {
namespace {

xla::XlaOp MaybeConvertF32ToBF16(xla::XlaOp input, DataType dtype) {
  // Mask the last 16 bit. With normal rounding, values near "maxval" would be
  // converted to "maxval" which is out of range ["minval", "maxval"). In
  // addition, the distribution near the limit is not uniform.
  if (dtype == DT_BFLOAT16) {
    xla::XlaBuilder* builder = input.builder();
    auto output = xla::BitcastConvertType(input, xla::U32) &
                  xla::ConstantR0<uint32>(builder, 0xFFFF0000);
    return xla::ConvertElementType(xla::BitcastConvertType(output, xla::F32),
                                   xla::BF16);
  } else {
    return input;
  }
}

class StatelessRandomUniformOp : public XlaOpKernel {
 public:
  explicit StatelessRandomUniformOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    xla::XlaOp seed = ctx->Input(1);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(DT_FLOAT, shape, &xla_shape));

    auto seed0 = xla::Reshape(xla::Slice(seed, {0}, {1}, {1}), {});
    auto seed1 = xla::Reshape(xla::Slice(seed, {1}, {2}, {1}), {});

    auto uniform = xla::StatelessRngUniform(
        {seed0, seed1}, xla_shape, xla::ConstantR0<float>(builder, 0.0),
        xla::ConstantR0<float>(builder, 1.0));
    uniform = MaybeConvertF32ToBF16(uniform, dtype_);
    ctx->SetOutput(0, uniform);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomUniformOp);
};

// TODO(phawkins): generalize to non-float, non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomUniform")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomUniformOp);

class StatelessRandomUniformIntOp : public XlaOpKernel {
 public:
  explicit StatelessRandomUniformIntOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    TensorShape minval_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval_shape),
                errors::InvalidArgument("minval must be scalar, got shape ",
                                        minval_shape.DebugString()));
    TensorShape maxval_shape = ctx->InputShape(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval_shape),
                errors::InvalidArgument("minval must be scalar, got shape ",
                                        maxval_shape.DebugString()));

    xla::XlaOp seed = ctx->Input(1);
    xla::XlaOp minval = ctx->Input(2);
    xla::XlaOp maxval = ctx->Input(3);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape, &xla_shape));

    auto seed0 = xla::Reshape(xla::Slice(seed, {0}, {1}, {1}), {});
    auto seed1 = xla::Reshape(xla::Slice(seed, {1}, {2}, {1}), {});

    auto uniform =
        xla::StatelessRngUniform({seed0, seed1}, xla_shape, minval, maxval);
    ctx->SetOutput(0, uniform);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomUniformIntOp);
};

// TODO(phawkins): generalize to non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomUniformInt")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_INT32, DT_INT64})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomUniformIntOp);

class StatelessRandomNormalOp : public XlaOpKernel {
 public:
  explicit StatelessRandomNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape == TensorShape({2}),
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    xla::XlaOp seed = ctx->Input(1);
    xla::XlaBuilder* builder = ctx->builder();
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(DT_FLOAT, shape, &xla_shape));

    auto seed0 = xla::Reshape(xla::Slice(seed, {0}, {1}, {1}), {});
    auto seed1 = xla::Reshape(xla::Slice(seed, {1}, {2}, {1}), {});

    auto uniform = xla::StatelessRngUniform(
        {seed0, seed1}, xla_shape,
        xla::ConstantR0<float>(builder, std::nextafter(-1.0f, 0.0f)),
        xla::ConstantR0<float>(builder, 1.0));
    // Convert uniform distribution to normal distribution by computing
    // sqrt(2) * erfinv(x)
    auto normal =
        xla::ScalarLike(uniform, std::sqrt(2.0)) * xla::ErfInv(uniform);
    normal = MaybeConvertF32ToBF16(normal, dtype_);
    ctx->SetOutput(0, normal);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomNormalOp);
};

// TODO(phawkins): generalize to non-float, non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomNormalOp);

class StatelessTruncatedNormalOp : public XlaOpKernel {
 public:
  explicit StatelessTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape == TensorShape({2}),
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    xla::XlaOp seed = ctx->Input(1);
    xla::XlaBuilder* builder = ctx->builder();

    auto seed0 = xla::Reshape(xla::Slice(seed, {0}, {1}, {1}), {});
    auto seed1 = xla::Reshape(xla::Slice(seed, {1}, {2}, {1}), {});

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(DT_FLOAT, shape, &xla_shape));
    auto uniform = xla::StatelessRngUniform(
        {seed0, seed1}, xla_shape,
        xla::ConstantR0<float>(builder, std::numeric_limits<float>::min()),
        xla::ConstantR0<float>(builder, 1.0));
    auto output = TruncatedNormal(uniform);
    output = MaybeConvertF32ToBF16(output, dtype_);
    ctx->SetOutput(0, output);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessTruncatedNormalOp);
};

REGISTER_XLA_OP(Name("StatelessTruncatedNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessTruncatedNormalOp);

}  // namespace
}  // namespace tensorflow
