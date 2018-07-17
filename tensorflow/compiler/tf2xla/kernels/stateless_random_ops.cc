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
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/numeric.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace tensorflow {
namespace {

class StatelessRandomUniformOp : public XlaOpKernel {
 public:
  explicit StatelessRandomUniformOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

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
    ctx->SetOutput(0, uniform);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomUniformOp);
};

// TODO(phawkins): generalize to non-float, non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomUniform")
                    .CompileTimeConstInput("shape")
                    .TypeConstraint("dtype", DT_FLOAT)
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomUniformOp);

class StatelessRandomNormalOp : public XlaOpKernel {
 public:
  explicit StatelessRandomNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

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
    ctx->SetOutput(0, normal);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomNormalOp);
};

// TODO(phawkins): generalize to non-float, non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomNormal")
                    .CompileTimeConstInput("shape")
                    .TypeConstraint("dtype", DT_FLOAT)
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomNormalOp);

class StatelessTruncatedNormalOp : public XlaOpKernel {
 public:
  explicit StatelessTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

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

    ctx->SetOutput(0, TruncatedNormal(uniform));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(StatelessTruncatedNormalOp);
};

REGISTER_XLA_OP(Name("StatelessTruncatedNormal")
                    .CompileTimeConstInput("shape")
                    .TypeConstraint("dtype", DT_FLOAT)
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessTruncatedNormalOp);

}  // namespace
}  // namespace tensorflow
