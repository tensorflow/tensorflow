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

#include "tensorflow/compiler/tf2xla/kernels/random_ops_util.h"
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

xla::XlaOp MaybeConvertF32ToBF16(xla::XlaOp input, DataType dtype) {
  if (dtype == DT_BFLOAT16) {
    xla::XlaBuilder* builder = input.builder();
    xla::XlaOp output = xla::BitcastConvertType(input, xla::U32) &
                        xla::ConstantR0<uint32>(builder, 0xFFFF0000);
    return xla::ConvertElementType(xla::BitcastConvertType(output, xla::F32),
                                   xla::BF16);
  } else {
    return input;
  }
}

xla::XlaOp StatelessRngUniform(xla::XlaOp seeds, const xla::Shape& shape,
                               xla::XlaOp minval, xla::XlaOp maxval) {
  xla::XlaBuilder* builder = seeds.builder();

  xla::XlaOp seed0 = xla::Reshape(xla::Slice(seeds, {0}, {1}, {1}), {});
  xla::XlaOp seed1 = xla::Reshape(xla::Slice(seeds, {1}, {2}, {1}), {});
  xla::XlaOp key = ConvertElementType(seed0, xla::U64) |
                   ShiftLeft(ConvertElementType(seed1, xla::U64),
                             ConstantR0WithType(builder, xla::U64, 32));
  xla::XlaOp initial_state = xla::ConstantR0WithType(builder, xla::U64, 0);
  xla::PrimitiveType type = shape.element_type();
  switch (type) {
    case xla::F32:
      return xla::UniformF32Distribution(key, initial_state,
                                         xla::ThreeFryBitGenerator, minval,
                                         maxval, shape)
          .value;
    case xla::S32:  // fall through
    case xla::S64:
      return UniformIntDistribution(key, initial_state,
                                    xla::ThreeFryBitGenerator, minval, maxval,
                                    shape)
          .value;
      break;
    default:
      return builder->ReportError(xla::Unimplemented(
          "Types other than F32, S32 and S64 are not implemented by "
          "StatelessRngUniform; got %s",
          xla::primitive_util::LowercasePrimitiveTypeName(type)));
  }
}

namespace {

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
    xla::XlaOp uniform = StatelessRngUniform(
        seed, xla_shape, xla::ConstantR0<float>(builder, 0.0),
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
    xla::XlaOp uniform = StatelessRngUniform(seed, xla_shape, minval, maxval);

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
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(DT_FLOAT, shape, &xla_shape));

    xla::XlaBuilder* builder = seed.builder();
    xla::XlaOp seed0 = xla::Reshape(xla::Slice(seed, {0}, {1}, {1}), {});
    xla::XlaOp seed1 = xla::Reshape(xla::Slice(seed, {1}, {2}, {1}), {});
    xla::XlaOp initial_state = xla::ConstantR0WithType(builder, xla::U64, 0);
    xla::XlaOp key = ConvertElementType(seed0, xla::U64) |
                     ShiftLeft(ConvertElementType(seed1, xla::U64),
                               ConstantR0WithType(builder, xla::U64, 32));
    xla::XlaOp normal =
        xla::NormalF32Distribution(key, initial_state,
                                   xla::ThreeFryBitGenerator, xla_shape)
            .value;
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

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(DT_FLOAT, shape, &xla_shape));
    xla::XlaOp uniform = StatelessRngUniform(
        seed, xla_shape,
        xla::MinPositiveNormalValue(builder, xla_shape.element_type()),
        xla::One(builder, xla_shape.element_type()));
    xla::XlaOp truncated_normal = TruncatedNormal(uniform);
    truncated_normal = MaybeConvertF32ToBF16(truncated_normal, dtype_);
    ctx->SetOutput(0, truncated_normal);
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
