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

#include <tuple>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/kernels/random_ops_util.h"
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/lib/random.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/prng.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

xla::BitGeneratorTy GetBitGeneratorForDevice(
    absl::string_view device_type_string) {
  // The Philox algorithm may cause performance regression on other devices.
  // Turn on the Philox algorithm for the CPU and GPU backends only.
  if (device_type_string == DEVICE_GPU_XLA_JIT ||
      device_type_string == DEVICE_CPU_XLA_JIT) {
    return [=](xla::XlaOp key, xla::XlaOp state, const xla::Shape& shape) {
      std::tie(state, key) = xla::ScramblePhiloxKey(key);
      xla::XlaOp philox_state =
          xla::ConcatInDim(key.builder(), {xla::Reshape(key, {1}), state}, 0);
      xla::XlaOp result = xla::RngBitGenerator(xla::RandomAlgorithm::RNG_PHILOX,
                                               philox_state, shape);
      return xla::RngOutput{/*value=*/xla::GetTupleElement(result, 1),
                            /*state=*/xla::GetTupleElement(result, 0)};
    };
  }
  return [=](xla::XlaOp key, xla::XlaOp state, const xla::Shape& shape) {
    state = xla::ConcatScalars(key.builder(), {key, state});
    xla::XlaOp result =
        xla::RngBitGenerator(xla::RandomAlgorithm::RNG_DEFAULT, state, shape);
    return xla::RngOutput{/*value=*/xla::GetTupleElement(result, 1),
                          /*state=*/xla::GetTupleElement(result, 0)};
  };
}

}  // namespace

xla::XlaOp MaybeConvertF32ToBF16(xla::XlaOp input, DataType dtype) {
  if (dtype == DT_BFLOAT16) {
    xla::XlaBuilder* builder = input.builder();
    // TODO(b/256243456): Instead of doing
    // `ConvertElementType(BitcastConvertType(u32, F32), BF16)` we should do
    // `BitcastConvertType(ConvertElementType(u32, U16), BF16)`, to avoid the
    // unclear `ConvertElementType(f32, BF16)` behavior.
    xla::XlaOp output = xla::BitcastConvertType(input, xla::U32) &
                        xla::ConstantR0<uint32>(builder, 0xFFFF0000);
    return xla::ConvertElementType(xla::BitcastConvertType(output, xla::F32),
                                   xla::BF16);
  } else {
    return input;
  }
}

xla::XlaOp StatelessRngUniform(absl::string_view device_type_string,
                               xla::XlaOp seeds, const xla::Shape& shape,
                               xla::XlaOp minval, xla::XlaOp maxval) {
  xla::XlaBuilder* builder = seeds.builder();

  xla::XlaOp seed0 = xla::Reshape(xla::Slice(seeds, {0}, {1}, {1}), {});
  xla::XlaOp seed1 = xla::Reshape(xla::Slice(seeds, {1}, {2}, {1}), {});
  xla::XlaOp key = GetU64FromS32Seeds(seed0, seed1);
  xla::XlaOp initial_state = xla::ConstantR0WithType(builder, xla::U64, 0);
  xla::PrimitiveType type = shape.element_type();
  switch (type) {
    case xla::F16:
    case xla::F32:
    case xla::F64:
      return xla::UniformFloatingPointDistribution(
                 key, initial_state,
                 GetBitGeneratorForDevice(device_type_string), minval, maxval,
                 shape)
          .value;
    case xla::S32:
    case xla::S64:
      return UniformIntDistribution(
                 key, initial_state,
                 GetBitGeneratorForDevice(device_type_string), minval, maxval,
                 shape)
          .value;
      break;
    default:
      return builder->ReportError(xla::Unimplemented(
          "Types other than F16, F32, S32 and S64 are not implemented by "
          "StatelessRngUniform; got %s",
          xla::primitive_util::LowercasePrimitiveTypeName(type)));
  }
}

namespace {

xla::XlaOp StatelessRngUniformFullInt(absl::string_view device_type_string,
                                      xla::XlaOp seeds,
                                      const xla::Shape& shape) {
  xla::XlaBuilder* builder = seeds.builder();

  xla::XlaOp seed0 = xla::Reshape(xla::Slice(seeds, {0}, {1}, {1}), {});
  xla::XlaOp seed1 = xla::Reshape(xla::Slice(seeds, {1}, {2}, {1}), {});
  xla::XlaOp key = GetU64FromS32Seeds(seed0, seed1);
  xla::XlaOp initial_state = xla::ConstantR0WithType(builder, xla::U64, 0);
  xla::PrimitiveType type = shape.element_type();
  xla::RngOutput output =
      GetBitGeneratorForDevice(device_type_string)(key, initial_state, shape);
  switch (type) {
    case xla::U32:
    case xla::U64:
      return output.value;
    case xla::S32:
    case xla::S64:
      return BitcastConvertType(output.value, type);
    default:
      return builder->ReportError(xla::Unimplemented(
          "Types other than U32, S32, U64 and S64 are not implemented by "
          "StatelessRngUniformFullInt; got: %s",
          xla::primitive_util::LowercasePrimitiveTypeName(type)));
  }
}

class StatelessRandomUniformOp : public XlaOpKernel {
 public:
  explicit StatelessRandomUniformOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(
        ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
        absl::InvalidArgumentError(absl::StrCat(
            "seed must have shape [2], not ", seed_shape.DebugString())));
    xla::XlaOp seed = ctx->Input(1);

    auto rng_dtype = MaybeConvertBF16ToF32(dtype_);
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));
    xla::PrimitiveType rng_primitive_type = xla_shape.element_type();

    xla::XlaOp uniform = StatelessRngUniform(
        device_type_string_, seed, xla_shape,
        xla::ConstantR0WithType(builder, rng_primitive_type, 0.0),
        xla::ConstantR0WithType(builder, rng_primitive_type, 1.0));
    uniform = MaybeConvertF32ToBF16(uniform, dtype_);
    ctx->SetOutput(0, uniform);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  StatelessRandomUniformOp(const StatelessRandomUniformOp&) = delete;
  void operator=(const StatelessRandomUniformOp&) = delete;
};

// TODO(phawkins): generalize to non-float, non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomUniform")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype",
                                    {DT_DOUBLE, DT_FLOAT, DT_HALF, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomUniformOp);

class StatelessRandomUniformIntOp : public XlaOpKernel {
 public:
  explicit StatelessRandomUniformIntOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(
        ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
        absl::InvalidArgumentError(absl::StrCat(
            "seed must have shape [2], not ", seed_shape.DebugString())));
    TensorShape minval_shape = ctx->InputShape(2);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(minval_shape),
        absl::InvalidArgumentError(absl::StrCat(
            "minval must be scalar, got shape ", minval_shape.DebugString())));
    TensorShape maxval_shape = ctx->InputShape(3);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(maxval_shape),
        absl::InvalidArgumentError(absl::StrCat(
            "minval must be scalar, got shape ", maxval_shape.DebugString())));

    xla::XlaOp seed = ctx->Input(1);
    xla::XlaOp minval = ctx->Input(2);
    xla::XlaOp maxval = ctx->Input(3);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape, &xla_shape));
    xla::XlaOp uniform = StatelessRngUniform(device_type_string_, seed,
                                             xla_shape, minval, maxval);

    ctx->SetOutput(0, uniform);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  StatelessRandomUniformIntOp(const StatelessRandomUniformIntOp&) = delete;
  void operator=(const StatelessRandomUniformIntOp&) = delete;
};

// TODO(phawkins): generalize to non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomUniformInt")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_INT32, DT_INT64})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomUniformIntOp);

class StatelessRandomUniformFullIntOp : public XlaOpKernel {
 public:
  explicit StatelessRandomUniformFullIntOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(
        ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
        absl::InvalidArgumentError(absl::StrCat(
            "seed must have shape [2], not ", seed_shape.DebugString())));

    xla::XlaOp seed = ctx->Input(1);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape, &xla_shape));
    xla::XlaOp uniform =
        StatelessRngUniformFullInt(device_type_string_, seed, xla_shape);

    ctx->SetOutput(0, uniform);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  StatelessRandomUniformFullIntOp(const StatelessRandomUniformFullIntOp&) =
      delete;
  void operator=(const StatelessRandomUniformFullIntOp&) = delete;
};

// TODO(phawkins): generalize to non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomUniformFullInt")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_INT32, DT_INT64})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomUniformFullIntOp);

class StatelessRandomNormalOp : public XlaOpKernel {
 public:
  explicit StatelessRandomNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(
        ctx, seed_shape == TensorShape({2}),
        absl::InvalidArgumentError(absl::StrCat(
            "seed must have shape [2], not ", seed_shape.DebugString())));
    xla::XlaOp seed = ctx->Input(1);
    auto rng_dtype = MaybeConvertBF16ToF32(dtype_);
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));

    xla::XlaBuilder* builder = seed.builder();
    xla::XlaOp seed0 = xla::Reshape(xla::Slice(seed, {0}, {1}, {1}), {});
    xla::XlaOp seed1 = xla::Reshape(xla::Slice(seed, {1}, {2}, {1}), {});
    xla::XlaOp initial_state = xla::ConstantR0WithType(builder, xla::U64, 0);

    xla::XlaOp key = GetU64FromS32Seeds(seed0, seed1);
    xla::XlaOp normal =
        xla::NormalFloatingPointDistribution(
            key, initial_state, GetBitGeneratorForDevice(device_type_string_),
            xla_shape)
            .value;
    normal = MaybeConvertF32ToBF16(normal, dtype_);
    ctx->SetOutput(0, normal);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  StatelessRandomNormalOp(const StatelessRandomNormalOp&) = delete;
  void operator=(const StatelessRandomNormalOp&) = delete;
};

// TODO(phawkins): generalize to non-float, non-int32 seed types.
REGISTER_XLA_OP(Name("StatelessRandomNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype",
                                    {DT_DOUBLE, DT_FLOAT, DT_HALF, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessRandomNormalOp);

class StatelessTruncatedNormalOp : public XlaOpKernel {
 public:
  explicit StatelessTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(
        ctx, seed_shape == TensorShape({2}),
        absl::InvalidArgumentError(absl::StrCat(
            "seed must have shape [2], not ", seed_shape.DebugString())));
    xla::XlaOp seed = ctx->Input(1);
    xla::XlaBuilder* builder = ctx->builder();

    auto rng_dtype = MaybeConvertBF16ToF32(dtype_);
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));
    xla::XlaOp uniform = StatelessRngUniform(
        device_type_string_, seed, xla_shape,
        xla::MinPositiveNormalValue(builder, xla_shape.element_type()),
        xla::One(builder, xla_shape.element_type()));
    xla::XlaOp truncated_normal = TruncatedNormal(uniform);
    truncated_normal = MaybeConvertF32ToBF16(truncated_normal, dtype_);
    ctx->SetOutput(0, truncated_normal);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  StatelessTruncatedNormalOp(const StatelessTruncatedNormalOp&) = delete;
  void operator=(const StatelessTruncatedNormalOp&) = delete;
};

REGISTER_XLA_OP(Name("StatelessTruncatedNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype",
                                    {DT_DOUBLE, DT_FLOAT, DT_HALF, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessTruncatedNormalOp);

class StatelessParameterizedTruncatedNormalOp : public XlaOpKernel {
 public:
  explicit StatelessParameterizedTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(
        ctx, seed_shape == TensorShape({2}),
        absl::InvalidArgumentError(absl::StrCat(
            "seed must have shape [2], not ", seed_shape.DebugString())));
    xla::XlaOp seed = ctx->Input(1);
    xla::XlaBuilder* builder = ctx->builder();

    auto rng_dtype = MaybeConvertBF16ToF32(dtype_);
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));

    auto bcasted_means = BroadcastTo(ctx->Input(2), shape.dim_sizes());
    OP_REQUIRES_OK(ctx, bcasted_means.status());
    auto means = bcasted_means.value();

    auto bcasted_stddevs = BroadcastTo(ctx->Input(3), shape.dim_sizes());
    OP_REQUIRES_OK(ctx, bcasted_stddevs.status());
    auto stddevs = bcasted_stddevs.value();

    auto bcasted_minvals = BroadcastTo(ctx->Input(4), shape.dim_sizes());
    OP_REQUIRES_OK(ctx, bcasted_minvals.status());
    auto minvals = bcasted_minvals.value();

    auto bcasted_maxvals = BroadcastTo(ctx->Input(5), shape.dim_sizes());
    OP_REQUIRES_OK(ctx, bcasted_maxvals.status());
    auto maxvals = bcasted_maxvals.value();

    xla::XlaOp uniform = StatelessRngUniform(
        device_type_string_, seed, xla_shape,
        xla::MinPositiveNormalValue(builder, xla_shape.element_type()),
        xla::One(builder, xla_shape.element_type()));
    xla::XlaOp truncated_normal =
        ParameterizedTruncatedNormal(uniform, means, stddevs, minvals, maxvals);
    truncated_normal = MaybeConvertF32ToBF16(truncated_normal, dtype_);
    ctx->SetOutput(0, truncated_normal);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  StatelessParameterizedTruncatedNormalOp(
      const StatelessParameterizedTruncatedNormalOp&) = delete;
  void operator=(const StatelessParameterizedTruncatedNormalOp&) = delete;
};

REGISTER_XLA_OP(Name("StatelessParameterizedTruncatedNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype",
                                    {DT_DOUBLE, DT_FLOAT, DT_HALF, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessParameterizedTruncatedNormalOp);

}  // namespace
}  // namespace tensorflow
