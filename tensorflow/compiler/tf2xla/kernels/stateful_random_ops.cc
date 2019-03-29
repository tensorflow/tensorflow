/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/stateful_random_ops.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace tensorflow {
namespace {

std::pair<xla::ThreeFry2x32State, xla::XlaOp> GetInputsFromCounter(
    xla::XlaOp counter, const int64 size) {
  auto builder = counter.builder();
  auto input_u64 = Iota(builder, xla::U64, size);
  input_u64 = input_u64 + counter;
  counter = counter + xla::ConstantR0<uint64>(builder, size);
  return std::make_pair(xla::Uint64ToUint32s(input_u64), counter);
}

// `StatelessRngUniformU32` uses ThreeFry2x32’s counter space too
// wastefully, only able to generate 2^32*2 int32 numbers for each key, while
// the real capacity is 2^64*2. Counter-space efficiency is important for
// stateful ops, hence the following 2 new functions.
std::pair<xla::XlaOp, xla::XlaOp> StatefulRngUniformU32(
    xla::XlaOp key, xla::XlaOp counter, const xla::Shape& shape) {
  auto builder = key.builder();
  const int64 size = xla::ShapeUtil::ElementsIn(shape);
  const int64 half_size = xla::CeilOfRatio<int64>(size, 2);
  const bool size_is_odd = (half_size * 2 != size);
  auto inputs_counter = GetInputsFromCounter(counter, half_size);
  auto inputs = inputs_counter.first;
  counter = inputs_counter.second;
  auto outputs = xla::ThreeFry2x32(inputs, xla::Uint64ToUint32s(key));
  if (size_is_odd) {
    outputs[1] = Slice(outputs[1], {0}, {half_size - 1}, {1});
  }
  auto result = ConcatInDim(builder, outputs, 0);
  return std::make_pair(Reshape(result, xla::AsInt64Slice(shape.dimensions())),
                        counter);
}

std::pair<xla::XlaOp, xla::XlaOp> StatefulRngUniformU64(
    xla::XlaOp key, xla::XlaOp counter, const xla::Shape& shape) {
  const int64 size = xla::ShapeUtil::ElementsIn(shape);
  auto inputs_counter = GetInputsFromCounter(counter, size);
  auto inputs = inputs_counter.first;
  counter = inputs_counter.second;
  auto outputs = ThreeFry2x32(inputs, Uint64ToUint32s(key));
  auto result = Uint32sToUint64(outputs);
  return std::make_pair(Reshape(result, xla::AsInt64Slice(shape.dimensions())),
                        counter);
}

std::pair<xla::XlaOp, xla::XlaOp> StatefulRngUniform(xla::XlaOp key,
                                                     xla::XlaOp counter,
                                                     const xla::Shape& shape,
                                                     xla::XlaOp minval,
                                                     xla::XlaOp maxval) {
  auto builder = key.builder();
  xla::PrimitiveType type = shape.element_type();
  switch (type) {
    case xla::F32: {
      auto bits_counter = StatefulRngUniformU32(key, counter, shape);
      auto bits = bits_counter.first;
      counter = bits_counter.second;
      return std::make_pair(xla::StatelessRngUniformF32(bits, minval, maxval),
                            counter);
    }
    case xla::U32:  // fall through
    case xla::S32: {
      auto bits_counter = StatefulRngUniformU32(key, counter, shape);
      auto bits = bits_counter.first;
      counter = bits_counter.second;
      return std::make_pair(
          xla::StatelessRngUniformInt(bits, minval, maxval, type, xla::U32),
          counter);
    }
    case xla::U64:  // fall through
    case xla::S64: {
      auto bits_counter = StatefulRngUniformU64(key, counter, shape);
      auto bits = bits_counter.first;
      counter = bits_counter.second;
      return std::make_pair(
          xla::StatelessRngUniformInt(bits, minval, maxval, type, xla::U64),
          counter);
    }
    default:
      return std::make_pair(
          builder->ReportError(xla::Unimplemented(
              "Types other than F32, U32, S32, U64 and S64 "
              "are not implemented by "
              "StatefulRngUniform; got: %s",
              xla::primitive_util::LowercasePrimitiveTypeName(type))),
          counter);
  }
}

template <typename A, typename B, typename A2>
std::pair<A2, B> map_first(std::function<A2(A)> f, std::pair<A, B> p) {
  return std::make_pair(f(p.first), p.second);
}

std::pair<xla::XlaOp, xla::XlaOp> StatefulRngUniformFullInt(
    xla::XlaOp key, xla::XlaOp counter, const xla::Shape& shape) {
  xla::PrimitiveType type = shape.element_type();
  switch (type) {
    case xla::U32:
      return StatefulRngUniformU32(key, counter, shape);
    case xla::S32: {
      // Needs explicit function type because of type-inference failure.
      std::function<xla::XlaOp(xla::XlaOp)> f = [](xla::XlaOp x) {
        return BitcastConvertType(x, xla::S32);
      };
      return map_first(f, StatefulRngUniformU32(key, counter, shape));
    }
    case xla::U64:
      return StatefulRngUniformU64(key, counter, shape);
    case xla::S64: {
      std::function<xla::XlaOp(xla::XlaOp)> f = [](xla::XlaOp x) {
        return BitcastConvertType(x, xla::S64);
      };
      return map_first(f, StatefulRngUniformU64(key, counter, shape));
    }
    default:
      auto builder = key.builder();
      return std::make_pair(
          builder->ReportError(xla::Unimplemented(
              "Types other than U32, S32, U64 and S64 are not implemented by "
              "StatefulRngUniformFullInt; got: %s",
              xla::primitive_util::LowercasePrimitiveTypeName(type))),
          counter);
  }
}

template <typename ListB, typename ListA, typename F>
ListB Map(F f, ListA const& list_a) {
  ListB list_b;
  for (auto a : list_a) {
    list_b.push_back(f(a));
  }
  return list_b;
}

xla::XlaOp ConcatScalars(xla::XlaBuilder* builder,
                         absl::Span<const xla::XlaOp> scalars) {
  return ConcatInDim(
      builder,
      Map<std::vector<xla::XlaOp>>(
          [](xla::XlaOp x) { return xla::Reshape(x, {1}); }, scalars),
      0);
}

using sampler_return_type = xla::StatusOr<std::pair<xla::XlaOp, xla::XlaOp>>;

// A helper function containing the common part of several kernels below.
// Precondition: 'algorithm' and 'shape' are compile-time constants.
Status CompileImpl(XlaOpKernelContext* ctx, int state_input_idx,
                   int alg_input_idx, int shape_input_idx,
                   std::function<sampler_return_type(xla::XlaOp, xla::XlaOp,
                                                     TensorShape)> const&
                       sample_with_threefry) {
  auto alg_shape = ctx->InputShape(alg_input_idx);
  if (alg_shape.dims() != 0) {
    return errors::InvalidArgument("algorithm must be of shape [], not ",
                                   alg_shape.DebugString());
  }
  xla::Literal alg_literal;
  TF_RETURN_IF_ERROR(ctx->ConstantInput(alg_input_idx, &alg_literal));
  auto alg = alg_literal.Get<Algorithm>({});

  if (alg == RNG_ALG_THREEFRY) {
    xla::XlaOp var;
    TensorShape var_shape;
    TF_RETURN_IF_ERROR(ctx->ReadVariableInput(
        state_input_idx, STATE_ELEMENT_DTYPE, &var_shape, &var));
    if (var_shape.dims() != 1) {
      return errors::InvalidArgument(
          "RNG state must have one and only one dimension, not ",
          var_shape.dims());
    }
    auto state_size = var_shape.dim_size(0);
    if (state_size < THREEFRY_MIN_STATE_SIZE) {
      return errors::InvalidArgument(
          "For the ThreeFry algorithm, the size of state"
          " must be at least ",
          THREEFRY_MIN_STATE_SIZE, "; got ", state_size);
    }
    TensorShape shape;
    TF_RETURN_IF_ERROR(ctx->ConstantInputAsShape(shape_input_idx, &shape));

    static constexpr int COUNTER_SIZE = 1;
    auto counter = BitcastConvertType(
        xla::Reshape(xla::Slice(var, {0}, {COUNTER_SIZE}, {1}), {}), xla::U64);
    auto key = BitcastConvertType(
        xla::Reshape(xla::Slice(var, {COUNTER_SIZE}, {COUNTER_SIZE + 1}, {1}),
                     {}),
        xla::U64);

    auto status_or_value = sample_with_threefry(counter, key, shape);
    if (!status_or_value.ok()) {
      return status_or_value.status();
    }
    auto output_counter = status_or_value.ConsumeValueOrDie();
    auto output = output_counter.first;
    counter = output_counter.second;
    ctx->SetOutput(0, output);
    auto builder = ctx->builder();
    var = ConcatScalars(builder, {counter, key});
    xla::PrimitiveType state_element_type;
    TF_RETURN_IF_ERROR(
        DataTypeToPrimitiveType(STATE_ELEMENT_DTYPE, &state_element_type));
    var = BitcastConvertType(var, state_element_type);
    TF_RETURN_IF_ERROR(
        ctx->AssignVariable(state_input_idx, STATE_ELEMENT_DTYPE, var));
    return Status::OK();
  } else {
    return errors::InvalidArgument("Unsupported algorithm id: ", alg);
  }
}

class StatefulUniformOp : public XlaOpKernel {
 public:
  explicit StatefulUniformOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    auto sample_with_threefry = [builder, this](
                                    xla::XlaOp counter, xla::XlaOp key,
                                    TensorShape shape) -> sampler_return_type {
      xla::Shape xla_shape;
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(DT_FLOAT, shape, &xla_shape));
      auto uniform_counter = StatefulRngUniform(
          key, counter, xla_shape, xla::ConstantR0<float>(builder, 0.0),
          xla::ConstantR0<float>(builder, 1.0));
      auto uniform = uniform_counter.first;
      counter = uniform_counter.second;
      uniform = MaybeConvertF32ToBF16(uniform, dtype_);
      return {{uniform, counter}};
    };
    OP_REQUIRES_OK(ctx,
                   CompileImpl(ctx, /*state_input_idx=*/0, /*alg_input_idx=*/1,
                               /*shape_input_idx=*/2, sample_with_threefry));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatefulUniformOp);
};

// TODO(wangpeng): Support plain float16 and float64 to get rid of the
//   `TypeConstraint`.
REGISTER_XLA_OP(Name("StatefulUniform")
                    .CompileTimeConstantInput("algorithm")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_BFLOAT16}),
                StatefulUniformOp);

class StatefulStandardNormalOp : public XlaOpKernel {
 public:
  explicit StatefulStandardNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    auto sample_with_threefry =
        // Needs explicit lambda return type because it fails to be inferred.
        [builder, this](xla::XlaOp counter, xla::XlaOp key,
                        TensorShape shape) -> sampler_return_type {
      xla::Shape xla_shape;
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(DT_FLOAT, shape, &xla_shape));

      auto uniform_counter = StatefulRngUniform(
          key, counter, xla_shape,
          xla::ConstantR0<float>(builder, std::nextafter(-1.0f, 0.0f)),
          xla::ConstantR0<float>(builder, 1.0));
      auto uniform = uniform_counter.first;
      counter = uniform_counter.second;
      // Convert uniform distribution to normal distribution by computing
      // sqrt(2) * erfinv(x)
      auto normal =
          xla::ScalarLike(uniform, std::sqrt(2.0)) * xla::ErfInv(uniform);
      normal = MaybeConvertF32ToBF16(normal, dtype_);
      return {{normal, counter}};
    };
    OP_REQUIRES_OK(ctx,
                   CompileImpl(ctx, /*state_input_idx=*/0, /*alg_input_idx=*/1,
                               /*shape_input_idx=*/2, sample_with_threefry));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatefulStandardNormalOp);
};

// TODO(wangpeng): Support plain float16 and float64 to get rid of the
//   `TypeConstraint`.
REGISTER_XLA_OP(Name("StatefulStandardNormalV2")
                    .CompileTimeConstantInput("algorithm")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_BFLOAT16}),
                StatefulStandardNormalOp);

class StatefulTruncatedNormalOp : public XlaOpKernel {
 public:
  explicit StatefulTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    auto sample_with_threefry =
        // Needs explicit lambda return type because it fails to be inferred.
        [builder, this](xla::XlaOp counter, xla::XlaOp key,
                        TensorShape shape) -> sampler_return_type {
      xla::Shape xla_shape;
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(DT_FLOAT, shape, &xla_shape));

      auto uniform_counter = StatefulRngUniform(
          key, counter, xla_shape,
          xla::MinPositiveNormalValue(builder, xla_shape.element_type()),
          xla::One(builder, xla_shape.element_type()));
      auto uniform = uniform_counter.first;
      counter = uniform_counter.second;
      xla::XlaOp truncated_normal = TruncatedNormal(uniform);
      truncated_normal = MaybeConvertF32ToBF16(truncated_normal, dtype_);
      return {{truncated_normal, counter}};
    };
    OP_REQUIRES_OK(ctx,
                   CompileImpl(ctx, /*state_input_idx=*/0, /*alg_input_idx=*/1,
                               /*shape_input_idx=*/2, sample_with_threefry));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatefulTruncatedNormalOp);
};

// TODO(wangpeng): Support plain float16 and float64 to get rid of the
//   `TypeConstraint`.
REGISTER_XLA_OP(Name("StatefulTruncatedNormal")
                    .CompileTimeConstantInput("algorithm")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_BFLOAT16}),
                StatefulTruncatedNormalOp);

class StatefulUniformIntOp : public XlaOpKernel {
 public:
  explicit StatefulUniformIntOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp minval = ctx->Input(3);
    xla::XlaOp maxval = ctx->Input(4);
    auto sample_with_threefry = [minval, maxval, this](
                                    xla::XlaOp counter, xla::XlaOp key,
                                    TensorShape shape) -> sampler_return_type {
      xla::Shape xla_shape;
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype_, shape, &xla_shape));
      return StatefulRngUniform(key, counter, xla_shape, minval, maxval);
    };
    OP_REQUIRES_OK(ctx,
                   CompileImpl(ctx, /*state_input_idx=*/0, /*alg_input_idx=*/1,
                               /*shape_input_idx=*/2, sample_with_threefry));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatefulUniformIntOp);
};

REGISTER_XLA_OP(Name("StatefulUniformInt")
                    .CompileTimeConstantInput("algorithm")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype",
                                    {DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}),
                StatefulUniformIntOp);

class StatefulUniformFullIntOp : public XlaOpKernel {
 public:
  explicit StatefulUniformFullIntOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto sample_with_threefry = [this](
                                    xla::XlaOp counter, xla::XlaOp key,
                                    TensorShape shape) -> sampler_return_type {
      xla::Shape xla_shape;
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype_, shape, &xla_shape));
      return StatefulRngUniformFullInt(key, counter, xla_shape);
    };
    OP_REQUIRES_OK(ctx,
                   CompileImpl(ctx, /*state_input_idx=*/0, /*alg_input_idx=*/1,
                               /*shape_input_idx=*/2, sample_with_threefry));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatefulUniformFullIntOp);
};

REGISTER_XLA_OP(Name("StatefulUniformFullInt")
                    .CompileTimeConstantInput("algorithm")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype",
                                    {DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}),
                StatefulUniformFullIntOp);

}  // namespace
}  // namespace tensorflow
