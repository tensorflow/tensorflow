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

#include "tensorflow/core/kernels/stateless_random_ops_v2.h"

#include <cmath>

#include "tensorflow/compiler/tf2xla/kernels/random_ops_util.h"
#include "tensorflow/compiler/tf2xla/lib/random.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace tensorflow {

namespace {

inline xla::RandomAlgorithm TensorFlowRngAlgToXla(Algorithm const& alg) {
  if (alg == RNG_ALG_PHILOX) {
    return xla::RandomAlgorithm::RNG_PHILOX;
  } else if (alg == RNG_ALG_THREEFRY) {
    return xla::RandomAlgorithm::RNG_THREE_FRY;
  } else if (alg == RNG_ALG_AUTO_SELECT) {
    return xla::RandomAlgorithm::RNG_DEFAULT;
  }
  return xla::RandomAlgorithm::RNG_THREE_FRY;
}

inline Algorithm XlaRngAlgToTensorFlow(xla::RandomAlgorithm const& alg) {
  if (alg == xla::RandomAlgorithm::RNG_PHILOX) {
    return RNG_ALG_PHILOX;
  } else if (alg == xla::RandomAlgorithm::RNG_THREE_FRY) {
    return RNG_ALG_THREEFRY;
  } else if (alg == xla::RandomAlgorithm::RNG_DEFAULT) {
    return RNG_ALG_AUTO_SELECT;
  }
  return RNG_ALG_THREEFRY;
}

xla::XlaOp GetCounter(xla::RandomAlgorithm const& alg, xla::XlaOp state) {
  Algorithm alg_ = XlaRngAlgToTensorFlow(alg);
  return xla::Slice(state, {RNG_KEY_SIZE},
                    {RNG_KEY_SIZE + GetCounterSize(alg_)}, {1});
}

xla::RngOutput BitGenerator(xla::RandomAlgorithm const& alg, xla::XlaOp key,
                            xla::XlaOp counter, const xla::Shape& shape) {
  key = BitcastConvertType(key, xla::U64);
  counter = BitcastConvertType(counter, xla::U64);
  xla::XlaOp state = xla::ConcatInDim(key.builder(), {key, counter}, 0);
  xla::XlaOp result = xla::RngBitGenerator(alg, state, shape);
  auto new_counter = GetCounter(alg, xla::GetTupleElement(result, 0));
  new_counter = BitcastConvertType(new_counter, xla::S64);
  return xla::RngOutput{/*value=*/xla::GetTupleElement(result, 1),
                        /*state=*/new_counter};
}

std::tuple<xla::XlaOp, xla::XlaOp> GetKeyCounter(
    absl::string_view device_type_string, xla::XlaOp key) {
  // The Philox algorithm may cause performance regression on other devices.
  // Turn on the Philox algorithm for the CPU and GPU backends only.
  if (device_type_string == DEVICE_GPU_XLA_JIT ||
      device_type_string == DEVICE_CPU_XLA_JIT) {
    auto counter_key = xla::ScramblePhiloxKey(key);
    return std::make_tuple(counter_key.second, counter_key.first);
  } else {
    auto counter_shape =
        xla::ShapeUtil::MakeShape(xla::U64, {RNG_MAX_COUNTER_SIZE});
    auto counter = xla::Zeros(key.builder(), counter_shape);
    return std::make_tuple(key, counter);
  }
}

Algorithm DefaultRngAlgForDeviceType(absl::string_view device_type_string) {
  // The Philox algorithm may cause performance regression on other devices.
  // Turn on the Philox algorithm for the CPU and GPU backends only.
  if (device_type_string == DEVICE_GPU_XLA_JIT ||
      device_type_string == DEVICE_CPU_XLA_JIT) {
    return RNG_ALG_PHILOX;
  } else {
    return RNG_ALG_AUTO_SELECT;
  }
}

}  // namespace

xla::RngOutput StatelessRngUniformV2(xla::RandomAlgorithm const& alg,
                                   xla::XlaOp key, xla::XlaOp counter,
                                   const xla::Shape& shape, xla::XlaOp minval,
                                   xla::XlaOp maxval) {
  xla::XlaBuilder* builder = key.builder();
  xla::PrimitiveType type = shape.element_type();
  using std::placeholders::_1;
  using std::placeholders::_2;
  using std::placeholders::_3;
  auto generator = std::bind(BitGenerator, alg, _1, _2, _3);
  switch (type) {
    case xla::F32:
    case xla::F64:
      return xla::UniformFloatingPointDistribution(key, counter, generator,
                                                   minval, maxval, shape);
    case xla::S32:
    case xla::S64:
    case xla::U32:
    case xla::U64:
      return UniformIntDistribution(key, counter, generator, minval, maxval,
                                    shape);
      break;
    default:
      return {builder->ReportError(xla::Unimplemented(
                  "Types other than F32, S32, S64, U32 and U64 are not "
                  "implemented by "
                  "StatelessRngUniformV2; got %s",
                  xla::primitive_util::LowercasePrimitiveTypeName(type))),
              counter};
  }
}

namespace {

xla::RngOutput StatelessRngUniformFullInt(xla::RandomAlgorithm const& alg,
                                          xla::XlaOp key, xla::XlaOp counter,
                                          const xla::Shape& shape) {
  xla::XlaBuilder* builder = key.builder();

  xla::PrimitiveType type = shape.element_type();
  xla::RngOutput output = BitGenerator(alg, key, counter, shape);
  switch (type) {
    case xla::U32:
    case xla::U64:
      return output;
    case xla::S32:
    case xla::S64:
      return xla::RngOutput{BitcastConvertType(output.value, type),
                            output.state};
    default:
      return {
          builder->ReportError(xla::Unimplemented(
              "Types other than U32, S32, U64 and S64 are not implemented by "
              "StatelessRngUniformFullInt; got: %s",
              xla::primitive_util::LowercasePrimitiveTypeName(type))),
          output.state};
  }
}

Status AlgorithmFromInput(XlaOpKernelContext* ctx, int alg_input_idx,
                          absl::string_view device_type_string,
                          xla::RandomAlgorithm* xla_alg) {
  auto alg_shape = ctx->InputShape(alg_input_idx);
  if (alg_shape.dims() != 0) {
    return errors::InvalidArgument("algorithm must be of shape [], not ",
                                   alg_shape.DebugString());
  }
  xla::Literal alg_literal;
  TF_RETURN_IF_ERROR(ctx->ConstantInput(alg_input_idx, &alg_literal));
  auto alg = Algorithm(alg_literal.Get<int>({}));
  if (alg == RNG_ALG_AUTO_SELECT) {
    alg = DefaultRngAlgForDeviceType(device_type_string);
  }
  *xla_alg = TensorFlowRngAlgToXla(alg);
  return OkStatus();
}

xla::XlaOp MaybeSliceCounter(xla::RandomAlgorithm const& alg,
                             TensorShape const& counter_shape,
                             xla::XlaOp counter) {
  auto input_counter_size = counter_shape.dim_size(0);
  // TODO(wangpeng): We shouldn't rely on
  // `GetCounterSize(XlaRngAlgToTensorFlow(alg))` to decide the size when
  // alg==RNG_DEFAULT, which happens to give the correct answer. We should
  // define a "get counter size" function for xla::RandomAlgorithm, independent
  // of `GetCounterSize`.
  auto real_counter_size = GetCounterSize(XlaRngAlgToTensorFlow(alg));
  if (input_counter_size > real_counter_size) {
    counter = xla::Slice(counter, {0}, {real_counter_size}, {1});
  }
  return counter;
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

    const int key_input_idx = 1;
    const int counter_input_idx = 2;
    const int alg_input_idx = 3;
    xla::XlaOp key = ctx->Input(key_input_idx);
    xla::XlaOp counter = ctx->Input(counter_input_idx);

    xla::RandomAlgorithm alg;
    OP_REQUIRES_OK(
        ctx, AlgorithmFromInput(ctx, alg_input_idx, device_type_string_, &alg));

    auto counter_shape = ctx->InputShape(counter_input_idx);
    OP_REQUIRES_OK(ctx, CheckKeyCounterShape(XlaRngAlgToTensorFlow(alg),
                                             ctx->InputShape(key_input_idx),
                                             counter_shape));

    DataType rng_dtype = dtype_ == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));
    xla::PrimitiveType rng_primitive_type = xla_shape.element_type();

    counter = MaybeSliceCounter(alg, counter_shape, counter);

    auto result = StatelessRngUniformV2(
        alg, key, counter, xla_shape,
        xla::ConstantR0WithType(builder, rng_primitive_type, 0.0),
        xla::ConstantR0WithType(builder, rng_primitive_type, 1.0));
    auto uniform = MaybeConvertF32ToBF16(result.value, dtype_);
    ctx->SetOutput(0, uniform);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomUniformOp);
};

REGISTER_XLA_OP(Name("StatelessRandomUniformV2")
                    .CompileTimeConstantInput("shape")
                    .CompileTimeConstantInput("alg")
                    .TypeConstraint("dtype",
                                    {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16}),
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

    const int key_input_idx = 1;
    const int counter_input_idx = 2;
    const int alg_input_idx = 3;
    xla::XlaOp key = ctx->Input(key_input_idx);
    xla::XlaOp counter = ctx->Input(counter_input_idx);

    xla::RandomAlgorithm alg;
    OP_REQUIRES_OK(
        ctx, AlgorithmFromInput(ctx, alg_input_idx, device_type_string_, &alg));

    auto counter_shape = ctx->InputShape(counter_input_idx);
    OP_REQUIRES_OK(ctx, CheckKeyCounterShape(XlaRngAlgToTensorFlow(alg),
                                             ctx->InputShape(key_input_idx),
                                             counter_shape));

    const int minval_input_idx = 4;
    const int maxval_input_idx = 5;
    TensorShape minval_shape = ctx->InputShape(minval_input_idx);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval_shape),
                errors::InvalidArgument("minval must be scalar, got shape ",
                                        minval_shape.DebugString()));
    TensorShape maxval_shape = ctx->InputShape(maxval_input_idx);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval_shape),
                errors::InvalidArgument("maxval must be scalar, got shape ",
                                        maxval_shape.DebugString()));

    xla::XlaOp minval = ctx->Input(minval_input_idx);
    xla::XlaOp maxval = ctx->Input(maxval_input_idx);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape, &xla_shape));

    counter = MaybeSliceCounter(alg, counter_shape, counter);
    auto result =
        StatelessRngUniformV2(alg, key, counter, xla_shape, minval, maxval);
    ctx->SetOutput(0, result.value);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomUniformIntOp);
};

REGISTER_XLA_OP(Name("StatelessRandomUniformIntV2")
                    .CompileTimeConstantInput("shape")
                    .CompileTimeConstantInput("alg")
                    .TypeConstraint("dtype",
                                    {DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}),
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

    const int key_input_idx = 1;
    const int counter_input_idx = 2;
    const int alg_input_idx = 3;
    xla::XlaOp key = ctx->Input(key_input_idx);
    xla::XlaOp counter = ctx->Input(counter_input_idx);

    xla::RandomAlgorithm alg;
    OP_REQUIRES_OK(
        ctx, AlgorithmFromInput(ctx, alg_input_idx, device_type_string_, &alg));

    auto counter_shape = ctx->InputShape(counter_input_idx);
    OP_REQUIRES_OK(ctx, CheckKeyCounterShape(XlaRngAlgToTensorFlow(alg),
                                             ctx->InputShape(key_input_idx),
                                             counter_shape));

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape, &xla_shape));

    counter = MaybeSliceCounter(alg, counter_shape, counter);
    auto result = StatelessRngUniformFullInt(alg, key, counter, xla_shape);
    ctx->SetOutput(0, result.value);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomUniformFullIntOp);
};

REGISTER_XLA_OP(Name("StatelessRandomUniformFullIntV2")
                    .CompileTimeConstantInput("shape")
                    .CompileTimeConstantInput("alg")
                    .TypeConstraint("dtype",
                                    {DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}),
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

    const int key_input_idx = 1;
    const int counter_input_idx = 2;
    const int alg_input_idx = 3;
    xla::XlaOp key = ctx->Input(key_input_idx);
    xla::XlaOp counter = ctx->Input(counter_input_idx);

    xla::RandomAlgorithm alg;
    OP_REQUIRES_OK(
        ctx, AlgorithmFromInput(ctx, alg_input_idx, device_type_string_, &alg));

    auto counter_shape = ctx->InputShape(counter_input_idx);
    OP_REQUIRES_OK(ctx, CheckKeyCounterShape(XlaRngAlgToTensorFlow(alg),
                                             ctx->InputShape(key_input_idx),
                                             counter_shape));

    DataType rng_dtype = dtype_ == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    auto generator = std::bind(BitGenerator, alg, _1, _2, _3);
    counter = MaybeSliceCounter(alg, counter_shape, counter);
    auto result = xla::NormalFloatingPointDistribution(key, counter, generator,
                                                       xla_shape);
    auto normal = MaybeConvertF32ToBF16(result.value, dtype_);
    ctx->SetOutput(0, normal);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomNormalOp);
};

REGISTER_XLA_OP(Name("StatelessRandomNormalV2")
                    .CompileTimeConstantInput("shape")
                    .CompileTimeConstantInput("alg")
                    .TypeConstraint("dtype",
                                    {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16}),
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

    const int key_input_idx = 1;
    const int counter_input_idx = 2;
    const int alg_input_idx = 3;
    xla::XlaOp key = ctx->Input(key_input_idx);
    xla::XlaOp counter = ctx->Input(counter_input_idx);

    xla::RandomAlgorithm alg;
    OP_REQUIRES_OK(
        ctx, AlgorithmFromInput(ctx, alg_input_idx, device_type_string_, &alg));

    auto counter_shape = ctx->InputShape(counter_input_idx);
    OP_REQUIRES_OK(ctx, CheckKeyCounterShape(XlaRngAlgToTensorFlow(alg),
                                             ctx->InputShape(key_input_idx),
                                             counter_shape));

    xla::XlaBuilder* builder = ctx->builder();

    DataType rng_dtype = dtype_ == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));

    counter = MaybeSliceCounter(alg, counter_shape, counter);
    auto result = StatelessRngUniformV2(
        alg, key, counter, xla_shape,
        xla::MinPositiveNormalValue(builder, xla_shape.element_type()),
        xla::One(builder, xla_shape.element_type()));
    xla::XlaOp truncated_normal = TruncatedNormal(result.value);
    truncated_normal = MaybeConvertF32ToBF16(truncated_normal, dtype_);
    ctx->SetOutput(0, truncated_normal);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessTruncatedNormalOp);
};

REGISTER_XLA_OP(Name("StatelessTruncatedNormalV2")
                    .CompileTimeConstantInput("shape")
                    .CompileTimeConstantInput("alg")
                    .TypeConstraint("dtype",
                                    {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16}),
                StatelessTruncatedNormalOp);

class GetKeyCounterAlgOp : public XlaOpKernel {
 public:
  explicit GetKeyCounterAlgOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {}

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape seed_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, seed_shape == TensorShape({2}),
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    xla::XlaOp seed = ctx->Input(0);

    xla::XlaBuilder* builder = seed.builder();
    xla::XlaOp seed0 = xla::Reshape(xla::Slice(seed, {0}, {1}, {1}), {});
    xla::XlaOp seed1 = xla::Reshape(xla::Slice(seed, {1}, {2}, {1}), {});
    xla::XlaOp key = GetU64FromS32Seeds(seed0, seed1);
    auto key_counter = GetKeyCounter(device_type_string_, key);
    key = std::get<0>(key_counter);
    auto counter = std::get<1>(key_counter);
    auto alg = DefaultRngAlgForDeviceType(device_type_string_);
    key = xla::Reshape(key, {RNG_KEY_SIZE});
    ctx->SetOutput(0, key);
    ctx->SetOutput(1, counter);
    ctx->SetOutput(2, ConstantR0(builder, static_cast<int>(alg)));
  }

 private:
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(GetKeyCounterAlgOp);
};

// TODO(hinsu): Dis-allow unsupported int64 seed types.
REGISTER_XLA_OP(Name("StatelessRandomGetKeyCounterAlg"), GetKeyCounterAlgOp);

class GetKeyCounterOp : public XlaOpKernel {
 public:
  explicit GetKeyCounterOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {}

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape seed_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, seed_shape == TensorShape({2}),
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    xla::XlaOp seed = ctx->Input(0);

    xla::XlaOp seed0 = xla::Reshape(xla::Slice(seed, {0}, {1}, {1}), {});
    xla::XlaOp seed1 = xla::Reshape(xla::Slice(seed, {1}, {2}, {1}), {});
    xla::XlaOp key = GetU64FromS32Seeds(seed0, seed1);
    auto key_counter = GetKeyCounter(device_type_string_, key);
    key = std::get<0>(key_counter);
    auto counter = std::get<1>(key_counter);
    key = xla::Reshape(key, {RNG_KEY_SIZE});
    ctx->SetOutput(0, key);
    ctx->SetOutput(1, counter);
  }

 private:
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(GetKeyCounterOp);
};

// TODO(hinsu): Dis-allow unsupported int64 seed types.
REGISTER_XLA_OP(Name("StatelessRandomGetKeyCounter"), GetKeyCounterOp);

class GetAlgOp : public XlaOpKernel {
 public:
  explicit GetAlgOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {}

  void Compile(XlaOpKernelContext* ctx) override {
    auto alg = DefaultRngAlgForDeviceType(device_type_string_);
    auto builder = ctx->builder();
    ctx->SetOutput(0, ConstantR0(builder, static_cast<int>(alg)));
  }

 private:
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(GetAlgOp);
};

REGISTER_XLA_OP(Name("StatelessRandomGetAlg"), GetAlgOp);

REGISTER_XLA_OP(Name("XlaRngBitGenerator")
                    .CompileTimeConstantInput("algorithm")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_UINT32, DT_UINT64}),
                MlirXlaOpKernel);

}  // namespace
}  // namespace tensorflow
