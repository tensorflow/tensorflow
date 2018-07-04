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
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace tensorflow {
namespace {

// Rotates a 32-bit integer 'v' left by 'distance' bits.
xla::XlaOp RotateLeftS32(xla::XlaBuilder* builder, const xla::XlaOp& v,
                         int distance) {
  return xla::Or(
      xla::ShiftLeft(v, xla::ConstantR0<int>(builder, distance)),
      xla::ShiftRightLogical(v, xla::ConstantR0<int>(builder, 32 - distance)));
}

using ThreeFry2x32State = std::array<xla::XlaOp, 2>;

// Implements the ThreeFry counter-based PRNG algorithm.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
ThreeFry2x32State ThreeFry2x32(xla::XlaBuilder* builder,
                               ThreeFry2x32State input, ThreeFry2x32State key) {
  // Rotation distances specified by the Threefry2x32 algorithm.
  constexpr std::array<int, 8> rotations = {13, 15, 26, 6, 17, 29, 16, 24};
  ThreeFry2x32State x;

  std::array<xla::XlaOp, 3> ks;
  // 0x1BD11BDA is a parity constant specified by the ThreeFry2x32 algorithm.
  ks[2] = xla::ConstantR0<int32>(builder, 0x1BD11BDA);
  for (int i = 0; i < 2; ++i) {
    ks[i] = key[i];
    x[i] = input[i];
    ks[2] = xla::Xor(ks[2], key[i]);
  }

  x[0] = xla::Add(x[0], ks[0]);
  x[1] = xla::Add(x[1], ks[1]);

  // Performs a single round of the Threefry2x32 algorithm, with a rotation
  // amount 'rotation'.
  auto round = [builder](ThreeFry2x32State v, int rotation) {
    v[0] = xla::Add(v[0], v[1]);
    v[1] = RotateLeftS32(builder, v[1], rotation);
    v[1] = xla::Xor(v[0], v[1]);
    return v;
  };

  // There are no known statistical flaws with 13 rounds of Threefry2x32.
  // We are conservative and use 20 rounds.
  x = round(x, rotations[0]);
  x = round(x, rotations[1]);
  x = round(x, rotations[2]);
  x = round(x, rotations[3]);
  x[0] = xla::Add(x[0], ks[1]);
  x[1] = xla::Add(xla::Add(x[1], ks[2]), xla::ConstantR0<int32>(builder, 1));

  x = round(x, rotations[4]);
  x = round(x, rotations[5]);
  x = round(x, rotations[6]);
  x = round(x, rotations[7]);
  x[0] = xla::Add(x[0], ks[2]);
  x[1] = xla::Add(xla::Add(x[1], ks[0]), xla::ConstantR0<int32>(builder, 2));

  x = round(x, rotations[0]);
  x = round(x, rotations[1]);
  x = round(x, rotations[2]);
  x = round(x, rotations[3]);
  x[0] = xla::Add(x[0], ks[0]);
  x[1] = xla::Add(xla::Add(x[1], ks[1]), xla::ConstantR0<int32>(builder, 3));

  x = round(x, rotations[4]);
  x = round(x, rotations[5]);
  x = round(x, rotations[6]);
  x = round(x, rotations[7]);
  x[0] = xla::Add(x[0], ks[1]);
  x[1] = xla::Add(xla::Add(x[1], ks[2]), xla::ConstantR0<int32>(builder, 4));

  x = round(x, rotations[0]);
  x = round(x, rotations[1]);
  x = round(x, rotations[2]);
  x = round(x, rotations[3]);
  x[0] = xla::Add(x[0], ks[2]);
  x[1] = xla::Add(xla::Add(x[1], ks[0]), xla::ConstantR0<int32>(builder, 5));

  return x;
}

// Returns a tensor of 'shape' random values uniformly distributed in the range
// [minval, maxval)
xla::XlaOp RandomUniform(xla::XlaBuilder* builder, const xla::XlaOp& seed,
                         const TensorShape& shape, double minval,
                         double maxval) {
  // Split the seed into two 32-bit scalars to form a key.
  auto seed0 = xla::Reshape(xla::Slice(seed, {0}, {1}, {1}), {});
  auto seed1 = xla::Reshape(xla::Slice(seed, {1}, {2}, {1}), {});
  ThreeFry2x32State key = {seed0, seed1};
  const int64 size = shape.num_elements();

  const int64 half_size = MathUtil::CeilOfRatio<int64>(size, 2);
  const bool size_is_odd = (half_size * 2 != size);

  // Fill the generator inputs with unique counter values.
  ThreeFry2x32State inputs;
  inputs[0] = xla::Iota(builder, xla::S32, half_size);
  inputs[1] = xla::Add(inputs[0], xla::ConstantR0<int32>(builder, half_size));
  ThreeFry2x32State outputs = ThreeFry2x32(builder, inputs, key);

  if (size_is_odd) {
    outputs[1] = xla::Slice(outputs[1], {0}, {half_size - 1}, {1});
  }

  auto bits =
      xla::Reshape(xla::ConcatInDim(builder, outputs, 0), shape.dim_sizes());

  // Form 22 random mantissa bits, with a leading 1 bit. The leading 1 bit
  // forces the random bits into the mantissa.
  constexpr int kFloatBits = 32;
  constexpr int kMantissaBits = 23;
  bits = xla::Or(
      xla::ShiftRightLogical(
          bits, xla::ConstantR0<int32>(builder, kFloatBits - kMantissaBits)),
      xla::ConstantR0<int32>(builder, bit_cast<int32>(1.0f)));
  auto floats = xla::BitcastConvertType(bits, xla::F32);

  // We have a floating point number in the range [1.0, 2.0).
  // Subtract 1.0f to shift to the range [0.0, 1.0)
  floats = xla::Sub(floats, xla::ConstantR0<float>(builder, 1.0f));
  // Multiply and add to shift to the range [minval, maxval).
  floats = xla::Mul(floats, xla::ConstantR0<float>(builder, maxval - minval));
  floats = xla::Add(floats, xla::ConstantR0<float>(builder, minval));
  return floats;
}

}  // namespace

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
    ctx->SetOutput(0, RandomUniform(builder, seed, shape, 0.0, 1.0));
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
    auto uniform =
        RandomUniform(builder, seed, shape, std::nextafter(-1.0f, 0.0f), 1.0);
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
    xla::XlaBuilder* b = ctx->builder();

    auto uniform =
        RandomUniform(b, seed, shape, std::numeric_limits<float>::min(), 1.0);
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

}  // namespace tensorflow
