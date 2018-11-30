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

#include "absl/base/casts.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace {

// Rotates a 32-bit integer 'v' left by 'distance' bits.
XlaOp RotateLeftU32(XlaOp v, int distance) {
  return (v << ConstantR0<uint32>(v.builder(), distance)) |
         ShiftRightLogical(v, ConstantR0<uint32>(v.builder(), 32 - distance));
}

using ThreeFry2x32State = std::array<XlaOp, 2>;

// Implements the ThreeFry counter-based PRNG algorithm.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
ThreeFry2x32State ThreeFry2x32(ThreeFry2x32State input, ThreeFry2x32State key) {
  XlaBuilder* builder = input[0].builder();
  key[0] = BitcastConvertType(key[0], U32);
  key[1] = BitcastConvertType(key[1], U32);

  // Rotation distances specified by the Threefry2x32 algorithm.
  constexpr std::array<int, 8> rotations = {13, 15, 26, 6, 17, 29, 16, 24};
  ThreeFry2x32State x;

  std::array<XlaOp, 3> ks;
  // 0x1BD11BDA is a parity constant specified by the ThreeFry2x32 algorithm.
  ks[2] = ConstantR0<uint32>(builder, 0x1BD11BDA);
  for (int i = 0; i < 2; ++i) {
    ks[i] = key[i];
    x[i] = input[i];
    ks[2] = ks[2] ^ key[i];
  }

  x[0] = x[0] + ks[0];
  x[1] = x[1] + ks[1];

  // Performs a single round of the Threefry2x32 algorithm, with a rotation
  // amount 'rotation'.
  auto round = [](ThreeFry2x32State v, int rotation) {
    v[0] = v[0] + v[1];
    v[1] = RotateLeftU32(v[1], rotation);
    v[1] = v[0] ^ v[1];
    return v;
  };

  // There are no known statistical flaws with 13 rounds of Threefry2x32.
  // We are conservative and use 20 rounds.
  x = round(x, rotations[0]);
  x = round(x, rotations[1]);
  x = round(x, rotations[2]);
  x = round(x, rotations[3]);
  x[0] = x[0] + ks[1];
  x[1] = x[1] + ks[2] + ConstantR0<uint32>(builder, 1);

  x = round(x, rotations[4]);
  x = round(x, rotations[5]);
  x = round(x, rotations[6]);
  x = round(x, rotations[7]);
  x[0] = x[0] + ks[2];
  x[1] = x[1] + ks[0] + ConstantR0<uint32>(builder, 2);

  x = round(x, rotations[0]);
  x = round(x, rotations[1]);
  x = round(x, rotations[2]);
  x = round(x, rotations[3]);
  x[0] = x[0] + ks[0];
  x[1] = x[1] + ks[1] + ConstantR0<uint32>(builder, 3);

  x = round(x, rotations[4]);
  x = round(x, rotations[5]);
  x = round(x, rotations[6]);
  x = round(x, rotations[7]);
  x[0] = x[0] + ks[1];
  x[1] = x[1] + ks[2] + ConstantR0<uint32>(builder, 4);

  x = round(x, rotations[0]);
  x = round(x, rotations[1]);
  x = round(x, rotations[2]);
  x = round(x, rotations[3]);
  x[0] = x[0] + ks[2];
  x[1] = x[1] + ks[0] + ConstantR0<uint32>(builder, 5);

  return x;
}

// Returns the inputs with unique counter values for ThreeFry2x32.
ThreeFry2x32State GetInputs(const int64 size, XlaBuilder* builder) {
  ThreeFry2x32State inputs;
  inputs[0] = Iota(builder, U32, size);
  inputs[1] = inputs[0] + ConstantR0<uint32>(builder, size);
  return inputs;
}

XlaOp StatelessRngUniformU32(std::array<XlaOp, 2> key, const Shape& shape) {
  XlaBuilder* builder = key[0].builder();
  const int64 size = ShapeUtil::ElementsIn(shape);
  const int64 half_size = CeilOfRatio<int64>(size, 2);
  const bool size_is_odd = (half_size * 2 != size);
  ThreeFry2x32State inputs = GetInputs(half_size, builder);
  ThreeFry2x32State outputs = ThreeFry2x32(inputs, key);
  if (size_is_odd) {
    outputs[1] = Slice(outputs[1], {0}, {half_size - 1}, {1});
  }
  auto result = ConcatInDim(builder, outputs, 0);
  return Reshape(result, AsInt64Slice(shape.dimensions()));
}

XlaOp StatelessRngUniformU64(std::array<XlaOp, 2> key, const Shape& shape) {
  XlaBuilder* builder = key[0].builder();
  const int64 size = ShapeUtil::ElementsIn(shape);
  ThreeFry2x32State inputs = GetInputs(size, builder);
  ThreeFry2x32State outputs = ThreeFry2x32(inputs, key);
  // low 32 bit: outputs[0], high 32 bit: outputs[1]
  auto result = ConvertElementType(outputs[0], U64) |
                ShiftLeft(ConvertElementType(outputs[1], U64),
                          ConstantR0WithType(builder, U64, 32));
  return Reshape(result, AsInt64Slice(shape.dimensions()));
}

XlaOp StatelessRngUniformF32(XlaOp bits, XlaOp minval, XlaOp maxval) {
  XlaBuilder* builder = bits.builder();

  // Form 23 random mantissa bits, with a leading 1 bit. The leading 1 bit
  // forces the random bits into the mantissa.
  constexpr int kFloatBits = 32;
  constexpr int kMantissaBits = 23;
  bits = ShiftRightLogical(
             bits, ConstantR0<uint32>(builder, kFloatBits - kMantissaBits)) |
         ConstantR0<uint32>(builder, absl::bit_cast<uint32>(1.0f));
  auto floats = BitcastConvertType(bits, F32);

  // We have a floating point number in the range [1.0, 2.0).
  // Subtract 1.0f to shift to the range [0.0, 1.0)
  floats = floats - ConstantR0<float>(builder, 1.0f);
  // Multiply and add to shift to the range [minval, maxval).
  return floats * (maxval - minval) + minval;
}

XlaOp StatelessRngUniformInt(XlaOp bits, XlaOp minval, XlaOp maxval,
                             PrimitiveType type, PrimitiveType unsigned_type) {
  XlaBuilder* builder = bits.builder();
  // TODO(b/72573764): Generate real uniform integer distribution.
  // The following algorithm is the same one that TF uses right now, but it's
  // uniform only when maxval - minval is a divisor of the range that bits is
  // generated from.
  auto range = BitcastConvertType(maxval, unsigned_type) -
               BitcastConvertType(minval, unsigned_type);
  auto dist = Rem(bits, range);
  auto dist_div_2 =
      ShiftRightLogical(dist, ConstantR0WithType(builder, unsigned_type, 1));

  return minval + BitcastConvertType(dist_div_2, type) +
         BitcastConvertType(dist - dist_div_2, type);
}

}  // namespace

XlaOp StatelessRngUniform(std::array<XlaOp, 2> seeds, const Shape& shape,
                          XlaOp minval, XlaOp maxval) {
  XlaBuilder* builder = seeds[0].builder();
  PrimitiveType type = shape.element_type();
  switch (type) {
    case F32: {
      auto bits = StatelessRngUniformU32(seeds, shape);
      return StatelessRngUniformF32(bits, minval, maxval);
    }
    case S32: {
      auto bits = StatelessRngUniformU32(seeds, shape);
      return StatelessRngUniformInt(bits, minval, maxval, type, U32);
    }
    case S64: {
      auto bits = StatelessRngUniformU64(seeds, shape);
      return StatelessRngUniformInt(bits, minval, maxval, type, U64);
    }
    default:
      return builder->ReportError(Unimplemented(
          "Types other than F32, S32 and S64 are not implemented by "
          "StatelessRngUniform."));
  }
}

}  // namespace xla
