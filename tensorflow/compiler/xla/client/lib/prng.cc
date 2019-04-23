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

#include "tensorflow/compiler/xla/client/lib/prng.h"

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

// The internal state of the Three Fry implementation.
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

// Converts a uint64 to two uint32s.
ThreeFry2x32State Uint64ToUint32s(XlaOp u64) {
  XlaBuilder* builder = u64.builder();
  XlaOp const32 = ConstantR0WithType(builder, U64, 32);
  XlaOp fst = ConvertElementType(u64, U32);
  XlaOp snd = ConvertElementType(ShiftRightLogical(u64, const32), U32);
  return {fst, snd};
}

// Converts two uint32s to a uint64.
XlaOp Uint32sToUint64(ThreeFry2x32State u32s) {
  XlaBuilder* builder = u32s[0].builder();
  return ConvertElementType(u32s[0], U64) |
         ShiftLeft(ConvertElementType(u32s[1], U64),
                   ConstantR0WithType(builder, U64, 32));
}

// Given the initial state and the request number of random numbers to be
// generated, returns the input for the random number generator and a new state.
std::pair<ThreeFry2x32State, XlaOp> GetThreeFryInputsAndUpdatedState(
    XlaOp initial_state, const int64 size) {
  XlaBuilder* builder = initial_state.builder();
  XlaOp input_u64 = Iota(builder, U64, size);
  input_u64 = input_u64 + initial_state;
  XlaOp new_state = initial_state + ConstantR0<uint64>(builder, size);
  return std::make_pair(Uint64ToUint32s(input_u64), new_state);
}

// Generates random 32bits with the given shape using the Three Fry
// implementation. Returns the random bits and the new state.
RngOutput ThreeFryRngBit32(XlaOp key, XlaOp initial_state, const Shape& shape) {
  XlaBuilder* builder = key.builder();
  const int64 size = ShapeUtil::ElementsIn(shape);
  const int64 half_size = CeilOfRatio<int64>(size, 2);
  const bool size_is_odd = (half_size * 2 != size);
  std::pair<ThreeFry2x32State, XlaOp> inputs_state =
      GetThreeFryInputsAndUpdatedState(initial_state, half_size);
  ThreeFry2x32State inputs = inputs_state.first;
  ThreeFry2x32State outputs = ThreeFry2x32(inputs, Uint64ToUint32s(key));
  if (size_is_odd) {
    outputs[1] = Slice(outputs[1], {0}, {half_size - 1}, {1});
  }
  XlaOp result = ConcatInDim(builder, outputs, 0);
  return {Reshape(result, AsInt64Slice(shape.dimensions())),
          inputs_state.second};
}

// Generates random 64bits with the given shape using the Three Fry
// implementation. Returns the random bits and the new state.
RngOutput ThreeFryRngBit64(XlaOp key, XlaOp initial_state, const Shape& shape) {
  const int64 size = ShapeUtil::ElementsIn(shape);
  std::pair<ThreeFry2x32State, XlaOp> inputs_state =
      GetThreeFryInputsAndUpdatedState(initial_state, size);
  ThreeFry2x32State inputs = inputs_state.first;
  ThreeFry2x32State outputs = ThreeFry2x32(inputs, Uint64ToUint32s(key));
  XlaOp result = Uint32sToUint64(outputs);
  return {Reshape(result, AsInt64Slice(shape.dimensions())),
          inputs_state.second};
}

XlaOp ConvertRandomBitsToUniformF32(XlaOp bits, XlaOp minval, XlaOp maxval) {
  XlaBuilder* builder = bits.builder();
  // Form 23 random mantissa bits, with a leading 1 bit. The leading 1 bit
  // forces the random bits into the mantissa.
  constexpr int kFloatBits = 32;
  constexpr int kMantissaBits = 23;
  bits = ShiftRightLogical(
             bits, ConstantR0<uint32>(builder, kFloatBits - kMantissaBits)) |
         ConstantR0<uint32>(builder, absl::bit_cast<uint32>(1.0f));
  XlaOp values = BitcastConvertType(bits, F32);

  // We have a floating point number in the range [1.0, 2.0).
  // Subtract 1.0f to shift to the range [0.0, 1.0)
  values = values - ConstantR0<float>(builder, 1.0f);
  // Multiply and add to shift to the range [minval, maxval).
  return values * (maxval - minval) + minval;
}

XlaOp ConvertRandomBitsToUniformInt(XlaOp bits, XlaOp minval, XlaOp maxval,
                                    PrimitiveType type,
                                    PrimitiveType unsigned_type) {
  XlaBuilder* builder = bits.builder();
  XlaOp range = BitcastConvertType(maxval, unsigned_type) -
                BitcastConvertType(minval, unsigned_type);
  XlaOp dist = Rem(bits, range);
  XlaOp dist_div_2 =
      ShiftRightLogical(dist, ConstantR0WithType(builder, unsigned_type, 1));

  return minval + BitcastConvertType(dist_div_2, type) +
         BitcastConvertType(dist - dist_div_2, type);
}

XlaOp UniformToNormalUsingSqrtErfInv(XlaOp uniform) {
  return ScalarLike(uniform, std::sqrt(2.0)) * ErfInv(uniform);
}

}  // namespace

RngOutput ThreeFryBitGenerator(XlaOp key, XlaOp initial_state,
                               const Shape& shape) {
  PrimitiveType type = shape.element_type();
  switch (type) {
    case F32:
    case U32:
    case S32:
      return ThreeFryRngBit32(key, initial_state, shape);
    case U64:
    case S64:
      return ThreeFryRngBit64(key, initial_state, shape);
    default:
      return {key.builder()->ReportError(Unimplemented(
                  "Types other than F32, U32, S32, U64 and S64 "
                  "are not implemented by ThreeFryBitGenerator; got %s",
                  xla::primitive_util::LowercasePrimitiveTypeName(type))),
              initial_state};
  }
}

RngOutput UniformF32Distribution(XlaOp key, XlaOp initial_state,
                                 BitGeneratorTy bit_generator, XlaOp minval,
                                 XlaOp maxval, const Shape& shape) {
  DCHECK_EQ(shape.element_type(), F32);
  RngOutput bits_state = bit_generator(key, initial_state, shape);
  XlaOp bits = bits_state.value;
  XlaOp new_state = bits_state.state;
  return {ConvertRandomBitsToUniformF32(bits, minval, maxval), new_state};
}

RngOutput UniformIntDistribution(XlaOp key, XlaOp initial_state,
                                 BitGeneratorTy bit_generator, XlaOp minval,
                                 XlaOp maxval, const Shape& shape) {
  RngOutput bits_state = bit_generator(key, initial_state, shape);
  XlaOp bits = bits_state.value;
  XlaOp new_state = bits_state.state;
  PrimitiveType type = shape.element_type();
  PrimitiveType unsigned_type;
  if (type == U32 || type == S32) {
    unsigned_type = U32;
  } else {
    DCHECK(type == U64 || type == S64);
    unsigned_type = U64;
  }
  return {
      ConvertRandomBitsToUniformInt(bits, minval, maxval, type, unsigned_type),
      new_state};
}

RngOutput NormalF32Distribution(XlaOp key, XlaOp initial_state,
                                BitGeneratorTy bit_generator,
                                const Shape& shape) {
  DCHECK_EQ(shape.element_type(), F32);
  XlaBuilder* builder = key.builder();
  RngOutput bits_state = UniformF32Distribution(
      key, initial_state, bit_generator,
      ConstantR0<float>(builder, std::nextafter(-1.0f, 0.0f)),
      ConstantR0<float>(builder, 1.0), shape);
  XlaOp normal = UniformToNormalUsingSqrtErfInv(bits_state.value);
  return {normal, bits_state.state};
}

}  // namespace xla
