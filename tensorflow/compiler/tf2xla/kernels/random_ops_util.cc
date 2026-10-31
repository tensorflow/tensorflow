/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/random_ops_util.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/kernels/rng_converter_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/prng.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/stateless_random_ops_v2.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

xla::XlaOp GetCounter(xla::RandomAlgorithm const& alg, xla::XlaOp state) {
  return xla::Slice(state, {RNG_KEY_SIZE},
                    {RNG_KEY_SIZE + xla::GetCounterSize(alg)}, {1});
}

absl::StatusOr<xla::RandomAlgorithm> ResolveAlg(
    int alg_id, absl::string_view device_type_string) {
  switch (alg_id) {
    case RNG_ALG_PHILOX:
      return xla::RandomAlgorithm::RNG_PHILOX;
    case RNG_ALG_THREEFRY:
      return xla::RandomAlgorithm::RNG_THREE_FRY;
    case RNG_ALG_AUTO_SELECT:
      return DefaultRngAlgForDeviceType(device_type_string);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported algorithm id: ", alg_id));
  }
}

// Converts 32-bit words to half, bfloat16 or float values in [0, 1) by putting
// their low mantissa bits under the exponent of 1.0 and subtracting 1.0, like
// Uint16ToHalf, Uint16ToGfloat16 and Uint32ToFloat.
xla::XlaOp WordsToUnitFloats(xla::XlaOp words, xla::PrimitiveType type) {
  xla::XlaBuilder* builder = words.builder();
  const int mantissa_bits = xla::primitive_util::SignificandWidth(type) - 1;
  const xla::PrimitiveType bits_type =
      xla::primitive_util::UnsignedIntegralTypeForBitWidth(
          xla::primitive_util::BitWidth(type));
  xla::XlaOp mantissa = xla::ConvertElementType(
      words & xla::ConstantR0<uint32_t>(builder, (1u << mantissa_bits) - 1),
      bits_type);
  xla::XlaOp one = xla::One(builder, type);
  return xla::BitcastConvertType(
             mantissa | xla::BitcastConvertType(one, bits_type), type) -
         one;
}

// Converts 64-bit words to double values in [0, 1) the same way. Each 64-bit
// word holds two consecutive 32-bit words, the first in its low half, and like
// Uint64ToDouble the first gives the upper 20 mantissa bits and the second the
// lower 32.
xla::XlaOp WordsToUnitDoubles(xla::XlaOp words) {
  xla::XlaBuilder* builder = words.builder();
  xla::XlaOp shift = xla::ConstantR0<uint64_t>(builder, 32);
  xla::XlaOp mantissa =
      xla::ShiftLeft(words & xla::ConstantR0<uint64_t>(builder, 0xfffff),
                     shift) |
      xla::ShiftRightLogical(words, shift);
  xla::XlaOp one = xla::One(builder, xla::F64);
  return xla::BitcastConvertType(
             mantissa | xla::BitcastConvertType(one, xla::U64), xla::F64) -
         one;
}

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
    case xla::F16:
    case xla::F32:
    case xla::F64:
      return xla::UniformFloatingPointDistribution(key, counter, generator,
                                                   minval, maxval, shape);
    case xla::S32:
    case xla::S64:
    case xla::U32:
    case xla::U64:
      return xla::UniformIntDistribution(key, counter, generator, minval,
                                         maxval, shape);
      break;
    default:
      return {builder->ReportError(xla::Unimplemented(
                  "Types other than F16, F32, S32, S64, U32 and U64 are not "
                  "implemented by "
                  "StatelessRngUniformV2; got %s",
                  xla::primitive_util::LowercasePrimitiveTypeName(type))),
              counter};
  }
}

}  // namespace

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
xla::XlaOp GetU64FromS32Seeds(xla::XlaOp seed0, xla::XlaOp seed1) {
  // Here, the seeds are cast to unsigned type of the same width to have leading
  // zeros in the 64 bit representation.
  xla::XlaOp u64_seed0 =
      ConvertElementType(ConvertElementType(seed0, xla::U32), xla::U64);
  xla::XlaOp u64_seed1 =
      ConvertElementType(ConvertElementType(seed1, xla::U32), xla::U64);
  return u64_seed0 |
         (u64_seed1 << ConstantR0WithType(seed0.builder(), xla::U64, 32));
}

std::pair<xla::XlaOp, xla::XlaOp> MixSeedsForEagerCompatibility(
    xla::XlaBuilder* builder, xla::XlaOp seed0, xla::XlaOp seed1) {
  xla::XlaOp k0 = xla::ConstantR0WithType(builder, xla::U32, 0x3ec8f720);
  xla::XlaOp k1 = xla::ConstantR0WithType(builder, xla::U32, 0x02461e29);

  // We pack our U32 words into U64 pairs (little-endian word order).
  auto pack_u32_to_u64 = [&](xla::XlaOp lo, xla::XlaOp hi) -> xla::XlaOp {
    xla::XlaOp lo64 = xla::ConvertElementType(lo, xla::U64);
    xla::XlaOp hi64 = xla::ConvertElementType(hi, xla::U64);
    return lo64 | (hi64 << xla::ConstantR0WithType(builder, xla::U64, 32));
  };
  // Widens a seed to 64 bits like the eager path, which assigns S32 and S64
  // seeds to uint64_t.
  auto seed_to_u64 = [](xla::XlaOp seed) -> xla::XlaOp {
    return xla::BitcastConvertType(xla::ConvertElementType(seed, xla::S64),
                                   xla::U64);
  };

  xla::XlaOp key_u64 = pack_u32_to_u64(k0, k1);
  xla::XlaOp ctr_lo_u64 = seed_to_u64(seed0);
  xla::XlaOp ctr_hi_u64 = seed_to_u64(seed1);

  // Philox state layout: [key(1 x U64), counter(2 x U64)] = 3 x U64
  xla::XlaOp philox_state = xla::ConcatInDim(
      builder,
      {xla::Reshape(key_u64, {1}), xla::Reshape(ctr_lo_u64, {1}),
       xla::Reshape(ctr_hi_u64, {1})},
      /*dimension=*/0);

  // Generate 4 x U32 mixed output words via one Philox round.
  xla::Shape mix_shape = xla::ShapeUtil::MakeShape(xla::U32, {4});
  xla::XlaOp mix_result = xla::RngBitGenerator(xla::RandomAlgorithm::RNG_PHILOX,
                                               philox_state, mix_shape);
  xla::XlaOp mixed = xla::GetTupleElement(mix_result, 1);

  // Reassemble mixed words into key and counter U64 values.
  // Mirrors the eager GenerateKey() post-mix assignment:
  //   key[0]     = mix[0];  key[1]     = mix[1]
  //   counter[2] = mix[2];  counter[3] = mix[3]
  //   counter[0] = counter[1] = 0
  auto slice_scalar = [&](int idx) -> xla::XlaOp {
    return xla::Reshape(xla::Slice(mixed, {idx}, {idx + 1}, {1}), {});
  };
  xla::XlaOp mix0 = slice_scalar(0);
  xla::XlaOp mix1 = slice_scalar(1);
  xla::XlaOp mix2 = slice_scalar(2);
  xla::XlaOp mix3 = slice_scalar(3);

  xla::XlaOp final_key = pack_u32_to_u64(mix0, mix1);
  xla::XlaOp final_state = pack_u32_to_u64(mix2, mix3);

  return {final_key, final_state};
}

xla::RngOutput EagerCompatibleUniformDistribution(
    xla::XlaOp key, xla::XlaOp counter, xla::BitGeneratorTy bit_generator,
    const xla::Shape& shape) {
  xla::PrimitiveType type = shape.element_type();
  const int64_t num_values = xla::ShapeUtil::ElementsIn(shape);
  const xla::PrimitiveType word_type = type == xla::F64 ? xla::U64 : xla::U32;
  xla::RngOutput words = bit_generator(
      key, counter, xla::ShapeUtil::MakeShape(word_type, {num_values}));
  xla::XlaOp values = type == xla::F64
                          ? WordsToUnitDoubles(words.value)
                          : WordsToUnitFloats(words.value, type);
  return {xla::Reshape(values, shape.dimensions()), words.state};
}

xla::RngOutput EagerCompatibleNormalDistribution(
    xla::XlaOp key, xla::XlaOp counter, xla::BitGeneratorTy bit_generator,
    const xla::Shape& shape) {
  xla::XlaBuilder* builder = key.builder();
  xla::PrimitiveType type = shape.element_type();
  const int64_t num_values = xla::ShapeUtil::ElementsIn(shape);
  const int64_t num_pairs = xla::CeilOfRatio<int64_t>(num_values, 2);
  const xla::PrimitiveType word_type = type == xla::F64 ? xla::U64 : xla::U32;
  xla::RngOutput words = bit_generator(
      key, counter, xla::ShapeUtil::MakeShape(word_type, {2 * num_pairs}));
  xla::XlaOp uniforms = type == xla::F64
                            ? WordsToUnitDoubles(words.value)
                            : WordsToUnitFloats(words.value, xla::F32);
  xla::XlaOp pairs = xla::Reshape(uniforms, {num_pairs, 2});
  auto column = [&](int64_t i) {
    return xla::Reshape(xla::SliceInDim(pairs, i, i + 1, 1, 1), {num_pairs});
  };
  // The Box-Muller transform, as in BoxMullerFloat and BoxMullerDouble.
  xla::XlaOp u = xla::Max(column(0), xla::ScalarLike(uniforms, 1.0e-7));
  xla::XlaOp v = column(1);
  xla::XlaOp radius = xla::Sqrt(xla::ScalarLike(u, -2.0) * xla::Log(u));
  xla::XlaOp angle = xla::ScalarLike(v, 2.0 * M_PI) * v;
  // Each pair gives two consecutive values, the sine one first.
  xla::XlaOp values = xla::ConcatInDim(
      builder,
      {xla::Reshape(xla::Sin(angle) * radius, {num_pairs, 1}),
       xla::Reshape(xla::Cos(angle) * radius, {num_pairs, 1})},
      1);
  values = xla::Slice(xla::Reshape(values, {2 * num_pairs}), {0}, {num_values},
                      {1});
  return {xla::Reshape(xla::ConvertElementType(values, type),
                       shape.dimensions()),
          words.state};
}

absl::StatusOr<int> GetAlgId(XlaOpKernelContext* ctx, int alg_input_idx) {
  TF_ASSIGN_OR_RETURN(auto alg_shape, ctx->InputXlaShape(alg_input_idx));
  if (alg_shape.dimensions().size() != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("The algorithm argument must be of shape [], not ",
                     alg_shape.ToString()));
  }
  auto alg_dtype = ctx->input_type(alg_input_idx);
  if (alg_dtype != DT_INT32 && alg_dtype != DT_INT64) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The algorithm argument must have dtype int32 or int64, not ",
        DataTypeString(alg_dtype)));
  }
  xla::Literal alg_literal;
  TF_RETURN_IF_ERROR(ctx->ConstantInput(alg_input_idx, &alg_literal));
  if (alg_dtype == DT_INT32) {
    return alg_literal.Get<int>({});
  } else {
    return alg_literal.Get<int64_t>({});
  }
}

absl::StatusOr<xla::RandomAlgorithm> AlgorithmFromInput(
    XlaOpKernelContext* ctx, int alg_input_idx,
    absl::string_view device_type_string) {
  TF_ASSIGN_OR_RETURN(auto alg_id, GetAlgId(ctx, alg_input_idx));
  return ResolveAlg(alg_id, device_type_string);
}

xla::XlaOp MaybeSliceCounter(xla::RandomAlgorithm const& alg,
                             TensorShape const& counter_shape,
                             xla::XlaOp counter) {
  auto input_counter_size = counter_shape.dim_size(0);
  auto real_counter_size = xla::GetCounterSize(alg);
  if (input_counter_size > real_counter_size) {
    counter = xla::Slice(counter, {0}, {real_counter_size}, {1});
  }
  return counter;
}

DataType MaybeConvertBF16ToF32(DataType const& dtype) {
  if (dtype == DT_BFLOAT16) {
    // We'll go through F32 to generate BF16.
    // TODO(b/256243456): Generate BF16 directly from U16.
    return DT_FLOAT;
  }
  return dtype;
}

absl::StatusOr<xla::XlaOp> BuildUniformRandoms(
    XlaOpKernelContext* ctx, DataType dtype, std::string device_type_string,
    TensorShape shape,
    std::function<xla::XlaOp(xla::XlaBuilder*, xla::PrimitiveType)> lo_fn,
    std::function<xla::XlaOp(xla::XlaBuilder*, xla::PrimitiveType)> hi_fn) {
  xla::XlaBuilder* builder = ctx->builder();
  auto rng_dtype = MaybeConvertBF16ToF32(dtype);
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));

  xla::PrimitiveType rng_primitive_type = xla_shape.element_type();
  xla::XlaOp lo = lo_fn(builder, rng_primitive_type);
  xla::XlaOp hi = hi_fn(builder, rng_primitive_type);

  return BuildUniformRandoms(ctx, dtype, device_type_string, xla_shape, lo, hi);
}

absl::StatusOr<xla::XlaOp> BuildUniformRandoms(XlaOpKernelContext* ctx,
                                               DataType dtype,
                                               std::string device_type_string,
                                               xla::Shape xla_shape,
                                               xla::XlaOp lo, xla::XlaOp hi) {
  xla::XlaOp key = ctx->Input(kRandomKeyInputIdx);
  xla::XlaOp counter = ctx->Input(kRandomCounterInputIdx);

  TF_ASSIGN_OR_RETURN(
      xla::RandomAlgorithm alg,
      AlgorithmFromInput(ctx, kRandomAlgInputIdx, device_type_string));
  TensorShape counter_shape = ctx->InputShape(kRandomCounterInputIdx);
  TF_RETURN_IF_ERROR(CheckKeyCounterShape(
      GetCounterSize(alg), ctx->InputShape(kRandomKeyInputIdx), counter_shape));
  counter = MaybeSliceCounter(alg, counter_shape, counter);

  xla::RngOutput result =
      StatelessRngUniformV2(alg, key, counter, xla_shape, lo, hi);
  return result.value;
}
}  // namespace tensorflow

namespace xla {

int GetCounterSize(RandomAlgorithm const& alg) {
  switch (alg) {
    case RandomAlgorithm::RNG_PHILOX:
      return 2;
    case RandomAlgorithm::RNG_THREE_FRY:  // fall through
    case RandomAlgorithm::RNG_DEFAULT:    // fall through
    default:
      return 1;
  }
}

}  // namespace xla
