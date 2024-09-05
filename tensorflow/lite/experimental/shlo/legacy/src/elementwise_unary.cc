/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <version>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/bf16.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/f16.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/util.h"

namespace stablehlo {

namespace {

template <typename Value>
absl::Status CheckParameters(const Value& operand, Value& result) {
  if (operand.baseline_type() != result.baseline_type()) {
    return absl::InvalidArgumentError(
        "Constraint violation: baseline_type(operand) = baseline_type(result)");
  }

  if constexpr (std::is_same_v<Value, QuantizedTensor>) {
    if (!(operand.is_per_tensor_quantized() and
          result.is_per_tensor_quantized())) {
      return absl::InvalidArgumentError("Expected per-tensor quantization");
    }
  }

  if (operand.layout().has_strides() || result.layout().has_strides()) {
    return absl::InvalidArgumentError("Stides not supported yet");
  }

  return absl::OkStatus();
}

template <ElementType storage_type, ElementType expressed_type, typename Value,
          typename Op>
absl::Status ElementwiseUnaryOp(const Value& operand, Value& result, Op&& op) {
  if (auto check = CheckParameters(operand, result); !check.ok()) {
    return check;
  }

  using S = Storage<storage_type>;

  auto operand_buffer = operand.buffer();
  auto result_buffer = result.buffer();

  size_t n = operand.num_elements();
  if constexpr (std::is_same_v<Value, Tensor>) {
    if (storage_type != operand.element_type()) {
      return absl::InvalidArgumentError("Unexpected tensor element type");
    }

    for (size_t i = 0; i < n; ++i) {
      auto x = S::Get(operand_buffer, i);
      auto y = op(x);
      S::Set(result_buffer, i, y);
    }

  } else {
    static_assert(std::is_same_v<Value, QuantizedTensor>);

    if (storage_type != result.storage_type()) {
      return absl::InvalidArgumentError("Unexpected storage type");
    } else if (expressed_type != result.expressed_type()) {
      return absl::InvalidArgumentError("Unexpected expressed type");
    }

    const QuantizedParameter& operand_quant_param =
        operand.type().element_type().parameters(0);
    const QuantizedParameter& result_quant_param =
        result.type().element_type().parameters(0);

    using ET = typename Storage<expressed_type>::Type;
    ET result_scale_inv = ET(1.0) / static_cast<ET>(result_quant_param.scale);

    for (size_t i = 0; i < n; ++i) {
      auto operand_storage = S::Get(operand_buffer, i);
      auto result_storage =
          DequantizeOpQuantizePartial<storage_type, expressed_type>(
              operand_storage, operand_quant_param, result_scale_inv,
              result_quant_param.zero_point, op);
      S::Set(result_buffer, i, result_storage);
    }

    if (auto status = CompleteQuantization<storage_type>(result);
        !status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

#define DEFINE_ELEMENTWISE_UNARY_OP(name, element_type, expression) \
  absl::Status name(const Tensor& operand, Tensor& result) {        \
    return ElementwiseUnaryOp<element_type, element_type>(          \
        operand, result, [](auto x) { return expression; });        \
  }

#define DEFINE_ELEMENTWISE_UNARY_QUANTIZED_OP(name, storage_type,              \
                                              expressed_type, expression)      \
  absl::Status name(const QuantizedTensor& operand, QuantizedTensor& result) { \
    return ElementwiseUnaryOp<storage_type, expressed_type>(                   \
        operand, result, [](auto x) { return expression; });                   \
  }

#define DEFINE_ELEMENTWISE_UNARY_OP_BOOL(name, expression) \
  DEFINE_ELEMENTWISE_UNARY_OP(name##_i1, ElementType::kI1, expression);

#define DEFINE_ELEMENTWISE_UNARY_OP_INT(name, expression)                   \
  DEFINE_ELEMENTWISE_UNARY_OP(name##_si8, ElementType::kSI8, expression);   \
  DEFINE_ELEMENTWISE_UNARY_OP(name##_si16, ElementType::kSI16, expression); \
  DEFINE_ELEMENTWISE_UNARY_OP(name##_si32, ElementType::kSI32, expression);

#define DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(name, expression)                    \
  DEFINE_ELEMENTWISE_UNARY_OP(name##_bf16, ElementType::kBF16, expression);    \
  DEFINE_ELEMENTWISE_UNARY_OP(name##_f16, ElementType::kF16, expression);      \
  DEFINE_ELEMENTWISE_UNARY_OP(name##_f32, ElementType::kF32, expression);      \
  DEFINE_ELEMENTWISE_UNARY_QUANTIZED_OP(name##_q_si8_bf16, ElementType::kSI8,  \
                                        ElementType::kBF16, expression);       \
  DEFINE_ELEMENTWISE_UNARY_QUANTIZED_OP(name##_q_si8_f16, ElementType::kSI8,   \
                                        ElementType::kF16, expression);        \
  DEFINE_ELEMENTWISE_UNARY_QUANTIZED_OP(name##_q_si8_f32, ElementType::kSI8,   \
                                        ElementType::kF32, expression);        \
  DEFINE_ELEMENTWISE_UNARY_QUANTIZED_OP(                                       \
      name##_q_si16_bf16, ElementType::kSI16, ElementType::kBF16, expression); \
  DEFINE_ELEMENTWISE_UNARY_QUANTIZED_OP(name##_q_si16_f16, ElementType::kSI16, \
                                        ElementType::kF16, expression);        \
  DEFINE_ELEMENTWISE_UNARY_QUANTIZED_OP(name##_q_si16_f32, ElementType::kSI16, \
                                        ElementType::kF32, expression);        \
  DEFINE_ELEMENTWISE_UNARY_QUANTIZED_OP(                                       \
      name##_q_si32_bf16, ElementType::kSI32, ElementType::kBF16, expression); \
  DEFINE_ELEMENTWISE_UNARY_QUANTIZED_OP(name##_q_si32_f16, ElementType::kSI32, \
                                        ElementType::kF16, expression);        \
  DEFINE_ELEMENTWISE_UNARY_QUANTIZED_OP(name##_q_si32_f32, ElementType::kSI32, \
                                        ElementType::kF32, expression);

#define CALL_UNARY_OP_BOOL_HELPER(name, operand, result) \
  case ElementType::kI1:                                 \
    return name##_i1(operand, result);

#define CALL_UNARY_OP_INT_HELPER(name, operand, result) \
  case ElementType::kSI8:                               \
    return name##_si8(operand, result);                 \
  case ElementType::kSI16:                              \
    return name##_si16(operand, result);                \
  case ElementType::kSI32:                              \
    return name##_si32(operand, result);

#define CALL_UNARY_OP_FLOAT_HELPER(name, operand, result) \
  case ElementType::kBF16:                                \
    return name##_bf16(operand, result);                  \
  case ElementType::kF16:                                 \
    return name##_f16(operand, result);                   \
  case ElementType::kF32:                                 \
    return name##_f32(operand, result);

#define CALL_UNARY_OP_BOOL_INT(name, operand, result)                        \
  {                                                                          \
    auto element_type = operand.element_type();                              \
    switch (element_type) {                                                  \
      CALL_UNARY_OP_BOOL_HELPER(name, operand, result);                      \
      CALL_UNARY_OP_INT_HELPER(name, operand, result);                       \
      default:                                                               \
        return absl::InvalidArgumentError("Unexpected tensor element type"); \
    }                                                                        \
  }

#define CALL_UNARY_OP_INT(name, operand, result)                             \
  {                                                                          \
    auto element_type = operand.element_type();                              \
    switch (element_type) {                                                  \
      CALL_UNARY_OP_INT_HELPER(name, operand, result);                       \
      default:                                                               \
        return absl::InvalidArgumentError("Unexpected tensor element type"); \
    }                                                                        \
  }

#define CALL_UNARY_OP_FLOAT(name, operand, result)                           \
  {                                                                          \
    auto element_type = operand.element_type();                              \
    switch (element_type) {                                                  \
      CALL_UNARY_OP_FLOAT_HELPER(name, operand, result);                     \
      default:                                                               \
        return absl::InvalidArgumentError("Unexpected tensor element type"); \
    }                                                                        \
  }

#define CALL_UNARY_OP_INT_FLOAT(name, operand, result)                       \
  {                                                                          \
    auto element_type = operand.element_type();                              \
    switch (element_type) {                                                  \
      CALL_UNARY_OP_INT_HELPER(name, operand, result);                       \
      CALL_UNARY_OP_FLOAT_HELPER(name, operand, result);                     \
      default:                                                               \
        return absl::InvalidArgumentError("Unexpected tensor element type"); \
    }                                                                        \
  }

#define CALL_UNARY_OP_BOOL_INT_FLOAT(name, operand, result)                  \
  {                                                                          \
    auto element_type = operand.element_type();                              \
    switch (element_type) {                                                  \
      CALL_UNARY_OP_BOOL_HELPER(name, operand, result);                      \
      CALL_UNARY_OP_INT_HELPER(name, operand, result);                       \
      CALL_UNARY_OP_FLOAT_HELPER(name, operand, result);                     \
      default:                                                               \
        return absl::InvalidArgumentError("Unexpected tensor element type"); \
    }                                                                        \
  }

#define CALL_UNARY_QUANTIZED_OP(name, operand, result)                      \
  {                                                                         \
    auto storage_type = operand.storage_type();                             \
    auto expressed_type = operand.expressed_type();                         \
    switch (storage_type) {                                                 \
      case ElementType::kSI8:                                               \
        switch (expressed_type) {                                           \
          case ElementType::kBF16:                                          \
            return name##_q_si8_bf16(operand, result);                      \
          case ElementType::kF16:                                           \
            return name##_q_si8_f16(operand, result);                       \
          case ElementType::kF32:                                           \
            return name##_q_si8_f32(operand, result);                       \
          default:                                                          \
            return absl::InvalidArgumentError("Unexpected expressed type"); \
        }                                                                   \
      case ElementType::kSI16:                                              \
        switch (expressed_type) {                                           \
          case ElementType::kBF16:                                          \
            return name##_q_si16_bf16(operand, result);                     \
          case ElementType::kF16:                                           \
            return name##_q_si16_f16(operand, result);                      \
          case ElementType::kF32:                                           \
            return name##_q_si16_f32(operand, result);                      \
          default:                                                          \
            return absl::InvalidArgumentError("Unexpected expressed type"); \
        }                                                                   \
      case ElementType::kSI32:                                              \
        switch (expressed_type) {                                           \
          case ElementType::kBF16:                                          \
            return name##_q_si32_bf16(operand, result);                     \
          case ElementType::kF16:                                           \
            return name##_q_si32_f16(operand, result);                      \
          case ElementType::kF32:                                           \
            return name##_q_si32_f32(operand, result);                      \
          default:                                                          \
            return absl::InvalidArgumentError("Unexpected expressed type"); \
        }                                                                   \
      default:                                                              \
        return absl::InvalidArgumentError("Unexpected storage type");       \
    }                                                                       \
  }

}  // namespace

// /////////////////////////////////////////////////////////////////////////////
// Abs
// /////////////////////////////////////////////////////////////////////////////

namespace {

DEFINE_ELEMENTWISE_UNARY_OP_INT(Abs, ((x > 0) ? x : -x));
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Abs, ((x > 0) ? x : -x));

}  // namespace

absl::Status Abs(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_INT_FLOAT(Abs, operand, result);
}

absl::Status Abs(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Abs, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Cbrt
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is
// inefficient for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Cbrt, std::cbrt(static_cast<float>(x)));

}  // namespace

absl::Status Cbrt(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(Cbrt, operand, result);
}

absl::Status Cbrt(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Cbrt, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Ceil
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is
// inefficient for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Ceil, std::ceil(static_cast<float>(x)));

}  // namespace

absl::Status Ceil(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(Ceil, operand, result);
}

absl::Status Ceil(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Ceil, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Cosine
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is
// inefficient for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Cosine, std::cos(static_cast<float>(x)));

}  // namespace

absl::Status Cosine(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(Cosine, operand, result);
}

absl::Status Cosine(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Cosine, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// CountLeadingZeros
// /////////////////////////////////////////////////////////////////////////////

namespace {

template <typename Int>
inline Int CountLeadingZeros(Int x) {
  using UInt = typename std::make_unsigned<Int>::type;
#if __cpp_lib_bitops >= 201907L
  return std::countl_zero(static_cast<UInt>(x));
#else
  if (!x) {
    return 8 * sizeof(x);
  }
  Int result = 0;
  auto mask = UInt(1) << (8 * (sizeof(x) - 1) + 7);
  for (auto t = static_cast<UInt>(x); t > 0; t <<= 1) {
    if (t & mask) break;
    result++;
  }
  return result;
#endif
}

DEFINE_ELEMENTWISE_UNARY_OP_INT(CountLeadingZeros, CountLeadingZeros(x));

}  // namespace

absl::Status CountLeadingZeros(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_INT(CountLeadingZeros, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Exponential
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is
// inefficient for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Exponential, std::exp(static_cast<float>(x)));

}  // namespace

absl::Status Exponential(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(Exponential, operand, result);
}

absl::Status Exponential(const QuantizedTensor& operand,
                         QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Exponential, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// ExponentialMinusOne
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is
// inefficient for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(ExponentialMinusOne,
                                  std::expm1(static_cast<float>(x)));

}  // namespace

absl::Status ExponentialMinusOne(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(ExponentialMinusOne, operand, result);
}

absl::Status ExponentialMinusOne(const QuantizedTensor& operand,
                                 QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(ExponentialMinusOne, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Floor
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is
// inefficient for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Floor, std::floor(static_cast<float>(x)));

}  // namespace

absl::Status Floor(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(Floor, operand, result);
}

absl::Status Floor(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Floor, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Log
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is
// inefficient for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Log, std::log(static_cast<float>(x)));

}  // namespace

absl::Status Log(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(Log, operand, result);
}

absl::Status Log(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Log, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// LogPlusOne
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is
// inefficient for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(LogPlusOne,
                                  std::log1p(static_cast<float>(x)));

}  // namespace

absl::Status LogPlusOne(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(LogPlusOne, operand, result);
}

absl::Status LogPlusOne(const QuantizedTensor& operand,
                        QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(LogPlusOne, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Logistic
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is
// inefficient for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Logistic,
                                  1.0f / (1.0f +
                                          std::exp(static_cast<float>(-x))));

}  // namespace

absl::Status Logistic(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(Logistic, operand, result);
}

absl::Status Logistic(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Logistic, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Negate
// /////////////////////////////////////////////////////////////////////////////

namespace {

DEFINE_ELEMENTWISE_UNARY_OP_INT(Negate, -x);
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Negate, -x);

}  // namespace

absl::Status Negate(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_INT_FLOAT(Negate, operand, result);
}

absl::Status Negate(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Negate, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Not
// /////////////////////////////////////////////////////////////////////////////

namespace {

DEFINE_ELEMENTWISE_UNARY_OP_BOOL(Not, !x);
DEFINE_ELEMENTWISE_UNARY_OP_INT(Not, ~x);

}  // namespace

absl::Status Not(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_BOOL_INT(Not, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Popcnt
// /////////////////////////////////////////////////////////////////////////////

namespace {

template <typename Int>
Int Popcount(Int x) {
#if __cpp_lib_bitops >= 201907L
  return std::popcount(static_cast<uint32_t>(x));
#else
  using UInt = typename std::make_unsigned<Int>::type;
  Int result = 0;
  UInt mask = 0x1;
  for (auto t = static_cast<UInt>(x); t > 0; t >>= 1) {
    result += (t & mask);
  }
  return result;
#endif
}

DEFINE_ELEMENTWISE_UNARY_OP_INT(Popcnt, Popcount(x));

}  // namespace

absl::Status Popcnt(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_INT(Popcnt, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// RoundNearestAfz
// /////////////////////////////////////////////////////////////////////////////

namespace {

template <typename Float>
inline Float RoundNearestAfz(Float x) {
  // TODO(cbasile): Performing the op with a conversion to float is
  // inefficient for bf16 and f16 types.
  return std::round(static_cast<float>(x));
}

DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(RoundNearestAfz, RoundNearestAfz(x));

}  // namespace

absl::Status RoundNearestAfz(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(RoundNearestAfz, operand, result);
}

absl::Status RoundNearestAfz(const QuantizedTensor& operand,
                             QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(RoundNearestAfz, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// RoundNearestEven
// /////////////////////////////////////////////////////////////////////////////

namespace {

template <typename Float>
inline Float RoundNearestEven(Float x) {
  // TODO(cbasile): Performing the op with a conversion to float is
  // inefficient for bf16 and f16 types.
  return x - static_cast<Float>(std::remainder(static_cast<float>(x), 1.0f));
}

DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(RoundNearestEven, RoundNearestEven(x));

}  // namespace

absl::Status RoundNearestEven(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(RoundNearestEven, operand, result);
}

absl::Status RoundNearestEven(const QuantizedTensor& operand,
                              QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(RoundNearestEven, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Rsqrt
// /////////////////////////////////////////////////////////////////////////////

namespace {

template <typename Float>
inline Float Rsqrt(Float x) {
  // TODO(cbasile): Performing the op with a conversion to float is
  // inefficient for bf16 and f16 types.
  return Float{1} / static_cast<Float>(std::sqrt(static_cast<float>(x)));
}

DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Rsqrt, Rsqrt(x));

}  // namespace

absl::Status Rsqrt(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(Rsqrt, operand, result);
}

absl::Status Rsqrt(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Rsqrt, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Sign
// /////////////////////////////////////////////////////////////////////////////

namespace {

template <typename Number>
inline Number Sign(Number x) {
  if constexpr (std::is_integral<Number>::value) {
    return x < 0 ? -1 : (x > 0 ? 1 : 0);
  } else {
    static_assert(std::is_floating_point<Number>::value ||
                  std::is_same_v<Number, BF16> || std::is_same_v<Number, F16>);
    if (std::isnan(x)) {
      return NAN;
    }
    return (x < 0 ? -1 : (x > 0 ? 1 : 0));
  }
}

DEFINE_ELEMENTWISE_UNARY_OP_INT(Sign, Sign(x));
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Sign, Sign(x));

}  // namespace

absl::Status Sign(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_INT_FLOAT(Sign, operand, result);
}

absl::Status Sign(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Sign, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Sine
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is
// inefficient for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Sine, std::sin(static_cast<float>(x)));

}  // namespace

absl::Status Sine(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(Sine, operand, result);
}

absl::Status Sine(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Sine, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Sqrt
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is inefficient
// for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Sqrt, std::sqrt(static_cast<float>(x)));

}  // namespace

absl::Status Sqrt(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(Sqrt, operand, result);
}

absl::Status Sqrt(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Sqrt, operand, result);
}

// /////////////////////////////////////////////////////////////////////////////
// Tanh
// /////////////////////////////////////////////////////////////////////////////

namespace {

// TODO(cbasile): Performing the op with a conversion to float is inefficient
// for bf16 and f16 types.
DEFINE_ELEMENTWISE_UNARY_OP_FLOAT(Tanh, std::tanh(static_cast<float>(x)));

}  // namespace

absl::Status Tanh(const Tensor& operand, Tensor& result) {
  CALL_UNARY_OP_FLOAT(Tanh, operand, result);
}

absl::Status Tanh(const QuantizedTensor& operand, QuantizedTensor& result) {
  CALL_UNARY_QUANTIZED_OP(Tanh, operand, result);
}

}  // namespace stablehlo
