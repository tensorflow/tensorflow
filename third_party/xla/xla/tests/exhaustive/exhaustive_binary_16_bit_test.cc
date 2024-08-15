/* Copyright 2020 The OpenXLA Authors.

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

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ios>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/client/xla_builder.h"
#include "xla/literal.h"
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/tests/test_macros.h"
#include "xla/types.h"
#include "tsl/platform/test.h"

#ifdef __FAST_MATH__
#error("Can't be compiled with fast math on");
#endif

namespace xla {
namespace exhaustive_op_test {
namespace {

// Exhaustive test for binary operations for 16 bit floating point types,
// including float16 and bfloat.
//
// Test parameter is a pair of (begin, end) for range under test.
template <PrimitiveType T, bool kLeftToRightPacking = false>
class Exhaustive16BitBinaryTest
    : public ExhaustiveBinaryTest<T>,
      public ::testing::WithParamInterface<std::pair<int64_t, int64_t>> {
 public:
  int64_t GetInputSize() override {
    int64_t begin, end;
    std::tie(begin, end) = GetParam();
    return end - begin;
  }

  // Given a range of uint64_t representation, uses bits 0..15 and bits 16..31
  // for the values of src0 and src1 (see below for ordering) for the 16 bit
  // binary operation being tested, and generates the cartesian product of the
  // two sets as the two inputs for the test.
  //
  // If `kLeftToRightPacking == true`, bit 31..16 become src0 and 15..0 becomes
  // src1. If `kLeftToRightPacking == false`, then bits 31..16 become src1
  // and 15..0 becomes src0.
  void FillInput(std::array<Literal, 2>* input_literals) override {
    int64_t input_size = GetInputSize();
    CHECK_EQ(input_size, (*input_literals)[0].element_count());
    CHECK_EQ(input_size, (*input_literals)[1].element_count());

    int64_t begin, end;
    std::tie(begin, end) = GetParam();

    uint16_t left_begin, left_end, right_begin, right_end;
    if constexpr (kLeftToRightPacking) {
      left_begin = std::bit_cast<uint16_t>(static_cast<int16_t>(begin >> 16));
      left_end = std::bit_cast<uint16_t>(static_cast<int16_t>(end >> 16));
      right_begin = std::bit_cast<uint16_t>(static_cast<int16_t>(begin));
      right_end = std::bit_cast<uint16_t>(static_cast<int16_t>(end));
    } else {
      left_begin = std::bit_cast<uint16_t>(static_cast<int16_t>(begin));
      left_end = std::bit_cast<uint16_t>(static_cast<int16_t>(end));
      right_begin = std::bit_cast<uint16_t>(static_cast<int16_t>(begin >> 16));
      right_end = std::bit_cast<uint16_t>(static_cast<int16_t>(end >> 16));
    }
    if (VLOG_IS_ON(2)) {
      LOG(INFO) << this->SuiteName() << this->TestName() << " Range:";
      LOG(INFO) << "\tfrom=(" << left_begin << ", " << right_begin << "); hex=("
                << std::hex << left_begin << ", " << right_begin << "); float=("
                << *reinterpret_cast<xla::bfloat16*>(&left_begin) << ", "
                << *reinterpret_cast<xla::bfloat16*>(&right_begin)
                << ") (inclusive)";
      LOG(INFO) << "\tto=(" << left_end << ", " << right_end << "); hex=("
                << std::hex << left_end << ", " << right_end << "); float=("
                << *reinterpret_cast<xla::bfloat16*>(&left_end) << ", "
                << *reinterpret_cast<xla::bfloat16*>(&right_end)
                << ") (exclusive)";
      LOG(INFO) << "\ttotal values to test=" << (end - begin);
    }

    absl::Span<NativeT> input_arr_0 = (*input_literals)[0].data<NativeT>();
    absl::Span<NativeT> input_arr_1 = (*input_literals)[1].data<NativeT>();
    for (int64_t i = 0; i < input_size; i++) {
      uint32_t input_val = i + begin;
      // Convert the packed bits to a pair of NativeT and replace known
      // incorrect input values with 0.
      //
      // In either case, we only use 32 bits out of the 64 bits possible.
      if constexpr (kLeftToRightPacking) {
        // Left is stored at higher 16 bits.
        input_arr_0[i] =
            ConvertAndReplaceKnownIncorrectValueWith(input_val >> 16, 0);
        input_arr_1[i] = ConvertAndReplaceKnownIncorrectValueWith(input_val, 0);
      } else {
        // Left is stored at lower 16 bits.
        input_arr_0[i] = ConvertAndReplaceKnownIncorrectValueWith(input_val, 0);
        input_arr_1[i] =
            ConvertAndReplaceKnownIncorrectValueWith(input_val >> 16, 0);
      }
    }
  }

 protected:
  using typename ExhaustiveBinaryTest<T>::NativeT;
  using ExhaustiveBinaryTest<T>::ConvertAndReplaceKnownIncorrectValueWith;
};

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
using ExhaustiveF16BinaryTest = Exhaustive16BitBinaryTest<F16>;
#define BINARY_TEST_F16(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF16BinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_F16(test_name, ...)
#endif

#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
using ExhaustiveBF16BinaryTest = Exhaustive16BitBinaryTest<BF16>;
#define BINARY_TEST_BF16(test_name, ...)          \
  XLA_TEST_P(ExhaustiveBF16BinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_BF16(test_name, ...)
#endif

#define BINARY_TEST_16BIT(test_name, ...) \
  BINARY_TEST_F16(test_name, __VA_ARGS__) \
  BINARY_TEST_BF16(test_name, __VA_ARGS__)

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
double AddCpuBf16AbsErr(xla::bfloat16 left, xla::bfloat16 right) {
  float output = static_cast<float>(left) + static_cast<float>(right);

  // Hardware flushes subnormal outputs to 0.
  if (IsSubnormal(output)) {
    return std::numeric_limits<float>::min();
  }

  return 0.0;
}

BINARY_TEST_16BIT(Add, {
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };
  if (IsCpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(AddCpuBf16AbsErr(static_cast<xla::bfloat16>(left),
                                      static_cast<xla::bfloat16>(right)))
            .strict_signed_zeros()
            .build();
      };
    }
  }
  Run(
      AddEmptyBroadcastDimension(Add), [](float x, float y) { return x + y; },
      error_spec_gen);
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
double SubCpuBf16AbsErr(xla::bfloat16 left, xla::bfloat16 right) {
  float output = static_cast<float>(left) - static_cast<float>(right);

  // Hardware flushes subnormal outputs to 0.
  if (IsSubnormal(output)) {
    return std::numeric_limits<float>::min();
  }

  return 0.0;
}

BINARY_TEST_16BIT(Sub, {
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };
  if (IsCpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(SubCpuBf16AbsErr(static_cast<xla::bfloat16>(left),
                                      static_cast<xla::bfloat16>(right)))
            .strict_signed_zeros()
            .build();
      };
    }
  }
  Run(
      AddEmptyBroadcastDimension(Sub), [](float x, float y) { return x - y; },
      error_spec_gen);
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
double MulCpuBf16AbsErr(xla::bfloat16 left, xla::bfloat16 right) {
  float output = static_cast<float>(left) * static_cast<float>(right);

  // Subnormals are flushed to 0 (as inputs or outputs). In these cases, we
  // calculate 0 instead of the expected very small number so we use the minimum
  // float value as the absolute error to give a buffer.
  auto left_is_subnormal = IsSubnormal(left);
  auto right_is_subnormal = IsSubnormal(right);
  auto output_is_subnormal = IsSubnormal(output);
  if (left_is_subnormal || right_is_subnormal || output_is_subnormal) {
    return std::numeric_limits<float>::min();
  }

  return 0.0;
}

bool MulCpuBf16Skip(xla::bfloat16 left, xla::bfloat16 right) {
  // For BF16, multiplying a subnormal by infinity will lead to calculating 0
  // multiplied by infinity due to subnormal flushing, which is defined to be
  // NaN. However, the calculation in higher precision does not flush the
  // subnormal value to 0, leading to a result of infinity.
  auto left_is_subnormal = IsSubnormal(left);
  auto left_is_infinite = std::isinf(left);
  auto right_is_subnormal = IsSubnormal(right);
  auto right_is_infinite = std::isinf(right);
  if ((left_is_subnormal && right_is_infinite) ||
      (left_is_infinite && right_is_subnormal)) {
    return true;
  }

  return false;
}

BINARY_TEST_16BIT(Mul, {
  ErrorSpecGen error_spec_gen = +[](NativeT left, NativeT right) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };
  if (IsCpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(MulCpuBf16AbsErr(static_cast<xla::bfloat16>(left),
                                      static_cast<xla::bfloat16>(right)))
            .strict_signed_zeros()
            .skip_comparison(MulCpuBf16Skip(static_cast<xla::bfloat16>(left),
                                            static_cast<xla::bfloat16>(right)))
            .build();
      };
    }
  }
  Run(
      AddEmptyBroadcastDimension(Mul), [](float x, float y) { return x * y; },
      error_spec_gen);
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
double DivCpuBf16AbsErr(xla::bfloat16 left, xla::bfloat16 right) {
  float output = static_cast<float>(left) / static_cast<float>(right);

  // Subnormals are flushed to 0 so we add a absolute error margin that is
  // larger than any subnormal.
  auto output_is_subnormal = IsSubnormal(output);
  if (output_is_subnormal) {
    return std::numeric_limits<float>::min();
  }

  return 0.0;
}

BINARY_TEST_16BIT(Div, {
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };
  if (IsCpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(DivCpuBf16AbsErr(static_cast<xla::bfloat16>(left),
                                      static_cast<xla::bfloat16>(right)))
            .strict_signed_zeros()
            .build();
      };
    }
  }
  if (IsGpu(platform_) && std::is_same_v<NativeT, xla::half>) {
    error_spec_gen = +[](NativeT, NativeT) {
      return ErrorSpec::Builder().distance_err(1).strict_signed_zeros().build();
    };
  }
  Run(
      AddEmptyBroadcastDimension(Div), [](float x, float y) { return x / y; },
      error_spec_gen);
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
double MaxMinCpuBf16AbsErr(xla::bfloat16 left, xla::bfloat16 right) {
  // It seems subnormals are treated as 0 and max returns the first if all are
  // 0.
  if (IsSubnormal(left) && (right == 0.0 || IsSubnormal(right))) {
    return std::abs(left);
  }
  return 0.0;
}

BINARY_TEST_16BIT(Max, {
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };
  if (IsCpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(MaxMinCpuBf16AbsErr(static_cast<xla::bfloat16>(left),
                                         static_cast<xla::bfloat16>(right)))
            .strict_signed_zeros()
            .build();
      };
    }
  }
  if (IsGpu(platform_)) {
    error_spec_gen = +[](NativeT, NativeT) {
      // A100 and H100 return -0 for max(-0,0).
      return ErrorSpec::Builder().strict_signed_zeros(false).build();
    };
  }
  Run(AddEmptyBroadcastDimension(Max), ReferenceMax<float>, error_spec_gen);
})

BINARY_TEST_16BIT(Min, {
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };
  if (IsCpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(MaxMinCpuBf16AbsErr(static_cast<xla::bfloat16>(left),
                                         static_cast<xla::bfloat16>(right)))
            .strict_signed_zeros()
            .build();
      };
    }
  }
  if (IsGpu(platform_)) {
    error_spec_gen = +[](NativeT, NativeT) {
      // A100 and H100 return 0 for min(0,-0).
      return ErrorSpec::Builder().strict_signed_zeros(false).build();
    };
  }
  Run(AddEmptyBroadcastDimension(Min), ReferenceMin<float>, error_spec_gen);
})

template <typename NativeT>
bool PowCpuGpuF16Skip(NativeT left, NativeT right) {
  // Hardware seems to always return 1 if right is 0, no matter if left is NaN.
  if (std::isnan(left) && right == 0) {
    return true;
  }
  // Hardware seems to always return 1 if left is 1, no matter if right is NaN.
  if (left == 1 && std::isnan(right)) {
    return true;
  }
  return false;
}

double PowCpuBf16AbsErr(xla::bfloat16 left, xla::bfloat16 right) {
  float output = std::pow(static_cast<float>(left), static_cast<float>(right));

  // Output is flushed to 0 if subnormal.
  if (IsSubnormal(output)) {
    return std::numeric_limits<float>::min();
  }

  // TODO(b/359325328): pow computation for subnormal bases is different from
  // std::pow.
  //
  // If the base is subnormal, the output computation selects a different base.
  // The minimum value ever chosen is slightly greater than the 1e-91 used
  // below. We return an absolute error from this value to the "real" output.
  //
  // Because the exponent (right) can be any floating point value, this allows
  // an arbitrary absolute error for subnormal values.
  if (IsSubnormal(left)) {
    xla::bfloat16 output_as_bf16 = static_cast<xla::bfloat16>(output);
    auto expected = std::pow(1e-91, static_cast<double>(right));
    auto err = std::abs(expected - output_as_bf16);
    if (!std::isnan(err)) {
      return err;
    }
  }

  return 0.0;
}

BINARY_TEST_16BIT(Pow, {
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };
  if (IsCpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::half>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .strict_signed_zeros()
            .skip_comparison(PowCpuGpuF16Skip(left, right))
            .build();
      };
    }
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(PowCpuBf16AbsErr(static_cast<xla::bfloat16>(left),
                                      static_cast<xla::bfloat16>(right)))
            .strict_signed_zeros()
            .build();
      };
    }
  }
  if (IsGpu(platform_)) {
    error_spec_gen = +[](NativeT left, NativeT right) {
      return ErrorSpec::Builder()
          .distance_err(1)
          .strict_signed_zeros()
          .skip_comparison(PowCpuGpuF16Skip(left, right))
          .build();
    };
  }
  Run(AddEmptyBroadcastDimension(Pow), std::pow, error_spec_gen);
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
double Atan2CpuBf16AbsErr(xla::bfloat16 left, xla::bfloat16 right) {
  float output =
      std::atan2(static_cast<float>(left), static_cast<float>(right));

  // If the output would be a subnormal float, we allow some error to account
  // for BF16 implementation flushing subnormals to zero.
  auto output_is_subnormal = IsSubnormal(output);
  if (output_is_subnormal) {
    return std::numeric_limits<float>::min();
  }

  return 0.0;
}

bool Atan2CpuBf16Skip(xla::bfloat16 left, xla::bfloat16 right) {
  // Subnormals are flushed to 0, but 0/0 returns NaN instead of
  // <subnormal>/<subnormal> which returns some positive number. We cannot set
  // an error to compare against NaN.
  if (IsSubnormal(left) && IsSubnormal(right)) {
    return true;
  }

  return false;
}

BINARY_TEST_16BIT(Atan2, {
  auto error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };
  if (IsCpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(Atan2CpuBf16AbsErr(static_cast<xla::bfloat16>(left),
                                        static_cast<xla::bfloat16>(right)))
            .strict_signed_zeros()
            .skip_comparison(
                Atan2CpuBf16Skip(static_cast<xla::bfloat16>(left),
                                 static_cast<xla::bfloat16>(right)))
            .build();
      };
    }
  }
  if (IsGpu(platform_)) {
    error_spec_gen = +[](NativeT, NativeT) {
      return ErrorSpec::Builder().distance_err(1).strict_signed_zeros().build();
    };
  }
  Run(AddEmptyBroadcastDimension(Atan2), std::atan2, error_spec_gen);
})

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
INSTANTIATE_TEST_SUITE_P(F16, ExhaustiveF16BinaryTest,
                         ::testing::ValuesIn(CreateExhaustiveF32Ranges()));
#endif

#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
INSTANTIATE_TEST_SUITE_P(BF16, ExhaustiveBF16BinaryTest,
                         ::testing::ValuesIn(CreateExhaustiveF32Ranges()));
#endif

}  // namespace
}  // namespace exhaustive_op_test
}  // namespace xla
