/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_UNARY_TEST_DEFINITIONS_H_
#define XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_UNARY_TEST_DEFINITIONS_H_

#include <array>
#include <cstdint>
#include <ios>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/test.h"

namespace xla {
namespace exhaustive_op_test {

// Exhaustive test for unary operations for <= 32bit floating point types.
//
// Test parameter is a tuple containing
//   - primitive type under test,
//   - (begin, end) range under test, as zero-extended int64_ts bitcast to the
//     primitive type under test.
template <PrimitiveType T>
class Exhaustive32BitOrLessUnaryTest
    : public ExhaustiveUnaryTest<T>,
      public ::testing::WithParamInterface<std::pair<int64_t, int64_t>> {
 public:
  // Sets error parameters appropriately for testing tan.
  void SetParamsForTan();

 protected:
  using typename ExhaustiveUnaryTest<T>::NativeT;

 private:
  int64_t GetInputSize() override {
    auto [begin, end] = GetParam();
    return end - begin;
  }

  // Generates all the input values for the test. The range of the bit
  // representation of the input values is described by the test parameter as
  // a pair of int64_t representing the starting bit pattern and the ending
  // pattern. Each bit representation is first truncated to the integral type of
  // the same bit as the type being tested, if needed, and then bitcasted to the
  // type being tested.
  void FillInput(std::array<Literal, 1>* input_literal) override {
    using IntegralT =
        typename ExhaustiveOpTestBase<T, 1>::ComponentIntegralNativeT;

    auto [begin, end] = GetParam();
    if (VLOG_IS_ON(2)) {
      // N.B.: Use INFO directly instead of doing another thread-safe VLOG
      // check.
      LOG(INFO) << this->SuiteName() << this->TestName() << " Range:";
      LOG(INFO) << "\tfrom=" << begin << "; hex=" << std::hex << begin
                << "; float=" << *reinterpret_cast<float*>(&begin)
                << " (inclusive)";
      LOG(INFO) << "\tto=" << end << "; hex=" << std::hex << end
                << "; float=" << *reinterpret_cast<float*>(&end)
                << " (exclusive)";
      LOG(INFO) << "\ttotal values to test=" << (end - begin);
    }

    int64_t input_size = (*input_literal)[0].element_count();
    CHECK_EQ(input_size, end - begin);

    absl::Span<NativeT> input_arr = (*input_literal)[0].data<NativeT>();
    for (int64_t i = 0; i < input_size; i++) {
      IntegralT input_val = i + begin;
      input_arr[i] = this->ConvertValue(input_val);
    }
  }
};

using ExhaustiveF32UnaryTest = Exhaustive32BitOrLessUnaryTest<F32>;
using ExhaustiveF16UnaryTest = Exhaustive32BitOrLessUnaryTest<F16>;
using ExhaustiveBF16UnaryTest = Exhaustive32BitOrLessUnaryTest<BF16>;

// Exhaustive test for unary operations for double.
//
// Test parameter is a tuple containing
//   - primitive type under test,
//   - FpValues representing a set of double values.
class ExhaustiveF64UnaryTest : public ExhaustiveUnaryTest<F64>,
                               public ::testing::WithParamInterface<FpValues> {
 private:
  int64_t GetInputSize() override {
    FpValues values = GetParam();
    return values.GetTotalNumValues();
  }

  void FillInput(std::array<Literal, 1>* input_literal) override {
    FpValues fp_values = GetParam();
    int64_t input_size = (*input_literal)[0].element_count();
    if (VLOG_IS_ON(2)) {
      // N.B.: Use INFO directly instead of doing another thread-safe VLOG
      // check.
      LOG(INFO) << this->SuiteName() << this->TestName() << " Values:";
      LOG(INFO) << "\t" << fp_values.ToString();
      LOG(INFO) << "\ttotal values to test=" << input_size;
    }

    uint64_t i = 0;
    absl::Span<double> input_arr = (*input_literal)[0].data<double>();
    for (auto bits : fp_values) {
      input_arr[i] = this->ConvertValue(bits);
      ++i;
    }
    CHECK_EQ(i, input_size);
  }
};

#ifdef XLA_BACKEND_SUPPORTS_BFLOAT16
#define UNARY_TEST_BF16(test_name, ...)          \
  XLA_TEST_P(ExhaustiveBF16UnaryTest, test_name) \
  __VA_ARGS__
#else
#define UNARY_TEST_BF16(test_name, ...)
#endif

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
#define UNARY_TEST_F16(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF16UnaryTest, test_name) \
  __VA_ARGS__
#else
#define UNARY_TEST_F16(test_name, ...)
#endif

#define UNARY_TEST_F32(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF32UnaryTest, test_name) \
  __VA_ARGS__

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
#define UNARY_TEST_F64(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF64UnaryTest, test_name) \
  __VA_ARGS__
#else
#define UNARY_TEST_F64(test_name, ...)
#endif

#define UNARY_TEST(test_name, ...)        \
  UNARY_TEST_BF16(test_name, __VA_ARGS__) \
  UNARY_TEST_F16(test_name, __VA_ARGS__)  \
  UNARY_TEST_F32(test_name, __VA_ARGS__)  \
  UNARY_TEST_F64(test_name, __VA_ARGS__)

}  // namespace exhaustive_op_test
}  // namespace xla

#endif  // XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_UNARY_TEST_DEFINITIONS_H_
