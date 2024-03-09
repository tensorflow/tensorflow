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

#include <cmath>

#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"

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
template <PrimitiveType T>
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
  // for the values of src0 and src1 for a 16 bit binary operation being tested,
  // and generates the cartesian product of the two sets as the two inputs for
  // the test.
  void FillInput(std::array<Literal, 2>* input_literals) override {
    int64_t input_size = GetInputSize();
    CHECK_EQ(input_size, (*input_literals)[0].element_count());
    CHECK_EQ(input_size, (*input_literals)[1].element_count());

    int64_t begin, end;
    std::tie(begin, end) = GetParam();
    VLOG(2) << "Checking range [" << begin << ", " << end << "]";

    absl::Span<NativeT> input_arr_0 = (*input_literals)[0].data<NativeT>();
    absl::Span<NativeT> input_arr_1 = (*input_literals)[1].data<NativeT>();
    for (int64_t i = 0; i < input_size; i++) {
      uint32_t input_val = i + begin;
      // Convert the lower 16 bits to the NativeT and replaced known incorrect
      // input values with 0.
      input_arr_0[i] = ConvertAndReplaceKnownIncorrectValueWith(input_val, 0);
      input_arr_1[i] =
          ConvertAndReplaceKnownIncorrectValueWith(input_val >> 16, 0);
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

BINARY_TEST_16BIT(Add, {
  auto host_add = [](float x, float y) { return x + y; };
  Run(AddEmptyBroadcastDimension(Add), host_add);
})

BINARY_TEST_16BIT(Sub, {
  auto host_sub = [](float x, float y) { return x - y; };
  Run(AddEmptyBroadcastDimension(Sub), host_sub);
})

// TODO(bixia): Mul fails with bfloat16 on CPU.
BINARY_TEST_16BIT(DISABLED_ON_CPU(Mul), {
  auto host_mul = [](float x, float y) { return x * y; };
  Run(AddEmptyBroadcastDimension(Mul), host_mul);
})

// TODO(bixia): Div fails with bfloat16 on CPU.
BINARY_TEST_16BIT(DISABLED_ON_CPU(Div), {
  auto host_div = [](float x, float y) { return x / y; };
  Run(AddEmptyBroadcastDimension(Div), host_div);
})

BINARY_TEST_16BIT(Max, {
  Run(AddEmptyBroadcastDimension(Max), ReferenceMax<float>);
})

BINARY_TEST_16BIT(Min, {
  Run(AddEmptyBroadcastDimension(Min), ReferenceMin<float>);
})

// TODO(bixia): Pow fails with bfloat16 on CPU.
BINARY_TEST_16BIT(DISABLED_ON_GPU(DISABLED_ON_CPU(Pow)), {
  // See b/162664705.
  known_incorrect_fn_ = [](int64_t val) {
    Eigen::bfloat16 f;
    uint16_t val_16 = val;
    memcpy(&f, &val_16, 2);
    return std::isnan(f);
  };
  Run(AddEmptyBroadcastDimension(Pow), std::pow);
})

// TODO(bixia): Atan2 fails with bfloat16 on CPU.
BINARY_TEST_16BIT(DISABLED_ON_CPU(Atan2),
                  { Run(AddEmptyBroadcastDimension(Atan2), std::atan2); })

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
