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

#include "tensorflow/compiler/xla/tests/exhaustive_op_test_utils.h"

#ifdef __FAST_MATH__
#error("Can't be compiled with fast math on");
#endif

namespace xla {
namespace {

template <PrimitiveType T>
using ExhaustiveBinaryTest = ExhaustiveOpTestBase<T, 2>;

// Exhaustive test for binary operations for 16 bit floating point types,
// including float16 and bfloat.
//
// Test parameter is a pair of (begin, end) for range under test.
template <
    PrimitiveType T,
    typename std::enable_if<
        std::is_same<typename primitive_util::PrimitiveTypeToNative<T>::type,
                     half>::value ||
        std::is_same<typename primitive_util::PrimitiveTypeToNative<T>::type,
                     bfloat16>::value>::type* = nullptr>
class Exhaustive16BitBinaryTest
    : public ExhaustiveBinaryTest<T>,
      public ::testing::WithParamInterface<std::pair<int64, int64>> {
 public:
  int64 GetInputSize() override {
    int64 begin, end;
    std::tie(begin, end) = GetParam();
    return end - begin;
  }

  // Given a range of uint64 representation, uses bits 0..15 and bits 16..31 for
  // the values of src0 and src1 for a 16 bit binary operation being tested,
  // and generates the cartesian product of the two sets as the two inputs for
  // the test.
  void FillInput(std::array<Literal, 2>* input_literals) override {
    int64 input_size = GetInputSize();
    CHECK_EQ(input_size, (*input_literals)[0].element_count());
    CHECK_EQ(input_size, (*input_literals)[1].element_count());

    int64 begin, end;
    std::tie(begin, end) = GetParam();
    VLOG(2) << "Checking range [" << begin << ", " << end << "]";

    absl::Span<NativeT> input_arr_0 = (*input_literals)[0].data<NativeT>();
    absl::Span<NativeT> input_arr_1 = (*input_literals)[1].data<NativeT>();
    for (int64 i = 0; i < input_size; i++) {
      uint32 input_val = i + begin;
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

using ExhaustiveF16BinaryTest = Exhaustive16BitBinaryTest<F16>;
using ExhaustiveBF16BinaryTest = Exhaustive16BitBinaryTest<BF16>;

// Returns a wrapper of the given build method, which build an HLO operation
// with an empty broadcast dimension.
inline std::function<XlaOp(XlaOp, XlaOp)> AddEmptyBroadcastDimension(
    std::function<XlaOp(XlaOp, XlaOp, absl::Span<const int64>)> build_method) {
  return [&](XlaOp src0, XlaOp src1) -> XlaOp {
    return build_method(src0, src1, {});
  };
}

#define XLA_TEST_16BIT(test_name, ...)            \
  XLA_TEST_P(ExhaustiveF16BinaryTest, test_name)  \
  __VA_ARGS__                                     \
  XLA_TEST_P(ExhaustiveBF16BinaryTest, test_name) \
  __VA_ARGS__

XLA_TEST_16BIT(Add, {
  auto host_add = [](float x, float y) { return x + y; };
  Run(AddEmptyBroadcastDimension(Add), host_add);
})

XLA_TEST_16BIT(Sub, {
  auto host_sub = [](float x, float y) { return x - y; };
  Run(AddEmptyBroadcastDimension(Sub), host_sub);
})

// TODO(bixia): Mul fails with bfloat16 on CPU.
XLA_TEST_16BIT(DISABLED_ON_CPU(Mul), {
  auto host_mul = [](float x, float y) { return x * y; };
  Run(AddEmptyBroadcastDimension(Mul), host_mul);
})

// TODO(bixia): Div fails with bfloat16 on CPU.
XLA_TEST_16BIT(DISABLED_ON_CPU(Div), {
  auto host_div = [](float x, float y) { return x / y; };
  Run(AddEmptyBroadcastDimension(Div), host_div);
})

template <typename T, typename std::enable_if<
                          std::is_same<T, float>::value ||
                          std::is_same<T, double>::value>::type* = nullptr>
T ReferenceMax(T x, T y) {
  // We need to propagate NAN here becasue std::max may not propagate NAN.
  if (std::fpclassify(x) == FP_NAN) {
    return x;
  }
  if (std::fpclassify(y) == FP_NAN) {
    return y;
  }

  return std::max<T>(x, y);
}

template <typename T, typename std::enable_if<
                          std::is_same<T, float>::value ||
                          std::is_same<T, double>::value>::type* = nullptr>
T ReferenceMin(T x, T y) {
  // We need to propagate NAN here becasue std::max may not propagate NAN.
  if (std::fpclassify(x) == FP_NAN) {
    return x;
  }
  if (std::fpclassify(y) == FP_NAN) {
    return y;
  }

  return std::min<T>(x, y);
}

XLA_TEST_16BIT(Max,
               { Run(AddEmptyBroadcastDimension(Max), ReferenceMax<float>); })

XLA_TEST_16BIT(Min,
               { Run(AddEmptyBroadcastDimension(Min), ReferenceMin<float>); })

// TODO(bixia): Pow fails with bfloat16 on CPU.
XLA_TEST_16BIT(DISABLED_ON_CPU(Pow),
               { Run(AddEmptyBroadcastDimension(Pow), std::powf); })

// TODO(bixia): Atan2 fails with bfloat16 on CPU.
XLA_TEST_16BIT(DISABLED_ON_CPU(Atan2),
               { Run(AddEmptyBroadcastDimension(Atan2), std::atan2f); })

#if defined(BINARY_TEST_TARGET_F16)
#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
INSTANTIATE_TEST_SUITE_P(F16, ExhaustiveF16BinaryTest,
                         ::testing::ValuesIn(CreateExhaustiveF32Ranges()));
#endif
#endif

#if defined(BINARY_TEST_TARGET_BF16)
#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
INSTANTIATE_TEST_SUITE_P(BF16, ExhaustiveBF16BinaryTest,
                         ::testing::ValuesIn(CreateExhaustiveF32Ranges()));
#endif
#endif

}  // namespace
}  // namespace xla
