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

#if defined(BINARY_TEST_TARGET_F16) && defined(BINARY_TEST_TARGET_BF16)
#error "Can't define both BINARY_TEST_TARGET_F16 and BINARY_TEST_TARGET_BF16"
#endif

#if defined(BINARY_TEST_TARGET_F16) && \
    !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
#define BINARY_TEST_16BIT(test_name, ...)        \
  XLA_TEST_P(ExhaustiveF16BinaryTest, test_name) \
  __VA_ARGS__
#elif defined(BINARY_TEST_TARGET_BF16) && defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
#define BINARY_TEST_16BIT(test_name, ...)         \
  XLA_TEST_P(ExhaustiveBF16BinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_16BIT(test_name, ...)
#endif

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

BINARY_TEST_16BIT(Max, {
  Run(AddEmptyBroadcastDimension(Max), ReferenceMax<float>);
})

BINARY_TEST_16BIT(Min, {
  Run(AddEmptyBroadcastDimension(Min), ReferenceMin<float>);
})

// TODO(bixia): Pow fails with bfloat16 on CPU.
BINARY_TEST_16BIT(DISABLED_ON_CPU(Pow),
                  { Run(AddEmptyBroadcastDimension(Pow), std::powf); })

// TODO(bixia): Atan2 fails with bfloat16 on CPU.
BINARY_TEST_16BIT(DISABLED_ON_CPU(Atan2),
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

// Exhaustive test for binary operations for float and double.
//
// Test parameter is a tuple of (FpValues, FpValues) describing the possible
// values for each operand. The inputs for the test are the Cartesian product
// of the possible values for the two operands.
template <PrimitiveType T>
class Exhaustive32BitOrMoreBinaryTest
    : public ExhaustiveBinaryTest<T>,
      public ::testing::WithParamInterface<std::tuple<FpValues, FpValues>> {
 protected:
  using typename ExhaustiveBinaryTest<T>::NativeT;
  using ExhaustiveBinaryTest<T>::ConvertAndReplaceKnownIncorrectValueWith;

 private:
  int64 GetInputSize() override {
    FpValues values_0;
    FpValues values_1;
    std::tie(values_0, values_1) = GetParam();
    return values_0.GetTotalNumValues() * values_1.GetTotalNumValues();
  }

  void FillInput(std::array<Literal, 2>* input_literals) override {
    int64 input_size = GetInputSize();
    FpValues values_0;
    FpValues values_1;
    std::tie(values_0, values_1) = GetParam();

    VLOG(2) << " testing " << values_0.ToString() << " " << values_1.ToString()
            << "total values " << input_size;
    CHECK(input_size == (*input_literals)[0].element_count() &&
          input_size == (*input_literals)[1].element_count());

    absl::Span<NativeT> input_arr_0 = (*input_literals)[0].data<NativeT>();
    absl::Span<NativeT> input_arr_1 = (*input_literals)[1].data<NativeT>();

    uint64 i = 0;
    for (auto src0 : values_0) {
      for (auto src1 : values_1) {
        input_arr_0[i] = ConvertAndReplaceKnownIncorrectValueWith(src0, 1);
        input_arr_1[i] = ConvertAndReplaceKnownIncorrectValueWith(src1, 1);
        ++i;
      }
    }
    CHECK_EQ(i, input_size);
  }
};

using ExhaustiveF32BinaryTest = Exhaustive32BitOrMoreBinaryTest<F32>;
using ExhaustiveF64BinaryTest = Exhaustive32BitOrMoreBinaryTest<F64>;

#if defined(BINARY_TEST_TARGET_F32)
#define BINARY_TEST_FLOAT_32(test_name, ...)     \
  XLA_TEST_P(ExhaustiveF32BinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_FLOAT_32(test_name, ...)
#endif

BINARY_TEST_FLOAT_32(Add, {
  auto host_add = [](float x, float y) { return x + y; };
  Run(AddEmptyBroadcastDimension(Add), host_add);
})

BINARY_TEST_FLOAT_32(Sub, {
  auto host_sub = [](float x, float y) { return x - y; };
  Run(AddEmptyBroadcastDimension(Sub), host_sub);
})

// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_32(DISABLED_ON_CPU(Mul), {
  auto host_mul = [](float x, float y) { return x * y; };
  Run(AddEmptyBroadcastDimension(Mul), host_mul);
})

// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_32(DISABLED_ON_CPU(Div), {
  auto host_div = [](float x, float y) { return x / y; };
  Run(AddEmptyBroadcastDimension(Div), host_div);
})

BINARY_TEST_FLOAT_32(Max, {
  Run(AddEmptyBroadcastDimension(Max), ReferenceMax<float>);
})

BINARY_TEST_FLOAT_32(Min, {
  Run(AddEmptyBroadcastDimension(Min), ReferenceMin<float>);
})

// It is more convenient to implement Abs(complex) as a binary op than a unary
// op, as the operations we currently support all have the same data type for
// the source operands and the results.
// TODO(bixia): May want to move this test to unary test if we will be able to
// implement Abs(complex) as unary conveniently.
//
// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_32(DISABLED_ON_CPU(AbsComplex), {
  auto host_abs_complex = [](float x, float y) {
    return std::abs(std::complex<float>(x, y));
  };
  auto device_abs_complex = [](XlaOp x, XlaOp y) { return Abs(Complex(x, y)); };

  Run(device_abs_complex, host_abs_complex);
})

INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    SpecialAndNormalValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::Values(GetNormals<float>(2000))));

INSTANTIATE_TEST_SUITE_P(
    NormalAndSpecialValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<float>(2000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    NormalAndNormalValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(::testing::Values(GetNormals<float>(2000)),
                       ::testing::Values(GetNormals<float>(2000))));

// Tests a total of 40000 ^ 2 inputs, with 2000 ^ 2 inputs in each sub-test.
// Comparing with the unary tests, the binary tests use a smaller set of inputs
// for each sub-test to avoid timeout because the implementation of ExpectNear
// more than 2x slower for binary test.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnituedNormalValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(GetFpValuesForMagnitudeExtremeNormals<float>(40000,
                                                                         2000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<float>(40000, 2000))));

#if defined(BINARY_TEST_TARGET_F64) && \
    !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
#define BINARY_TEST_FLOAT_64(test_name, ...)     \
  XLA_TEST_P(ExhaustiveF64BinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_FLOAT_64(test_name, ...)
#endif

BINARY_TEST_FLOAT_64(Add, {
  auto host_add = [](double x, double y) { return x + y; };
  Run(AddEmptyBroadcastDimension(Add), host_add);
})

BINARY_TEST_FLOAT_64(Sub, {
  auto host_sub = [](double x, double y) { return x - y; };
  Run(AddEmptyBroadcastDimension(Sub), host_sub);
})

// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_64(DISABLED_ON_CPU(Mul), {
  auto host_mul = [](double x, double y) { return x * y; };
  Run(AddEmptyBroadcastDimension(Mul), host_mul);
})

// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_64(DISABLED_ON_CPU(Div), {
  auto host_div = [](double x, double y) { return x / y; };
  Run(AddEmptyBroadcastDimension(Div), host_div);
})

BINARY_TEST_FLOAT_64(Max, {
  Run(AddEmptyBroadcastDimension(Max), ReferenceMax<double>);
})

BINARY_TEST_FLOAT_64(Min, {
  Run(AddEmptyBroadcastDimension(Min), ReferenceMin<double>);
})

// TODO(bixia): Need to investigate the failure on CPU and file bugs.
BINARY_TEST_FLOAT_64(DISABLED_ON_CPU(AbsComplex), {
  auto host_abs_complex = [](double x, double y) {
    return std::abs(std::complex<double>(x, y));
  };
  auto device_abs_complex = [](XlaOp x, XlaOp y) { return Abs(Complex(x, y)); };

  Run(device_abs_complex, host_abs_complex);
})

INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    SpecialAndNormalValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::Values(GetNormals<double>(1000))));

INSTANTIATE_TEST_SUITE_P(
    NormalAndSpecialValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<double>(1000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    NormalAndNormalValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(::testing::Values(GetNormals<double>(1000)),
                       ::testing::Values(GetNormals<double>(1000))));

// Tests a total of 40000 ^ 2 inputs, with 1000 ^ 2 inputs in each sub-test.
// Similar to ExhaustiveF64BinaryTest, we use a smaller set of inputs for each
// for each sub-test comparing with the unary test to avoid timeout.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnituedNormalValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000))));

}  // namespace
}  // namespace xla
