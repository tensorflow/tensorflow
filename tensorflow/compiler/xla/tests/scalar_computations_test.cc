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
#include <limits>
#include <memory>

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ScalarComputationsTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001};

 protected:
  // A template for building and running a binary comparison test.
  template <typename NativeT>
  void TestCompare(NativeT lhs, NativeT rhs, bool expected,
                   ComputationDataHandle (ComputationBuilder::*op)(
                       const ComputationDataHandle&,
                       const ComputationDataHandle&,
                       tensorflow::gtl::ArraySlice<int64>)) {
    ComputationBuilder builder(client_, TestName());
    ComputationDataHandle lhs_op = builder.ConstantR0<NativeT>(lhs);
    ComputationDataHandle rhs_op = builder.ConstantR0<NativeT>(rhs);
    ComputationDataHandle result = (builder.*op)(lhs_op, rhs_op, {});
    ComputeAndCompareR0<bool>(&builder, expected, {});
  }

  template <typename NativeT>
  void TestMinMax(NativeT lhs, NativeT rhs, NativeT expected,
                  ComputationDataHandle (ComputationBuilder::*op)(
                      const ComputationDataHandle&,
                      const ComputationDataHandle&,
                      tensorflow::gtl::ArraySlice<int64>)) {
    ComputationBuilder builder(client_, TestName());
    ComputationDataHandle lhs_op = builder.ConstantR0<NativeT>(lhs);
    ComputationDataHandle rhs_op = builder.ConstantR0<NativeT>(rhs);
    ComputationDataHandle result = (builder.*op)(lhs_op, rhs_op, {});
    ComputeAndCompareR0<NativeT>(&builder, expected, {});
  }
};

TEST_F(ScalarComputationsTest, NegateScalarF32) {
  ComputationBuilder builder(client_, TestName());
  builder.Neg(builder.ConstantR0<float>(2.1f));

  ComputeAndCompareR0<float>(&builder, -2.1f, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, NegateScalarS32) {
  ComputationBuilder builder(client_, TestName());
  builder.Neg(builder.ConstantR0<int32>(2));

  ComputeAndCompareR0<int32>(&builder, -2, {});
}

TEST_F(ScalarComputationsTest, AddTwoScalarsF32) {
  ComputationBuilder builder(client_, TestName());
  builder.Add(builder.ConstantR0<float>(2.1f), builder.ConstantR0<float>(5.5f));

  ComputeAndCompareR0<float>(&builder, 7.6f, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, AddTwoScalarsS32) {
  ComputationBuilder builder(client_, TestName());
  builder.Add(builder.ConstantR0<int32>(2), builder.ConstantR0<int32>(5));

  ComputeAndCompareR0<int32>(&builder, 7, {});
}

TEST_F(ScalarComputationsTest, AddTwoScalarsU32) {
  ComputationBuilder builder(client_, TestName());
  builder.Add(builder.ConstantR0<uint32>(35), builder.ConstantR0<uint32>(57));

  ComputeAndCompareR0<uint32>(&builder, 92, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsU8) {
  ComputationBuilder builder(client_, TestName());
  builder.Add(builder.ConstantR0<uint8>(35), builder.ConstantR0<uint8>(57));

  ComputeAndCompareR0<uint8>(&builder, 92, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsU64) {
  ComputationBuilder builder(client_, TestName());
  const uint64 a = static_cast<uint64>(1) << 63;
  const uint64 b = a + 1;
  builder.Add(builder.ConstantR0<uint64>(a), builder.ConstantR0<uint64>(b));

  ComputeAndCompareR0<uint64>(&builder, a + b, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsS64) {
  ComputationBuilder builder(client_, TestName());
  const int64 a = static_cast<int64>(1) << 62;
  const int64 b = a + 1;
  builder.Add(builder.ConstantR0<int64>(a), builder.ConstantR0<int64>(b));

  ComputeAndCompareR0<int64>(&builder, a + b, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsF64) {
  ComputationBuilder builder(client_, TestName());
  builder.Add(builder.ConstantR0<double>(0.25),
              builder.ConstantR0<double>(3.5));

  ComputeAndCompareR0<double>(&builder, 3.75, {});
}

TEST_F(ScalarComputationsTest, SubtractTwoScalarsF32) {
  ComputationBuilder builder(client_, TestName());
  builder.Sub(builder.ConstantR0<float>(2.1f), builder.ConstantR0<float>(5.5f));

  ComputeAndCompareR0<float>(&builder, -3.4f, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, SubtractTwoScalarsS32) {
  ComputationBuilder builder(client_, TestName());
  builder.Sub(builder.ConstantR0<int32>(2), builder.ConstantR0<int32>(5));

  ComputeAndCompareR0<int32>(&builder, -3, {});
}

TEST_F(ScalarComputationsTest, MulThreeScalarsF32) {
  ComputationBuilder builder(client_, TestName());
  builder.Mul(builder.Mul(builder.ConstantR0<float>(2.1f),
                          builder.ConstantR0<float>(5.5f)),
              builder.ConstantR0<float>(0.5f));

  ComputeAndCompareR0<float>(&builder, 5.775f, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, MulTwoScalarsS32) {
  std::vector<int32> data = {0,
                             1,
                             -1,
                             1234,
                             0x1a243514,
                             std::numeric_limits<int32>::max(),
                             std::numeric_limits<int32>::min()};

  for (int32 x : data) {
    for (int32 y : data) {
      ComputationBuilder builder(client_, TestName());
      builder.Mul(builder.ConstantR0<int32>(x), builder.ConstantR0<int32>(y));

      // Signed integer overflow is undefined behavior in C++. Convert the input
      // integers to unsigned, perform the multiplication unsigned, and convert
      // back.
      int32 expected = static_cast<uint32>(x) * static_cast<uint32>(y);

      ComputeAndCompareR0<int32>(&builder, expected, {});
    }
  }
}

TEST_F(ScalarComputationsTest, MulTwoScalarsU32) {
  std::vector<uint32> data = {0,          1,          0xDEADBEEF, 1234,
                              0x1a243514, 0xFFFFFFFF, 0x80808080};

  for (uint32 x : data) {
    for (uint32 y : data) {
      ComputationBuilder builder(client_, TestName());
      builder.Mul(builder.ConstantR0<uint32>(x), builder.ConstantR0<uint32>(y));

      uint32 expected = x * y;
      ComputeAndCompareR0<uint32>(&builder, expected, {});
    }
  }
}

TEST_F(ScalarComputationsTest, MulThreeScalarsS32) {
  ComputationBuilder builder(client_, TestName());
  builder.Mul(
      builder.Mul(builder.ConstantR0<int32>(2), builder.ConstantR0<int32>(5)),
      builder.ConstantR0<int32>(1));

  ComputeAndCompareR0<int32>(&builder, 10, {});
}

TEST_F(ScalarComputationsTest, MulThreeScalarsF32Params) {
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> a_literal = LiteralUtil::CreateR0<float>(2.1f);
  std::unique_ptr<Literal> b_literal = LiteralUtil::CreateR0<float>(5.5f);
  std::unique_ptr<Literal> c_literal = LiteralUtil::CreateR0<float>(0.5f);

  std::unique_ptr<GlobalData> a_data =
      client_->TransferToServer(*a_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> b_data =
      client_->TransferToServer(*b_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> c_data =
      client_->TransferToServer(*c_literal).ConsumeValueOrDie();

  ComputationDataHandle a = builder.Parameter(0, a_literal->shape(), "a");
  ComputationDataHandle b = builder.Parameter(1, b_literal->shape(), "b");
  ComputationDataHandle c = builder.Parameter(2, c_literal->shape(), "c");
  builder.Mul(builder.Mul(a, b), c);

  ComputeAndCompareR0<float>(&builder, 5.775f,
                             {a_data.get(), b_data.get(), c_data.get()},
                             error_spec_);
}

TEST_F(ScalarComputationsTest, DivideTwoScalarsF32) {
  ComputationBuilder builder(client_, TestName());
  builder.Div(builder.ConstantR0<float>(5.0f), builder.ConstantR0<float>(2.5f));

  ComputeAndCompareR0<float>(&builder, 2.0f, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, RemTwoScalarsF32) {
  ComputationBuilder builder(client_, TestName());
  builder.Rem(builder.ConstantR0<float>(2.5f), builder.ConstantR0<float>(5.0f));

  ComputeAndCompareR0<float>(&builder, 2.5f, {}, error_spec_);
}

struct DivS32Params {
  int32 dividend;
  int32 divisor;
  int32 quotient;
  int32 remainder;
};

void PrintTo(const DivS32Params& p, std::ostream* os) {
  *os << "{" << p.dividend << ", " << p.divisor << ", " << p.quotient << ", "
      << p.remainder << "}";
}

class DivS32Test : public ClientLibraryTestBase,
                   public ::testing::WithParamInterface<DivS32Params> {};

XLA_TEST_P(DivS32Test, DivideTwoScalarsS32) {
  DivS32Params p = GetParam();
  ComputationBuilder builder(client_, TestName());
  builder.Div(builder.ConstantR0<int32>(p.dividend),
              builder.ConstantR0<int32>(p.divisor));

  ComputeAndCompareR0<int32>(&builder, p.quotient, {});
}

XLA_TEST_P(DivS32Test, RemainderTwoScalarsS32) {
  DivS32Params p = GetParam();
  ComputationBuilder builder(client_, TestName());
  builder.Rem(builder.ConstantR0<int32>(p.dividend),
              builder.ConstantR0<int32>(p.divisor));

  ComputeAndCompareR0<int32>(&builder, p.remainder, {});
}

INSTANTIATE_TEST_CASE_P(
    DivS32Test_Instantiation, DivS32Test,
    ::testing::Values(
        // Positive divisors.
        DivS32Params{5, 2, 2, 1},      //
        DivS32Params{-5, 2, -2, -1},   //
        DivS32Params{17, 3, 5, 2},     //
        DivS32Params{-17, 3, -5, -2},  //
        // Negative divisors.
        DivS32Params{5, -2, -2, 1},    //
        DivS32Params{-5, -2, 2, -1},   //
        DivS32Params{17, -3, -5, 2},   //
        DivS32Params{-17, -3, 5, -2},  //
        // Large positive divisors.
        DivS32Params{INT32_MIN, 7919, -271181, -1309},             //
        DivS32Params{INT32_MIN, INT32_MAX, -1, -1},                //
        DivS32Params{INT32_MIN + 1, INT32_MAX, -1, 0},             //
        DivS32Params{INT32_MIN + 2, INT32_MAX, 0, INT32_MIN + 2},  //
        DivS32Params{INT32_MIN, 0x40000000, -2, 0},                //
        DivS32Params{INT32_MIN + 1, 0x40000000, -1, -0x3fffffff},  //
        // Large negative divisors.
        DivS32Params{INT32_MIN, INT32_MIN, 1, 0},                  //
        DivS32Params{INT32_MIN, INT32_MIN + 1, 1, -1},             //
        DivS32Params{INT32_MIN + 1, INT32_MIN, 0, INT32_MIN + 1},  //
        DivS32Params{INT32_MAX, INT32_MIN, 0, INT32_MAX},          //
        DivS32Params{INT32_MAX, INT32_MIN + 1, -1, 0},             //
        DivS32Params{INT32_MIN, -0x40000000, 2, 0},                //
        DivS32Params{INT32_MIN + 1, -0x40000000, 1, -0x3fffffff}));

TEST_F(ScalarComputationsTest, RemainderTwoScalarsNonConstDividendS32) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(S32, {}), "x");
  builder.Rem(x, builder.ConstantR0<int32>(80000));

  std::unique_ptr<Literal> literal = LiteralUtil::CreateR0<int32>(87919);
  TF_ASSIGN_OR_ASSERT_OK(auto input_data, client_->TransferToServer(*literal));
  ComputeAndCompareR0<int32>(&builder, 7919, {input_data.get()});
}

XLA_TEST_F(ScalarComputationsTest, DivideTwoScalarsU32) {
  ComputationBuilder builder(client_, TestName());
  // This verifies 0xFFFFFFFE / 2 = 0x7FFFFFFF. If XLA incorrectly treated U32
  // as S32, it would output -2 / 2 = -1 (0xFFFFFFFF).
  builder.Div(builder.ConstantR0<uint32>(0xFFFFFFFE),
              builder.ConstantR0<uint32>(2));

  ComputeAndCompareR0<uint32>(&builder, 0x7FFFFFFF, {});
}

XLA_TEST_F(ScalarComputationsTest, RemTwoScalarsU32) {
  ComputationBuilder builder(client_, TestName());
  builder.Rem(builder.ConstantR0<uint32>(11), builder.ConstantR0<uint32>(3));

  ComputeAndCompareR0<uint32>(&builder, 2, {});
}

TEST_F(ScalarComputationsTest, LogicalAnd) {
  for (bool x : {false, true}) {
    for (bool y : {false, true}) {
      ComputationBuilder builder(client_, TestName());
      builder.LogicalAnd(builder.ConstantR0<bool>(x),
                         builder.ConstantR0<bool>(y));

      ComputeAndCompareR0<bool>(&builder, x && y, {});
    }
  }
}

TEST_F(ScalarComputationsTest, LogicalOr) {
  for (bool x : {false, true}) {
    for (bool y : {false, true}) {
      ComputationBuilder builder(client_, TestName());
      builder.LogicalOr(builder.ConstantR0<bool>(x),
                        builder.ConstantR0<bool>(y));

      ComputeAndCompareR0<bool>(&builder, x || y, {});
    }
  }
}

TEST_F(ScalarComputationsTest, LogicalNot) {
  for (bool x : {false, true}) {
    ComputationBuilder builder(client_, TestName());
    builder.LogicalNot(builder.ConstantR0<bool>(x));

    ComputeAndCompareR0<bool>(&builder, !x, {});
  }
}

TEST_F(ScalarComputationsTest, SelectScalarTrue) {
  ComputationBuilder builder(client_, TestName());
  builder.Select(builder.ConstantR0<bool>(true),     // The predicate.
                 builder.ConstantR0<float>(123.0f),  // The value on true.
                 builder.ConstantR0<float>(42.0f));  // The value on false.

  ComputeAndCompareR0<float>(&builder, 123.0f, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, SelectScalarFalse) {
  ComputationBuilder builder(client_, TestName());
  builder.Select(builder.ConstantR0<bool>(false),    // The predicate.
                 builder.ConstantR0<float>(123.0f),  // The value on true.
                 builder.ConstantR0<float>(42.0f));  // The value on false.

  ComputeAndCompareR0<float>(&builder, 42.0f, {}, error_spec_);
}

// This test is an explicit version of what is happening in the following
// templatized comparison tests.
TEST_F(ScalarComputationsTest, CompareGtScalar) {
  ComputationBuilder builder(client_, TestName());
  builder.Gt(builder.ConstantR0<float>(2.0f), builder.ConstantR0<float>(1.0f));

  ComputeAndCompareR0<bool>(&builder, true, {});
}

// S32 comparisons.
TEST_F(ScalarComputationsTest, CompareEqS32Greater) {
  TestCompare<int32>(2, 1, false, &ComputationBuilder::Eq);
}
TEST_F(ScalarComputationsTest, CompareEqS32Equal) {
  TestCompare<int32>(3, 3, true, &ComputationBuilder::Eq);
}

TEST_F(ScalarComputationsTest, CompareNeS32) {
  TestCompare<int32>(2, 1, true, &ComputationBuilder::Ne);
}

TEST_F(ScalarComputationsTest, CompareGeS32) {
  TestCompare<int32>(2, 1, true, &ComputationBuilder::Ge);
}

TEST_F(ScalarComputationsTest, CompareGtS32) {
  TestCompare<int32>(1, 5, false, &ComputationBuilder::Gt);
}

TEST_F(ScalarComputationsTest, CompareLeS32) {
  TestCompare<int32>(2, 1, false, &ComputationBuilder::Le);
}

TEST_F(ScalarComputationsTest, CompareLtS32) {
  TestCompare<int32>(9, 7, false, &ComputationBuilder::Lt);
  TestCompare<int32>(std::numeric_limits<int32>::min(),
                     std::numeric_limits<int32>::max(), true,
                     &ComputationBuilder::Lt);
}

// U32 comparisons.
TEST_F(ScalarComputationsTest, CompareEqU32False) {
  TestCompare<uint32>(2, 1, false, &ComputationBuilder::Eq);
}

TEST_F(ScalarComputationsTest, CompareNeU32) {
  TestCompare<uint32>(2, 1, true, &ComputationBuilder::Ne);
}

TEST_F(ScalarComputationsTest, CompareGeU32Greater) {
  TestCompare<uint32>(2, 1, true, &ComputationBuilder::Ge);
}

TEST_F(ScalarComputationsTest, CompareGeU32Equal) {
  TestCompare<uint32>(3, 3, true, &ComputationBuilder::Ge);
}

TEST_F(ScalarComputationsTest, CompareGtU32) {
  TestCompare<uint32>(1, 5, false, &ComputationBuilder::Gt);
  TestCompare<uint32>(5, 5, false, &ComputationBuilder::Gt);
  TestCompare<uint32>(5, 1, true, &ComputationBuilder::Gt);
}

TEST_F(ScalarComputationsTest, CompareLeU32) {
  TestCompare<uint32>(2, 1, false, &ComputationBuilder::Le);
}

TEST_F(ScalarComputationsTest, CompareLtU32) {
  TestCompare<uint32>(9, 7, false, &ComputationBuilder::Lt);
  TestCompare<uint32>(0, std::numeric_limits<uint32>::max(), true,
                      &ComputationBuilder::Lt);
}

// F32 comparisons.
TEST_F(ScalarComputationsTest, CompareEqF32False) {
  TestCompare<float>(2.0, 1.3, false, &ComputationBuilder::Eq);
}

TEST_F(ScalarComputationsTest, CompareNeF32) {
  TestCompare<float>(2.0, 1.3, true, &ComputationBuilder::Ne);
}

TEST_F(ScalarComputationsTest, CompareGeF32Greater) {
  TestCompare<float>(2.0, 1.9, true, &ComputationBuilder::Ge);
}
TEST_F(ScalarComputationsTest, CompareGeF32Equal) {
  TestCompare<float>(3.5, 3.5, true, &ComputationBuilder::Ge);
}

TEST_F(ScalarComputationsTest, CompareGtF32) {
  TestCompare<float>(1.0, 5.2, false, &ComputationBuilder::Gt);
}

TEST_F(ScalarComputationsTest, CompareLeF32) {
  TestCompare<float>(2.0, 1.2, false, &ComputationBuilder::Le);
}

TEST_F(ScalarComputationsTest, CompareLtF32) {
  TestCompare<float>(9.0, 7.2, false, &ComputationBuilder::Lt);
}

// F32 comparisons with exceptional values.  The test names encode the
// left/right operands at the end, and use Minf and Mzero for -inf and -0.0.
TEST_F(ScalarComputationsTest, CompareLtF32MinfMzero) {
  TestCompare<float>(-INFINITY, -0.0, true, &ComputationBuilder::Lt);
}
TEST_F(ScalarComputationsTest, CompareLtF32MzeroZero) {
  // Comparisons of 0.0 to -0.0 consider them equal in IEEE 754.
  TestCompare<float>(-0.0, 0.0, false, &ComputationBuilder::Lt);
}
TEST_F(ScalarComputationsTest, CompareLtF32ZeroInf) {
  TestCompare<float>(0.0, INFINITY, true, &ComputationBuilder::Lt);
}

TEST_F(ScalarComputationsTest, CompareGeF32MinfMzero) {
  TestCompare<float>(-INFINITY, -0.0, false, &ComputationBuilder::Ge);
}
TEST_F(ScalarComputationsTest, CompareGeF32MzeroZero) {
  // Comparisons of 0.0 to -0.0 consider them equal in IEEE 754.
  TestCompare<float>(-0.0, 0.0, true, &ComputationBuilder::Ge);
}
TEST_F(ScalarComputationsTest, CompareGeF32ZeroInf) {
  TestCompare<float>(0.0, INFINITY, false, &ComputationBuilder::Ge);
}

TEST_F(ScalarComputationsTest, ExpScalar) {
  ComputationBuilder builder(client_, TestName());
  builder.Exp(builder.ConstantR0<float>(2.0f));

  ComputeAndCompareR0<float>(&builder, 7.3890562, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, LogScalar) {
  ComputationBuilder builder(client_, "log");
  builder.Log(builder.ConstantR0<float>(2.0f));

  ComputeAndCompareR0<float>(&builder, 0.6931471, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, TanhScalar) {
  ComputationBuilder builder(client_, TestName());
  builder.Tanh(builder.ConstantR0<float>(2.0f));

  ComputeAndCompareR0<float>(&builder, 0.96402758, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, TanhDoubleScalar) {
  ComputationBuilder builder(client_, TestName());
  builder.Tanh(builder.ConstantR0<double>(2.0));

  ComputeAndCompareR0<double>(&builder, 0.96402758, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, PowScalar) {
  ComputationBuilder builder(client_, TestName());
  builder.Pow(builder.ConstantR0<float>(2.0f), builder.ConstantR0<float>(3.0f));

  ComputeAndCompareR0<float>(&builder, 8.0, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, ClampScalarHigh) {
  ComputationBuilder builder(client_, TestName());
  builder.Clamp(builder.ConstantR0<float>(2.0f),   // The lower bound.
                builder.ConstantR0<float>(5.0f),   // The operand to be clamped.
                builder.ConstantR0<float>(3.0f));  // The upper bound.

  ComputeAndCompareR0<float>(&builder, 3.0, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, ClampScalarMiddle) {
  ComputationBuilder builder(client_, TestName());
  builder.Clamp(builder.ConstantR0<float>(2.0f),   // The lower bound.
                builder.ConstantR0<float>(2.5f),   // The operand to be clamped.
                builder.ConstantR0<float>(3.0f));  // The upper bound.

  ComputeAndCompareR0<float>(&builder, 2.5, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, ClampScalarLow) {
  ComputationBuilder builder(client_, TestName());
  builder.Clamp(builder.ConstantR0<float>(2.0f),   // The lower bound.
                builder.ConstantR0<float>(-5.0f),  // The operand to be clamped.
                builder.ConstantR0<float>(3.0f));  // The upper bound.

  ComputeAndCompareR0<float>(&builder, 2.0, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, MinS32Above) {
  TestMinMax<int32>(10, 3, 3, &ComputationBuilder::Min);
}

TEST_F(ScalarComputationsTest, MinS32Below) {
  TestMinMax<int32>(-100, 3, -100, &ComputationBuilder::Min);
}

TEST_F(ScalarComputationsTest, MaxS32Above) {
  TestMinMax<int32>(10, 3, 10, &ComputationBuilder::Max);
}

TEST_F(ScalarComputationsTest, MaxS32Below) {
  TestMinMax<int32>(-100, 3, 3, &ComputationBuilder::Max);
}

TEST_F(ScalarComputationsTest, MinU32Above) {
  const uint32 large = std::numeric_limits<int32>::max();
  TestMinMax<uint32>(large, 3, 3, &ComputationBuilder::Min);
}

TEST_F(ScalarComputationsTest, MinU32Below) {
  TestMinMax<uint32>(0, 5, 0, &ComputationBuilder::Min);
}

TEST_F(ScalarComputationsTest, MaxU32Above) {
  const uint32 large = std::numeric_limits<int32>::max();
  TestMinMax<uint32>(large, 3, large, &ComputationBuilder::Max);
}

TEST_F(ScalarComputationsTest, MaxU32Below) {
  TestMinMax<uint32>(0, 5, 5, &ComputationBuilder::Max);
}

TEST_F(ScalarComputationsTest, MinF32Above) {
  TestMinMax<float>(10.1f, 3.1f, 3.1f, &ComputationBuilder::Min);
}

TEST_F(ScalarComputationsTest, MinF32Below) {
  TestMinMax<float>(-100.1f, 3.1f, -100.1f, &ComputationBuilder::Min);
}

TEST_F(ScalarComputationsTest, MaxF32Above) {
  TestMinMax<float>(10.1f, 3.1f, 10.1f, &ComputationBuilder::Max);
}

TEST_F(ScalarComputationsTest, MaxF32Below) {
  TestMinMax<float>(-100.1f, 3.1f, 3.1f, &ComputationBuilder::Max);
}

TEST_F(ScalarComputationsTest, ComplicatedArithmeticExpressionF32) {
  // Compute the expression (1 * (3 - 1) * (7 + 0) - 4) / 20.
  ComputationBuilder b(client_, TestName());
  b.Div(
      b.Sub(b.Mul(b.ConstantR0<float>(1),
                  b.Mul(b.Sub(b.ConstantR0<float>(3), b.ConstantR0<float>(1)),
                        b.Add(b.ConstantR0<float>(7), b.ConstantR0<float>(0)))),
            b.ConstantR0<float>(4)),
      b.ConstantR0<float>(20));

  ComputeAndCompareR0<float>(&b, 0.5, {}, error_spec_);
}

TEST_F(ScalarComputationsTest, ComplicatedArithmeticExpressionS32) {
  // Compute the expression 1 * (3 - 1) * (7 + 0) - 4.
  ComputationBuilder b(client_, TestName());
  b.Sub(b.Mul(b.ConstantR0<int32>(1),
              b.Mul(b.Sub(b.ConstantR0<int32>(3), b.ConstantR0<int32>(1)),
                    b.Add(b.ConstantR0<int32>(7), b.ConstantR0<int32>(0)))),
        b.ConstantR0<int32>(4));

  ComputeAndCompareR0<int32>(&b, 10, {});
}

TEST_F(ScalarComputationsTest, SqrtF320) {
  ComputationBuilder builder(client_, TestName());
  Literal zero_literal = LiteralUtil::Zero(PrimitiveType::F32);

  std::unique_ptr<GlobalData> zero_data =
      client_->TransferToServer(zero_literal).ConsumeValueOrDie();

  ComputationDataHandle zero =
      builder.Parameter(0, zero_literal.shape(), "zero");
  builder.SqrtF32(zero);

  ComputeAndCompareR0<float>(&builder, 0.0f, {zero_data.get()}, error_spec_);
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
