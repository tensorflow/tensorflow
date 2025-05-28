/* Copyright 2017 The OpenXLA Authors.

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
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

constexpr ErrorSpec kErrorSpec{0.0001};

class ScalarComputationsTest
    : public ClientLibraryTestRunnerMixin<
          HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>> {
 protected:
  // A template for building and running a binary comparison test.
  template <typename NativeT>
  void TestCompare(NativeT lhs, NativeT rhs, bool expected,
                   const std::function<XlaOp(const XlaOp&, const XlaOp&,
                                             absl::Span<const int64_t>)>& op) {
    XlaBuilder builder(TestName());
    XlaOp lhs_op = ConstantR0<NativeT>(&builder, lhs);
    XlaOp rhs_op = ConstantR0<NativeT>(&builder, rhs);
    op(lhs_op, rhs_op, {});
    ComputeAndCompareR0<bool>(&builder, expected, {});
  }

  template <typename NativeT>
  void TestMinMax(NativeT lhs, NativeT rhs, NativeT expected,
                  const std::function<XlaOp(const XlaOp&, const XlaOp&,
                                            absl::Span<const int64_t>)>& op) {
    XlaBuilder builder(TestName());
    XlaOp lhs_op = ConstantR0<NativeT>(&builder, lhs);
    XlaOp rhs_op = ConstantR0<NativeT>(&builder, rhs);
    XlaOp minmax_op = op(lhs_op, rhs_op, {});
    // Canonicalize NaNs so we can do a bitwise compare without caring about
    // payloads.
    if constexpr (std::is_floating_point_v<NativeT>) {
      XlaOp isnan_op = Ne(minmax_op, minmax_op);
      Select(isnan_op, ConstantR0<NativeT>(&builder, NAN), minmax_op);
    }
    ComputeAndCompareR0<NativeT>(&builder, expected, {});
  }
};

XLA_TEST_F(ScalarComputationsTest, ReturnScalarF32) {
  XlaBuilder builder(TestName());
  ConstantR0<float>(&builder, 2.1f);

  ComputeAndCompareR0<float>(&builder, 2.1f, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, NegateScalarF32) {
  XlaBuilder builder(TestName());
  Neg(ConstantR0<float>(&builder, 2.1f));

  ComputeAndCompareR0<float>(&builder, -2.1f, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, NegateScalarS32) {
  XlaBuilder builder(TestName());
  Neg(ConstantR0<int32_t>(&builder, 2));

  ComputeAndCompareR0<int32_t>(&builder, -2, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsF32) {
  XlaBuilder builder(TestName());
  Add(ConstantR0<float>(&builder, 2.1f), ConstantR0<float>(&builder, 5.5f));

  ComputeAndCompareR0<float>(&builder, 7.6f, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsS32) {
  XlaBuilder builder(TestName());
  Add(ConstantR0<int32_t>(&builder, 2), ConstantR0<int32_t>(&builder, 5));

  ComputeAndCompareR0<int32_t>(&builder, 7, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsU32) {
  XlaBuilder builder(TestName());
  Add(ConstantR0<uint32_t>(&builder, 35), ConstantR0<uint32_t>(&builder, 57));

  ComputeAndCompareR0<uint32_t>(&builder, 92, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsU8) {
  XlaBuilder builder(TestName());
  Add(ConstantR0<uint8_t>(&builder, 35), ConstantR0<uint8_t>(&builder, 57));

  ComputeAndCompareR0<uint8_t>(&builder, 92, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsU64) {
  XlaBuilder builder(TestName());
  const uint64_t a = static_cast<uint64_t>(1) << 63;
  const uint64_t b = a + 1;
  Add(ConstantR0<uint64_t>(&builder, a), ConstantR0<uint64_t>(&builder, b));

  ComputeAndCompareR0<uint64_t>(&builder, a + b, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsS64) {
  XlaBuilder builder(TestName());
  const int64_t a = static_cast<int64_t>(1) << 62;
  const int64_t b = a - 1;
  Add(ConstantR0<int64_t>(&builder, a), ConstantR0<int64_t>(&builder, b));

  ComputeAndCompareR0<int64_t>(&builder, a + b, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsF64) {
  XlaBuilder builder(TestName());
  Add(ConstantR0<double>(&builder, 0.25), ConstantR0<double>(&builder, 3.5));

  ComputeAndCompareR0<double>(&builder, 3.75, {});
}

XLA_TEST_F(ScalarComputationsTest, SubtractTwoScalarsF32) {
  XlaBuilder builder(TestName());
  Sub(ConstantR0<float>(&builder, 2.1f), ConstantR0<float>(&builder, 5.5f));

  ComputeAndCompareR0<float>(&builder, -3.4f, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, SubtractTwoScalarsS32) {
  XlaBuilder builder(TestName());
  Sub(ConstantR0<int32_t>(&builder, 2), ConstantR0<int32_t>(&builder, 5));

  ComputeAndCompareR0<int32_t>(&builder, -3, {});
}

XLA_TEST_F(ScalarComputationsTest, CastS64ToF32) {
  XlaBuilder builder(TestName());
  auto a = Parameter(&builder, 0, ShapeUtil::MakeShape(S64, {}), "a");
  ConvertElementType(a, F32);

  int64_t value = 3LL << 35;
  Literal a_literal = LiteralUtil::CreateR0<int64_t>(value);
  ComputeAndCompareR0<float>(&builder, static_cast<float>(value), {&a_literal});
}

XLA_TEST_F(ScalarComputationsTest, MulThreeScalarsF32) {
  XlaBuilder builder(TestName());
  Mul(Mul(ConstantR0<float>(&builder, 2.1f), ConstantR0<float>(&builder, 5.5f)),
      ConstantR0<float>(&builder, 0.5f));

  ComputeAndCompareR0<float>(&builder, 5.775f, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, MulThreeScalarsF64) {
  XlaBuilder builder(TestName());
  Mul(Mul(ConstantR0<double>(&builder, 3.1415926535897932),
          ConstantR0<double>(&builder, 2.7182818284590452)),
      ConstantR0<double>(&builder, 0.5772156649015328));

  ComputeAndCompareR0<double>(&builder, 4.929268367422896, {},
                              ErrorSpec{3.6e-15});
}

XLA_TEST_F(ScalarComputationsTest, MulTwoScalarsS32) {
  std::vector<int32_t> data = {0,
                               1,
                               -1,
                               1234,
                               0x1a243514,
                               std::numeric_limits<int32_t>::max(),
                               std::numeric_limits<int32_t>::min()};

  for (int32_t x : data) {
    for (int32_t y : data) {
      XlaBuilder builder(TestName());
      Mul(ConstantR0<int32_t>(&builder, x), ConstantR0<int32_t>(&builder, y));

      // Signed integer overflow is undefined behavior in C++. Convert the input
      // integers to unsigned, perform the multiplication unsigned, and convert
      // back.
      int32_t expected = static_cast<uint32_t>(x) * static_cast<uint32_t>(y);

      ComputeAndCompareR0<int32_t>(&builder, expected, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, MulTwoScalarsU32) {
  std::vector<uint32_t> data = {0,          1,          0xDEADBEEF, 1234,
                                0x1a243514, 0xFFFFFFFF, 0x80808080};

  for (uint32_t x : data) {
    for (uint32_t y : data) {
      XlaBuilder builder(TestName());
      Mul(ConstantR0<uint32_t>(&builder, x), ConstantR0<uint32_t>(&builder, y));

      uint32_t expected = x * y;
      ComputeAndCompareR0<uint32_t>(&builder, expected, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, MulThreeScalarsS32) {
  XlaBuilder builder(TestName());
  Mul(Mul(ConstantR0<int32_t>(&builder, 2), ConstantR0<int32_t>(&builder, 5)),
      ConstantR0<int32_t>(&builder, 1));

  ComputeAndCompareR0<int32_t>(&builder, 10, {});
}

XLA_TEST_F(ScalarComputationsTest, MulThreeScalarsF32Params) {
  XlaBuilder builder(TestName());
  const Literal a_literal = LiteralUtil::CreateR0<float>(2.1f);
  const Literal b_literal = LiteralUtil::CreateR0<float>(5.5f);
  const Literal c_literal = LiteralUtil::CreateR0<float>(0.5f);

  XlaOp a = Parameter(&builder, 0, a_literal.shape(), "a");
  XlaOp b = Parameter(&builder, 1, b_literal.shape(), "b");
  XlaOp c = Parameter(&builder, 2, c_literal.shape(), "c");
  Mul(Mul(a, b), c);

  ComputeAndCompareR0<float>(&builder, 5.775f,
                             {&a_literal, &b_literal, &c_literal}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, DivideTwoScalarsF32) {
  XlaBuilder builder(TestName());
  Div(ConstantR0<float>(&builder, 5.0f), ConstantR0<float>(&builder, 2.5f));

  ComputeAndCompareR0<float>(&builder, 2.0f, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, RemTwoScalarsF32) {
  XlaBuilder builder(TestName());
  Rem(ConstantR0<float>(&builder, 2.5f), ConstantR0<float>(&builder, 5.0f));

  ComputeAndCompareR0<float>(&builder, 2.5f, {}, kErrorSpec);
}

struct DivS32Params {
  int32_t dividend;
  int32_t divisor;
  int32_t quotient;
  int32_t remainder;
};

void PrintTo(const DivS32Params& p, std::ostream* os) {
  *os << "{" << p.dividend << ", " << p.divisor << ", " << p.quotient << ", "
      << p.remainder << "}";
}

class DivS32Test : public ClientLibraryTestRunnerMixin<
                       HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>,
                   public ::testing::WithParamInterface<DivS32Params> {};

XLA_TEST_P(DivS32Test, DivideTwoScalarsS32) {
  DivS32Params p = GetParam();
  XlaBuilder builder(TestName());
  Div(ConstantR0<int32_t>(&builder, p.dividend),
      ConstantR0<int32_t>(&builder, p.divisor));

  ComputeAndCompareR0<int32_t>(&builder, p.quotient, {});
}

XLA_TEST_P(DivS32Test, RemainderTwoScalarsS32) {
  DivS32Params p = GetParam();
  XlaBuilder builder(TestName());
  Rem(ConstantR0<int32_t>(&builder, p.dividend),
      ConstantR0<int32_t>(&builder, p.divisor));

  ComputeAndCompareR0<int32_t>(&builder, p.remainder, {});
}

XLA_TEST_P(DivS32Test, DivideTwoScalarsNonConstS32) {
  DivS32Params p = GetParam();
  XlaBuilder builder(TestName());
  XlaOp dividend;
  XlaOp divisor;
  auto dividendd = CreateR0Parameter<int32_t>(p.dividend, 0, "dividend",
                                              &builder, &dividend);
  auto divisord =
      CreateR0Parameter<int32_t>(p.divisor, 1, "divisor", &builder, &divisor);
  Div(dividend, divisor);

  ComputeAndCompareR0<int32_t>(&builder, p.quotient, {&dividendd, &divisord});
}

XLA_TEST_P(DivS32Test, RemainderTwoScalarsNonConstDivisorS32) {
  DivS32Params p = GetParam();
  XlaBuilder builder(TestName());
  XlaOp dividend;
  XlaOp divisor;
  auto dividendd = CreateR0Parameter<int32_t>(p.dividend, 0, "dividend",
                                              &builder, &dividend);
  auto divisord =
      CreateR0Parameter<int32_t>(p.divisor, 1, "divisor", &builder, &divisor);
  Rem(dividend, divisor);

  ComputeAndCompareR0<int32_t>(&builder, p.remainder, {&dividendd, &divisord});
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

XLA_TEST_F(ScalarComputationsTest, DivU32s) {
  // clang-format off
  // Some interesting values to test.
  std::vector<uint32_t> vals = {
    0, 1, 2, 17, 101, 3333, 0x7FFFFFFF, 0x80000000, UINT32_MAX - 1, UINT32_MAX};
  // clang-format on

  XlaComputation div_computation;
  {
    XlaBuilder builder(TestName());

    XlaOp dividend =
        Parameter(&builder, 0, ShapeUtil::MakeShape(U32, {}), "dividend");
    XlaOp divisor =
        Parameter(&builder, 1, ShapeUtil::MakeShape(U32, {}), "divisor");
    Div(dividend, divisor);
    TF_ASSERT_OK_AND_ASSIGN(div_computation, builder.Build());
  }

  for (uint32_t divisor : vals) {
    if (divisor != 0) {
      for (uint32_t dividend : vals) {
        const Literal dividend_literal =
            LiteralUtil::CreateR0<uint32_t>(dividend);
        const Literal divisor_literal =
            LiteralUtil::CreateR0<uint32_t>(divisor);
        TF_ASSERT_OK_AND_ASSIGN(
            const Literal actual_literal,
            ExecuteAndTransfer(div_computation,
                               {&dividend_literal, &divisor_literal}));
        const Literal expected_literal =
            LiteralUtil::CreateR0<uint32_t>(dividend / divisor);
        EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, actual_literal));
      }
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, RemU32s) {
  // clang-format off
  // Some interesting values to test.
  std::vector<uint32_t> vals = {
    0, 1, 2, 17, 101, 3333, 0x7FFFFFFF, 0x80000000, UINT32_MAX - 1, UINT32_MAX};
  // clang-format on

  XlaComputation rem_computation;
  {
    XlaBuilder builder(TestName());

    XlaOp dividend =
        Parameter(&builder, 0, ShapeUtil::MakeShape(U32, {}), "dividend");
    XlaOp divisor =
        Parameter(&builder, 1, ShapeUtil::MakeShape(U32, {}), "divisor");
    Rem(dividend, divisor);
    TF_ASSERT_OK_AND_ASSIGN(rem_computation, builder.Build());
  }

  for (uint32_t divisor : vals) {
    if (divisor != 0) {
      for (uint32_t dividend : vals) {
        const Literal dividend_literal =
            LiteralUtil::CreateR0<uint32_t>(dividend);
        const Literal divisor_literal =
            LiteralUtil::CreateR0<uint32_t>(divisor);
        TF_ASSERT_OK_AND_ASSIGN(
            const Literal actual_literal,
            ExecuteAndTransfer(rem_computation,
                               {&dividend_literal, &divisor_literal}));
        const Literal expected_literal =
            LiteralUtil::CreateR0<uint32_t>(dividend % divisor);
        EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, actual_literal));
      }
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, RemainderTwoScalarsNonConstDividendS32) {
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(S32, {}), "x");
  Rem(x, ConstantR0<int32_t>(&builder, 80000));

  Literal literal = LiteralUtil::CreateR0<int32_t>(87919);
  ComputeAndCompareR0<int32_t>(&builder, 7919, {&literal});
}

XLA_TEST_F(ScalarComputationsTest, DivideTwoScalarsU32) {
  XlaBuilder builder(TestName());
  // This verifies 0xFFFFFFFE / 2 = 0x7FFFFFFF. If XLA incorrectly treated U32
  // as S32, it would output -2 / 2 = -1 (0xFFFFFFFF).
  Div(ConstantR0<uint32_t>(&builder, 0xFFFFFFFE),
      ConstantR0<uint32_t>(&builder, 2));

  ComputeAndCompareR0<uint32_t>(&builder, 0x7FFFFFFF, {});
}

XLA_TEST_F(ScalarComputationsTest, RemTwoScalarsU32) {
  XlaBuilder builder(TestName());
  Rem(ConstantR0<uint32_t>(&builder, 11), ConstantR0<uint32_t>(&builder, 3));

  ComputeAndCompareR0<uint32_t>(&builder, 2, {});
}

XLA_TEST_F(ScalarComputationsTest, AndBool) {
  for (bool x : {false, true}) {
    for (bool y : {false, true}) {
      XlaBuilder builder(TestName());
      And(ConstantR0<bool>(&builder, x), ConstantR0<bool>(&builder, y));

      ComputeAndCompareR0<bool>(&builder, x && y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, AndS32) {
  for (int32_t x : {0, 8}) {
    for (int32_t y : {1, -16}) {
      XlaBuilder builder(TestName());
      And(ConstantR0<int32_t>(&builder, x), ConstantR0<int32_t>(&builder, y));

      ComputeAndCompareR0<int32_t>(&builder, x & y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, AndU32) {
  for (uint32_t x : {0, 8}) {
    for (uint32_t y : {1, 16}) {
      XlaBuilder builder(TestName());
      And(ConstantR0<uint32_t>(&builder, x), ConstantR0<uint32_t>(&builder, y));

      ComputeAndCompareR0<uint32_t>(&builder, x & y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, OrBool) {
  for (bool x : {false, true}) {
    for (bool y : {false, true}) {
      XlaBuilder builder(TestName());
      Or(ConstantR0<bool>(&builder, x), ConstantR0<bool>(&builder, y));

      ComputeAndCompareR0<bool>(&builder, x || y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, OrS32) {
  for (int32_t x : {0, 8}) {
    for (int32_t y : {1, -16}) {
      XlaBuilder builder(TestName());
      Or(ConstantR0<int32_t>(&builder, x), ConstantR0<int32_t>(&builder, y));

      ComputeAndCompareR0<int32_t>(&builder, x | y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, OrU32) {
  for (uint32_t x : {0, 8}) {
    for (uint32_t y : {1, 16}) {
      XlaBuilder builder(TestName());
      Or(ConstantR0<uint32_t>(&builder, x), ConstantR0<uint32_t>(&builder, y));

      ComputeAndCompareR0<uint32_t>(&builder, x | y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, NotBool) {
  for (bool x : {false, true}) {
    XlaBuilder builder(TestName());
    Not(ConstantR0<bool>(&builder, x));

    ComputeAndCompareR0<bool>(&builder, !x, {});
  }
}

XLA_TEST_F(ScalarComputationsTest, NotS32) {
  for (int32_t x : {-1, 0, 1}) {
    XlaBuilder builder(TestName());
    Not(ConstantR0<int32_t>(&builder, x));

    ComputeAndCompareR0<int32_t>(&builder, ~x, {});
  }
}

XLA_TEST_F(ScalarComputationsTest, NotU32) {
  for (uint32_t x : {0, 1, 2}) {
    XlaBuilder builder(TestName());
    Not(ConstantR0<uint32_t>(&builder, x));

    ComputeAndCompareR0<uint32_t>(&builder, ~x, {});
  }
}

XLA_TEST_F(ScalarComputationsTest, SelectScalarTrue) {
  XlaBuilder builder(TestName());
  Select(ConstantR0<bool>(&builder, true),     // The predicate.
         ConstantR0<float>(&builder, 123.0f),  // The value on true.
         ConstantR0<float>(&builder, 42.0f));  // The value on false.

  ComputeAndCompareR0<float>(&builder, 123.0f, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, SelectScalarFalse) {
  XlaBuilder builder(TestName());
  Select(ConstantR0<bool>(&builder, false),    // The predicate.
         ConstantR0<float>(&builder, 123.0f),  // The value on true.
         ConstantR0<float>(&builder, 42.0f));  // The value on false.

  ComputeAndCompareR0<float>(&builder, 42.0f, {}, kErrorSpec);
}

// This test is an explicit version of what is happening in the following
// templatized comparison tests.
XLA_TEST_F(ScalarComputationsTest, CompareGtScalar) {
  XlaBuilder builder(TestName());
  Gt(ConstantR0<float>(&builder, 2.0f), ConstantR0<float>(&builder, 1.0f));

  ComputeAndCompareR0<bool>(&builder, true, {});
}

// S32 comparisons.
XLA_TEST_F(ScalarComputationsTest, CompareEqS32Greater) {
  TestCompare<int32_t>(2, 1, false, &Eq);
}
XLA_TEST_F(ScalarComputationsTest, CompareEqS32Equal) {
  TestCompare<int32_t>(3, 3, true, &Eq);
}

XLA_TEST_F(ScalarComputationsTest, CompareNeS32) {
  TestCompare<int32_t>(2, 1, true, &Ne);
}

XLA_TEST_F(ScalarComputationsTest, CompareGeS32) {
  TestCompare<int32_t>(2, 1, true, &Ge);
}

XLA_TEST_F(ScalarComputationsTest, CompareGtS32) {
  TestCompare<int32_t>(1, 5, false, &Gt);
}

XLA_TEST_F(ScalarComputationsTest, CompareLeS32) {
  TestCompare<int32_t>(2, 1, false, &Le);
}

XLA_TEST_F(ScalarComputationsTest, CompareLtS32) {
  TestCompare<int32_t>(9, 7, false, &Lt);
  TestCompare<int32_t>(std::numeric_limits<int32_t>::min(),
                       std::numeric_limits<int32_t>::max(), true, &Lt);
}

// U32 comparisons.
XLA_TEST_F(ScalarComputationsTest, CompareEqU32False) {
  TestCompare<uint32_t>(2, 1, false, &Eq);
}

XLA_TEST_F(ScalarComputationsTest, CompareNeU32) {
  TestCompare<uint32_t>(2, 1, true, &Ne);
}

XLA_TEST_F(ScalarComputationsTest, CompareGeU32Greater) {
  TestCompare<uint32_t>(2, 1, true, &Ge);
}

XLA_TEST_F(ScalarComputationsTest, CompareGeU32Equal) {
  TestCompare<uint32_t>(3, 3, true, &Ge);
}

XLA_TEST_F(ScalarComputationsTest, CompareGtU32) {
  TestCompare<uint32_t>(1, 5, false, &Gt);
  TestCompare<uint32_t>(5, 5, false, &Gt);
  TestCompare<uint32_t>(5, 1, true, &Gt);
}

XLA_TEST_F(ScalarComputationsTest, CompareLeU32) {
  TestCompare<uint32_t>(2, 1, false, &Le);
}

XLA_TEST_F(ScalarComputationsTest, CompareLtU32) {
  TestCompare<uint32_t>(9, 7, false, &Lt);
  TestCompare<uint32_t>(0, std::numeric_limits<uint32_t>::max(), true, &Lt);
}

// F32 comparisons.
XLA_TEST_F(ScalarComputationsTest, CompareEqF32False) {
  TestCompare<float>(2.0, 1.3, false, &Eq);
}

XLA_TEST_F(ScalarComputationsTest, CompareNeF32) {
  TestCompare<float>(2.0, 1.3, true, &Ne);
}

XLA_TEST_F(ScalarComputationsTest, CompareGeF32Greater) {
  TestCompare<float>(2.0, 1.9, true, &Ge);
}
XLA_TEST_F(ScalarComputationsTest, CompareGeF32Equal) {
  TestCompare<float>(3.5, 3.5, true, &Ge);
}

XLA_TEST_F(ScalarComputationsTest, CompareGtF32) {
  TestCompare<float>(1.0, 5.2, false, &Gt);
}

XLA_TEST_F(ScalarComputationsTest, CompareLeF32) {
  TestCompare<float>(2.0, 1.2, false, &Le);
}

XLA_TEST_F(ScalarComputationsTest, CompareLtF32) {
  TestCompare<float>(9.0, 7.2, false, &Lt);
}

// F32 comparisons with exceptional values.  The test names encode the
// left/right operands at the end, and use Minf and Mzero for -inf and -0.0.
XLA_TEST_F(ScalarComputationsTest, CompareLtF32MinfMzero) {
  TestCompare<float>(-INFINITY, -0.0, true, &Lt);
}
XLA_TEST_F(ScalarComputationsTest, CompareLtF32MzeroZero) {
  // Comparisons of 0.0 to -0.0 consider them equal in IEEE 754.
  TestCompare<float>(-0.0, 0.0, false, &Lt);
}
XLA_TEST_F(ScalarComputationsTest, CompareLtF32ZeroInf) {
  TestCompare<float>(0.0, INFINITY, true, &Lt);
}

XLA_TEST_F(ScalarComputationsTest, CompareGeF32MinfMzero) {
  TestCompare<float>(-INFINITY, -0.0, false, &Ge);
}
XLA_TEST_F(ScalarComputationsTest, CompareGeF32MzeroZero) {
  // Comparisons of 0.0 to -0.0 consider them equal in IEEE 754.
  TestCompare<float>(-0.0, 0.0, true, &Ge);
}
XLA_TEST_F(ScalarComputationsTest, CompareGeF32ZeroInf) {
  TestCompare<float>(0.0, INFINITY, false, &Ge);
}

XLA_TEST_F(ScalarComputationsTest, ExpScalar) {
  XlaBuilder builder(TestName());
  Exp(ConstantR0<float>(&builder, 2.0f));

  ComputeAndCompareR0<float>(&builder, 7.3890562, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, LogScalar) {
  XlaBuilder builder("log");
  Log(ConstantR0<float>(&builder, 2.0f));

  ComputeAndCompareR0<float>(&builder, 0.6931471, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, TanhScalar) {
  XlaBuilder builder(TestName());
  Tanh(ConstantR0<float>(&builder, 2.0f));

  ComputeAndCompareR0<float>(&builder, 0.96402758, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, TanhDoubleScalar) {
  XlaBuilder builder(TestName());
  Tanh(ConstantR0<double>(&builder, 2.0));

  ComputeAndCompareR0<double>(&builder, 0.96402758, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, PowScalar) {
  XlaBuilder builder(TestName());
  Pow(ConstantR0<float>(&builder, 2.0f), ConstantR0<float>(&builder, 3.0f));

  ComputeAndCompareR0<float>(&builder, 8.0, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, CbrtScalar) {
  XlaBuilder builder(TestName());
  Cbrt(ConstantR0<float>(&builder, 2.0f));

  ComputeAndCompare(&builder, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarHighS32) {
  XlaBuilder builder(TestName());
  Clamp(ConstantR0<int32_t>(&builder, -1),  // The lower bound.
        ConstantR0<int32_t>(&builder, 5),   // The operand to be clamped.
        ConstantR0<int32_t>(&builder, 3));  // The upper bound.

  ComputeAndCompareR0<int32_t>(&builder, 3, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarMiddleS32) {
  XlaBuilder builder(TestName());
  Clamp(ConstantR0<int32_t>(&builder, -1),  // The lower bound.
        ConstantR0<int32_t>(&builder, 2),   // The operand to be clamped.
        ConstantR0<int32_t>(&builder, 3));  // The upper bound.

  ComputeAndCompareR0<int32_t>(&builder, 2, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarLowS32) {
  XlaBuilder builder(TestName());
  Clamp(ConstantR0<int32_t>(&builder, -1),  // The lower bound.
        ConstantR0<int32_t>(&builder, -5),  // The operand to be clamped.
        ConstantR0<int32_t>(&builder, 3));  // The upper bound.

  ComputeAndCompareR0<int32_t>(&builder, -1, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarHighU32) {
  XlaBuilder builder(TestName());
  Clamp(ConstantR0<uint32_t>(&builder, 1),   // The lower bound.
        ConstantR0<uint32_t>(&builder, 5),   // The operand to be clamped.
        ConstantR0<uint32_t>(&builder, 3));  // The upper bound.

  ComputeAndCompareR0<uint32_t>(&builder, 3, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarMiddleU32) {
  XlaBuilder builder(TestName());
  Clamp(ConstantR0<uint32_t>(&builder, 1),   // The lower bound.
        ConstantR0<uint32_t>(&builder, 2),   // The operand to be clamped.
        ConstantR0<uint32_t>(&builder, 3));  // The upper bound.

  ComputeAndCompareR0<uint32_t>(&builder, 2, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarLowU32) {
  XlaBuilder builder(TestName());
  Clamp(ConstantR0<uint32_t>(&builder, 1),   // The lower bound.
        ConstantR0<uint32_t>(&builder, 0),   // The operand to be clamped.
        ConstantR0<uint32_t>(&builder, 3));  // The upper bound.

  ComputeAndCompareR0<uint32_t>(&builder, 1, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarHighF32) {
  XlaBuilder builder(TestName());
  Clamp(ConstantR0<float>(&builder, 2.0f),   // The lower bound.
        ConstantR0<float>(&builder, 5.0f),   // The operand to be clamped.
        ConstantR0<float>(&builder, 3.0f));  // The upper bound.

  ComputeAndCompareR0<float>(&builder, 3.0, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarMiddleF32) {
  XlaBuilder builder(TestName());
  Clamp(ConstantR0<float>(&builder, 2.0f),   // The lower bound.
        ConstantR0<float>(&builder, 2.5f),   // The operand to be clamped.
        ConstantR0<float>(&builder, 3.0f));  // The upper bound.

  ComputeAndCompareR0<float>(&builder, 2.5, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarLowF32) {
  XlaBuilder builder(TestName());
  Clamp(ConstantR0<float>(&builder, 2.0f),   // The lower bound.
        ConstantR0<float>(&builder, -5.0f),  // The operand to be clamped.
        ConstantR0<float>(&builder, 3.0f));  // The upper bound.

  ComputeAndCompareR0<float>(&builder, 2.0, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, MinS32Above) {
  TestMinMax<int32_t>(10, 3, 3, &Min);
}

XLA_TEST_F(ScalarComputationsTest, MinS32Below) {
  TestMinMax<int32_t>(-100, 3, -100, &Min);
}

XLA_TEST_F(ScalarComputationsTest, MaxS32Above) {
  TestMinMax<int32_t>(10, 3, 10, &Max);
}

XLA_TEST_F(ScalarComputationsTest, MaxS32Below) {
  TestMinMax<int32_t>(-100, 3, 3, &Max);
}

XLA_TEST_F(ScalarComputationsTest, MinU32Above) {
  const uint32_t large = std::numeric_limits<int32_t>::max();
  TestMinMax<uint32_t>(large, 3, 3, &Min);
}

XLA_TEST_F(ScalarComputationsTest, MinU32Below) {
  TestMinMax<uint32_t>(0, 5, 0, &Min);
}

XLA_TEST_F(ScalarComputationsTest, MaxU32Above) {
  const uint32_t large = std::numeric_limits<int32_t>::max();
  TestMinMax<uint32_t>(large, 3, large, &Max);
}

XLA_TEST_F(ScalarComputationsTest, MaxU32Below) {
  TestMinMax<uint32_t>(0, 5, 5, &Max);
}

XLA_TEST_F(ScalarComputationsTest, MinF32Above) {
  TestMinMax<float>(10.1f, 3.1f, 3.1f, &Min);
}

XLA_TEST_F(ScalarComputationsTest, MinF32Below) {
  TestMinMax<float>(-100.1f, 3.1f, -100.1f, &Min);
}

XLA_TEST_F(ScalarComputationsTest, MinPropagatesNan) {
  SetFastMathDisabled(true);
  TestMinMax<float>(NAN, 3.1f, NAN, &Min);
  TestMinMax<float>(-3.1f, NAN, NAN, &Min);
}

XLA_TEST_F(ScalarComputationsTest, MaxF32Above) {
  TestMinMax<float>(10.1f, 3.1f, 10.1f, &Max);
}

XLA_TEST_F(ScalarComputationsTest, MaxF32Below) {
  TestMinMax<float>(-100.1f, 3.1f, 3.1f, &Max);
}

XLA_TEST_F(ScalarComputationsTest, MaxPropagatesNan) {
  SetFastMathDisabled(true);
  TestMinMax<float>(NAN, 3.1f, NAN, &Max);
  TestMinMax<float>(-3.1f, NAN, NAN, &Max);
}

XLA_TEST_F(ScalarComputationsTest, ComplicatedArithmeticExpressionF32) {
  // Compute the expression (1 * (3 - 1) * (7 + 0) - 4) / 20.
  XlaBuilder b(TestName());
  Div(Sub(Mul(ConstantR0<float>(&b, 1),
              Mul(Sub(ConstantR0<float>(&b, 3), ConstantR0<float>(&b, 1)),
                  Add(ConstantR0<float>(&b, 7), ConstantR0<float>(&b, 0)))),
          ConstantR0<float>(&b, 4)),
      ConstantR0<float>(&b, 20));

  ComputeAndCompareR0<float>(&b, 0.5, {}, kErrorSpec);
}

XLA_TEST_F(ScalarComputationsTest, ComplicatedArithmeticExpressionS32) {
  // Compute the expression 1 * (3 - 1) * (7 + 0) - 4.
  XlaBuilder b(TestName());
  Sub(Mul(ConstantR0<int32_t>(&b, 1),
          Mul(Sub(ConstantR0<int32_t>(&b, 3), ConstantR0<int32_t>(&b, 1)),
              Add(ConstantR0<int32_t>(&b, 7), ConstantR0<int32_t>(&b, 0)))),
      ConstantR0<int32_t>(&b, 4));

  ComputeAndCompareR0<int32_t>(&b, 10, {});
}

XLA_TEST_F(ScalarComputationsTest, RoundScalar) {
  XlaBuilder builder(TestName());
  Round(ConstantR0<float>(&builder, 1.4f));

  ComputeAndCompareR0<float>(&builder, 1.0f, {}, kErrorSpec);
}

}  // namespace
}  // namespace xla
