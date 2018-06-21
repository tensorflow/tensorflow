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

#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
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
  void TestCompare(
      NativeT lhs, NativeT rhs, bool expected,
      XlaOp (XlaBuilder::*op)(const XlaOp&, const XlaOp&,
                              tensorflow::gtl::ArraySlice<int64>)) {
    XlaBuilder builder(TestName());
    XlaOp lhs_op = builder.ConstantR0<NativeT>(lhs);
    XlaOp rhs_op = builder.ConstantR0<NativeT>(rhs);
    XlaOp result = (builder.*op)(lhs_op, rhs_op, {});
    ComputeAndCompareR0<bool>(&builder, expected, {});
  }

  template <typename NativeT>
  void TestMinMax(NativeT lhs, NativeT rhs, NativeT expected,
                  XlaOp (XlaBuilder::*op)(const XlaOp&, const XlaOp&,
                                          tensorflow::gtl::ArraySlice<int64>)) {
    XlaBuilder builder(TestName());
    XlaOp lhs_op = builder.ConstantR0<NativeT>(lhs);
    XlaOp rhs_op = builder.ConstantR0<NativeT>(rhs);
    XlaOp result = (builder.*op)(lhs_op, rhs_op, {});
    ComputeAndCompareR0<NativeT>(&builder, expected, {});
  }
};

XLA_TEST_F(ScalarComputationsTest, ReturnScalarF32) {
  XlaBuilder builder(TestName());
  builder.ConstantR0<float>(2.1f);

  ComputeAndCompareR0<float>(&builder, 2.1f, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, NegateScalarF32) {
  XlaBuilder builder(TestName());
  builder.Neg(builder.ConstantR0<float>(2.1f));

  ComputeAndCompareR0<float>(&builder, -2.1f, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, NegateScalarS32) {
  XlaBuilder builder(TestName());
  builder.Neg(builder.ConstantR0<int32>(2));

  ComputeAndCompareR0<int32>(&builder, -2, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsF32) {
  XlaBuilder builder(TestName());
  builder.Add(builder.ConstantR0<float>(2.1f), builder.ConstantR0<float>(5.5f));

  ComputeAndCompareR0<float>(&builder, 7.6f, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsS32) {
  XlaBuilder builder(TestName());
  builder.Add(builder.ConstantR0<int32>(2), builder.ConstantR0<int32>(5));

  ComputeAndCompareR0<int32>(&builder, 7, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsU32) {
  XlaBuilder builder(TestName());
  builder.Add(builder.ConstantR0<uint32>(35), builder.ConstantR0<uint32>(57));

  ComputeAndCompareR0<uint32>(&builder, 92, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsU8) {
  XlaBuilder builder(TestName());
  builder.Add(builder.ConstantR0<uint8>(35), builder.ConstantR0<uint8>(57));

  ComputeAndCompareR0<uint8>(&builder, 92, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsU64) {
  XlaBuilder builder(TestName());
  const uint64 a = static_cast<uint64>(1) << 63;
  const uint64 b = a + 1;
  builder.Add(builder.ConstantR0<uint64>(a), builder.ConstantR0<uint64>(b));

  ComputeAndCompareR0<uint64>(&builder, a + b, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsS64) {
  XlaBuilder builder(TestName());
  const int64 a = static_cast<int64>(1) << 62;
  const int64 b = a - 1;
  builder.Add(builder.ConstantR0<int64>(a), builder.ConstantR0<int64>(b));

  ComputeAndCompareR0<int64>(&builder, a + b, {});
}

XLA_TEST_F(ScalarComputationsTest, AddTwoScalarsF64) {
  XlaBuilder builder(TestName());
  builder.Add(builder.ConstantR0<double>(0.25),
              builder.ConstantR0<double>(3.5));

  ComputeAndCompareR0<double>(&builder, 3.75, {});
}

XLA_TEST_F(ScalarComputationsTest, SubtractTwoScalarsF32) {
  XlaBuilder builder(TestName());
  builder.Sub(builder.ConstantR0<float>(2.1f), builder.ConstantR0<float>(5.5f));

  ComputeAndCompareR0<float>(&builder, -3.4f, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, SubtractTwoScalarsS32) {
  XlaBuilder builder(TestName());
  builder.Sub(builder.ConstantR0<int32>(2), builder.ConstantR0<int32>(5));

  ComputeAndCompareR0<int32>(&builder, -3, {});
}

XLA_TEST_F(ScalarComputationsTest, CastS64ToF32) {
  XlaBuilder builder(TestName());
  auto a = builder.Parameter(0, ShapeUtil::MakeShape(S64, {}), "a");
  builder.ConvertElementType(a, F32);

  int64 value = 3LL << 35;
  std::unique_ptr<Literal> a_literal = Literal::CreateR0<int64>(value);
  std::unique_ptr<GlobalData> a_data =
      client_->TransferToServer(*a_literal).ConsumeValueOrDie();
  ComputeAndCompareR0<float>(&builder, static_cast<float>(value),
                             {a_data.get()});
}

XLA_TEST_F(ScalarComputationsTest, MulThreeScalarsF32) {
  XlaBuilder builder(TestName());
  builder.Mul(builder.Mul(builder.ConstantR0<float>(2.1f),
                          builder.ConstantR0<float>(5.5f)),
              builder.ConstantR0<float>(0.5f));

  ComputeAndCompareR0<float>(&builder, 5.775f, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, MulTwoScalarsS32) {
  std::vector<int32> data = {0,
                             1,
                             -1,
                             1234,
                             0x1a243514,
                             std::numeric_limits<int32>::max(),
                             std::numeric_limits<int32>::min()};

  for (int32 x : data) {
    for (int32 y : data) {
      XlaBuilder builder(TestName());
      builder.Mul(builder.ConstantR0<int32>(x), builder.ConstantR0<int32>(y));

      // Signed integer overflow is undefined behavior in C++. Convert the input
      // integers to unsigned, perform the multiplication unsigned, and convert
      // back.
      int32 expected = static_cast<uint32>(x) * static_cast<uint32>(y);

      ComputeAndCompareR0<int32>(&builder, expected, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, MulTwoScalarsU32) {
  std::vector<uint32> data = {0,          1,          0xDEADBEEF, 1234,
                              0x1a243514, 0xFFFFFFFF, 0x80808080};

  for (uint32 x : data) {
    for (uint32 y : data) {
      XlaBuilder builder(TestName());
      builder.Mul(builder.ConstantR0<uint32>(x), builder.ConstantR0<uint32>(y));

      uint32 expected = x * y;
      ComputeAndCompareR0<uint32>(&builder, expected, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, MulThreeScalarsS32) {
  XlaBuilder builder(TestName());
  builder.Mul(
      builder.Mul(builder.ConstantR0<int32>(2), builder.ConstantR0<int32>(5)),
      builder.ConstantR0<int32>(1));

  ComputeAndCompareR0<int32>(&builder, 10, {});
}

XLA_TEST_F(ScalarComputationsTest, MulThreeScalarsF32Params) {
  XlaBuilder builder(TestName());
  std::unique_ptr<Literal> a_literal = Literal::CreateR0<float>(2.1f);
  std::unique_ptr<Literal> b_literal = Literal::CreateR0<float>(5.5f);
  std::unique_ptr<Literal> c_literal = Literal::CreateR0<float>(0.5f);

  std::unique_ptr<GlobalData> a_data =
      client_->TransferToServer(*a_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> b_data =
      client_->TransferToServer(*b_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> c_data =
      client_->TransferToServer(*c_literal).ConsumeValueOrDie();

  XlaOp a = builder.Parameter(0, a_literal->shape(), "a");
  XlaOp b = builder.Parameter(1, b_literal->shape(), "b");
  XlaOp c = builder.Parameter(2, c_literal->shape(), "c");
  builder.Mul(builder.Mul(a, b), c);

  ComputeAndCompareR0<float>(&builder, 5.775f,
                             {a_data.get(), b_data.get(), c_data.get()},
                             error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, DivideTwoScalarsF32) {
  XlaBuilder builder(TestName());
  builder.Div(builder.ConstantR0<float>(5.0f), builder.ConstantR0<float>(2.5f));

  ComputeAndCompareR0<float>(&builder, 2.0f, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, RemTwoScalarsF32) {
  XlaBuilder builder(TestName());
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
  XlaBuilder builder(TestName());
  builder.Div(builder.ConstantR0<int32>(p.dividend),
              builder.ConstantR0<int32>(p.divisor));

  ComputeAndCompareR0<int32>(&builder, p.quotient, {});
}

XLA_TEST_P(DivS32Test, RemainderTwoScalarsS32) {
  DivS32Params p = GetParam();
  XlaBuilder builder(TestName());
  builder.Rem(builder.ConstantR0<int32>(p.dividend),
              builder.ConstantR0<int32>(p.divisor));

  ComputeAndCompareR0<int32>(&builder, p.remainder, {});
}

XLA_TEST_P(DivS32Test, DivideTwoScalarsNonConstS32) {
  DivS32Params p = GetParam();
  XlaBuilder builder(TestName());
  XlaOp dividend;
  XlaOp divisor;
  auto dividendd =
      CreateR0Parameter<int32>(p.dividend, 0, "dividend", &builder, &dividend);
  auto divisord =
      CreateR0Parameter<int32>(p.divisor, 1, "divisor", &builder, &divisor);
  builder.Div(dividend, divisor);

  ComputeAndCompareR0<int32>(&builder, p.quotient,
                             {dividendd.get(), divisord.get()});
}

XLA_TEST_P(DivS32Test, RemainderTwoScalarsNonConstDivisorS32) {
  DivS32Params p = GetParam();
  XlaBuilder builder(TestName());
  XlaOp dividend;
  XlaOp divisor;
  auto dividendd =
      CreateR0Parameter<int32>(p.dividend, 0, "dividend", &builder, &dividend);
  auto divisord =
      CreateR0Parameter<int32>(p.divisor, 1, "divisor", &builder, &divisor);
  builder.Rem(dividend, divisor);

  ComputeAndCompareR0<int32>(&builder, p.remainder,
                             {dividendd.get(), divisord.get()});
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
  std::vector<uint32> vals = {
    0, 1, 2, 17, 101, 3333, 0x7FFFFFFF, 0x80000000, UINT32_MAX - 1, UINT32_MAX};
  // clang-format on

  XlaComputation div_computation;
  {
    XlaBuilder builder(TestName());

    XlaOp dividend =
        builder.Parameter(0, ShapeUtil::MakeShape(U32, {}), "dividend");
    XlaOp divisor =
        builder.Parameter(1, ShapeUtil::MakeShape(U32, {}), "divisor");
    builder.Div(dividend, divisor);
    TF_ASSERT_OK_AND_ASSIGN(div_computation, builder.Build());
  }

  for (uint32 divisor : vals) {
    if (divisor != 0) {
      for (uint32 dividend : vals) {
        auto dividend_literal = Literal::CreateR0<uint32>(dividend);
        auto divisor_literal = Literal::CreateR0<uint32>(divisor);
        TF_ASSERT_OK_AND_ASSIGN(auto dividend_data,
                                client_->TransferToServer(*dividend_literal));
        TF_ASSERT_OK_AND_ASSIGN(auto divisor_data,
                                client_->TransferToServer(*divisor_literal));
        auto actual_literal =
            client_
                ->ExecuteAndTransfer(div_computation,
                                     {dividend_data.get(), divisor_data.get()},
                                     &execution_options_)
                .ConsumeValueOrDie();
        auto expected_literal = Literal::CreateR0<uint32>(dividend / divisor);
        EXPECT_TRUE(LiteralTestUtil::Equal(*expected_literal, *actual_literal));
      }
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, RemU32s) {
  // clang-format off
  // Some interesting values to test.
  std::vector<uint32> vals = {
    0, 1, 2, 17, 101, 3333, 0x7FFFFFFF, 0x80000000, UINT32_MAX - 1, UINT32_MAX};
  // clang-format on

  XlaComputation rem_computation;
  {
    XlaBuilder builder(TestName());

    XlaOp dividend =
        builder.Parameter(0, ShapeUtil::MakeShape(U32, {}), "dividend");
    XlaOp divisor =
        builder.Parameter(1, ShapeUtil::MakeShape(U32, {}), "divisor");
    builder.Rem(dividend, divisor);
    TF_ASSERT_OK_AND_ASSIGN(rem_computation, builder.Build());
  }

  for (uint32 divisor : vals) {
    if (divisor != 0) {
      for (uint32 dividend : vals) {
        auto dividend_literal = Literal::CreateR0<uint32>(dividend);
        auto divisor_literal = Literal::CreateR0<uint32>(divisor);
        TF_ASSERT_OK_AND_ASSIGN(auto dividend_data,
                                client_->TransferToServer(*dividend_literal));
        TF_ASSERT_OK_AND_ASSIGN(auto divisor_data,
                                client_->TransferToServer(*divisor_literal));
        auto actual_literal =
            client_
                ->ExecuteAndTransfer(rem_computation,
                                     {dividend_data.get(), divisor_data.get()},
                                     &execution_options_)
                .ConsumeValueOrDie();
        auto expected_literal = Literal::CreateR0<uint32>(dividend % divisor);
        EXPECT_TRUE(LiteralTestUtil::Equal(*expected_literal, *actual_literal));
      }
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, RemainderTwoScalarsNonConstDividendS32) {
  XlaBuilder builder(TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(S32, {}), "x");
  builder.Rem(x, builder.ConstantR0<int32>(80000));

  std::unique_ptr<Literal> literal = Literal::CreateR0<int32>(87919);
  TF_ASSERT_OK_AND_ASSIGN(auto input_data, client_->TransferToServer(*literal));
  ComputeAndCompareR0<int32>(&builder, 7919, {input_data.get()});
}

XLA_TEST_F(ScalarComputationsTest, DivideTwoScalarsU32) {
  XlaBuilder builder(TestName());
  // This verifies 0xFFFFFFFE / 2 = 0x7FFFFFFF. If XLA incorrectly treated U32
  // as S32, it would output -2 / 2 = -1 (0xFFFFFFFF).
  builder.Div(builder.ConstantR0<uint32>(0xFFFFFFFE),
              builder.ConstantR0<uint32>(2));

  ComputeAndCompareR0<uint32>(&builder, 0x7FFFFFFF, {});
}

XLA_TEST_F(ScalarComputationsTest, RemTwoScalarsU32) {
  XlaBuilder builder(TestName());
  builder.Rem(builder.ConstantR0<uint32>(11), builder.ConstantR0<uint32>(3));

  ComputeAndCompareR0<uint32>(&builder, 2, {});
}

XLA_TEST_F(ScalarComputationsTest, AndBool) {
  for (bool x : {false, true}) {
    for (bool y : {false, true}) {
      XlaBuilder builder(TestName());
      builder.And(builder.ConstantR0<bool>(x), builder.ConstantR0<bool>(y));

      ComputeAndCompareR0<bool>(&builder, x && y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, AndS32) {
  for (int32 x : {0, 8}) {
    for (int32 y : {1, -16}) {
      XlaBuilder builder(TestName());
      builder.And(builder.ConstantR0<int32>(x), builder.ConstantR0<int32>(y));

      ComputeAndCompareR0<int32>(&builder, x & y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, AndU32) {
  for (uint32 x : {0, 8}) {
    for (uint32 y : {1, 16}) {
      XlaBuilder builder(TestName());
      builder.And(builder.ConstantR0<uint32>(x), builder.ConstantR0<uint32>(y));

      ComputeAndCompareR0<uint32>(&builder, x & y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, OrBool) {
  for (bool x : {false, true}) {
    for (bool y : {false, true}) {
      XlaBuilder builder(TestName());
      builder.Or(builder.ConstantR0<bool>(x), builder.ConstantR0<bool>(y));

      ComputeAndCompareR0<bool>(&builder, x || y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, OrS32) {
  for (int32 x : {0, 8}) {
    for (int32 y : {1, -16}) {
      XlaBuilder builder(TestName());
      builder.Or(builder.ConstantR0<int32>(x), builder.ConstantR0<int32>(y));

      ComputeAndCompareR0<int32>(&builder, x | y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, OrU32) {
  for (uint32 x : {0, 8}) {
    for (uint32 y : {1, 16}) {
      XlaBuilder builder(TestName());
      builder.Or(builder.ConstantR0<uint32>(x), builder.ConstantR0<uint32>(y));

      ComputeAndCompareR0<uint32>(&builder, x | y, {});
    }
  }
}

XLA_TEST_F(ScalarComputationsTest, NotBool) {
  for (bool x : {false, true}) {
    XlaBuilder builder(TestName());
    builder.Not(builder.ConstantR0<bool>(x));

    ComputeAndCompareR0<bool>(&builder, !x, {});
  }
}

XLA_TEST_F(ScalarComputationsTest, NotS32) {
  for (int32 x : {-1, 0, 1}) {
    XlaBuilder builder(TestName());
    builder.Not(builder.ConstantR0<int32>(x));

    ComputeAndCompareR0<int32>(&builder, ~x, {});
  }
}

XLA_TEST_F(ScalarComputationsTest, NotU32) {
  for (uint32 x : {0, 1, 2}) {
    XlaBuilder builder(TestName());
    builder.Not(builder.ConstantR0<uint32>(x));

    ComputeAndCompareR0<uint32>(&builder, ~x, {});
  }
}

XLA_TEST_F(ScalarComputationsTest, SelectScalarTrue) {
  XlaBuilder builder(TestName());
  builder.Select(builder.ConstantR0<bool>(true),     // The predicate.
                 builder.ConstantR0<float>(123.0f),  // The value on true.
                 builder.ConstantR0<float>(42.0f));  // The value on false.

  ComputeAndCompareR0<float>(&builder, 123.0f, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, SelectScalarFalse) {
  XlaBuilder builder(TestName());
  builder.Select(builder.ConstantR0<bool>(false),    // The predicate.
                 builder.ConstantR0<float>(123.0f),  // The value on true.
                 builder.ConstantR0<float>(42.0f));  // The value on false.

  ComputeAndCompareR0<float>(&builder, 42.0f, {}, error_spec_);
}

// This test is an explicit version of what is happening in the following
// templatized comparison tests.
XLA_TEST_F(ScalarComputationsTest, CompareGtScalar) {
  XlaBuilder builder(TestName());
  builder.Gt(builder.ConstantR0<float>(2.0f), builder.ConstantR0<float>(1.0f));

  ComputeAndCompareR0<bool>(&builder, true, {});
}

// S32 comparisons.
XLA_TEST_F(ScalarComputationsTest, CompareEqS32Greater) {
  TestCompare<int32>(2, 1, false, &XlaBuilder::Eq);
}
XLA_TEST_F(ScalarComputationsTest, CompareEqS32Equal) {
  TestCompare<int32>(3, 3, true, &XlaBuilder::Eq);
}

XLA_TEST_F(ScalarComputationsTest, CompareNeS32) {
  TestCompare<int32>(2, 1, true, &XlaBuilder::Ne);
}

XLA_TEST_F(ScalarComputationsTest, CompareGeS32) {
  TestCompare<int32>(2, 1, true, &XlaBuilder::Ge);
}

XLA_TEST_F(ScalarComputationsTest, CompareGtS32) {
  TestCompare<int32>(1, 5, false, &XlaBuilder::Gt);
}

XLA_TEST_F(ScalarComputationsTest, CompareLeS32) {
  TestCompare<int32>(2, 1, false, &XlaBuilder::Le);
}

XLA_TEST_F(ScalarComputationsTest, CompareLtS32) {
  TestCompare<int32>(9, 7, false, &XlaBuilder::Lt);
  TestCompare<int32>(std::numeric_limits<int32>::min(),
                     std::numeric_limits<int32>::max(), true, &XlaBuilder::Lt);
}

// U32 comparisons.
XLA_TEST_F(ScalarComputationsTest, CompareEqU32False) {
  TestCompare<uint32>(2, 1, false, &XlaBuilder::Eq);
}

XLA_TEST_F(ScalarComputationsTest, CompareNeU32) {
  TestCompare<uint32>(2, 1, true, &XlaBuilder::Ne);
}

XLA_TEST_F(ScalarComputationsTest, CompareGeU32Greater) {
  TestCompare<uint32>(2, 1, true, &XlaBuilder::Ge);
}

XLA_TEST_F(ScalarComputationsTest, CompareGeU32Equal) {
  TestCompare<uint32>(3, 3, true, &XlaBuilder::Ge);
}

XLA_TEST_F(ScalarComputationsTest, CompareGtU32) {
  TestCompare<uint32>(1, 5, false, &XlaBuilder::Gt);
  TestCompare<uint32>(5, 5, false, &XlaBuilder::Gt);
  TestCompare<uint32>(5, 1, true, &XlaBuilder::Gt);
}

XLA_TEST_F(ScalarComputationsTest, CompareLeU32) {
  TestCompare<uint32>(2, 1, false, &XlaBuilder::Le);
}

XLA_TEST_F(ScalarComputationsTest, CompareLtU32) {
  TestCompare<uint32>(9, 7, false, &XlaBuilder::Lt);
  TestCompare<uint32>(0, std::numeric_limits<uint32>::max(), true,
                      &XlaBuilder::Lt);
}

// F32 comparisons.
XLA_TEST_F(ScalarComputationsTest, CompareEqF32False) {
  TestCompare<float>(2.0, 1.3, false, &XlaBuilder::Eq);
}

XLA_TEST_F(ScalarComputationsTest, CompareNeF32) {
  TestCompare<float>(2.0, 1.3, true, &XlaBuilder::Ne);
}

XLA_TEST_F(ScalarComputationsTest, CompareGeF32Greater) {
  TestCompare<float>(2.0, 1.9, true, &XlaBuilder::Ge);
}
XLA_TEST_F(ScalarComputationsTest, CompareGeF32Equal) {
  TestCompare<float>(3.5, 3.5, true, &XlaBuilder::Ge);
}

XLA_TEST_F(ScalarComputationsTest, CompareGtF32) {
  TestCompare<float>(1.0, 5.2, false, &XlaBuilder::Gt);
}

XLA_TEST_F(ScalarComputationsTest, CompareLeF32) {
  TestCompare<float>(2.0, 1.2, false, &XlaBuilder::Le);
}

XLA_TEST_F(ScalarComputationsTest, CompareLtF32) {
  TestCompare<float>(9.0, 7.2, false, &XlaBuilder::Lt);
}

// F32 comparisons with exceptional values.  The test names encode the
// left/right operands at the end, and use Minf and Mzero for -inf and -0.0.
XLA_TEST_F(ScalarComputationsTest, CompareLtF32MinfMzero) {
  TestCompare<float>(-INFINITY, -0.0, true, &XlaBuilder::Lt);
}
XLA_TEST_F(ScalarComputationsTest, CompareLtF32MzeroZero) {
  // Comparisons of 0.0 to -0.0 consider them equal in IEEE 754.
  TestCompare<float>(-0.0, 0.0, false, &XlaBuilder::Lt);
}
XLA_TEST_F(ScalarComputationsTest, CompareLtF32ZeroInf) {
  TestCompare<float>(0.0, INFINITY, true, &XlaBuilder::Lt);
}

XLA_TEST_F(ScalarComputationsTest, CompareGeF32MinfMzero) {
  TestCompare<float>(-INFINITY, -0.0, false, &XlaBuilder::Ge);
}
XLA_TEST_F(ScalarComputationsTest, CompareGeF32MzeroZero) {
  // Comparisons of 0.0 to -0.0 consider them equal in IEEE 754.
  TestCompare<float>(-0.0, 0.0, true, &XlaBuilder::Ge);
}
XLA_TEST_F(ScalarComputationsTest, CompareGeF32ZeroInf) {
  TestCompare<float>(0.0, INFINITY, false, &XlaBuilder::Ge);
}

XLA_TEST_F(ScalarComputationsTest, ExpScalar) {
  XlaBuilder builder(TestName());
  builder.Exp(builder.ConstantR0<float>(2.0f));

  ComputeAndCompareR0<float>(&builder, 7.3890562, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, LogScalar) {
  XlaBuilder builder("log");
  builder.Log(builder.ConstantR0<float>(2.0f));

  ComputeAndCompareR0<float>(&builder, 0.6931471, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, TanhScalar) {
  XlaBuilder builder(TestName());
  builder.Tanh(builder.ConstantR0<float>(2.0f));

  ComputeAndCompareR0<float>(&builder, 0.96402758, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, TanhDoubleScalar) {
  XlaBuilder builder(TestName());
  builder.Tanh(builder.ConstantR0<double>(2.0));

  ComputeAndCompareR0<double>(&builder, 0.96402758, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, PowScalar) {
  XlaBuilder builder(TestName());
  builder.Pow(builder.ConstantR0<float>(2.0f), builder.ConstantR0<float>(3.0f));

  ComputeAndCompareR0<float>(&builder, 8.0, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarHighS32) {
  XlaBuilder builder(TestName());
  builder.Clamp(builder.ConstantR0<int32>(-1),  // The lower bound.
                builder.ConstantR0<int32>(5),   // The operand to be clamped.
                builder.ConstantR0<int32>(3));  // The upper bound.

  ComputeAndCompareR0<int32>(&builder, 3, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarMiddleS32) {
  XlaBuilder builder(TestName());
  builder.Clamp(builder.ConstantR0<int32>(-1),  // The lower bound.
                builder.ConstantR0<int32>(2),   // The operand to be clamped.
                builder.ConstantR0<int32>(3));  // The upper bound.

  ComputeAndCompareR0<int32>(&builder, 2, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarLowS32) {
  XlaBuilder builder(TestName());
  builder.Clamp(builder.ConstantR0<int32>(-1),  // The lower bound.
                builder.ConstantR0<int32>(-5),  // The operand to be clamped.
                builder.ConstantR0<int32>(3));  // The upper bound.

  ComputeAndCompareR0<int32>(&builder, -1, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarHighU32) {
  XlaBuilder builder(TestName());
  builder.Clamp(builder.ConstantR0<uint32>(1),   // The lower bound.
                builder.ConstantR0<uint32>(5),   // The operand to be clamped.
                builder.ConstantR0<uint32>(3));  // The upper bound.

  ComputeAndCompareR0<uint32>(&builder, 3, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarMiddleU32) {
  XlaBuilder builder(TestName());
  builder.Clamp(builder.ConstantR0<uint32>(1),   // The lower bound.
                builder.ConstantR0<uint32>(2),   // The operand to be clamped.
                builder.ConstantR0<uint32>(3));  // The upper bound.

  ComputeAndCompareR0<uint32>(&builder, 2, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarLowU32) {
  XlaBuilder builder(TestName());
  builder.Clamp(builder.ConstantR0<uint32>(1),   // The lower bound.
                builder.ConstantR0<uint32>(0),   // The operand to be clamped.
                builder.ConstantR0<uint32>(3));  // The upper bound.

  ComputeAndCompareR0<uint32>(&builder, 1, {});
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarHighF32) {
  XlaBuilder builder(TestName());
  builder.Clamp(builder.ConstantR0<float>(2.0f),   // The lower bound.
                builder.ConstantR0<float>(5.0f),   // The operand to be clamped.
                builder.ConstantR0<float>(3.0f));  // The upper bound.

  ComputeAndCompareR0<float>(&builder, 3.0, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarMiddleF32) {
  XlaBuilder builder(TestName());
  builder.Clamp(builder.ConstantR0<float>(2.0f),   // The lower bound.
                builder.ConstantR0<float>(2.5f),   // The operand to be clamped.
                builder.ConstantR0<float>(3.0f));  // The upper bound.

  ComputeAndCompareR0<float>(&builder, 2.5, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, ClampScalarLowF32) {
  XlaBuilder builder(TestName());
  builder.Clamp(builder.ConstantR0<float>(2.0f),   // The lower bound.
                builder.ConstantR0<float>(-5.0f),  // The operand to be clamped.
                builder.ConstantR0<float>(3.0f));  // The upper bound.

  ComputeAndCompareR0<float>(&builder, 2.0, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, MinS32Above) {
  TestMinMax<int32>(10, 3, 3, &XlaBuilder::Min);
}

XLA_TEST_F(ScalarComputationsTest, MinS32Below) {
  TestMinMax<int32>(-100, 3, -100, &XlaBuilder::Min);
}

XLA_TEST_F(ScalarComputationsTest, MaxS32Above) {
  TestMinMax<int32>(10, 3, 10, &XlaBuilder::Max);
}

XLA_TEST_F(ScalarComputationsTest, MaxS32Below) {
  TestMinMax<int32>(-100, 3, 3, &XlaBuilder::Max);
}

XLA_TEST_F(ScalarComputationsTest, MinU32Above) {
  const uint32 large = std::numeric_limits<int32>::max();
  TestMinMax<uint32>(large, 3, 3, &XlaBuilder::Min);
}

XLA_TEST_F(ScalarComputationsTest, MinU32Below) {
  TestMinMax<uint32>(0, 5, 0, &XlaBuilder::Min);
}

XLA_TEST_F(ScalarComputationsTest, MaxU32Above) {
  const uint32 large = std::numeric_limits<int32>::max();
  TestMinMax<uint32>(large, 3, large, &XlaBuilder::Max);
}

XLA_TEST_F(ScalarComputationsTest, MaxU32Below) {
  TestMinMax<uint32>(0, 5, 5, &XlaBuilder::Max);
}

XLA_TEST_F(ScalarComputationsTest, MinF32Above) {
  TestMinMax<float>(10.1f, 3.1f, 3.1f, &XlaBuilder::Min);
}

XLA_TEST_F(ScalarComputationsTest, MinF32Below) {
  TestMinMax<float>(-100.1f, 3.1f, -100.1f, &XlaBuilder::Min);
}

XLA_TEST_F(ScalarComputationsTest, MinPropagatesNan) {
  SetFastMathDisabled(true);
  TestMinMax<float>(NAN, 3.1f, NAN, &XlaBuilder::Min);
  TestMinMax<float>(-3.1f, NAN, NAN, &XlaBuilder::Min);
}

XLA_TEST_F(ScalarComputationsTest, MaxF32Above) {
  TestMinMax<float>(10.1f, 3.1f, 10.1f, &XlaBuilder::Max);
}

XLA_TEST_F(ScalarComputationsTest, MaxF32Below) {
  TestMinMax<float>(-100.1f, 3.1f, 3.1f, &XlaBuilder::Max);
}

XLA_TEST_F(ScalarComputationsTest, MaxPropagatesNan) {
  SetFastMathDisabled(true);
  TestMinMax<float>(NAN, 3.1f, NAN, &XlaBuilder::Max);
  TestMinMax<float>(-3.1f, NAN, NAN, &XlaBuilder::Max);
}

XLA_TEST_F(ScalarComputationsTest, ComplicatedArithmeticExpressionF32) {
  // Compute the expression (1 * (3 - 1) * (7 + 0) - 4) / 20.
  XlaBuilder b(TestName());
  b.Div(
      b.Sub(b.Mul(b.ConstantR0<float>(1),
                  b.Mul(b.Sub(b.ConstantR0<float>(3), b.ConstantR0<float>(1)),
                        b.Add(b.ConstantR0<float>(7), b.ConstantR0<float>(0)))),
            b.ConstantR0<float>(4)),
      b.ConstantR0<float>(20));

  ComputeAndCompareR0<float>(&b, 0.5, {}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, ComplicatedArithmeticExpressionS32) {
  // Compute the expression 1 * (3 - 1) * (7 + 0) - 4.
  XlaBuilder b(TestName());
  b.Sub(b.Mul(b.ConstantR0<int32>(1),
              b.Mul(b.Sub(b.ConstantR0<int32>(3), b.ConstantR0<int32>(1)),
                    b.Add(b.ConstantR0<int32>(7), b.ConstantR0<int32>(0)))),
        b.ConstantR0<int32>(4));

  ComputeAndCompareR0<int32>(&b, 10, {});
}

XLA_TEST_F(ScalarComputationsTest, SqrtF320) {
  XlaBuilder builder(TestName());
  Literal zero_literal = Literal::Zero(PrimitiveType::F32);

  std::unique_ptr<GlobalData> zero_data =
      client_->TransferToServer(zero_literal).ConsumeValueOrDie();

  XlaOp zero = builder.Parameter(0, zero_literal.shape(), "zero");
  builder.SqrtF32(zero);

  ComputeAndCompareR0<float>(&builder, 0.0f, {zero_data.get()}, error_spec_);
}

XLA_TEST_F(ScalarComputationsTest, RoundScalar) {
  XlaBuilder builder(TestName());
  builder.Round(builder.ConstantR0<float>(1.4f));

  ComputeAndCompareR0<float>(&builder, 1.0f, {}, error_spec_);
}

}  // namespace
}  // namespace xla
