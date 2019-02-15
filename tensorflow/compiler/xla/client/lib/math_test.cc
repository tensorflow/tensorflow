/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

class MathTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001};
};

// Write TYPED_TESTs within the class definition so that we don't have to litter
// "this->" everywhere.
template <typename T>
class MathTypedTest : public MathTest {
 public:
  void TestLogEdgeCases() {
    SetFastMathDisabled(true);

    XlaBuilder b(TestName());
    Log(AddParam(LiteralUtil::CreateR1<T>({T{0.0}, T{-0.0}}), &b));
    ComputeAndCompareR1<T>(&b,
                           {-std::numeric_limits<T>::infinity(),
                            -std::numeric_limits<T>::infinity()},
                           {}, error_spec_);
  }

  void TestLog1pEdgeCases() {
    SetFastMathDisabled(true);

    XlaBuilder b(TestName());
    Log1p(AddParam(LiteralUtil::CreateR1<T>({T{0.0}, T{-0.0}, T{-1.0}}), &b));
    ComputeAndCompareR1<T>(
        &b, {T{0.0}, T{-0.0}, -std::numeric_limits<T>::infinity()}, {},
        error_spec_);
  }

  void TestIsInfOrNan() {
    SetFastMathDisabled(true);

    XlaBuilder b(TestName());
    auto x =
        ConstantR1<T>(&b, {
                              T{0},
                              T{100},
                              T{-1000},
                              T{std::numeric_limits<T>::max()},
                              T{std::numeric_limits<T>::lowest()},
                              T{std::numeric_limits<float>::infinity()},
                              T{-std::numeric_limits<float>::infinity()},
                              T{std::numeric_limits<float>::quiet_NaN()},
                              T{std::numeric_limits<float>::signaling_NaN()},
                          });
    Tuple(&b, {IsFinite(x), IsInf(x), IsPosInf(x), IsNegInf(x), IsNan(x)});

    auto expected = LiteralUtil::MakeTupleOwned(
        LiteralUtil::CreateR1<bool>(
            {true, true, true, true, true, false, false, false, false}),
        LiteralUtil::CreateR1<bool>(
            {false, false, false, false, false, true, true, false, false}),
        LiteralUtil::CreateR1<bool>(
            {false, false, false, false, false, true, false, false, false}),
        LiteralUtil::CreateR1<bool>(
            {false, false, false, false, false, false, true, false, false}),
        LiteralUtil::CreateR1<bool>(
            {false, false, false, false, false, false, false, true, true}));
    ComputeAndCompareLiteral(&b, expected, {});
  }

  void TestIsNegZero() {
    SetFastMathDisabled(true);
    XlaBuilder b(TestName());
    T inf(std::numeric_limits<float>::infinity());
    T nan(std::numeric_limits<float>::quiet_NaN());
    IsNegZero(AddParam(
        LiteralUtil::CreateR1<T>({T{-0.0}, T{0}, T{1}, T{-1}, inf, -inf, nan}),
        &b));

    ComputeAndCompareLiteral(
        &b,
        LiteralUtil::CreateR1<bool>(
            {true, false, false, false, false, false, false}),
        {}, error_spec_);
  }
};

// TODO(b/123355973): Add bfloat16 to TestTypes once it's working.
#ifdef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
using TestTypes = ::testing::Types<float>;
#else
using TestTypes = ::testing::Types<float, Eigen::half>;
#endif

TYPED_TEST_CASE(MathTypedTest, TestTypes);

XLA_TYPED_TEST(MathTypedTest, LogEdgeCases) { this->TestLogEdgeCases(); }
XLA_TYPED_TEST(MathTypedTest, Log1pEdgeCases) { this->TestLog1pEdgeCases(); }
XLA_TYPED_TEST(MathTypedTest, IsInfOrNan) { this->TestIsInfOrNan(); }
XLA_TYPED_TEST(MathTypedTest, IsNegZero) { this->TestIsNegZero(); }

// Check that certain ops only support real, floating-point inputs.
//
// TODO(jlebar): Expand this test to cover more ops.
XLA_TEST_F(MathTest, RealFpOnlyOps) {
  for (int64 i = PrimitiveType_MIN; i <= PrimitiveType_MAX; ++i) {
    auto ty = static_cast<PrimitiveType>(i);
    SCOPED_TRACE(PrimitiveType_Name(ty));
    Shape shape;
    if (primitive_util::IsArrayType(ty)) {
      shape = ShapeUtil::MakeShape(ty, {42});
    } else if (ty == PrimitiveType::TUPLE) {
      shape = ShapeUtil::MakeTupleShape({});
    } else if (ty == PrimitiveType::OPAQUE) {
      shape = ShapeUtil::MakeOpaqueShape();
    } else if (ty == PrimitiveType::TOKEN) {
      shape = ShapeUtil::MakeTokenShape();
    } else {
      continue;
    }

    for (const auto& test :
         std::vector<std::pair<std::function<XlaOp(XlaOp)>, string>>({
             {IsFinite, "is_finite"},
             {IsInf, "is_inf"},
             {IsPosInf, "is_pos_inf"},
             {IsNegInf, "is_neg_inf"},
             {IsNan, "is_nan"},
             {Erf, "erf"},
             {Erfc, "erfc"},
             {Lgamma, "lgamma"},
             {Digamma, "digamma"},
             {RoundToEven, "round_to_even"},
         })) {
      SCOPED_TRACE(test.second);
      XlaBuilder b(TestName());
      XlaOp p = Parameter(&b, 0, shape, "p0");
      test.first(p);

      EXPECT_EQ(b.first_error().ok(), primitive_util::IsFloatingPointType(ty));
    }
  }
}

XLA_TEST_F(MathTest, SqrtF32) {
  XlaBuilder builder(TestName());
  Literal zero_literal = LiteralUtil::Zero(PrimitiveType::F32);

  std::unique_ptr<GlobalData> zero_data =
      client_->TransferToServer(zero_literal).ConsumeValueOrDie();

  XlaOp zero = Parameter(&builder, 0, zero_literal.shape(), "zero");
  Sqrt(zero);

  ComputeAndCompareR0<float>(&builder, 0.0f, {zero_data.get()}, error_spec_);
}

XLA_TEST_F(MathTest, SquareTenValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  Square(x);

  std::vector<float> expected = {4.41, 6.76, 6.76, 16.,  4.41,
                                 5.29, 25.,  0.81, 5.76, 2.56};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, ReciprocalTenValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  Reciprocal(x);

  std::vector<float> expected = {
      0.47619048, -0.38461538, 0.38461538,  -0.25,       0.47619048,
      0.43478261, -0.2,        -1.11111111, -0.41666667, 0.625};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, SqrtZeroes) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder, {0.0, -0.0});
  Sqrt(x);

  ComputeAndCompareR1<float>(&builder, {0, 0}, {}, error_spec_);
}

XLA_TEST_F(MathTest, SqrtSixValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder, {16.0, 1.0, 1024.0, 0.16, 0.2, 12345});
  Sqrt(x);

  std::vector<float> expected = {4, 1, 32, 0.4, 0.4472, 111.1080};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, Lgamma) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.5,
                                        2.5, -1.5, -3.5, -5.5});
  Lgamma(x);

  std::vector<float> expected = {
      0,
      0,
      static_cast<float>(std::log(2)),
      static_cast<float>(std::log(6)),
      static_cast<float>(std::log(24)),
      static_cast<float>(std::log(120)),
      static_cast<float>(std::log(M_PI) / 2),
      static_cast<float>(std::log(M_PI) / 2 - std::log(2)),
      static_cast<float>(std::log(M_PI) / 2 - std::log(4) + std::log(3)),
      static_cast<float>(std::log(M_PI) / 2 - std::log(3) + std::log(4)),
      static_cast<float>(std::log(M_PI) / 2 - std::log(105) + std::log(16)),
      static_cast<float>(std::log(M_PI) / 2 - std::log(10395) + std::log(64))};
  error_spec_ = ErrorSpec{0.001};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, LgammaF16) {
  SetFastMathDisabled(true);

  XlaBuilder b(TestName());

  // These seemingly arbitrary inputs came from debugging the lgamma
  // implementation against a test which tried all possible f16 values.
  auto x = ConstantR1<half>(&b, {
                                    half(-7360.0),
                                    half(-4066.0),
                                    half(-5.9605e-08),
                                });
  Lgamma(x);
  std::vector<half> expected = {
      std::numeric_limits<half>::infinity(),
      std::numeric_limits<half>::infinity(),
      half(16.64),
  };
  ComputeAndCompareR1<half>(&b, expected, {}, ErrorSpec{0.1});
}

XLA_TEST_F(MathTest, Digamma) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder, {1.0, 0.5, 1 / 3.0, 0.25, 1 / 6.0, 0.125,
                                        2.0, 3.0, 4.0, 6.0, 8.0, 9.0});
  Digamma(x);

  constexpr double euler_mascheroni =
      0.57721566490153286060651209008240243104215933593992;
  std::vector<float> expected = {
      static_cast<float>(-euler_mascheroni),
      static_cast<float>(-2 * std::log(2) - euler_mascheroni),
      static_cast<float>(-M_PI / 2 / std::sqrt(3) - 3 * std::log(3) / 2 -
                         euler_mascheroni),
      static_cast<float>(-M_PI / 2 - 3 * std::log(2) - euler_mascheroni),
      static_cast<float>(-M_PI * std::sqrt(3) / 2 - 2 * std::log(2) -
                         3 * std::log(3) / 2 - euler_mascheroni),
      static_cast<float>(
          -M_PI / 2 - 4 * std::log(2) -
          (M_PI + std::log(2 + std::sqrt(2)) - std::log(2 - std::sqrt(2))) /
              std::sqrt(2) -
          euler_mascheroni),
      static_cast<float>(1 - euler_mascheroni),
      static_cast<float>(1.5 - euler_mascheroni),
      static_cast<float>(11 / 6.0 - euler_mascheroni),
      static_cast<float>(137 / 60.0 - euler_mascheroni),
      static_cast<float>(363 / 140.0 - euler_mascheroni),
      static_cast<float>(761 / 280.0 - euler_mascheroni)};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, RoundToEven) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {-1.4, -1.5, -2.5, -0.5, 0, 0.5, 1.5, 2.5, 3.5, 4.5});
  RoundToEven(x);

  std::vector<float> expected = {-1.0, -2.0, -2.0, -0.0, 0,
                                 0.0,  2.0,  2.0,  4.0,  4.0};

  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, ErfRejectsComplexInputs) {
  XlaBuilder b(TestName());
  auto x = ConstantR1<std::complex<float>>(&b, {{0, 0}});
  Erf(x);
  EXPECT_FALSE(b.Build().status().ok());
}

XLA_TEST_F(MathTest, ErfcRejectsComplexInputs) {
  XlaBuilder b(TestName());
  auto x = ConstantR1<std::complex<float>>(&b, {{0, 0}});
  Erfc(x);
  EXPECT_FALSE(b.Build().status().ok());
}

XLA_TEST_F(MathTest, LgammaRejectsComplexInputs) {
  XlaBuilder b(TestName());
  auto x = ConstantR1<std::complex<float>>(&b, {{0, 0}});
  Lgamma(x);
  EXPECT_FALSE(b.Build().status().ok());
}

XLA_TEST_F(MathTest, DigammaRejectsComplexInputs) {
  XlaBuilder b(TestName());
  auto x = ConstantR1<std::complex<float>>(&b, {{0, 0}});
  Digamma(x);
  EXPECT_FALSE(b.Build().status().ok());
}

XLA_TEST_F(MathTest, RoundToEvenRejectsComplexInputs) {
  XlaBuilder b(TestName());
  auto x = ConstantR1<std::complex<float>>(&b, {{0, 0}});
  RoundToEven(x);
  EXPECT_FALSE(b.Build().status().ok());
}

}  // namespace
}  // namespace xla
