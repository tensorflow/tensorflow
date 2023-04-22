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

#include <limits>

#include "tensorflow/compiler/xla/client/lib/constants.h"
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

  // sqrt(x) == pow(x, 0.5) except that
  //
  //   pow(-inf, 0.5) == inf, while
  //   sqrt(-inf)     == nan.
  //
  // Check that none of our backends are incorrectly assuming that sqrt(x) ==
  // pow(x, 0.5) without checking this edge case.
  //
  // For good measure, we also check pow with an exponent other than 0.5.
  void TestSqrtPowInequivalence() {
    // TODO(b/145798892): test fails on GPU for double values.
    if (std::is_same<T, double>::value) {
      return;
    }
    SetFastMathDisabled(true);

    // Tests disable constant folding by default, but this test needs it
    // enabled, otherwise we don't tickle the bug we're trying to catch.
    // Specifically, without constant folding, the constants we pass to Pow
    // below are hidden behind a reshape that's never folded away!
    mutable_debug_options()->clear_xla_disable_hlo_passes();

    const T inf(std::numeric_limits<float>::infinity());
    const T nan(std::numeric_limits<float>::quiet_NaN());

    XlaBuilder b(TestName());
    auto x = AddParam(LiteralUtil::CreateR1<T>({-inf}), &b);
    ConcatInDim(
        &b, {Sqrt(x), Pow(x, ScalarLike(x, 0.5)), Pow(x, ScalarLike(x, 0.3))},
        0);
    std::vector<T> expected = {nan, inf, inf};
    ComputeAndCompareR1<T>(&b, expected, {}, error_spec_);
  }

  void TestErfEdgeCases() {
    SetFastMathDisabled(true);

    XlaBuilder b(TestName());
    auto x = AddParam(LiteralUtil::CreateR1<T>({T{-1}, T{1}, T{0}}), &b);
    ErfInv(x);

    const T inf(std::numeric_limits<float>::infinity());
    std::vector<T> expected = {-inf, inf, T{0}};

    ComputeAndCompareR1<T>(&b, expected, {}, error_spec_);
  }
};

// TODO(b/123355973): Add bfloat16 to TestTypes once it's working.
using TestTypes = ::testing::Types<float
#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
                                   ,
                                   Eigen::half
#endif
#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64
                                   ,
                                   double
#endif
                                   >;

TYPED_TEST_CASE(MathTypedTest, TestTypes);

XLA_TYPED_TEST(MathTypedTest, LogEdgeCases) { this->TestLogEdgeCases(); }
XLA_TYPED_TEST(MathTypedTest, Log1pEdgeCases) { this->TestLog1pEdgeCases(); }
XLA_TYPED_TEST(MathTypedTest, IsInfOrNan) { this->TestIsInfOrNan(); }
XLA_TYPED_TEST(MathTypedTest, IsNegZero) { this->TestIsNegZero(); }
XLA_TYPED_TEST(MathTypedTest, SqrtPowInequivalence) {
  this->TestSqrtPowInequivalence();
}
XLA_TYPED_TEST(MathTypedTest, ErfInvEdgeCases) { this->TestErfEdgeCases(); }

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
    } else if (ty == PrimitiveType::OPAQUE_TYPE) {
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

XLA_TEST_F(MathTest, SqrtF64) {
  XlaBuilder builder(TestName());
  Literal zero_literal = LiteralUtil::Zero(PrimitiveType::F64);

  std::unique_ptr<GlobalData> zero_data =
      client_->TransferToServer(zero_literal).ConsumeValueOrDie();

  XlaOp zero = Parameter(&builder, 0, zero_literal.shape(), "zero");
  Sqrt(zero);

  ComputeAndCompareR0<double>(&builder, 0.0f, {zero_data.get()}, error_spec_);
}

#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64
XLA_TEST_F(MathTest, ErfInvF64) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<double>(
      &builder, {-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1,
                 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9});
  ErfInv(x);

  std::vector<double> expected = {-1.163087153676674,   -0.9061938024368231,
                                  -0.732869077959217,   -0.5951160814499948,
                                  -0.4769362762044698,  -0.37080715859355795,
                                  -0.27246271472675443, -0.1791434546212916,
                                  -0.08885599049425767, 0.,
                                  0.08885599049425777,  0.1791434546212916,
                                  0.27246271472675443,  0.37080715859355784,
                                  0.4769362762044698,   0.5951160814499948,
                                  0.732869077959217,    0.9061938024368231,
                                  1.1630871536766736};
  ComputeAndCompareR1<double>(&builder, expected, {}, ErrorSpec{1e-15});
}
#endif

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

XLA_TEST_F(MathTest, CbrtSixValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder, {8.0, 1.0, 4096.0, -64.0, 1.728, 1331});
  Cbrt(x);

  std::vector<float> expected = {2, 1, 16, -4, 1.2, 11};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.001));
}

XLA_TEST_F(MathTest, SinhSmallValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder, {1e-3, 1e-5, 1e-7, 1e-9, 1e-11});
  Sinh(x);
  std::vector<float> expected = {1e-3, 1e-5, 1e-7, 1e-9, 1e-11};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, AsinhSmallValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder, {1e-3, 1e-5, 1e-7, 1e-9, 1e-11});
  Asinh(x);
  std::vector<float> expected = {1e-3, 1e-5, 1e-7, 1e-9, 1e-11};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, AtanhSmallValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder, {1e-8, 1e-9, 1e-10, 1e-11});
  Atanh(x);
  std::vector<float> expected = {1e-8, 1e-9, 1e-10, 1e-11};
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

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
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
#endif

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

XLA_TEST_F(MathTest, Igamma) {
  XlaBuilder builder(TestName());
  auto a = ConstantR3FromArray3D<float>(
      &builder,
      {{{0.3760359, 1.62685306, 0.53327996, 1.5111382, 0.3521143},
        {1.79378175, 1.05317882, 0.85049253, 1.399534, 0.22073882},
        {1.17725309, 0.90727209, 1.32418503, 1.53238533, 0.51984756}}});
  auto x = ConstantR3FromArray3D<float>(
      &builder,
      {{{0.56420934, 8.97671773, 2.81068609, 4.50655124, 2.88178617},
        {1.01795164, 8.86298411, 0.29232942, 8.17661015, 5.67652269},
        {1.59959565, 0.54463897, 0.6585252, 9.83192283, 3.93372669}}});

  Igamma(a, x);
  // Golden values generated by scipy.special.gammainc
  Array3D<float> expected = {
      {{0.78746926, 0.99940502, 0.98028261, 0.97033807, 0.99054696},
       {0.33265522, 0.99983558, 0.32599159, 0.99923275, 0.99980893},
       {0.74343963, 0.46703197, 0.33923541, 0.99978511, 0.99460685}}};
  ComputeAndCompareR3<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, IgammaSpecialValues) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto a =
      ConstantR1<float>(&builder, {nan, nan, 0.53327996, -6.00773744602e+37,
                                   -1.3937809742e+31, -23.351348877});
  auto x = ConstantR1<float>(
      &builder, {nan, 8.97671773, nan, nan, 0.0, 6.02455484352e-39});

  Igamma(a, x);
  std::vector<float> expected = {nan, nan, nan, nan, nan, nan};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
XLA_TEST_F(MathTest, IgammaF16) {
  SetFastMathDisabled(true);

  XlaBuilder builder(TestName());

  auto a = ConstantR3FromArray3D<half>(
      &builder,
      {{{half(0.37603), half(1.6268), half(0.53327), half(1.5111)},
        {half(1.79378), half(1.05317), half(0.85049), half(1.3995)},
        {half(1.17725), half(0.90727), half(1.32418), half(1.5323)}}});

  Igamma(a, a);

  // Golden values generated by scipy.special.gammainc
  Array3D<half> expected = {
      {{half(0.7068214), half(0.6041154), half(0.67748886), half(0.60799426)},
       {half(0.599202), half(0.6288743), half(0.64280254), half(0.6121421)},
       {half(0.6220287), half(0.6384635), half(0.6152258), half(0.6072449)}}};
  ComputeAndCompareR3<half>(&builder, expected, {}, ErrorSpec{1e-3});
}
#endif

XLA_TEST_F(MathTest, Igammac) {
  XlaBuilder builder(TestName());
  auto a = ConstantR3FromArray3D<float>(
      &builder,
      {{{0.3760359, 1.62685306, 0.53327996, 1.5111382, 0.3521143},
        {1.79378175, 1.05317882, 0.85049253, 1.399534, 0.22073882},
        {1.17725309, 0.90727209, 1.32418503, 1.53238533, 0.51984756}}});
  auto x = ConstantR3FromArray3D<float>(
      &builder,
      {{{0.56420934, 8.97671773, 2.81068609, 4.50655124, 2.88178617},
        {1.01795164, 8.86298411, 0.29232942, 8.17661015, 5.67652269},
        {1.59959565, 0.54463897, 0.6585252, 9.83192283, 3.93372669}}});

  Igammac(a, x);
  // Golden values generated by scipy.special.gammaincc
  Array3D<float> expected = {{{2.12530741e-01, 5.94977775e-04, 1.97173867e-02,
                               2.96619296e-02, 9.45303689e-03},
                              {6.67344782e-01, 1.64421996e-04, 6.74008406e-01,
                               7.67252602e-04, 1.91071108e-04},
                              {2.56560373e-01, 5.32968026e-01, 6.60764593e-01,
                               2.14889688e-04, 5.39314824e-03}}};
  ComputeAndCompareR3<float>(&builder, expected, {}, error_spec_);
}

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
XLA_TEST_F(MathTest, IgammacF16) {
  SetFastMathDisabled(true);

  XlaBuilder builder(TestName());

  auto a = ConstantR3FromArray3D<half>(
      &builder,
      {{{half(0.37603), half(1.6268), half(0.53327), half(1.5111)},
        {half(1.79378), half(1.05317), half(0.85049), half(1.3995)},
        {half(1.17725), half(0.90727), half(1.32418), half(1.5323)}}});

  Igammac(a, a);

  // Golden values generated by scipy.special.gammaincc
  Array3D<half> expected = {
      {{half(0.29317862), half(0.39588454), half(0.32251117), half(0.39200574)},
       {half(0.40079802), half(0.37112573), half(0.35719746), half(0.3878579)},
       {half(0.3779713), half(0.36153653), half(0.38477424),
        half(0.39275512)}}};
  ComputeAndCompareR3<half>(&builder, expected, {}, ErrorSpec{1e-4});
}
#endif

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

XLA_TEST_F(MathTest, BesselI0eFloat) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder,
      {-20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0,
       2.0,   4.0,   6.0,   8.0,   10.0,  12.0,  14.0, 16.0, 18.0, 20.0});
  BesselI0e(x);

  // Values were generated through scipy via scipy.special.i0e.
  std::vector<float> expected = {0.0897803118848,
                                 0.0947062952128,
                                 0.100544127361,
                                 0.107615251671,
                                 0.116426221213,
                                 0.127833337163,
                                 0.143431781857,
                                 0.16665743264,
                                 0.207001921224,
                                 0.308508322554,
                                 1.0,
                                 0.308508322554,
                                 0.207001921224,
                                 0.16665743264,
                                 0.143431781857,
                                 0.127833337163,
                                 0.116426221213,
                                 0.107615251671,
                                 0.100544127361,
                                 0.0947062952128,
                                 0.0897803118848};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, BesselI0eDouble) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<double>(
      &builder,
      {-20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0,
       2.0,   4.0,   6.0,   8.0,   10.0,  12.0,  14.0, 16.0, 18.0, 20.0});
  BesselI0e(x);

  // Values were generated through scipy via scipy.special.i0e.
  std::vector<double> expected = {0.0897803118848,
                                  0.0947062952128,
                                  0.100544127361,
                                  0.107615251671,
                                  0.116426221213,
                                  0.127833337163,
                                  0.143431781857,
                                  0.16665743264,
                                  0.207001921224,
                                  0.308508322554,
                                  1.0,
                                  0.308508322554,
                                  0.207001921224,
                                  0.16665743264,
                                  0.143431781857,
                                  0.127833337163,
                                  0.116426221213,
                                  0.107615251671,
                                  0.100544127361,
                                  0.0947062952128,
                                  0.0897803118848};
  ComputeAndCompareR1<double>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, BesselI1eFloat) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder,
      {-20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0,
       2.0,   4.0,   6.0,   8.0,   10.0,  12.0,  14.0, 16.0, 18.0, 20.0});
  BesselI1e(x);

  // Values were generated through scipy via scipy.special.i1e.
  std::vector<float> expected = {-0.0875062221833,
                                 -0.092036796872,
                                 -0.0973496147565,
                                 -0.103697667463,
                                 -0.11146429929,
                                 -0.121262681384,
                                 -0.134142493293,
                                 -0.152051459309,
                                 -0.178750839502,
                                 -0.215269289249,
                                 0.0,
                                 0.215269289249,
                                 0.178750839502,
                                 0.152051459309,
                                 0.134142493293,
                                 0.121262681384,
                                 0.11146429929,
                                 0.103697667463,
                                 0.0973496147565,
                                 0.092036796872,
                                 0.0875062221833};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, BesselI1eDouble) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<double>(
      &builder,
      {-20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0,
       2.0,   4.0,   6.0,   8.0,   10.0,  12.0,  14.0, 16.0, 18.0, 20.0});
  BesselI1e(x);

  // Values were generated through scipy via scipy.special.i1e.
  std::vector<double> expected = {-0.0875062221833,
                                  -0.092036796872,
                                  -0.0973496147565,
                                  -0.103697667463,
                                  -0.11146429929,
                                  -0.121262681384,
                                  -0.134142493293,
                                  -0.152051459309,
                                  -0.178750839502,
                                  -0.215269289249,
                                  0.0,
                                  0.215269289249,
                                  0.178750839502,
                                  0.152051459309,
                                  0.134142493293,
                                  0.121262681384,
                                  0.11146429929,
                                  0.103697667463,
                                  0.0973496147565,
                                  0.092036796872,
                                  0.0875062221833};
  ComputeAndCompareR1<double>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(MathTest, AcosComplexValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<std::complex<float>>(
      &builder, {{0, 0}, {0, 1}, {1, 1}, {0.8, 0.2}});

  Acos(x);
  std::vector<std::complex<float>> expected = {
      {1.5707963267948966, 0},
      {1.5707963267948966, -0.881373587019543},
      {0.9045568943023814, -1.0612750619050357},
      {0.7011246914497526, -0.30527648462436596}};
  ComputeAndCompareR1<std::complex<float>>(&builder, expected, {}, error_spec_);
}

}  // namespace
}  // namespace xla
