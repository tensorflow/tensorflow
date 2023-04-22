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

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class UnaryOpTest : public ClientLibraryTestBase {
 protected:
  template <typename T>
  T inf() {
    return std::numeric_limits<T>::infinity();
  }
  template <typename T>
  void AbsSize0TestHelper() {
    XlaBuilder builder(TestName());
    auto arg = ConstantR1<T>(&builder, {});
    Abs(arg);

    if (primitive_util::NativeToPrimitiveType<T>() == C64) {
      ComputeAndCompareR1<float>(&builder, {}, {});
    } else {
      ComputeAndCompareR1<T>(&builder, {}, {});
    }
  }

  template <typename T>
  void AbsTestHelper() {
    XlaBuilder builder(TestName());
    auto arg = ConstantR1<T>(&builder, {-2, 25, 0, -123, inf<T>(), -inf<T>()});
    Abs(arg);

    ComputeAndCompareR1<T>(&builder, {2, 25, 0, 123, inf<T>(), inf<T>()}, {});
  }

  template <typename T>
  void SignTestHelper() {
    XlaBuilder builder(TestName());
    auto arg = ConstantR1<T>(
        &builder, {-2, 25, 0, static_cast<T>(-0.0), -123, inf<T>(), -inf<T>()});
    Sign(arg);

    ComputeAndCompareR1<T>(
        &builder,
        {-1, 1, static_cast<T>(+0.0), static_cast<T>(-0.0), -1, 1, -1}, {});
  }

  template <typename T>
  void SignAbsTestHelper() {
    XlaBuilder builder(TestName());
    auto arg = ConstantR1<T>(&builder, {-2, 25, 0, -123});
    auto sign = Sign(arg);
    auto abs = Abs(arg);
    Sub(Mul(sign, abs), arg);

    ComputeAndCompareR1<T>(&builder, {0, 0, 0, 0}, {});
  }
};

template <>
int UnaryOpTest::inf<int>() {
  return 2147483647;
}

template <>
int64 UnaryOpTest::inf<int64>() {
  return 0x7FFFFFFFFFFFFFFFl;
}

template <>
void UnaryOpTest::AbsTestHelper<complex64>() {
  XlaBuilder builder(TestName());
  auto arg = ConstantR1<complex64>(&builder, {{-2, 0},
                                              {0, 25},
                                              {0, 0},
                                              {-0.3f, 0.4f},
                                              {0, inf<float>()},
                                              {-inf<float>(), 0}});
  Abs(arg);

  Literal expected =
      LiteralUtil::CreateR1<float>({2, 25, 0, 0.5, inf<float>(), inf<float>()});
  ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-6f));
}

template <>
void UnaryOpTest::SignTestHelper<complex64>() {
  XlaBuilder builder(TestName());
  auto arg = ConstantR1<complex64>(
      &builder,
      {{-2, 0}, {0, 25}, {0, 0}, {static_cast<float>(-0.0), 0}, {-1, 1}});
  Sign(arg);

  Literal expected = LiteralUtil::CreateR1<complex64>(
      {{-1, 0}, {0, 1}, {0, 0}, {0, 0}, {-std::sqrt(0.5f), std::sqrt(0.5f)}});
  ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-6f));
}

template <>
void UnaryOpTest::SignAbsTestHelper<complex64>() {
  XlaBuilder builder(TestName());
  auto arg =
      ConstantR1<complex64>(&builder, {{-2, 0}, {0, 25}, {0, 0}, {-0.4, 0.3}});
  auto sign = Sign(arg);
  auto abs = Abs(arg);
  Sub(Mul(sign, ConvertElementType(abs, C64)), arg);

  Literal expected = LiteralUtil::CreateR1<complex64>({0, 0, 0, 0});
  ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-6f));
}

XLA_TEST_F(UnaryOpTest, AbsTestR1Size0) {
  AbsSize0TestHelper<int>();
  AbsSize0TestHelper<float>();
  AbsSize0TestHelper<complex64>();
}

XLA_TEST_F(UnaryOpTest, AbsTestR1) {
  AbsTestHelper<int>();
  AbsTestHelper<float>();
  AbsTestHelper<complex64>();
}

XLA_TEST_F(UnaryOpTest, AbsTestR0) {
  XlaBuilder builder(TestName());
  auto argi = ConstantR0<int>(&builder, -5);
  auto absi = Abs(argi);
  auto argf = ConstantR0<float>(&builder, -3.0f);
  auto absf = Abs(argf);
  auto argf0 = ConstantR0<float>(&builder, -0.0f);
  auto absf0 = Abs(argf0);
  auto argc = ConstantR0<complex64>(&builder, {-0.3f, 0.4f});
  auto absc = Abs(argc);
  Add(Add(absc, absf0), Add(absf, ConvertElementType(absi, F32)));

  ComputeAndCompareR0<float>(&builder, 8.5f, {});
}

XLA_TEST_F(UnaryOpTest, SignTestR0) {
  XlaBuilder builder(TestName());
  auto argi = ConstantR0<int>(&builder, -5);
  auto sgni = Sign(argi);  // -1
  auto argf = ConstantR0<float>(&builder, -4.0f);
  auto sgnf = Sign(argf);  // -1
  auto argf0 = ConstantR0<float>(&builder, -0.0f);
  auto sgnf0 = Sign(argf0);  // 0
  auto argc = ConstantR0<complex64>(&builder, {-.3, .4});
  auto sgnc = Sign(argc);  // (-.6, .8)
  Add(sgnc, ConvertElementType(
                Add(Add(sgnf0, sgnf), ConvertElementType(sgni, F32)), C64));

  Literal expected = LiteralUtil::CreateR0<complex64>({-2.6f, 0.8f});
  ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-6f));
}

XLA_TEST_F(UnaryOpTest, SignTestR1) {
  SignTestHelper<int>();
  SignTestHelper<int64>();
  SignTestHelper<float>();
  SignTestHelper<complex64>();
}

XLA_TEST_F(UnaryOpTest, SignAbsTestR1) {
  SignAbsTestHelper<int>();
  SignAbsTestHelper<float>();
  SignAbsTestHelper<complex64>();
}

XLA_TEST_F(UnaryOpTest, SignAbsTestR2) {
  XlaBuilder builder(TestName());
  auto arg = ConstantR2<float>(&builder, {{1.0, -2.0}, {-3.0, 4.0}});
  auto sign = Sign(arg);
  auto abs = Abs(arg);
  Sub(Mul(sign, abs), arg);

  ComputeAndCompareR2<float>(&builder, {{0, 0}, {0, 0}}, {});
}

XLA_TEST_F(UnaryOpTest, ConvertElementTypePredToS32) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<int32>(&builder, {0, 1});
  auto rhs = ConstantR1<int32>(&builder, {1, 1});
  ConvertElementType(Eq(lhs, rhs), S32);

  ComputeAndCompareR1<int32>(&builder, {0, 1}, {});
}

XLA_TEST_F(UnaryOpTest, ConvertElementTypePredToF32) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<int32>(&builder, {0, 1});
  auto rhs = ConstantR1<int32>(&builder, {1, 1});
  ConvertElementType(Eq(lhs, rhs), F32);

  ComputeAndCompareR1<float>(&builder, {0.0, 1.0}, {});
}

}  // namespace
}  // namespace xla
