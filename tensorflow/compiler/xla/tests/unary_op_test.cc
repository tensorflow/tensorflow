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

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
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
    ComputationBuilder builder(client_, TestName());
    auto arg = builder.ConstantR1<T>({});
    auto abs = builder.Abs(arg);

    if (primitive_util::NativeToPrimitiveType<T>() == C64) {
      ComputeAndCompareR1<float>(&builder, {}, {});
    } else {
      ComputeAndCompareR1<T>(&builder, {}, {});
    }
  }

  template <typename T>
  void AbsTestHelper() {
    ComputationBuilder builder(client_, TestName());
    auto arg = builder.ConstantR1<T>({-2, 25, 0, -123, inf<T>(), -inf<T>()});
    auto abs = builder.Abs(arg);

    ComputeAndCompareR1<T>(&builder, {2, 25, 0, 123, inf<T>(), inf<T>()}, {});
  }

  template <typename T>
  void SignTestHelper() {
    ComputationBuilder builder(client_, TestName());
    auto arg = builder.ConstantR1<T>(
        {-2, 25, 0, static_cast<T>(-0.0), -123, inf<T>(), -inf<T>()});
    auto sign = builder.Sign(arg);

    ComputeAndCompareR1<T>(&builder, {-1, 1, 0, 0, -1, 1, -1}, {});
  }

  template <typename T>
  void SignAbsTestHelper() {
    ComputationBuilder builder(client_, TestName());
    auto arg = builder.ConstantR1<T>({-2, 25, 0, -123});
    auto sign = builder.Sign(arg);
    auto abs = builder.Abs(arg);
    builder.Sub(builder.Mul(sign, abs), arg);

    ComputeAndCompareR1<T>(&builder, {0, 0, 0, 0}, {});
  }
};

template <>
int UnaryOpTest::inf<int>() {
  return 2147483647;
}

template <>
void UnaryOpTest::AbsTestHelper<complex64>() {
  ComputationBuilder builder(client_, TestName());
  auto arg = builder.ConstantR1<complex64>({{-2, 0},
                                            {0, 25},
                                            {0, 0},
                                            {-0.3f, 0.4f},
                                            {0, inf<float>()},
                                            {-inf<float>(), 0}});
  auto abs = builder.Abs(arg);

  std::unique_ptr<Literal> expected =
      Literal::CreateR1<float>({2, 25, 0, 0.5, inf<float>(), inf<float>()});
  ComputeAndCompareLiteral(&builder, *expected, {}, ErrorSpec(1e-6f));
}

template <>
void UnaryOpTest::SignTestHelper<complex64>() {
  ComputationBuilder builder(client_, TestName());
  auto arg = builder.ConstantR1<complex64>(
      {{-2, 0}, {0, 25}, {0, 0}, {static_cast<float>(-0.0), 0}, {-1, 1}});
  auto sign = builder.Sign(arg);

  std::unique_ptr<Literal> expected = Literal::CreateR1<complex64>(
      {{-1, 0}, {0, 1}, {0, 0}, {0, 0}, {-std::sqrt(0.5f), std::sqrt(0.5f)}});
  ComputeAndCompareLiteral(&builder, *expected, {}, ErrorSpec(1e-6f));
}

template <>
void UnaryOpTest::SignAbsTestHelper<complex64>() {
  ComputationBuilder builder(client_, TestName());
  auto arg =
      builder.ConstantR1<complex64>({{-2, 0}, {0, 25}, {0, 0}, {-0.4, 0.3}});
  auto sign = builder.Sign(arg);
  auto abs = builder.Abs(arg);
  builder.Sub(builder.Mul(sign, builder.ConvertElementType(abs, C64)), arg);

  std::unique_ptr<Literal> expected =
      Literal::CreateR1<complex64>({0, 0, 0, 0});
  ComputeAndCompareLiteral(&builder, *expected, {}, ErrorSpec(1e-6f));
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
  ComputationBuilder builder(client_, TestName());
  auto argi = builder.ConstantR0<int>(-5);
  auto absi = builder.Abs(argi);
  auto argf = builder.ConstantR0<float>(-3.0f);
  auto absf = builder.Abs(argf);
  auto argf0 = builder.ConstantR0<float>(-0.0f);
  auto absf0 = builder.Abs(argf0);
  auto argc = builder.ConstantR0<complex64>({-0.3f, 0.4f});
  auto absc = builder.Abs(argc);
  builder.Add(builder.Add(absc, absf0),
              builder.Add(absf, builder.ConvertElementType(absi, F32)));

  ComputeAndCompareR0<float>(&builder, 8.5f, {});
}

XLA_TEST_F(UnaryOpTest, SignTestR0) {
  ComputationBuilder builder(client_, TestName());
  auto argi = builder.ConstantR0<int>(-5);
  auto sgni = builder.Sign(argi);  // -1
  auto argf = builder.ConstantR0<float>(-4.0f);
  auto sgnf = builder.Sign(argf);  // -1
  auto argf0 = builder.ConstantR0<float>(-0.0f);
  auto sgnf0 = builder.Sign(argf0);  // 0
  auto argc = builder.ConstantR0<complex64>({-.3, .4});
  auto sgnc = builder.Sign(argc);  // (-.6, .8)
  builder.Add(sgnc, builder.ConvertElementType(
                        builder.Add(builder.Add(sgnf0, sgnf),
                                    builder.ConvertElementType(sgni, F32)),
                        C64));

  std::unique_ptr<Literal> expected =
      Literal::CreateR0<complex64>({-2.6f, 0.8f});
  ComputeAndCompareLiteral(&builder, *expected, {}, ErrorSpec(1e-6f));
}

XLA_TEST_F(UnaryOpTest, SignTestR1) {
  SignTestHelper<int>();
  SignTestHelper<float>();
  SignTestHelper<complex64>();
}

XLA_TEST_F(UnaryOpTest, SignAbsTestR1) {
  SignAbsTestHelper<int>();
  SignAbsTestHelper<float>();
  SignAbsTestHelper<complex64>();
}

XLA_TEST_F(UnaryOpTest, UnsignedAbsTestR1) {
  ComputationBuilder builder(client_, TestName());
  auto arg = builder.ConstantR1<unsigned int>(
      {2, 25, 0, 123, std::numeric_limits<unsigned int>::max()});
  auto abs = builder.Abs(arg);

  ComputeAndCompareR1<unsigned int>(
      &builder, {2, 25, 0, 123, std::numeric_limits<unsigned int>::max()}, {});
}

XLA_TEST_F(UnaryOpTest, UnsignedSignTestR1) {
  ComputationBuilder builder(client_, TestName());
  auto arg = builder.ConstantR1<unsigned int>(
      {2, 25, 0, 123, std::numeric_limits<unsigned int>::max()});
  auto sign = builder.Sign(arg);

  ComputeAndCompareR1<unsigned int>(&builder, {1, 1, 0, 1, 1}, {});
}

XLA_TEST_F(UnaryOpTest, SignAbsTestR2) {
  ComputationBuilder builder(client_, TestName());
  auto arg = builder.ConstantR2<float>({{1.0, -2.0}, {-3.0, 4.0}});
  auto sign = builder.Sign(arg);
  auto abs = builder.Abs(arg);
  builder.Sub(builder.Mul(sign, abs), arg);

  ComputeAndCompareR2<float>(&builder, {{0, 0}, {0, 0}}, {});
}

}  // namespace
}  // namespace xla
