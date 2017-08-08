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

    ComputeAndCompareR1<T>(&builder, {}, {});
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

XLA_TEST_F(UnaryOpTest, AbsTestR1Size0) {
  AbsSize0TestHelper<int>();
  AbsSize0TestHelper<float>();
}

XLA_TEST_F(UnaryOpTest, AbsTestR1) {
  AbsTestHelper<int>();
  AbsTestHelper<float>();
}

XLA_TEST_F(UnaryOpTest, AbsTestR0) {
  ComputationBuilder builder(client_, TestName());
  auto argi = builder.ConstantR0<int>(-5);
  auto absi = builder.Abs(argi);
  auto argf = builder.ConstantR0<float>(-3.0f);
  auto absf = builder.Abs(argf);
  auto argf0 = builder.ConstantR0<float>(-0.0f);
  auto absf0 = builder.Abs(argf0);
  builder.Add(absf0, builder.Add(absf, builder.ConvertElementType(
                                           absi, PrimitiveType::F32)));

  ComputeAndCompareR0<float>(&builder, 8.0f, {});
}

XLA_TEST_F(UnaryOpTest, SignTestR0) {
  ComputationBuilder builder(client_, TestName());
  auto argi = builder.ConstantR0<int>(-5);
  auto absi = builder.Sign(argi);
  auto argf = builder.ConstantR0<float>(-4.0f);
  auto absf = builder.Sign(argf);
  auto argf0 = builder.ConstantR0<float>(-0.0f);
  auto absf0 = builder.Sign(argf0);
  builder.Add(absf0, builder.Add(absf, builder.ConvertElementType(
                                           absi, PrimitiveType::F32)));

  ComputeAndCompareR0<float>(&builder, -2.0f, {});
}

XLA_TEST_F(UnaryOpTest, SignTestR1) {
  SignTestHelper<int>();
  SignTestHelper<float>();
}

XLA_TEST_F(UnaryOpTest, SignAbsTestR1) {
  SignAbsTestHelper<int>();
  SignAbsTestHelper<float>();
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
