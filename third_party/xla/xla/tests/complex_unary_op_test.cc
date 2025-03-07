/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/client/local_client.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/complex_unary_op_samples.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

template <class>
constexpr bool dependent_false = false;

class ComplexUnaryOpTest : public ClientLibraryTestBase {
 protected:
  template <typename T, size_t index, typename... Types>
  std::vector<T> get_column(const std::vector<std::tuple<Types...>>& table) {
    std::vector<T> column;
    std::transform(
        table.cbegin(), table.cend(), std::back_inserter(column),
        [](const auto& item) { return static_cast<T>(std::get<index>(item)); });
    return column;
  }

  template <typename T, typename S>
  void scale_column(std::vector<T>& column, const std::vector<S>& scales) {
    std::transform(column.begin(), column.end(), scales.begin(), column.begin(),
                   [](const T& lhs, const S& rhs) { return lhs * rhs; });
  }

  template <typename C>
  void UnaryTestHelper(XlaOp (*Op)(const XlaOp operand)) {
    using InputType = typename C::InputType;
    using OutputType = typename C::OutputType;
    using FloatType = typename C::FloatType;

    float atol;
    // log(10)/log(2) = 3.3219...
    constexpr int precision_deficiency =
        static_cast<int>(C::dps_deficiency * 3.3219280948873626);
    // precision_deficiency defines a slack allowed when comparing a
    // result value against expected value that is known to be
    // inaccurate to some extent.
    if constexpr (std::is_same_v<FloatType, float>) {
      atol = std::ldexp(1e-6f, precision_deficiency);
    } else if constexpr (std::is_same_v<FloatType, double>) {
      atol = std::ldexp(1e-15f, precision_deficiency);
    } else {
      static_assert(dependent_false<FloatType>);
    }

    XlaBuilder builder(TestName());
    auto table = C().get();
    auto inputs_vec = get_column<InputType, 0>(table);
    auto expected_vec = get_column<OutputType, 1>(table);
    auto scales_vec = get_column<FloatType, 2>(table);
    scale_column(expected_vec, scales_vec);

    auto inputs = ConstantR1<InputType>(&builder, inputs_vec);
    auto scales = ConstantR1<FloatType>(&builder, scales_vec);
    Literal expected = LiteralUtil::CreateR1<OutputType>(expected_vec);

    if constexpr (std::is_same_v<OutputType, FloatType>) {
      auto results = Op(inputs);
      Mul(results, scales);
      ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(atol));
    } else {
      auto results = Op(inputs);
      auto re = Mul(Real(results), scales);
      auto im = Mul(Imag(results), scales);
      Complex(re, im);
      ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(atol));
    }
  }
};

XLA_TEST_F(ComplexUnaryOpTest, Log1pTest) {
  UnaryTestHelper<complex_unary_op_samples::Log1p<float>>(
      [](XlaOp x) { return Log1p(x); });
  UnaryTestHelper<complex_unary_op_samples::Log1p<double>>(
      [](XlaOp x) { return Log1p(x); });
}

XLA_TEST_F(ComplexUnaryOpTest, TanTest) {
  UnaryTestHelper<complex_unary_op_samples::Tan<float>>(
      [](XlaOp x) { return Tan(x); });
  UnaryTestHelper<complex_unary_op_samples::Tan<double>>(
      [](XlaOp x) { return Tan(x); });
}

XLA_TEST_F(ComplexUnaryOpTest, AsinTest) {
  UnaryTestHelper<complex_unary_op_samples::Asin<float>>(Asin);
  UnaryTestHelper<complex_unary_op_samples::Asin<double>>(Asin);
}

XLA_TEST_F(ComplexUnaryOpTest, AsinhTest) {
  UnaryTestHelper<complex_unary_op_samples::Asinh<float>>(Asinh);
  UnaryTestHelper<complex_unary_op_samples::Asinh<double>>(Asinh);
}

}  // namespace
}  // namespace xla
