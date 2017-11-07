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
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ArrayElementwiseOpTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001, 0.0001};
};

class ArrayElementwiseOpTestParamCount
    : public ArrayElementwiseOpTest,
      public ::testing::WithParamInterface<int> {};

XLA_TEST_F(ArrayElementwiseOpTest, NegConstantZeroElementF32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({});
  auto result = builder.Neg(a);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, NegConstantF32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({-2.5f, 3.14f, 2.25f, -10.0f, 6.0f});
  auto result = builder.Neg(a);

  ComputeAndCompareR1<float>(&builder, {2.5f, -3.14f, -2.25f, 10.0f, -6.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, NegConstantS32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({-1, 0, 1, 324,
                                      std::numeric_limits<int32>::min(),
                                      std::numeric_limits<int32>::max()});
  auto result = builder.Neg(a);

  // -min == min for int32 due to an overflow. In C++ it is undefined behavior
  // to do this calculation. For XLA we have not specified that, so it
  // ought to work.
  ComputeAndCompareR1<int32>(&builder,
                             {1, 0, -1, -324, std::numeric_limits<int32>::min(),
                              -std::numeric_limits<int32>::max()},
                             {});
}

XLA_TEST_F(ArrayElementwiseOpTest, IsFiniteZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({});
  auto result = builder.IsFinite(a);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

// A non-canonical quiet NaN value.
static const float kNonCanonicalNaN = tensorflow::bit_cast<float>(0x7FD01234);

XLA_TEST_F(ArrayElementwiseOpTest, IsFiniteScalarF32) {
  ComputationBuilder builder(client_, TestName());
  auto result = builder.IsFinite(builder.ConstantR0<float>(NAN));
  ComputeAndCompareR0<bool>(&builder, false, {});

  EXPECT_TRUE(std::isnan(kNonCanonicalNaN));
  auto result_non_canonical =
      builder.IsFinite(builder.ConstantR0<float>(kNonCanonicalNaN));
  ComputeAndCompareR0<bool>(&builder, false, {});

  const float inf = std::numeric_limits<float>::infinity();
  auto result_inf = builder.IsFinite(builder.ConstantR0<float>(inf));
  ComputeAndCompareR0<bool>(&builder, false, {});

  auto result_neg_inf = builder.IsFinite(builder.ConstantR0<float>(-inf));
  ComputeAndCompareR0<bool>(&builder, false, {});

  auto result_zero = builder.IsFinite(builder.ConstantR0<float>(0.0f));
  ComputeAndCompareR0<bool>(&builder, true, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, IsFiniteR1F32s) {
  ComputationBuilder builder(client_, TestName());
  const float inf = std::numeric_limits<float>::infinity();
  EXPECT_TRUE(std::isnan(kNonCanonicalNaN));
  auto a = builder.ConstantR1<float>(
      {{NAN, 7.0f, kNonCanonicalNaN, -1.0f, inf, -inf}});
  auto result = builder.IsFinite(a);

  ComputeAndCompareR1<bool>(&builder, {false, true, false, true, false, false},
                            {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoConstantF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({-2.5f, 3.14f, 2.25f, -10.0f, 6.0f});
  auto b = builder.ConstantR1<float>({100.0f, 3.13f, 2.75f, 10.5f, -999.0f});
  auto add = builder.Add(a, b);

  ComputeAndCompareR1<float>(&builder, {97.5f, 6.27f, 5.0f, 0.5f, -993.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoConstantZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({});
  auto b = builder.ConstantR1<float>({});
  auto add = builder.Add(a, b);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

TEST_P(ArrayElementwiseOpTestParamCount, AddManyValues) {
  const int count = GetParam();
  ComputationBuilder builder(client_, TestName());
  std::vector<float> a_values;
  std::vector<float> b_values;
  for (int i = 0; i < count; ++i) {
    a_values.push_back(i / static_cast<float>(count));
    b_values.push_back(2 * i / static_cast<float>(count + 2));
  }

  std::unique_ptr<Literal> a_literal = Literal::CreateR1<float>({a_values});
  std::unique_ptr<GlobalData> a_data =
      client_->TransferToServer(*a_literal).ConsumeValueOrDie();
  auto a_constant = builder.ConstantR1<float>(a_values);
  auto a_param = builder.Parameter(0, a_literal->shape(), "a_param");

  std::unique_ptr<Literal> b_literal = Literal::CreateR1<float>({b_values});
  std::unique_ptr<GlobalData> b_data =
      client_->TransferToServer(*b_literal).ConsumeValueOrDie();
  auto b_constant = builder.Parameter(1, a_literal->shape(), "b_param");
  auto b_param = builder.ConstantR1<float>(b_values);

  auto sum1 = builder.Add(a_constant, b_constant);
  auto sum2 = builder.Add(a_constant, b_param);
  auto sum3 = builder.Add(a_param, b_constant);
  auto sum4 = builder.Add(a_param, b_param);

  auto sum = builder.Add(sum1, sum2);
  sum = builder.Add(sum, sum3);
  sum = builder.Add(sum, sum4);

  std::vector<float> expected;
  for (int64 i = 0; i < count; ++i) {
    expected.push_back(4 * (a_values[i] + b_values[i]));
  }

  ComputeAndCompareR1<float>(&builder, expected, {a_data.get(), b_data.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({-2.5f, 3.14f, 2.25f, -10.0f, 6.0f});
  auto b = builder.ConstantR1<float>({100.0f, 3.13f, 2.75f, 10.5f, -999.0f});
  auto add = builder.Sub(a, b);

  ComputeAndCompareR1<float>(&builder, {-102.5f, 0.01f, -0.5f, -20.5f, 1005.0f},
                             {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({});
  auto b = builder.ConstantR1<float>({});
  auto add = builder.Sub(a, b);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantS32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({-1, 0, 2, 1000000000});
  auto b = builder.ConstantR1<int32>({-1, 2, 1, -1});
  auto add = builder.Sub(a, b);

  ComputeAndCompareR1<int32>(&builder, {0, -2, 1, 1000000001}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantZeroElementS32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({});
  auto b = builder.ConstantR1<int32>({});
  auto add = builder.Sub(a, b);

  ComputeAndCompareR1<int32>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, DivTwoConstantF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({-2.5f, 25.5f, 2.25f, -10.0f, 6.0f});
  auto b = builder.ConstantR1<float>({10.0f, 5.1f, 1.0f, 10.0f, -6.0f});
  auto add = builder.Div(a, b);

  ComputeAndCompareR1<float>(&builder, {-0.25f, 5.0f, 2.25f, -1.0f, -1.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, DivTwoConstantZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({});
  auto b = builder.ConstantR1<float>({});
  auto add = builder.Div(a, b);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, DivS32s) {
  // clang-format off
  // Some interesting values to test.
  std::vector<int32> vals = {
    INT32_MIN, INT32_MIN + 1, INT32_MIN + 2, -0x40000000, -0x3fffffff,
    -271181, -1309, -17, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 17, 26, 101,
    7919, 0x40000000, INT32_MAX - 2, INT32_MAX - 1, INT32_MAX};
  // clang-format on

  std::vector<int32> dividends, divisors, quotients, remainders;
  for (int32 divisor : vals) {
    if (divisor != 0) {
      for (int32 dividend : vals) {
        // Avoid integer overflow.
        if (dividend != INT32_MIN || divisor != -1) {
          dividends.push_back(dividend);
          divisors.push_back(divisor);
          quotients.push_back(dividend / divisor);
          remainders.push_back(dividend % divisor);
        }
      }
    }
  }

  {
    ComputationBuilder builder(client_, TestName());
    ComputationDataHandle dividend;
    ComputationDataHandle divisor;
    auto dividend_data =
        CreateR1Parameter<int32>(dividends, 0, "dividend", &builder, &dividend);
    auto divisor_data =
        CreateR1Parameter<int32>(divisors, 1, "divisor", &builder, &divisor);
    builder.Div(dividend, divisor);

    ComputeAndCompareR1<int32>(&builder, quotients,
                               {dividend_data.get(), divisor_data.get()});
  }

  // Test with a compile-time constant divisor.
  {
    ComputationBuilder builder(client_, TestName());
    ComputationDataHandle dividend;
    auto dividend_data =
        CreateR1Parameter<int32>(dividends, 0, "dividend", &builder, &dividend);
    builder.Div(dividend, builder.ConstantR1<int32>(divisors));

    ComputeAndCompareR1<int32>(&builder, quotients, {dividend_data.get()});
  }

  {
    ComputationBuilder builder(client_, TestName());
    ComputationDataHandle dividend;
    ComputationDataHandle divisor;
    auto dividend_data =
        CreateR1Parameter<int32>(dividends, 0, "dividend", &builder, &dividend);
    auto divisor_data =
        CreateR1Parameter<int32>(divisors, 1, "divisor", &builder, &divisor);
    builder.Rem(dividend, divisor);

    ComputeAndCompareR1<int32>(&builder, remainders,
                               {dividend_data.get(), divisor_data.get()});
  }

  // Test with a compile-time constant divisor.
  {
    ComputationBuilder builder(client_, TestName());
    ComputationDataHandle dividend;
    auto dividend_data =
        CreateR1Parameter<int32>(dividends, 0, "dividend", &builder, &dividend);
    builder.Rem(dividend, builder.ConstantR1<int32>(divisors));

    ComputeAndCompareR1<int32>(&builder, remainders, {dividend_data.get()});
  }
}

XLA_TEST_F(ArrayElementwiseOpTest, DivU32s) {
  // clang-format off
  // Some interesting values to test.
  std::vector<uint32> vals = {
    0, 1, 2, 17, 101, 3333, 0x7FFFFFFF, 0xABCDEF12, 0xCAFEBEEF, 0x80000000,
    0x80000001, UINT32_MAX - 2, UINT32_MAX - 1, UINT32_MAX};
  // clang-format on

  std::vector<uint32> dividends, divisors, quotients, remainders;
  for (uint32 divisor : vals) {
    if (divisor != 0) {
      for (uint32 dividend : vals) {
        dividends.push_back(dividend);
        divisors.push_back(divisor);
        quotients.push_back(dividend / divisor);
        remainders.push_back(dividend % divisor);
      }
    }
  }

  {
    ComputationBuilder builder(client_, TestName());
    ComputationDataHandle dividend;
    ComputationDataHandle divisor;
    auto dividend_data = CreateR1Parameter<uint32>(dividends, 0, "dividend",
                                                   &builder, &dividend);
    auto divisor_data =
        CreateR1Parameter<uint32>(divisors, 1, "divisor", &builder, &divisor);
    builder.Div(dividend, divisor);

    ComputeAndCompareR1<uint32>(&builder, quotients,
                                {dividend_data.get(), divisor_data.get()});
  }

  {
    ComputationBuilder builder(client_, TestName());
    ComputationDataHandle dividend;
    auto dividend_data = CreateR1Parameter<uint32>(dividends, 0, "dividend",
                                                   &builder, &dividend);
    builder.Div(dividend, builder.ConstantR1<uint32>(divisors));

    ComputeAndCompareR1<uint32>(&builder, quotients, {dividend_data.get()});
  }

  {
    ComputationBuilder builder(client_, TestName());
    ComputationDataHandle dividend;
    ComputationDataHandle divisor;
    auto dividend_data = CreateR1Parameter<uint32>(dividends, 0, "dividend",
                                                   &builder, &dividend);
    auto divisor_data =
        CreateR1Parameter<uint32>(divisors, 1, "divisor", &builder, &divisor);
    builder.Rem(dividend, divisor);

    ComputeAndCompareR1<uint32>(&builder, remainders,
                                {dividend_data.get(), divisor_data.get()});
  }

  {
    ComputationBuilder builder(client_, TestName());
    ComputationDataHandle dividend;
    auto dividend_data = CreateR1Parameter<uint32>(dividends, 0, "dividend",
                                                   &builder, &dividend);
    builder.Rem(dividend, builder.ConstantR1<uint32>(divisors));

    ComputeAndCompareR1<uint32>(&builder, remainders, {dividend_data.get()});
  }
}

XLA_TEST_F(ArrayElementwiseOpTest, RemF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>(
      {-2.5f, 25.5f, 2.25f, -10.0f, 6.0f, 3.0f, 3.0f, -1.0f, -8.0f});
  auto b = builder.ConstantR1<float>(
      {10.0f, 5.1f, 1.0f, 10.0f, -6.0f, 2.0f, -2.0f, 7.0f, -4.0f});
  auto add = builder.Rem(a, b);

  ComputeAndCompareR1<float>(
      &builder, {-2.5f, 0.0f, 0.25f, 0.0f, -0.0f, 1.0f, 1.0f, -1.0f, -0.0f}, {},
      error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, RemZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({});
  auto b = builder.ConstantR1<float>({});
  auto add = builder.Rem(a, b);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, RemF64s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<double>(
      {-2.5, 25.5, 2.25, -10.0, 6.0, 3.0, 3.0, -1.0, -8.0});
  auto b = builder.ConstantR1<double>(
      {10.0, 5.1, 1.0, 10.0, -6.0, 2.0, -2.0, 7.0, -4.0});
  auto add = builder.Rem(a, b);

  ComputeAndCompareR1<double>(
      &builder, {-2.5, 0.0, 0.25, 0.0, -0.0, 1.0, 1.0, -1.0, -0.0}, {},
      error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({-2.5f, 25.5f, 2.25f, -10.0f, 6.0f});
  auto b = builder.ConstantR1<float>({10.0f, 5.0f, 1.0f, 10.0f, -6.0f});
  auto add = builder.Mul(a, b);

  ComputeAndCompareR1<float>(&builder, {-25.0f, 127.5f, 2.25f, -100.0f, -36.0f},
                             {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({});
  auto b = builder.ConstantR1<float>({});
  auto add = builder.Mul(a, b);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantS32s) {
  std::vector<int32> data = {0,
                             1,
                             -1,
                             1234,
                             0x1a243514,
                             std::numeric_limits<int32>::max(),
                             std::numeric_limits<int32>::min()};
  // Form the test data set using all products of 'data' with itself.
  std::vector<int32> a_data, b_data, expected;
  for (int32 a : data) {
    for (int32 b : data) {
      a_data.push_back(a);
      b_data.push_back(b);
      expected.push_back(static_cast<uint32>(a) * static_cast<uint32>(b));
    }
  }

  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>(a_data);
  auto b = builder.ConstantR1<int32>(b_data);
  auto add = builder.Mul(a, b);

  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantZeroElementS32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({});
  auto b = builder.ConstantR1<int32>({});
  auto add = builder.Mul(a, b);

  ComputeAndCompareR1<int32>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantU32s) {
  std::vector<uint32> data = {0,          1,          0xDEADBEEF, 1234,
                              0x1a243514, 0xFFFFFFFF, 0x80808080};

  // Form the test data set using all products of 'data' with itself.
  std::vector<uint32> a_data, b_data, expected;
  for (uint32 a : data) {
    for (uint32 b : data) {
      a_data.push_back(a);
      b_data.push_back(b);
      expected.push_back(a * b);
    }
  }

  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint32>(a_data);
  auto b = builder.ConstantR1<uint32>(b_data);
  auto add = builder.Mul(a, b);

  ComputeAndCompareR1<uint32>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndPredR1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<bool>({false, false, true, true});
  auto b = builder.ConstantR1<bool>({false, true, false, true});
  auto out = builder.And(a, b);

  ComputeAndCompareR1<bool>(&builder, {false, false, false, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndPredR2) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2<bool>({{false, false}, {true, true}});
  auto b = builder.ConstantR2<bool>({{false, true}, {false, true}});
  auto out = builder.And(a, b);

  Array2D<bool> expected_array({{false, false}, {false, true}});
  ComputeAndCompareR2<bool>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndZeroElementPredR1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<bool>({});
  auto b = builder.ConstantR1<bool>({});
  auto out = builder.And(a, b);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndS32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({0, -1, -8});
  auto b = builder.ConstantR1<int32>({5, -7, 12});
  auto out = builder.And(a, b);

  ComputeAndCompareR1<int32>(&builder, {0, -7, 8}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndS32R2) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2<int32>({{0, -5}, {-1, 5}});
  auto b = builder.ConstantR2<int32>({{1, -6}, {4, 5}});
  auto out = builder.And(a, b);

  Array2D<int32> expected_array({{0, -6}, {4, 5}});
  ComputeAndCompareR2<int32>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndZeroElementS32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({});
  auto b = builder.ConstantR1<int32>({});
  auto out = builder.And(a, b);

  ComputeAndCompareR1<int32>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndU32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({0, 1, 8});
  auto b = builder.ConstantR1<int32>({5, 7, 12});
  auto out = builder.And(a, b);

  ComputeAndCompareR1<int32>(&builder, {0, 1, 8}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndU32R2) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2<uint32>({{0, 1}, {3, 8}});
  auto b = builder.ConstantR2<uint32>({{1, 0}, {7, 6}});
  auto out = builder.And(a, b);

  Array2D<uint32> expected_array({{0, 0}, {3, 0}});
  ComputeAndCompareR2<uint32>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndZeroElementU32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint32>({});
  auto b = builder.ConstantR1<uint32>({});
  auto out = builder.And(a, b);

  ComputeAndCompareR1<uint32>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrPredR1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<bool>({false, false, true, true});
  auto b = builder.ConstantR1<bool>({false, true, false, true});
  auto out = builder.Or(a, b);

  ComputeAndCompareR1<bool>(&builder, {false, true, true, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrPredR2) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2<bool>({{false, false}, {true, true}});
  auto b = builder.ConstantR2<bool>({{false, true}, {false, true}});
  auto out = builder.Or(a, b);

  Array2D<bool> expected_array({{false, true}, {true, true}});
  ComputeAndCompareR2<bool>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrZeroElementPredR1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<bool>({});
  auto b = builder.ConstantR1<bool>({});
  auto out = builder.Or(a, b);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrS32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({0, -1, 8});
  auto b = builder.ConstantR1<int32>({5, -7, 4});
  auto out = builder.Or(a, b);

  ComputeAndCompareR1<int32>(&builder, {5, -1, 12}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrS32R2) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2<int32>({{0, -1}, {8, 8}});
  auto b = builder.ConstantR2<int32>({{5, -7}, {4, 1}});
  auto out = builder.Or(a, b);

  Array2D<int32> expected_array({{5, -1}, {12, 9}});
  ComputeAndCompareR2<int32>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrZeroElementS32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({});
  auto b = builder.ConstantR1<int32>({});
  auto out = builder.Or(a, b);

  ComputeAndCompareR1<int32>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrU32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint32>({0, 1, 8});
  auto b = builder.ConstantR1<uint32>({5, 7, 4});
  auto out = builder.Or(a, b);

  ComputeAndCompareR1<uint32>(&builder, {5, 7, 12}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrU32R2) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2<uint32>({{0, 1}, {8, 8}});
  auto b = builder.ConstantR2<uint32>({{5, 7}, {4, 1}});
  auto out = builder.Or(a, b);

  Array2D<uint32> expected_array({{5, 7}, {12, 9}});
  ComputeAndCompareR2<uint32>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrZeroElementU32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint32>({});
  auto b = builder.ConstantR1<uint32>({});
  auto out = builder.Or(a, b);

  ComputeAndCompareR1<uint32>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotPredR1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<bool>({false, true, true, false});
  auto out = builder.Not(a);

  ComputeAndCompareR1<bool>(&builder, {true, false, false, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotPredR2) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2<bool>({{false, true}, {true, false}});
  auto out = builder.Not(a);

  Array2D<bool> expected_array({{true, false}, {false, true}});
  ComputeAndCompareR2<bool>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotZeroElementPredR1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<bool>({});
  auto out = builder.Not(a);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotS32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({-1, 0, 1});
  auto out = builder.Not(a);

  ComputeAndCompareR1<int32>(&builder, {0, -1, -2}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotS32R2) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2<int32>({{-1, 0}, {1, 8}});
  auto out = builder.Not(a);

  Array2D<int32> expected_array({{0, -1}, {-2, -9}});
  ComputeAndCompareR2<int32>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotZeroElementS32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({});
  auto out = builder.Not(a);

  ComputeAndCompareR1<int32>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotU32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint32>({0, 4294967295});
  auto out = builder.Not(a);

  ComputeAndCompareR1<uint32>(&builder, {4294967295, 0}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotU32R2) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2<uint32>({{0, 4294967295}, {1, 4294967294}});
  auto out = builder.Not(a);

  Array2D<uint32> expected_array({{4294967295, 0}, {4294967294, 1}});
  ComputeAndCompareR2<uint32>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotZeroElementU32R1) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint32>({});
  auto out = builder.Not(a);

  ComputeAndCompareR1<uint32>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftLeftS32) {
  ComputationBuilder builder(client_, TestName());
  auto a =
      builder.ConstantR1<int32>({static_cast<int32>(0x12345678),
                                 static_cast<int32>(0xF0001000), 1, 3, 77});
  auto b = builder.ConstantR1<int32>({4, 8, 2, 7, 15});
  auto out = builder.ShiftLeft(a, b);

  ComputeAndCompareR1<int32>(
      &builder,
      {static_cast<int32>(0x23456780), 0x00100000, 0x4, 0x180, 2523136}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftRightArithmeticS32) {
  ComputationBuilder builder(client_, TestName());
  auto a =
      builder.ConstantR1<int32>({static_cast<int32>(0x92345678),
                                 static_cast<int32>(0x10001000), 1, 3, 77});
  auto b = builder.ConstantR1<int32>({4, 8, 2, 7, 2});
  auto out = builder.ShiftRightArithmetic(a, b);

  ComputeAndCompareR1<int32>(&builder,
                             {static_cast<int32>(0xF9234567),
                              static_cast<int32>(0x00100010), 0, 0, 19},
                             {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftRightLogicalS32) {
  ComputationBuilder builder(client_, TestName());
  auto a =
      builder.ConstantR1<int32>({static_cast<int32>(0x92345678),
                                 static_cast<int32>(0x10001000), 1, 3, 77});
  auto b = builder.ConstantR1<int32>({4, 8, 2, 7, 5});
  auto out = builder.ShiftRightLogical(a, b);

  ComputeAndCompareR1<int32>(&builder, {0x09234567, 0x00100010, 0, 0, 2}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftLeftU32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint32>({0x12345678, 0xF0001000, 1, 3, 77});
  auto b = builder.ConstantR1<uint32>({4, 8, 2, 7, 15});
  auto out = builder.ShiftLeft(a, b);

  ComputeAndCompareR1<uint32>(
      &builder, {0x23456780, 0x00100000, 0x4, 0x180, 2523136}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftRightArithmeticU32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint32>({0x92345678, 0x10001000, 1, 3, 77});
  auto b = builder.ConstantR1<uint32>({4, 8, 2, 7, 2});
  auto out = builder.ShiftRightArithmetic(a, b);

  ComputeAndCompareR1<uint32>(&builder, {0xF9234567, 0x00100010, 0, 0, 19}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftRightLogicalU32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint32>({0x92345678, 0x10001000, 1, 3, 77});
  auto b = builder.ConstantR1<uint32>({4, 8, 2, 7, 5});
  auto out = builder.ShiftRightLogical(a, b);

  ComputeAndCompareR1<uint32>(&builder, {0x09234567, 0x00100010, 0, 0, 2}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqF32s) {
  SetFastMathDisabled(true);
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({-2.5f, 25.5f, 2.25f, NAN, 6.0f});
  auto rhs = builder.ConstantR1<float>({10.0f, 5.0f, 2.25f, 10.0f, NAN});
  auto compare = builder.Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {false, false, true, false, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({});
  auto rhs = builder.ConstantR1<float>({});
  auto compare = builder.Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGeF32s) {
  SetFastMathDisabled(true);
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({-2.5f, 25.5f, 2.25f, NAN, 6.0f});
  auto rhs = builder.ConstantR1<float>({10.0f, 5.0f, 1.0f, 10.0f, NAN});
  auto compare = builder.Ge(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {false, true, true, false, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGtF32s) {
  SetFastMathDisabled(true);
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({-2.5f, 25.5f, 2.25f, NAN, 6.0f});
  auto rhs = builder.ConstantR1<float>({10.0f, 5.0f, 1.0f, 10.0f, NAN});
  auto compare = builder.Gt(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {false, true, true, false, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLeF32s) {
  SetFastMathDisabled(true);
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({-2.5f, 5.0f, 2.25f, NAN, 6.0f});
  auto rhs = builder.ConstantR1<float>({10.0f, 5.0f, 1.0f, 10.0f, NAN});
  auto compare = builder.Le(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {true, true, false, false, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLtF32s) {
  SetFastMathDisabled(true);
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({-2.5f, 25.5f, 2.25f, NAN, 6.0f});
  auto rhs = builder.ConstantR1<float>({10.0f, 5.0f, 1.0f, 10.0f, NAN});
  auto compare = builder.Lt(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {true, false, false, false, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqS32s) {
  const int32 min = std::numeric_limits<int32>::min();
  const int32 max = std::numeric_limits<int32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<int32>({min, min, min, 0, 0, 0, max, max, max});
  auto rhs = builder.ConstantR1<int32>({min, 0, max, -1, 0, 1, min, 0, max});
  auto compare = builder.Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, false, false, false, true, false, false, false, true},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqZeroElementS32s) {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<int32>({});
  auto rhs = builder.ConstantR1<int32>({});
  auto compare = builder.Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareNeF32s) {
  // Disable fast-math because we're operating on NaNs.
  SetFastMathDisabled(true);

  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({-2.5f, 25.5f, 2.25f, NAN, 6.0f});
  auto rhs = builder.ConstantR1<float>({10.0f, 25.5f, 1.0f, 10.0f, NAN});
  auto compare = builder.Ne(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {true, false, true, true, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareNeS32s) {
  const int32 min = std::numeric_limits<int32>::min();
  const int32 max = std::numeric_limits<int32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<int32>({min, min, min, 0, 0, 0, max, max, max});
  auto rhs = builder.ConstantR1<int32>({min, 0, max, -1, 0, 1, min, 0, max});
  auto compare = builder.Ne(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, true, true, true, false, true, true, true, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGeS32s) {
  const int32 min = std::numeric_limits<int32>::min();
  const int32 max = std::numeric_limits<int32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<int32>({min, min, min, 0, 0, 0, max, max, max});
  auto rhs = builder.ConstantR1<int32>({min, 0, max, -1, 0, 1, min, 0, max});
  auto compare = builder.Ge(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, false, false, true, true, false, true, true, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGtS32s) {
  const int32 min = std::numeric_limits<int32>::min();
  const int32 max = std::numeric_limits<int32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<int32>({min, min, min, 0, 0, 0, max, max, max});
  auto rhs = builder.ConstantR1<int32>({min, 0, max, -1, 0, 1, min, 0, max});
  auto compare = builder.Gt(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, false, false, true, false, false, true, true, false},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLeS32s) {
  const int32 min = std::numeric_limits<int32>::min();
  const int32 max = std::numeric_limits<int32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<int32>({min, min, min, 0, 0, 0, max, max, max});
  auto rhs = builder.ConstantR1<int32>({min, 0, max, -1, 0, 1, min, 0, max});
  auto compare = builder.Le(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, true, true, false, true, true, false, false, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLtS32s) {
  const int32 min = std::numeric_limits<int32>::min();
  const int32 max = std::numeric_limits<int32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<int32>({min, min, min, 0, 0, 0, max, max, max});
  auto rhs = builder.ConstantR1<int32>({min, 0, max, -1, 0, 1, min, 0, max});
  auto compare = builder.Lt(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, true, true, false, false, true, false, false, false},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqU32s) {
  const uint32 max = std::numeric_limits<uint32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<uint32>({0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = builder.ConstantR1<uint32>({0, 1, max, 4, 5, 6, 0, 1, max});
  auto compare = builder.Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, false, false, false, true, false, false, false, true},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareNeU32s) {
  const uint32 max = std::numeric_limits<uint32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<uint32>({0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = builder.ConstantR1<uint32>({0, 1, max, 4, 5, 6, 0, 1, max});
  auto compare = builder.Ne(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, true, true, true, false, true, true, true, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGeU32s) {
  const uint32 max = std::numeric_limits<uint32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<uint32>({0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = builder.ConstantR1<uint32>({0, 1, max, 4, 5, 6, 0, 1, max});
  auto compare = builder.Ge(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, false, false, true, true, false, true, true, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGtU32s) {
  const uint32 max = std::numeric_limits<uint32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<uint32>({0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = builder.ConstantR1<uint32>({0, 1, max, 4, 5, 6, 0, 1, max});
  auto compare = builder.Gt(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, false, false, true, false, false, true, true, false},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLeU32s) {
  const uint32 max = std::numeric_limits<uint32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<uint32>({0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = builder.ConstantR1<uint32>({0, 1, max, 4, 5, 6, 0, 1, max});
  auto compare = builder.Le(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, true, true, false, true, true, false, false, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLtU32s) {
  const uint32 max = std::numeric_limits<uint32>::max();
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<uint32>({0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = builder.ConstantR1<uint32>({0, 1, max, 4, 5, 6, 0, 1, max});
  auto compare = builder.Lt(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, true, true, false, false, true, false, false, false},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, PowF32s) {
  SetFastMathDisabled(true);
  ComputationBuilder builder(client_, TestName());
  auto lhs =
      builder.ConstantR1<float>({4.0f, 2.0f, 2.0f, NAN, 6.0f, -2.0f, -2.0f});
  auto rhs =
      builder.ConstantR1<float>({2.0f, -2.0f, 3.0f, 10.0f, NAN, 3.0f, 4.0f});
  auto minimum = builder.Pow(lhs, rhs);

  ComputeAndCompareR1<float>(
      &builder, {16.0f, 0.25f, 8.0f, NAN, NAN, -8.0f, 16.0f}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, PowNonIntegerF32s) {
  SetFastMathDisabled(true);
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({-2.0f, -0.6f, -0.6f, 0.0f});
  auto rhs = builder.ConstantR1<float>({0.5f, 0.6f, -0.6f, -0.6f});
  auto minimum = builder.Pow(lhs, rhs);

  ComputeAndCompareR1<float>(&builder, {NAN, NAN, NAN, INFINITY}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, PowZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({});
  auto rhs = builder.ConstantR1<float>({});
  auto minimum = builder.Pow(lhs, rhs);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

// Some Pow cases that can be implemented more efficiently.
XLA_TEST_F(ArrayElementwiseOpTest, PowSpecialF32) {
  ComputationBuilder b(client_, TestName());

  std::vector<float> values = {1.0f, 2.0f, 3.2f, -4.0f};
  std::vector<float> exponents = {0.0f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};

  std::unique_ptr<Literal> param_literal = Literal::CreateR1<float>(values);
  std::unique_ptr<GlobalData> param_data =
      client_->TransferToServer(*param_literal).ConsumeValueOrDie();

  auto sum = b.ConstantR0<float>(0.0f);
  auto param = b.Parameter(0, param_literal->shape(), "param");
  for (float exponent : exponents) {
    sum = b.Add(sum, b.Pow(param, b.ConstantR0<float>(exponent)));
  }

  std::vector<float> expected;
  for (auto value : values) {
    float sum = 0.0f;
    for (float exponent : exponents) {
      sum += std::pow(value, exponent);
    }
    expected.push_back(sum);
  }

  ComputeAndCompareR1<float>(&b, expected, {param_data.get()}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, PowOfExpF32) {
  ComputationBuilder b(client_, TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.0f, 5.7f};
  std::vector<float> values1 = {0.0f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};

  std::unique_ptr<Literal> literal0 = Literal::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(*literal0).ConsumeValueOrDie();
  std::unique_ptr<Literal> literal1 = Literal::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(*literal1).ConsumeValueOrDie();
  auto param0 = b.Parameter(0, literal0->shape(), "param0");
  auto param1 = b.Parameter(1, literal1->shape(), "param1");
  b.Pow(b.Exp(param0), param1);

  std::vector<float> expected(values0.size());
  for (int64 i = 0; i < values0.size(); ++i) {
    expected[i] = std::pow(std::exp(values0[i]), values1[i]);
  }

  ComputeAndCompareR1<float>(&b, expected, {data0.get(), data1.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, LogOfPowerF32) {
  ComputationBuilder b(client_, TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, 4.0f, 0.5f, 5.7f};
  std::vector<float> values1 = {0.0f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};

  std::unique_ptr<Literal> literal0 = Literal::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(*literal0).ConsumeValueOrDie();
  std::unique_ptr<Literal> literal1 = Literal::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(*literal1).ConsumeValueOrDie();
  auto param0 = b.Parameter(0, literal0->shape(), "param0");
  auto param1 = b.Parameter(1, literal1->shape(), "param1");
  b.Log(b.Pow(param0, param1));

  std::vector<float> expected(values0.size());
  for (int64 i = 0; i < values0.size(); ++i) {
    expected[i] = std::log(std::pow(values0[i], values1[i]));
  }

  ComputeAndCompareR1<float>(&b, expected, {data0.get(), data1.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MulOfExpF32) {
  ComputationBuilder b(client_, TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.0f, 5.7f};
  std::vector<float> values1 = {0.0f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};

  std::unique_ptr<Literal> literal0 = Literal::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(*literal0).ConsumeValueOrDie();
  std::unique_ptr<Literal> literal1 = Literal::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(*literal1).ConsumeValueOrDie();
  auto param0 = b.Parameter(0, literal0->shape(), "param0");
  auto param1 = b.Parameter(1, literal1->shape(), "param1");
  b.Mul(b.Exp(param0), b.Exp(param1));

  std::vector<float> expected(values0.size());
  for (int64 i = 0; i < values0.size(); ++i) {
    expected[i] = std::exp(values0[i]) * std::exp(values1[i]);
  }

  ComputeAndCompareR1<float>(&b, expected, {data0.get(), data1.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, DivOfExpF32) {
  ComputationBuilder b(client_, TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.0f, 5.7f};
  std::vector<float> values1 = {0.0f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};

  std::unique_ptr<Literal> literal0 = Literal::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(*literal0).ConsumeValueOrDie();
  std::unique_ptr<Literal> literal1 = Literal::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(*literal1).ConsumeValueOrDie();
  auto param0 = b.Parameter(0, literal0->shape(), "param0");
  auto param1 = b.Parameter(1, literal1->shape(), "param1");
  b.Div(param0, b.Exp(param1));

  std::vector<float> expected(values0.size());
  for (int64 i = 0; i < values0.size(); ++i) {
    expected[i] = values0[i] / std::exp(values1[i]);
  }

  ComputeAndCompareR1<float>(&b, expected, {data0.get(), data1.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Div3_lhs_F32) {
  ComputationBuilder b(client_, TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.45f, 5.7f};
  std::vector<float> values1 = {0.1f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};
  std::vector<float> values2 = {0.1f, 1.1f, 6.9f, 12.5f, -15.0f, -0.5f};

  std::unique_ptr<Literal> literal0 = Literal::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(*literal0).ConsumeValueOrDie();

  std::unique_ptr<Literal> literal1 = Literal::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(*literal1).ConsumeValueOrDie();

  std::unique_ptr<Literal> literal2 = Literal::CreateR1<float>(values2);
  std::unique_ptr<GlobalData> data2 =
      client_->TransferToServer(*literal2).ConsumeValueOrDie();
  auto param0 = b.Parameter(0, literal0->shape(), "param0");
  auto param1 = b.Parameter(1, literal1->shape(), "param1");
  auto param2 = b.Parameter(2, literal2->shape(), "param2");
  b.Div(b.Div(param0, param1), param2);

  std::vector<float> expected(values0.size());
  for (int64 i = 0; i < values0.size(); ++i) {
    expected[i] = (values0[i] / values1[i]) / values2[i];
  }

  ComputeAndCompareR1<float>(
      &b, expected, {data0.get(), data1.get(), data2.get()}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Div3_rhs_F32) {
  ComputationBuilder b(client_, TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.45f, 5.7f};
  std::vector<float> values1 = {0.1f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};
  std::vector<float> values2 = {0.1f, 1.1f, 6.9f, 12.5f, -15.0f, -0.5f};

  std::unique_ptr<Literal> literal0 = Literal::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(*literal0).ConsumeValueOrDie();

  std::unique_ptr<Literal> literal1 = Literal::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(*literal1).ConsumeValueOrDie();

  std::unique_ptr<Literal> literal2 = Literal::CreateR1<float>(values2);
  std::unique_ptr<GlobalData> data2 =
      client_->TransferToServer(*literal2).ConsumeValueOrDie();

  auto param0 = b.Parameter(0, literal0->shape(), "param0");
  auto param1 = b.Parameter(1, literal1->shape(), "param1");
  auto param2 = b.Parameter(2, literal2->shape(), "param2");
  b.Div(param0, b.Div(param1, param2));

  std::vector<float> expected(values0.size());
  for (int64 i = 0; i < values0.size(); ++i) {
    expected[i] = values0[i] / (values1[i] / values2[i]);
  }

  ComputeAndCompareR1<float>(
      &b, expected, {data0.get(), data1.get(), data2.get()}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, DivOfPowerF32) {
  ComputationBuilder b(client_, TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.45f, 5.7f};
  std::vector<float> values1 = {0.1f, 1.0f, 2.0f, 0.5f, 1.0f, 0.5f};
  std::vector<float> values2 = {0.1f, 1.1f, 6.9f, 9.5f, -11.0f, -0.5f};

  std::unique_ptr<Literal> literal0 = Literal::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(*literal0).ConsumeValueOrDie();

  std::unique_ptr<Literal> literal1 = Literal::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(*literal1).ConsumeValueOrDie();

  std::unique_ptr<Literal> literal2 = Literal::CreateR1<float>(values2);
  std::unique_ptr<GlobalData> data2 =
      client_->TransferToServer(*literal2).ConsumeValueOrDie();

  auto param0 = b.Parameter(0, literal0->shape(), "param0");
  auto param1 = b.Parameter(1, literal1->shape(), "param1");
  auto param2 = b.Parameter(2, literal2->shape(), "param2");
  b.Div(param0, b.Pow(param1, param2));

  std::vector<float> expected(values0.size());
  for (int64 i = 0; i < values0.size(); ++i) {
    expected[i] = values0[i] / std::pow(values1[i], values2[i]);
  }

  ComputeAndCompareR1<float>(
      &b, expected, {data0.get(), data1.get(), data2.get()}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Div4F32) {
  ComputationBuilder b(client_, TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.45f, 5.7f};
  std::vector<float> values1 = {0.1f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};
  std::vector<float> values2 = {0.1f, 1.1f, 6.9f, 12.5f, -15.0f, -0.5f};
  std::vector<float> values3 = {2.1f, 3.1f, 9.9f, -4.5f, -11.0f, -21.5f};

  std::unique_ptr<Literal> literal0 = Literal::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(*literal0).ConsumeValueOrDie();

  std::unique_ptr<Literal> literal1 = Literal::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(*literal1).ConsumeValueOrDie();

  std::unique_ptr<Literal> literal2 = Literal::CreateR1<float>(values2);
  std::unique_ptr<GlobalData> data2 =
      client_->TransferToServer(*literal2).ConsumeValueOrDie();

  std::unique_ptr<Literal> literal3 = Literal::CreateR1<float>(values3);
  std::unique_ptr<GlobalData> data3 =
      client_->TransferToServer(*literal3).ConsumeValueOrDie();

  auto param0 = b.Parameter(0, literal0->shape(), "param0");
  auto param1 = b.Parameter(1, literal1->shape(), "param1");
  auto param2 = b.Parameter(2, literal2->shape(), "param2");
  auto param3 = b.Parameter(3, literal3->shape(), "param2");
  b.Div(b.Div(param0, param1), b.Div(param2, param3));

  std::vector<float> expected(values0.size());
  for (int64 i = 0; i < values0.size(); ++i) {
    expected[i] = (values0[i] / values1[i]) / (values2[i] / values3[i]);
  }

  ComputeAndCompareR1<float>(
      &b, expected, {data0.get(), data1.get(), data2.get(), data3.get()},
      error_spec_);
}

TEST_P(ArrayElementwiseOpTestParamCount, SquareManyValues) {
  const int count = GetParam();
  ComputationBuilder builder(client_, TestName());
  std::vector<float> values;
  values.reserve(count);
  for (int i = 0; i < count; ++i) {
    values.push_back(i / static_cast<float>(count));
  }
  auto x = builder.ConstantR1<float>(values);
  auto exp = builder.Pow(x, builder.ConstantR0<float>(2.0f));

  std::vector<float> expected;
  expected.reserve(values.size());
  for (float value : values) {
    expected.push_back(value * value);
  }

  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SquareIn4D) {
  ComputationBuilder builder(client_, TestName());
  Array4D<float> values(2, 2, 2, 2);

  std::vector<float> values_vector;
  std::vector<float> expected_vector;
  for (int i = 0; i < values.num_elements(); ++i) {
    values_vector.push_back(static_cast<float>(i) / values.num_elements());
    expected_vector.push_back(values_vector.back() * values_vector.back());
  }
  values.SetValues(values_vector);

  Array4D<float> expected(2, 2, 2, 2, expected_vector);

  auto x = builder.ConstantR4FromArray4D<float>(values);
  auto exp = builder.Pow(x, builder.ConstantR0<float>(2.0f));

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SquareIn4DZeroElements) {
  ComputationBuilder builder(client_, TestName());
  Array4D<float> values(2, 2, 0, 2);
  Array4D<float> expected(2, 2, 0, 2);

  auto x = builder.ConstantR4FromArray4D<float>(values);
  auto exp = builder.Pow(x, builder.ConstantR0<float>(2.0f));

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

// GPU backend emits nvvm intrinsic for fmin and fmax, whose semantics is NOT
// such
// * fmin(NaN, x) = x
// * fmax(NaN, x) = x
// so we only test NAN on CPU.
//
// TODO(b/28180546): Make this compile in a way that is consistent
// among backends.
XLA_TEST_F(ArrayElementwiseOpTest, MinF32s) {
  ComputationBuilder builder(client_, TestName());
#if !defined(XLA_TEST_BACKEND_CPU)
  auto lhs = builder.ConstantR1<float>({1.0f, 1.0f, 2.25f});
  auto rhs = builder.ConstantR1<float>({2.0f, -5.0f, 1.0f});
#else
  SetFastMathDisabled(true);
  auto lhs = builder.ConstantR1<float>({1.0f, 1.0f, 2.25f, NAN, 6.0f});
  auto rhs = builder.ConstantR1<float>({2.0f, -5.0f, 1.0f, 10.0f, NAN});
#endif
  auto minimum = builder.Min(lhs, rhs);

  ComputeAndCompareR1<float>(&builder,
#if !defined(XLA_TEST_BACKEND_CPU)
                             {1.0f, -5.0f, 1.0f},
#else
                             {1.0f, -5.0f, 1.0f, 10.0f, 6.0f},
#endif
                             {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MinZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({});
  auto rhs = builder.ConstantR1<float>({});
  auto minimum = builder.Min(lhs, rhs);
  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

// TODO(b/28180546): Make this compile in a way that is consistent
// among backends. See comment on MinF32s test above.
XLA_TEST_F(ArrayElementwiseOpTest, MinF64s) {
  ComputationBuilder builder(client_, TestName());
#if !defined(XLA_TEST_BACKEND_CPU)
  auto lhs = builder.ConstantR1<double>({1.0, 1.0, 2.25});
  auto rhs = builder.ConstantR1<double>({2.0, -5.0, 1.0});
#else
  SetFastMathDisabled(true);
  auto lhs = builder.ConstantR1<double>({1.0, 1.0, 2.25, NAN, 6.0});
  auto rhs = builder.ConstantR1<double>({2.0, -5.0, 1.0, 10.0, NAN});
#endif
  auto minimum = builder.Min(lhs, rhs);

  ComputeAndCompareR1<double>(&builder,
#if !defined(XLA_TEST_BACKEND_CPU)
                              {1.0, -5.0, 1.0},
#else
                              {1.0, -5.0, 1.0, 10.0, 6.0},
#endif
                              {}, error_spec_);
}

// TODO(b/28180546): Make this compile in a way that is consistent
// among backends. See comment on MinF32s test above.
XLA_TEST_F(ArrayElementwiseOpTest, MaxF32s) {
  ComputationBuilder builder(client_, TestName());
#if !defined(XLA_TEST_BACKEND_CPU)
  auto lhs = builder.ConstantR1<float>({1.0f, 1.0f, 2.25f});
  auto rhs = builder.ConstantR1<float>({2.0f, -5.0f, 1.0f});
#else
  SetFastMathDisabled(true);
  auto lhs = builder.ConstantR1<float>({1.0f, 1.0f, 2.25f, NAN, 6.0f});
  auto rhs = builder.ConstantR1<float>({2.0f, -5.0f, 1.0f, 10.0f, NAN});
#endif
  auto maximum = builder.Max(lhs, rhs);

  ComputeAndCompareR1<float>(&builder,
#if !defined(XLA_TEST_BACKEND_CPU)
                             {2.0f, 1.0f, 2.25f},
#else
                             {2.0f, 1.0f, 2.25f, 10.0f, 6.0f},
#endif
                             {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({});
  auto rhs = builder.ConstantR1<float>({});
  auto minimum = builder.Max(lhs, rhs);
  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

// TODO(b/28180546): Make this compile in a way that is consistent
// among backends. See comment on MinF32s test above.
XLA_TEST_F(ArrayElementwiseOpTest, MaxF64s) {
  ComputationBuilder builder(client_, TestName());
#if !defined(XLA_TEST_BACKEND_CPU)
  auto lhs = builder.ConstantR1<double>({1.0, 1.0, 2.25});
  auto rhs = builder.ConstantR1<double>({2.0, -5.0, 1.0});
#else
  SetFastMathDisabled(true);
  auto lhs = builder.ConstantR1<double>({1.0, 1.0, 2.25, NAN, 6.0});
  auto rhs = builder.ConstantR1<double>({2.0, -5.0, 1.0, 10.0, NAN});
#endif
  auto maximum = builder.Max(lhs, rhs);

  ComputeAndCompareR1<double>(&builder,
#if !defined(XLA_TEST_BACKEND_CPU)
                              {2.0, 1.0, 2.25},
#else
                              {2.0, 1.0, 2.25, 10.0, 6.0},
#endif
                              {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxS32s) {
  const int32 min = std::numeric_limits<int32>::min();
  const int32 max = std::numeric_limits<int32>::max();
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<int32>(
      {min, min, min, -1, -1, 0, 0, 0, 1, 1, max, max, max});
  auto y = builder.ConstantR1<int32>(
      {min, max, 0, -10, 0, -1, 0, 1, 0, 10, 0, max, min});
  builder.Max(x, y);

  std::vector<int32> expected = {min, max, 0,  -1,  0,   0,  0,
                                 1,   1,   10, max, max, max};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MinS32s) {
  const int32 min = std::numeric_limits<int32>::min();
  const int32 max = std::numeric_limits<int32>::max();
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<int32>(
      {min, min, min, -1, -1, 0, 0, 0, 1, 1, max, max, max});
  auto y = builder.ConstantR1<int32>(
      {min, max, 0, -10, 0, -1, 0, 1, 0, 10, 0, max, min});
  builder.Min(x, y);

  std::vector<int32> expected = {min, min, min, -10, -1,  -1, 0,
                                 0,   0,   1,   0,   max, min};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxU32s) {
  const uint32 max = std::numeric_limits<uint32>::max();
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<uint32>({0, 0, 1, 1, 1, max, max, max});
  auto y = builder.ConstantR1<uint32>({0, 1, 0, 1, 10, 0, 234234, max});
  builder.Max(x, y);

  std::vector<uint32> expected = {0, 1, 1, 1, 10, max, max, max};
  ComputeAndCompareR1<uint32>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MinU32s) {
  const uint32 max = std::numeric_limits<uint32>::max();
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<uint32>({0, 0, 1, 1, 1, max, max, max});
  auto y = builder.ConstantR1<uint32>({0, 1, 0, 1, 10, 0, 234234, max});
  builder.Min(x, y);

  std::vector<uint32> expected = {0, 0, 0, 1, 1, 0, 234234, max};
  ComputeAndCompareR1<uint32>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxTenF32s) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<float>(
      {-0.0, 1.0, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0, 9.0});
  auto y = builder.ConstantR1<float>(
      {-0.0, -1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0, -9.0});
  builder.Max(x, y);

  std::vector<float> expected = {-0.0, 1.0, 2.0, 3.0, 4.0,
                                 5.0,  6.0, 7.0, 8.0, 9.0};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxR1S1AndR1S0F32s) {
  ComputationBuilder builder(client_, TestName());
  auto u = builder.ConstantR1<float>({3.5});
  auto v = builder.ConstantR1<float>({});
  builder.Max(u, v);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxR1S0AndR2S0x2F32s) {
  for (int broadcast_dim : {0, 1}) {
    ComputationBuilder builder(client_, TestName());
    auto u = builder.ConstantR1<float>({3.5});
    auto v = builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 2));
    builder.Max(u, v, /*broadcast_dimensions=*/{broadcast_dim});

    ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 2), {}, error_spec_);
  }
}

XLA_TEST_F(ArrayElementwiseOpTest, Max1DAnd2DF32s) {
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<float>({2.0f, 3.0f, 4.0f});
  auto m =
      builder.ConstantR2<float>({{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  builder.Max(v, m, /*broadcast_dimensions=*/{1});

  Array2D<float> expected({{2.0f, 3.14f, 4.0f}, {2.25f, 3.0f, 4.0f}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Max1DAnd2DZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<float>({});
  auto m = builder.ConstantR2<float>({{}, {}});
  builder.Max(v, m, /*broadcast_dimensions=*/{1});

  Array2D<float> expected({{}, {}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Max3DAndScalarS32s) {
  ComputationBuilder builder(client_, TestName());
  auto scalar = builder.ConstantR0<int32>(2);
  Array3D<int32> a_3d({{{3, 9, -1}, {2, -10, 3}}, {{-2, 2, 8}, {12, 10, 4}}});
  auto array = builder.ConstantR3FromArray3D<int32>(a_3d);
  builder.Max(array, scalar, /*broadcast_dimensions=*/{});

  Array3D<int32> expected({{{3, 9, 2}, {2, 2, 3}}, {{2, 2, 8}, {12, 10, 4}}});
  ComputeAndCompareR3<int32>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, Max3DAndScalarZeroElementS32s) {
  ComputationBuilder builder(client_, TestName());
  auto scalar = builder.ConstantR0<int32>(2);
  Array3D<int32> a_3d(2, 0, 3);
  auto array = builder.ConstantR3FromArray3D<int32>(a_3d);
  builder.Max(array, scalar, /*broadcast_dimensions=*/{});

  Array3D<int32> expected(2, 0, 3);
  ComputeAndCompareR3<int32>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, Min2DTo1DF32s) {
  ComputationBuilder builder(client_, TestName());
  auto m =
      builder.ConstantR2<float>({{-10.4f, 64.0f, 6.0f}, {0.1f, 32.0f, 16.1f}});
  auto v = builder.ConstantR1<float>({-10.2f, 16.4f});
  builder.Min(m, v, /*broadcast_dimensions=*/{0});

  Array2D<float> expected({{-10.4f, -10.2f, -10.2f}, {0.1f, 16.4f, 16.1f}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Min2DTo1DZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto m = builder.ConstantR2<float>({{}, {}});
  auto v = builder.ConstantR1<float>({-10.2f, 16.4f});
  builder.Min(m, v, /*broadcast_dimensions=*/{0});

  Array2D<float> expected({{}, {}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Min2DTo4DF32s) {
  ComputationBuilder builder(client_, TestName());
  auto array2d =
      builder.ConstantR2<float>({{-12.2f, 64.3f, 6.1f}, {0.0f, 32.2f, 2.5f}});
  auto array4d = builder.ConstantR4FromArray4D<float>(
      {{{{-12.1f, 32.3f, 6.2f}}, {{0.0f, 32.5f, 3.0f}}},
       {{{-2.5f, 64.29f, 6.5f}}, {{-0.01f, 32.25f, 2.6f}}}});
  builder.Min(array2d, array4d, /*broadcast_dimensions=*/{1, 3});

  Array4D<float> expected(
      {{{{-12.2f, 32.3f, 6.1f}}, {{0.0f, 32.2f, 2.5f}}},
       {{{-12.2f, 64.29f, 6.1f}}, {{-0.01f, 32.2f, 2.5f}}}});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Min2DTo4DZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());
  auto array2d =
      builder.ConstantR2<float>({{-12.2f, 64.3f, 6.1f}, {0.0f, 32.2f, 2.5f}});
  Array4D<float> arg(2, 2, 0, 3);
  auto array4d = builder.ConstantR4FromArray4D<float>(arg);
  builder.Min(array2d, array4d, /*broadcast_dimensions=*/{1, 3});

  Array4D<float> expected(2, 2, 0, 3);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MinTenS32s) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<int32>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto y = builder.ConstantR1<int32>({9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
  builder.Min(x, y);

  std::vector<int32> expected = {0, 1, 2, 3, 4, 4, 3, 2, 1, 0};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxTenS32s) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<int32>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto y = builder.ConstantR1<int32>({9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
  builder.Max(x, y);

  std::vector<int32> expected = {9, 8, 7, 6, 5, 5, 6, 7, 8, 9};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, RemTwoConstantS32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({-3, 26, 2, -1, 1});
  auto b = builder.ConstantR1<int32>({10, 5, 1, 10, -10});
  auto add = builder.Rem(a, b);

  ComputeAndCompareR1<int32>(&builder, {-3, 1, 0, -1, 1}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NonNanClampF32) {
  ComputationBuilder builder(client_, TestName());
  auto minimum = builder.ConstantR1<float>({1.0f, -6.5f, 1.0f, 2.25f, 0.0f});
  auto argument = builder.ConstantR1<float>({2.0f, 10.0f, -5.0f, 1.0f, 10.0f});
  auto maximum = builder.ConstantR1<float>({3.0f, 0.5f, 25.5f, 5.0f, 123.0});
  auto clamp = builder.Clamp(minimum, argument, maximum);

  ComputeAndCompareR1<float>(&builder, {2.0f, 0.5f, 1.0f, 2.25f, 10.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, ClampF32Scalar) {
  ComputationBuilder builder(client_, TestName());
  auto minimum = builder.ConstantR0<float>(0.0f);
  auto argument = builder.ConstantR1<float>({2.0f, 10.0f, -5.0f, 1.0f, 4.0f});
  auto maximum = builder.ConstantR0<float>(5.0f);
  auto clamp = builder.Clamp(minimum, argument, maximum);

  ComputeAndCompareR1<float>(&builder, {2.0f, 5.0f, 0.0f, 1.0f, 4.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, ClampF32ScalarVector) {
  ComputationBuilder builder(client_, TestName());
  auto min_scalar = builder.ConstantR0<float>(0.0f);
  auto min_vector = builder.ConstantR1<float>({1.0f, -6.5f, 1.0f, 2.25f, 0.0f});
  auto arg_vector = builder.ConstantR1<float>({2.0f, 10.0f, -5.0f, 1.0f, 4.0f});
  auto arg_scalar = builder.ConstantR1<float>({2.0f, 10.0f, -5.0f, 1.0f, 4.0f});
  auto max_scalar = builder.ConstantR0<float>(3.0f);
  auto max_vector = builder.ConstantR1<float>({3.0f, 0.5f, 25.5f, 5.0f, 123.0});
  // Perform clamp with broadcasted scalar and vector.
  auto clamp = builder.Add(
      builder.Add(builder.Clamp(min_vector, arg_vector, max_scalar),
                  builder.Clamp(min_scalar, arg_vector, max_vector)),
      builder.Add(builder.Clamp(min_vector, arg_scalar, max_vector),
                  builder.Clamp(min_scalar, arg_scalar, max_vector)));

  ComputeAndCompareR1<float>(&builder, {8.0f, 4.5f, 2.0f, 6.5f, 15.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoParametersF32s) {
  ComputationBuilder builder(client_, TestName());

  std::unique_ptr<Literal> param0_literal =
      Literal::CreateR1<float>({1.1f, 2.2f, 3.3f, 5.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  std::unique_ptr<Literal> param1_literal =
      Literal::CreateR1<float>({7.2f, 2.3f, 3.4f, 5.6f});
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(*param1_literal).ConsumeValueOrDie();

  auto p0 = builder.Parameter(0, param0_literal->shape(), "param0");
  auto p1 = builder.Parameter(1, param1_literal->shape(), "param1");
  auto add = builder.Add(p0, p1);

  ComputeAndCompareR1<float>(&builder, {8.3f, 4.5f, 6.7f, 11.1f},
                             {param0_data.get(), param1_data.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoParametersZeroElementF32s) {
  ComputationBuilder builder(client_, TestName());

  std::unique_ptr<Literal> param0_literal =
      Literal::CreateR3FromArray3D<float>(Array3D<float>(0, 7, 0));
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  std::unique_ptr<Literal> param1_literal =
      Literal::CreateR3FromArray3D<float>(Array3D<float>(0, 7, 0));
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(*param1_literal).ConsumeValueOrDie();

  auto p0 = builder.Parameter(0, param0_literal->shape(), "param0");
  auto p1 = builder.Parameter(1, param1_literal->shape(), "param1");
  auto add = builder.Add(p0, p1);

  Array3D<float> expected(0, 7, 0);
  ComputeAndCompareR3<float>(
      &builder, expected, {param0_data.get(), param1_data.get()}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddParameterToConstantF32s) {
  ComputationBuilder builder(client_, TestName());

  std::unique_ptr<Literal> param0_literal =
      Literal::CreateR1<float>({1.1f, 2.2f, 3.3f, 5.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  auto a = builder.ConstantR1<float>({1.1f, 2.2f, 3.3f, 4.4f});
  auto p = builder.Parameter(0, param0_literal->shape(), "param0");
  auto add = builder.Add(a, p);

  ComputeAndCompareR1<float>(&builder, {2.2f, 4.4f, 6.6f, 9.9f},
                             {param0_data.get()}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, CosF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({3.14159f, 0.0f, 1.570796f, -0.78539f});
  auto result = builder.Cos(a);

  ComputeAndCompareR1<float>(&builder, {-1.0f, 1.0f, 0.0f, 0.707107f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SinF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({3.14159f, 0.0f, 1.570796f, -0.78539f});
  auto result = builder.Sin(a);

  ComputeAndCompareR1<float>(&builder, {0.0f, 0.0f, 1.0f, -0.707107f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, TanhF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({-2.5f, 3.14f, 2.25f});
  auto result = builder.Tanh(a);

  ComputeAndCompareR1<float>(&builder, {-0.986614f, 0.996260f, 0.978026}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, TanhF32sVector) {
  // This is like the test ArrayElementwiseOpTest.TanhF32s above, except that
  // the input tensor is large enough to exercise the vectorized tanh
  // implementation.
  ComputationBuilder builder(client_, TestName());
  auto input_literal = Literal::CreateR2<float>(
      {{1.02, -0.32, 0.85, 0.90, 1.23, -0.91, -0.49, 0.80},
       {-0.67, 0.16, -0.07, 0.39, -0.41, 0.04, 1.36, 1.25},
       {0.41, 0.65, -1.08, 0.32, -1.45, -0.77, -1.09, 0.91},
       {-1.03, -0.30, -1.11, -1.17, 1.50, -0.85, 0.04, 1.02},
       {0.34, -0.61, 0.41, 0.07, -0.02, 1.42, -0.62, 0.81},
       {0.08, 0.81, -0.30, 1.17, -0.65, -0.44, 0.92, 1.26},
       {-1.29, 1.35, 0.08, -1.24, -0.92, 0.49, 1.17, -0.45},
       {-1.31, -1.44, -0.13, -1.31, -0.79, 1.41, 1.21, 1.05}});
  auto input_data =
      client_->TransferToServer(*input_literal).ConsumeValueOrDie();

  auto input = builder.Parameter(0, input_literal->shape(), "input");
  builder.Tanh(input);

  ComputeAndCompareR2<float>(
      &builder,
      {{0.77009583, -0.30665702, 0.69070244, 0.71401149, 0.84400684,
        -0.71985596, -0.45764771, 0.66664988},
       {-0.58278900, 0.16050975, -0.06770509, 0.36843640, -0.38476998,
        0.04018109, 0.87562293, 0.84788644},
       {0.38603750, 0.57294142, -0.79140943, 0.31032649, -0.89590985,
        -0.64770776, -0.79625875, 0.72234446},
       {-0.77389336, -0.28871772, -0.80428445, -0.82541436, 0.90456349,
        -0.68856895, 0.03877772, 0.76877952},
       {0.32561871, -0.54546672, 0.39072621, 0.07273290, -0.01924866,
        0.88924897, -0.55283129, 0.67183107},
       {0.08006320, 0.66944766, -0.29068485, 0.82573754, -0.57170743,
        -0.41581789, 0.72739530, 0.85025692},
       {-0.85931867, 0.87357593, 0.07782833, -0.84597743, -0.72748238,
        0.45396307, 0.82449573, -0.42462519},
       {-0.86363792, -0.89368379, -0.12621804, -0.86445558, -0.65565848,
        0.88789743, 0.83566397, 0.78287679}},
      {input_data.get()},
      // The error spec is unusually high here to account for the fact that we
      // use a rational interpolant to approximate tanh.
      ErrorSpec(0.004, 0.004));
}

XLA_TEST_F(ArrayElementwiseOpTest, AddChainFoldLeft) {
  // a ------ (add) --------- (add)
  //         /               /
  // b -----/               /
  // c---------------------/
  ComputationBuilder builder(client_, TestName());

  auto a = builder.ConstantR1<float>({1.1f, 2.2f, 3.3f, 4.4f});
  auto b = builder.ConstantR1<float>({2.1f, 3.2f, 4.3f, 5.4f});
  auto c = builder.ConstantR1<float>({-3.3f, -15.5f, -7.7f, -29.9f});

  auto add = builder.Add(a, b);
  auto add2 = builder.Add(add, c);

  ComputeAndCompareR1<float>(&builder, {-0.1f, -10.1f, -0.1f, -20.1f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddChainFoldRight) {
  // b ------ (add) --------- (add)
  //         /               /
  // c -----/               /
  // a---------------------/
  ComputationBuilder builder(client_, TestName());

  auto a = builder.ConstantR1<float>({91.1f, 2.2f, 3.3f, 4.4f});
  auto b = builder.ConstantR1<float>({2.1f, 3.2f, 4.3f, 5.4f});
  auto c = builder.ConstantR1<float>({-3.3f, -15.5f, -7.7f, -29.9f});

  auto add = builder.Add(b, c);
  auto add2 = builder.Add(a, add);

  ComputeAndCompareR1<float>(&builder, {89.9f, -10.1f, -0.1f, -20.1f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddWithNeg) {
  // a ----- (neg) ----- (add)
  //                    /
  // b ----- (neg) ----/
  ComputationBuilder builder(client_, TestName());

  auto a = builder.ConstantR1<float>({91.1f, 2.2f, 3.3f, 4.4f});
  auto b = builder.ConstantR1<float>({2.1f, 3.2f, 4.3f, 5.4f});

  auto neg_a = builder.Neg(a);
  auto neg_b = builder.Neg(b);
  auto result = builder.Add(neg_a, neg_b);

  ComputeAndCompareR1<float>(&builder, {-93.2f, -5.4f, -7.6f, -9.8f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddChainTwoSide) {
  // a ------ (add) ------------\
  //         /                   \
  // b -----/                    (add)
  //                             /
  // c ------ (add) ------------/
  //         /
  // d -----/
  ComputationBuilder builder(client_, TestName());

  auto a = builder.ConstantR1<float>({91.1f, 2.2f, 3.3f, 4.4f});
  auto b = builder.ConstantR1<float>({2.1f, 3.2f, 4.3f, 5.4f});
  auto c = builder.ConstantR1<float>({-3.3f, -15.5f, -7.7f, -29.9f});
  auto d = builder.ConstantR1<float>({-19.0f, 10.0f, -40.0f, 20.2f});

  auto add_ab = builder.Add(a, b);
  auto add_cd = builder.Add(c, d);
  auto add_all = builder.Add(add_ab, add_cd);

  ComputeAndCompareR1<float>(&builder, {70.9f, -0.1f, -40.1f, 0.1f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, 2DBinaryOpF32s) {
  ComputationBuilder builder(client_, TestName());
  auto a =
      builder.ConstantR2<float>({{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto b =
      builder.ConstantR2<float>({{-1.5f, 8.14f, 42.0}, {-1.0f, -4.0f, 5.55f}});
  auto add = builder.Add(a, b);

  Array2D<float> expected_array(
      {{-4.0f, 11.28f, 43.0f}, {1.25f, -14.0f, 8.88f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, ScalarPlus2DF32) {
  // Add a scalar + matrix.
  ComputationBuilder builder(client_, TestName());
  auto a =
      builder.ConstantR2<float>({{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto scalar = builder.ConstantR0<float>(3.0f);
  auto add = builder.Add(scalar, a);

  Array2D<float> expected_array({{0.5f, 6.14f, 4.0f}, {5.25f, -7.0f, 6.33f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, 2DPlusScalarF32) {
  // Add a matrix + scalar.
  ComputationBuilder builder(client_, TestName());
  auto a =
      builder.ConstantR2<float>({{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto scalar = builder.ConstantR0<float>(3.0f);
  auto add = builder.Add(a, scalar);

  Array2D<float> expected_array({{0.5f, 6.14f, 4.0f}, {5.25f, -7.0f, 6.33f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add1DTo2DF32) {
  // Test simple broadcasting of a R1F32 over R2F32. The vector's size matches
  // only dim 0 of the matrix.
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<float>({20.0f, 40.0f, 60.0f});
  // clang-format off
  auto m = builder.ConstantR2<float>({
    {-2.5f, 3.14f, 1.0f},
    {2.25f, -10.0f, 3.33f}});
  // clang-format on
  auto add = builder.Add(v, m, /*broadcast_dimensions=*/{1});
  Array2D<float> expected_array(
      {{17.5f, 43.14f, 61.0f}, {22.25f, 30.0f, 63.33f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Eq) {
  // Test broadcasting in Eq comparison.
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<int32>({42, 73});
  auto m = builder.ConstantR2<int32>({{42, 73}, {42, 52}});

  // This test exercises both possible broadcast dimensions for a vector/matrix
  // comparison.
  auto cmp_dim_0 = builder.Eq(v, m, /*broadcast_dimensions=*/{1});
  auto cmp_dim_1 = builder.Eq(v, m, /*broadcast_dimensions=*/{0});
  auto result = builder.Tuple({cmp_dim_0, cmp_dim_1});

  auto expected = Literal::MakeTuple(
      {Literal::CreateR2<bool>({{true, true}, {true, false}}).get(),
       Literal::CreateR2<bool>({{true, false}, {false, false}}).get()});
  ComputeAndCompareTuple(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Ne) {
  // Test broadcasting in Ne comparison.
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<int32>({42, 73});
  auto m = builder.ConstantR2<int32>({{42, 73}, {42, 52}});
  auto cmp = builder.Ne(v, m, /*broadcast_dimensions=*/{1});

  const string expected = R"(pred[2,2] {
  { 00 },
  { 01 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Ge) {
  // Test broadcasting in Ge comparison.
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<int32>({1, 2, 3, 4});
  auto m = builder.ConstantR2<int32>({{1, 0, 5, 6}, {42, 52, 10, 4}});
  auto cmp = builder.Ge(v, m, /*broadcast_dimensions=*/{1});

  const string expected = R"(pred[2,4] {
  { 1100 },
  { 0001 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Gt) {
  // Test broadcasting in Gt comparison.
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<int32>({1, 2, 3, 4});
  auto m = builder.ConstantR2<int32>({{1, 0, 5, 6}, {42, 52, 10, 4}});
  auto cmp = builder.Gt(v, m, /*broadcast_dimensions=*/{1});

  const string expected = R"(pred[2,4] {
  { 0100 },
  { 0000 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Le) {
  // Test broadcasting in Le comparison.
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<int32>({1, 2, 3, 4});
  auto m = builder.ConstantR2<int32>({{1, 0, 5, 6}, {42, 52, 10, 4}});
  auto cmp = builder.Le(v, m, /*broadcast_dimensions=*/{1});

  const string expected = R"(pred[2,4] {
  { 1011 },
  { 1111 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Lt) {
  // Test broadcasting in Lt comparison.
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<int32>({1, 2, 3, 4});
  auto m = builder.ConstantR2<int32>({{1, 0, 5, 6}, {42, 52, 10, 4}});
  auto cmp = builder.Lt(v, m, /*broadcast_dimensions=*/{1});

  const string expected = R"(pred[2,4] {
  { 0011 },
  { 1110 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, Mul2Dby1DF32) {
  // Test simple broadcasting of a R1F32 over R2F32 when the order of binary op
  // arguments is reversed.
  ComputationBuilder builder(client_, TestName());
  auto m = builder.ConstantR2<float>({{1.5f, 2.5f, 3.5f}, {4.5f, 5.5f, 6.5f}});
  auto v = builder.ConstantR1<float>({2.0f, 4.0f, 6.0f});
  auto add = builder.Mul(m, v, /*broadcast_dimensions=*/{1});
  Array2D<float> expected_array({{3.0f, 10.0f, 21.0f}, {9.0f, 22.0f, 39.0f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add2DTo2DWithDegenerateDim1) {
  // Tests broadcasting for arrays with degenerate (size == 1) dimensions.
  ComputationBuilder builder(client_, TestName());
  // m's shape in XLA notation is {3, 2}
  // md's shape in XLA notation is {3, 1}
  // The result has shape {3, 2}, where md is broadcast over m
  auto m =
      builder.ConstantR2<float>({{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto md = builder.ConstantR2<float>({{10.0f, 20.0f, 30.0f}});
  auto add = builder.Add(m, md);
  Array2D<float> expected_array(
      {{7.5f, 23.14f, 31.0f}, {12.25f, 10.0f, 33.33f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add2DTo2DWithDegenerateDim0) {
  // Tests broadcasting for arrays with degenerate (size == 1) dimensions.
  ComputationBuilder builder(client_, TestName());
  // m's shape in XLA notation is {3, 2}
  // md's shape in XLA notation is {1, 2}
  // The result has shape {3, 2}, where md is broadcast over m
  auto m =
      builder.ConstantR2<float>({{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto md = builder.ConstantR2<float>({{10.0f}, {20.0f}});
  auto add = builder.Add(m, md);
  Array2D<float> expected_array(
      {{7.5f, 13.14f, 11.0f}, {22.25f, 10.0f, 23.33f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add2DsWithDegenerateDimsOuterProduct) {
  // Tests broadcasting for two degenerate arrays. This kind of broadcasting
  // effectively creates an "outer product" operation.
  // This is taken from the Numpy docs example at:
  // http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html
  ComputationBuilder builder(client_, TestName());
  // a's shape in XLA notation is {1, 4}
  // b's shape in XLA notation is {3, 1}
  // The result has shape {3, 4}.
  auto a = builder.ConstantR2<float>({{0.0f}, {10.0f}, {20.0f}, {30.0f}});
  auto b = builder.ConstantR2<float>({{1.0f, 2.0f, 3.0f}});
  auto add = builder.Add(a, b);
  Array2D<float> expected_array({{1.0f, 2.0f, 3.0f},
                                 {11.0f, 12.0f, 13.0f},
                                 {21.0f, 22.0f, 23.0f},
                                 {31.0f, 32.0f, 33.0f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add1DTo2DF32TwoWaysOver1) {
  // Add together a (2,2) array and a (2) array, using dimension 0 for
  // broadcasting (though there are two ways to broadcast these shapes).
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<float>({20.0f, 40.0f});
  auto m = builder.ConstantR2<float>({{10.0f, 50.0f}, {77.0f, 88.0f}});
  auto add = builder.Add(v, m, /*broadcast_dimensions=*/{1});
  Array2D<float> expected_array({{30.0f, 90.0f}, {97.0f, 128.0f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add1DTo2DF32TwoWaysOver0) {
  // Add together a (2,2) array and a (2) array, using dimension 1 for
  // broadcasting (though there are two ways to broadcast these shapes).
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<float>({20.0f, 40.0f});
  auto m = builder.ConstantR2<float>({{10.0f, 50.0f}, {77.0f, 88.0f}});
  auto add = builder.Add(v, m, /*broadcast_dimensions=*/{0});
  Array2D<float> expected_array({{30.0f, 70.0f}, {117.0f, 128.0f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, 3DBinaryOpF32s) {
  // Binary add of two R3s together
  ComputationBuilder builder(client_, TestName());
  Array3D<float> a_3d({{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                       {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}});
  auto a = builder.ConstantR3FromArray3D<float>(a_3d);

  Array3D<float> b_3d({{{2.0f, 4.0f}, {6.0f, 8.0f}, {10.0f, 12.0f}},
                       {{14.0f, 16.0f}, {18.0f, 20.0f}, {22.0f, 24.0f}}});
  auto b = builder.ConstantR3FromArray3D<float>(b_3d);
  auto add = builder.Add(a, b);

  Array3D<float> expected_3d(
      {{{3.0f, 6.0f}, {9.0f, 12.0f}, {15.0f, 18.0f}},
       {{21.0f, 24.0f}, {27.0f, 30.0f}, {33.0f, 36.0f}}});
  ComputeAndCompareR3<float>(&builder, expected_3d, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add1DTo3DTwoWaysOver2) {
  // Add together a (2, 3, 2) array with a (2) array, using dimension 0 for
  // broadcasting (though there are two ways to broadcast these shapes).
  ComputationBuilder builder(client_, TestName());
  // clang-format off
  Array3D<float> a_3d({
    {{1.0f, 2.0f},
     {3.0f, 4.0f},
     {5.0f, 6.0f}},
    {{7.0f, 8.0f},
     {9.0f, 10.0f},
     {11.0f, 12.0f}},
  });
  // clang-format on
  auto a = builder.ConstantR3FromArray3D<float>(a_3d);
  auto v = builder.ConstantR1<float>({10.0f, 20.0f});
  auto add = builder.Add(a, v, /*broadcast_dimensions=*/{2});

  Array3D<float> expected_3d(
      {{{11.0f, 22.0f}, {13.0f, 24.0f}, {15.0f, 26.0f}},
       {{17.0f, 28.0f}, {19.0f, 30.0f}, {21.0f, 32.0f}}});
  ComputeAndCompareR3<float>(&builder, expected_3d, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add1DTo3DTwoWaysOver0) {
  // Add together a (2, 3, 2) array with a (2) array, using dimension 2 for
  // broadcasting (though there are two ways to broadcast these shapes).
  ComputationBuilder builder(client_, TestName());
  // clang-format off
  Array3D<float> a_3d({
    {{1.0f, 2.0f},
     {3.0f, 4.0f},
     {5.0f, 6.0f}},
    {{7.0f, 8.0f},
     {9.0f, 10.0f},
     {11.0f, 12.0f}},
  });
  // clang-format on
  auto a = builder.ConstantR3FromArray3D<float>(a_3d);
  auto v = builder.ConstantR1<float>({10.0f, 20.0f});
  auto add = builder.Add(a, v, /*broadcast_dimensions=*/{0});

  // clang-format off
  Array3D<float> expected_3d({
    {{11.0f, 12.0f},
     {13.0f, 14.0f},
     {15.0f, 16.0f}},
    {{27.0f, 28.0f},
     {29.0f, 30.0f},
     {31.0f, 32.0f}},
  });
  // clang-format on
  ComputeAndCompareR3<float>(&builder, expected_3d, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add2DTo3D) {
  // Add together a (2, 3, 2) array with a (3, 2) array, using dimensions {1,2}
  // for broadcasting.
  ComputationBuilder builder(client_, TestName());
  // clang-format off
  Array3D<float> a_3d({
    {{1.0f, 2.0f},
     {3.0f, 4.0f},
     {5.0f, 6.0f}},
    {{7.0f, 8.0f},
     {9.0f, 10.0f},
     {11.0f, 12.0f}},
  });
  auto a = builder.ConstantR3FromArray3D<float>(a_3d);
  auto m = builder.ConstantR2<float>({
    {10.0f, 20.0f, 30.0f},
    {40.0f, 50.0f, 60.0f},
  });
  auto add = builder.Add(a, m, /*broadcast_dimensions=*/{0, 1});

  Array3D<float> expected_3d({
    {{11.0f, 12.0f},
     {23.0f, 24.0f},
     {35.0f, 36.0f}},
    {{47.0f, 48.0f},
     {59.0f, 60.0f},
     {71.0f, 72.0f}},
  });
  // clang-format on
  ComputeAndCompareR3<float>(&builder, expected_3d, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGtR3F32sWithDegenerateDim2) {
  // Comparison between two 3D arrays of compatible shapes:
  // (2, 3, 2) and (2, 3, 1): expected to produce a (2, 3, 2) shape of PREDs.
  ComputationBuilder builder(client_, TestName());
  Array3D<float> a_3d({{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                       {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}});
  auto a = builder.ConstantR3FromArray3D<float>(a_3d);

  Array3D<float> b_3d({{{7.0f, 1.0f}, {3.0f, 10.0f}, {15.0f, 6.0f}}});
  auto b = builder.ConstantR3FromArray3D<float>(b_3d);

  auto compare = builder.Gt(a, b);

  Array3D<int> expected_3d(
      {{{0, 1}, {0, 0}, {0, 0}}, {{0, 1}, {1, 0}, {0, 1}}});
  const string expected = R"(pred[2,3,2] {
{ { 01 },
  { 00 },
  { 00 } },
{ { 01 },
  { 10 },
  { 01 } }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, 4DBinaryOpF32s) {
  ComputationBuilder builder(client_, TestName());

  std::unique_ptr<Array4D<float>> operand_a_4d(new Array4D<float>(2, 3, 4, 5));
  std::unique_ptr<Array4D<float>> operand_b_4d(new Array4D<float>(2, 3, 4, 5));
  std::unique_ptr<Array4D<float>> expected_4d(new Array4D<float>(2, 3, 4, 5));
  float value = 0.0;
  for (int64 p = 0; p < 2; ++p) {
    for (int64 z = 0; z < 3; ++z) {
      for (int64 y = 0; y < 4; ++y) {
        for (int64 x = 0; x < 5; ++x) {
          (*operand_a_4d)(p, z, y, x) = value;
          (*operand_b_4d)(p, z, y, x) = 2.0 * value;
          (*expected_4d)(p, z, y, x) = 3.0 * value;
          value += 0.1;
        }
      }
    }
  }

  auto a = builder.ConstantR4FromArray4D<float>(*operand_a_4d);
  auto b = builder.ConstantR4FromArray4D<float>(*operand_b_4d);
  auto add = builder.Add(a, b);

  ComputeAndCompareR4<float>(&builder, *expected_4d, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, R4PlusR1InDim1) {
  ComputationBuilder builder(client_, TestName());

  std::unique_ptr<Array4D<float>> operand_a_4d(new Array4D<float>(2, 3, 4, 5));
  std::unique_ptr<Array4D<float>> expected_4d(new Array4D<float>(2, 3, 4, 5));
  std::vector<float> operand_b_1d(3);
  std::iota(operand_b_1d.begin(), operand_b_1d.end(), 1.0);

  float value = 0.0;
  for (int64 p = 0; p < 2; ++p) {
    for (int64 z = 0; z < 3; ++z) {
      for (int64 y = 0; y < 4; ++y) {
        for (int64 x = 0; x < 5; ++x) {
          (*operand_a_4d)(p, z, y, x) = value;
          (*expected_4d)(p, z, y, x) = value + operand_b_1d[z];
          value += 0.1;
        }
      }
    }
  }

  auto a = builder.ConstantR4FromArray4D<float>(*operand_a_4d);
  auto b = builder.ConstantR1<float>(operand_b_1d);
  auto add = builder.Add(a, b, {1});

  ComputeAndCompareR4<float>(&builder, *expected_4d, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, R4_16x16x2x2_Plus_R1_16) {
  constexpr int d0 = 16;
  constexpr int d1 = 16;
  constexpr int d2 = 2;
  constexpr int d3 = 2;
  Array4D<float> r4(d0, d1, d2, d3);
  r4.Fill(1.0);
  std::vector<float> r1(d1);
  std::iota(r1.begin(), r1.end(), 1.0);

  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> a_literal = Literal::CreateR4FromArray4D(r4);
  *a_literal->mutable_shape()->mutable_layout() =
      LayoutUtil::MakeLayout({0, 1, 2, 3});
  auto a = builder.ConstantLiteral(*a_literal);
  auto b = builder.ConstantR1<float>(r1);
  builder.Add(a, b, {1});

  for (int i0 = 0; i0 < d0; ++i0) {
    for (int i1 = 0; i1 < d1; ++i1) {
      for (int i2 = 0; i2 < d2; ++i2) {
        for (int i3 = 0; i3 < d3; ++i3) {
          r4(i0, i1, i2, i3) += r1[i1];
        }
      }
    }
  }
  ComputeAndCompareR4<float>(&builder, r4, {}, error_spec_);
}

// Show that we can't add two opaques.
XLA_TEST_F(ArrayElementwiseOpTest, CannotAddOpaques) {
  ComputationBuilder builder(client_, TestName());
  auto shape = ShapeUtil::MakeOpaqueShape();
  auto x = builder.Parameter(0, shape, "x");
  auto concatenated = builder.Add(x, x);
  StatusOr<Computation> computation_status = builder.Build();
  ASSERT_FALSE(computation_status.ok());
  EXPECT_THAT(computation_status.status().ToString(),
              ::testing::ContainsRegex(
                  "Expected non-opaque argument for lhs of binary operation"));
}

XLA_TEST_F(ArrayElementwiseOpTest, IdentityBroadcastOfSameRankIsAllowed) {
  ComputationBuilder builder(client_, TestName());
  auto a =
      builder.ConstantR2<float>({{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto b =
      builder.ConstantR2<float>({{-1.5f, 8.14f, 42.0}, {-1.0f, -4.0f, 5.55f}});
  auto add = builder.Add(a, b, /*broadcast_dimensions=*/{0, 1});

  Array2D<float> expected_array(
      {{-4.0f, 11.28f, 43.0f}, {1.25f, -14.0f, 8.88f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, NonIdentityBroadcastOfSameRankIsDisallowed) {
  ComputationBuilder builder(client_, TestName());
  auto a =
      builder.ConstantR2<float>({{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto b =
      builder.ConstantR2<float>({{-1.5f, 8.14f, 42.0}, {-1.0f, -4.0f, 5.55f}});
  auto add = builder.Add(a, b, /*broadcast_dimensions=*/{1, 0});

  StatusOr<Computation> computation_status = builder.Build();
  ASSERT_FALSE(computation_status.ok());
  EXPECT_THAT(computation_status.status().error_message(),
              ::testing::ContainsRegex("must.*be the identity"));
}

// Regression test for b/31927799. "slice - y" is fused and requires implicit
// broadcast.
XLA_TEST_F(ArrayElementwiseOpTest, ImplictBroadcastInFusedExpressions) {
  ComputationBuilder builder(client_, TestName());
  auto x_literal = Literal::CreateR1<float>({1, 2, 3});
  auto y_literal = Literal::CreateR1<float>({4, 5});
  auto x_data = client_->TransferToServer(*x_literal).ConsumeValueOrDie();
  auto y_data = client_->TransferToServer(*y_literal).ConsumeValueOrDie();

  auto x = builder.Parameter(0, x_literal->shape(), "x");
  auto y = builder.Parameter(1, y_literal->shape(), "y");
  auto slice = builder.Slice(x, {1}, {2}, {1});
  builder.Sub(slice, y);

  ComputeAndCompareR1<float>(&builder, {-2, -3}, {x_data.get(), y_data.get()},
                             error_spec_);
}

INSTANTIATE_TEST_CASE_P(ArrayElementwiseOpTestParamCount,
                        ArrayElementwiseOpTestParamCount,
                        ::testing::Values(127, 128, 129, 17 * 4096));

}  // namespace
}  // namespace xla
