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

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

template <typename T>
std::vector<T> AppendSpecialPositiveValues(std::vector<T> values) {
  const std::vector<T> kSpecialValues = {0,
                                         1,
                                         std::numeric_limits<T>::min(),
                                         std::numeric_limits<T>::epsilon(),
                                         std::numeric_limits<T>::max(),
                                         std::numeric_limits<T>::infinity(),
                                         std::numeric_limits<T>::quiet_NaN()};
  values.insert(values.end(), kSpecialValues.begin(), kSpecialValues.end());
  // Remove duplicate values.
  auto last = std::unique(values.begin(), values.end());
  values.erase(last, values.end());
  return values;
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> AllSignedPairs(
    absl::Span<const T> abs_vals) {
  std::vector<T> ys;
  std::vector<T> xs;
  const size_t n = 4 * abs_vals.size() * abs_vals.size();
  ys.reserve(n);
  xs.reserve(n);
  for (auto abs_y : abs_vals) {
    for (auto y : {-abs_y, abs_y}) {
      for (auto abs_x : abs_vals) {
        for (auto x : {-abs_x, abs_x}) {
          ys.push_back(y);
          xs.push_back(x);
        }
      }
    }
  }
  return {xs, ys};
}

class ArrayElementwiseOpTest : public ClientLibraryTestBase {
 public:
  static constexpr float kEpsF32 = std::numeric_limits<float>::epsilon();
  static constexpr double kEpsF64 = std::numeric_limits<double>::epsilon();
  ErrorSpec error_spec_{60 * kEpsF32, 60 * kEpsF32};
  ErrorSpec strict_error_spec_{100 * kEpsF64, 100 * kEpsF64};
};

class ArrayElementwiseOpTestParamCount
    : public ArrayElementwiseOpTest,
      public ::testing::WithParamInterface<int> {};

XLA_TEST_F(ArrayElementwiseOpTest, NegConstantZeroElementF32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {});
  Neg(a);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, NegConstantF32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {-2.5f, 3.14f, 2.25f, -10.0f, 6.0f});
  Neg(a);

  ComputeAndCompareR1<float>(&builder, {2.5f, -3.14f, -2.25f, 10.0f, -6.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, NegConstantF64) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<double>(&builder, {-2.5, 3.14, 2.25, -10.0, 6.0});
  Neg(a);

  ComputeAndCompare(&builder, {}, strict_error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, NegConstantS32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(
      &builder, {-1, 0, 1, 324, std::numeric_limits<int32_t>::min(),
                 std::numeric_limits<int32_t>::max()});
  Neg(a);

  // -min == min for int32_t due to an overflow. In C++ it is undefined behavior
  // to do this calculation. For XLA we have not specified that, so it
  // ought to work.
  ComputeAndCompareR1<int32_t>(
      &builder,
      {1, 0, -1, -324, std::numeric_limits<int32_t>::min(),
       -std::numeric_limits<int32_t>::max()},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NegConstantZeroElementC64) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<complex64>(&builder, {});
  Neg(a);

  ComputeAndCompareR1<complex64>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, NegConstantC64) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<complex64>(
      &builder, {{-2.5f, 1.0f}, {0.0f, 3.14f}, {2.25f, -1.0f}, {-10.0f, 0.0f}});
  Neg(a);

  ComputeAndCompareR1<complex64>(
      &builder, {{2.5f, -1.0f}, {0.0f, -3.14f}, {-2.25f, 1.0f}, {10.0f, 0.0f}},
      {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, NegConstantS64) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int64_t>(&builder,
                               {
                                   -1,
                                   1,
                                   0,
                                   0x12345678,
                                   static_cast<int64_t>(0xffffffff12345678l),
                                   static_cast<int64_t>(0x8000000000000000LL),
                                   static_cast<int64_t>(0x8000000000000001LL),
                               });
  Neg(a);
  LOG(INFO) << -static_cast<int64_t>(0x7FFFFFFFFFFFFFFFLL);

  ComputeAndCompareR1<int64_t>(&builder,
                               {
                                   1,
                                   -1,
                                   0,
                                   -0x12345678,
                                   0xedcba988,
                                   static_cast<int64_t>(0x8000000000000000LL),
                                   -static_cast<int64_t>(0x8000000000000001LL),
                               },
                               {});
}

XLA_TEST_F(ArrayElementwiseOpTest, IsFiniteZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {});
  IsFinite(a);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, IntPow) {
  XlaBuilder builder(TestName());
  XlaOp lhs =
      ConstantR1<int32_t>(&builder, {0, 1, 2, 3, 4, 5, -1, -2, 3, 5, 3, 1});
  XlaOp rhs =
      ConstantR1<int32_t>(&builder, {0, 3, 3, 3, 3, 3, 2, 3, 2, 10, -100, -2});
  Pow(lhs, rhs);

  std::vector<int32_t> expected = {1, 1,  8, 27,      64, 125,
                                   1, -8, 9, 9765625, 0,  1};

  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, IntPowLarge) {
  XlaBuilder builder(TestName());
  XlaOp lhs = ConstantR1<int64_t>(&builder, {2});
  XlaOp rhs = ConstantR1<int64_t>(&builder, {62});
  Pow(lhs, rhs);

  std::vector<int64_t> expected = {4611686018427387904};

  ComputeAndCompareR1<int64_t>(&builder, expected, {});
}

// A non-canonical quiet NaN value.
static const float kNonCanonicalNaN = absl::bit_cast<float>(0x7FD01234);

XLA_TEST_F(ArrayElementwiseOpTest, IsFiniteScalarF32) {
  XlaBuilder builder(TestName());
  IsFinite(ConstantR0<float>(&builder, NAN));
  ComputeAndCompareR0<bool>(&builder, false, {});

  EXPECT_TRUE(std::isnan(kNonCanonicalNaN));
  IsFinite(ConstantR0<float>(&builder, kNonCanonicalNaN));
  ComputeAndCompareR0<bool>(&builder, false, {});

  const float kInf = std::numeric_limits<float>::infinity();
  IsFinite(ConstantR0<float>(&builder, kInf));
  ComputeAndCompareR0<bool>(&builder, false, {});

  IsFinite(ConstantR0<float>(&builder, -kInf));
  ComputeAndCompareR0<bool>(&builder, false, {});

  IsFinite(ConstantR0<float>(&builder, 0.0f));
  ComputeAndCompareR0<bool>(&builder, true, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, IsFiniteR1F32s) {
  XlaBuilder builder(TestName());
  const float kInf = std::numeric_limits<float>::infinity();
  EXPECT_TRUE(std::isnan(kNonCanonicalNaN));
  auto a = ConstantR1<float>(
      &builder, {{NAN, 7.0f, kNonCanonicalNaN, -1.0f, kInf, -kInf}});
  IsFinite(a);

  ComputeAndCompareR1<bool>(&builder, {false, true, false, true, false, false},
                            {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoConstantF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {-2.5f, 3.14f, 2.25f, -10.0f, 6.0f});
  auto b = ConstantR1<float>(&builder, {100.0f, 3.13f, 2.75f, 10.5f, -999.0f});
  Add(a, b);

  ComputeAndCompareR1<float>(&builder, {97.5f, 6.27f, 5.0f, 0.5f, -993.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoConstantZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {});
  auto b = ConstantR1<float>(&builder, {});
  Add(a, b);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoConstantC64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<complex64>(
      &builder, {{-2.5f, 0.0f}, {0.0f, 3.14f}, {2.25f, 0.0f}, {1.0f, -10.0f}});
  auto b = ConstantR1<complex64>(
      &builder, {{100.0f, 0.0f}, {3.13f, 0.0f}, {2.75f, 1.0f}, {-2.0f, 10.5f}});
  Add(a, b);

  ComputeAndCompareR1<complex64>(
      &builder, {97.5f, {3.13f, 3.14f}, {5.0f, 1.0f}, {-1.0f, 0.5f}}, {},
      error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoConstantZeroElementC64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<complex64>(&builder, {});
  auto b = ConstantR1<complex64>(&builder, {});
  Add(a, b);

  ComputeAndCompareR1<complex64>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoConstantU64s) {
  XlaBuilder b(TestName());

  std::vector<uint64_t> lhs{0xFFFFFFFF,
                            static_cast<uint64_t>(-1),
                            0,
                            0,
                            0x7FFFFFFFFFFFFFFFLL,
                            0x7FFFFFFFFFFFFFFLL,
                            0x8000000000000000ULL,
                            0x8000000000000000ULL,
                            1};
  Literal lhs_literal = LiteralUtil::CreateR1<uint64_t>({lhs});
  auto lhs_param = Parameter(&b, 0, lhs_literal.shape(), "lhs_param");
  std::unique_ptr<GlobalData> lhs_data =
      client_->TransferToServer(lhs_literal).value();

  std::vector<uint64_t> rhs{1,
                            0x7FFFFFFFFFFFFFFLL,
                            0x7FFFFFFFFFFFFFFFLL,
                            0x8000000000000000ULL,
                            0,
                            static_cast<uint64_t>(-1),
                            0,
                            1,
                            0x8000000000000000ULL};
  Literal rhs_literal = LiteralUtil::CreateR1<uint64_t>({rhs});
  auto rhs_param = Parameter(&b, 1, rhs_literal.shape(), "rhs_param");
  std::unique_ptr<GlobalData> rhs_data =
      client_->TransferToServer(rhs_literal).value();

  Add(lhs_param, rhs_param);

  std::vector<uint64_t> expected(lhs.size());
  for (int64_t i = 0; i < lhs.size(); ++i) {
    expected[i] = lhs[i] + rhs[i];
  }

  ComputeAndCompareR1<uint64_t>(&b, expected, {lhs_data.get(), rhs_data.get()});
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantS64s) {
  XlaBuilder b(TestName());

  std::vector<int64_t> lhs{static_cast<int64_t>(0x8000000000000000LL),
                           static_cast<int64_t>(0x8000000000000000LL),
                           -1,
                           0x7FFFFFFFFFFFFFFLL,
                           0x7FFFFFFFFFFFFFFFLL,
                           1,
                           0,
                           -1};
  Literal lhs_literal = LiteralUtil::CreateR1<int64_t>({lhs});
  auto lhs_param = Parameter(&b, 0, lhs_literal.shape(), "lhs_param");
  std::unique_ptr<GlobalData> lhs_data =
      client_->TransferToServer(lhs_literal).value();

  std::vector<int64_t> rhs{-1,
                           0,
                           static_cast<int64_t>(0x8000000000000000LL),
                           1,
                           0,
                           0x7FFFFFFFFFFFFFFLL,
                           0x7FFFFFFFFFFFFFFFLL,
                           0x7FFFFFFFFFFFFFFFLL};
  Literal rhs_literal = LiteralUtil::CreateR1<int64_t>({rhs});
  auto rhs_param = Parameter(&b, 1, rhs_literal.shape(), "rhs_param");
  std::unique_ptr<GlobalData> rhs_data =
      client_->TransferToServer(rhs_literal).value();

  Sub(lhs_param, rhs_param);

  std::vector<int64_t> expected(lhs.size());
  for (int64_t i = 0; i < lhs.size(); ++i) {
    expected[i] = lhs[i] - rhs[i];
  }

  ComputeAndCompareR1<int64_t>(&b, expected, {lhs_data.get(), rhs_data.get()});
}

XLA_TEST_F(ArrayElementwiseOpTest, CmpTwoConstantU64s) {
  XlaBuilder b(TestName());

  std::vector<uint64_t> lhs{static_cast<uint64_t>(0x8000000000000000ULL)};
  Literal lhs_literal = LiteralUtil::CreateR1<uint64_t>({lhs});
  auto lhs_param = Parameter(&b, 0, lhs_literal.shape(), "lhs_param");

  std::vector<uint64_t> rhs{static_cast<uint64_t>(0x7FFFFFFFFFFFFFFFULL)};
  Literal rhs_literal = LiteralUtil::CreateR1<uint64_t>({rhs});
  auto rhs_param = Parameter(&b, 1, rhs_literal.shape(), "rhs_param");

  Lt(lhs_param, rhs_param);

  ComputeAndCompare(&b, {std::move(lhs_literal), std::move(rhs_literal)});
}

TEST_P(ArrayElementwiseOpTestParamCount, AddManyValues) {
  const int count = GetParam();
  XlaBuilder builder(TestName());
  std::vector<float> a_values;
  std::vector<float> b_values;
  a_values.reserve(count);
  b_values.reserve(count);
  for (int i = 0; i < count; ++i) {
    a_values.push_back(i / static_cast<float>(count));
    b_values.push_back(2 * i / static_cast<float>(count + 2));
  }

  Literal a_literal = LiteralUtil::CreateR1<float>({a_values});
  std::unique_ptr<GlobalData> a_data =
      client_->TransferToServer(a_literal).value();
  auto a_constant = ConstantR1<float>(&builder, a_values);
  auto a_param = Parameter(&builder, 0, a_literal.shape(), "a_param");

  Literal b_literal = LiteralUtil::CreateR1<float>({b_values});
  std::unique_ptr<GlobalData> b_data =
      client_->TransferToServer(b_literal).value();
  auto b_param = Parameter(&builder, 1, a_literal.shape(), "b_param");
  auto b_constant = ConstantR1<float>(&builder, b_values);

  auto sum1 = Add(a_constant, b_param);
  auto sum2 = Add(a_constant, b_constant);
  auto sum3 = Add(a_param, b_param);
  auto sum4 = Add(a_param, b_constant);

  auto sum = Add(sum1, sum2);
  sum = Add(sum, sum3);
  sum = Add(sum, sum4);

  std::vector<float> expected;
  expected.reserve(count);
  for (int64_t i = 0; i < count; ++i) {
    expected.push_back(4 * (a_values[i] + b_values[i]));
  }

  ComputeAndCompareR1<float>(&builder, expected, {a_data.get(), b_data.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, DeeplyNestedAddWithSlices) {
  XlaBuilder builder(TestName());
  std::vector<float> values(30, 0.0);
  auto a_literal = LiteralUtil::CreateR1<float>(values);
  auto a = Parameter(&builder, 0, a_literal.shape(), "x");
  auto b_literal = LiteralUtil::CreateR1<float>(values);
  auto b = Parameter(&builder, 1, b_literal.shape(), "x");

  // Construct a sequence of diamond-shaped gadgets like this:
  //
  //      add
  //    /    \
  //  slice  slice
  //     \   /
  //      add
  //
  // Each 'left' slice removes the last element, each 'right' slice removes the
  // first element. In this way, we index into the add with different
  // multi-dimensional index arrays, which defeats the caching we use to avoid
  // exponential compile time.
  std::function<XlaOp(int64_t)> generate_recursive =
      [&](int64_t slice_size) -> XlaOp {
    if (slice_size == values.size()) {
      return Add(a, b);
    }
    XlaOp param = generate_recursive(slice_size + 1);
    auto slice1 = Slice(param, {0}, {slice_size}, {1});
    auto slice2 = Slice(param, {1}, {slice_size + 1}, {1});
    return Add(slice1, slice2);
  };
  generate_recursive(1);
  auto a_data = client_->TransferToServer(a_literal).value();
  auto b_data = client_->TransferToServer(b_literal).value();
  ComputeAndCompareR1<float>(&builder, {0.0}, {a_data.get(), b_data.get()});
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {-2.5f, 3.14f, 2.25f, -10.0f, 6.0f});
  auto b = ConstantR1<float>(&builder, {100.0f, 3.13f, 2.75f, 10.5f, -999.0f});
  Sub(a, b);

  ComputeAndCompareR1<float>(&builder, {-102.5f, 0.01f, -0.5f, -20.5f, 1005.0f},
                             {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {});
  auto b = ConstantR1<float>(&builder, {});
  Sub(a, b);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantS32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {-1, 0, 2, 1000000000});
  auto b = ConstantR1<int32_t>(&builder, {-1, 2, 1, -1});
  Sub(a, b);

  ComputeAndCompareR1<int32_t>(&builder, {0, -2, 1, 1000000001}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantZeroElementS32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {});
  auto b = ConstantR1<int32_t>(&builder, {});
  Sub(a, b);

  ComputeAndCompareR1<int32_t>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantC64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<complex64>(&builder,
                                 {{-2.5f, 0.0f}, {0.0f, 3.14f}, {3.0f, 2.25f}});
  auto b = ConstantR1<complex64>(
      &builder, {{0.0f, 10.0f}, {3.13f, 0.0f}, {2.75f, -0.25f}});
  Sub(a, b);

  ComputeAndCompareR1<complex64>(
      &builder, {{-2.5f, -10.0f}, {-3.13f, 3.14f}, {0.25f, 2.5f}}, {},
      error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantZeroElementC64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<complex64>(&builder, {});
  auto b = ConstantR1<complex64>(&builder, {});
  Sub(a, b);

  ComputeAndCompareR1<complex64>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SubTwoConstantF64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<double>(&builder, {-2.5, 3.14, 2.25, -10.0, 6.0});
  auto b = ConstantR1<double>(&builder, {100.0, 3.13, 2.75, 10.5, -999.0});
  Sub(a, b);

  ComputeAndCompare(&builder, {}, strict_error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, DivTwoConstantF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {-2.5f, 25.5f, 2.25f, -10.0f, 6.0f});
  auto b = ConstantR1<float>(&builder, {10.0f, 5.1f, 1.0f, 10.0f, -6.0f});
  Div(a, b);

  ComputeAndCompareR1<float>(&builder, {-0.25f, 5.0f, 2.25f, -1.0f, -1.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, DivTwoConstantZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {});
  auto b = ConstantR1<float>(&builder, {});
  Div(a, b);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, DivTwoConstantF64s) {
  auto kInf = std::numeric_limits<double>::infinity();
  auto kNaN = std::numeric_limits<double>::quiet_NaN();
  std::array<double, 7> vals{0.0, 0.1, 1.0, 2.0, 1e20, kNaN, kInf};
  std::vector<double> a_vals;
  std::vector<double> b_vals;
  a_vals.reserve(vals.size() * vals.size());
  b_vals.reserve(vals.size() * vals.size());
  for (auto abs_a_val : vals) {
    for (auto a_val : {-abs_a_val, abs_a_val}) {
      for (auto abs_b_val : vals) {
        for (auto b_val : {-abs_b_val, abs_b_val}) {
          a_vals.push_back(a_val);
          b_vals.push_back(b_val);
        }
      }
    }
  }
  XlaBuilder builder(TestName());
  auto a = ConstantR1<double>(&builder, a_vals);
  auto b = ConstantR1<double>(&builder, b_vals);
  Div(a, b);

  ComputeAndCompare(&builder, {}, strict_error_spec_);
}

class IntegerDivideOpTest : public ArrayElementwiseOpTest {
 protected:
  template <typename T>
  void TestDivRem(absl::Span<const T> dividends, absl::Span<const T> divisors,
                  absl::Span<const T> quotients,
                  absl::Span<const T> remainders) {
    {
      XlaBuilder builder(TestName());
      XlaOp dividend;
      XlaOp divisor;
      auto dividend_data =
          CreateR1Parameter<T>(dividends, 0, "dividend", &builder, &dividend);
      auto divisor_data =
          CreateR1Parameter<T>(divisors, 1, "divisor", &builder, &divisor);
      Div(dividend, divisor);

      ComputeAndCompareR1<T>(&builder, quotients,
                             {dividend_data.get(), divisor_data.get()});
    }

    // Test with a compile-time constant divisor.
    {
      XlaBuilder builder(TestName());
      XlaOp dividend;
      auto dividend_data =
          CreateR1Parameter<T>(dividends, 0, "dividend", &builder, &dividend);
      Div(dividend, ConstantR1<T>(&builder, divisors));

      ComputeAndCompareR1<T>(&builder, quotients, {dividend_data.get()});
    }

    {
      XlaBuilder builder(TestName());
      XlaOp dividend;
      XlaOp divisor;
      auto dividend_data =
          CreateR1Parameter<T>(dividends, 0, "dividend", &builder, &dividend);
      auto divisor_data =
          CreateR1Parameter<T>(divisors, 1, "divisor", &builder, &divisor);
      Rem(dividend, divisor);

      ComputeAndCompareR1<T>(&builder, remainders,
                             {dividend_data.get(), divisor_data.get()});
    }

    // Test with a compile-time constant divisor.
    {
      XlaBuilder builder(TestName());
      XlaOp dividend;
      auto dividend_data =
          CreateR1Parameter<T>(dividends, 0, "dividend", &builder, &dividend);
      Rem(dividend, ConstantR1<T>(&builder, divisors));

      ComputeAndCompareR1<T>(&builder, remainders, {dividend_data.get()});
    }
  }
};

XLA_TEST_F(IntegerDivideOpTest, DivS32s) {
  // clang-format off
  // Some interesting values to test.
  std::vector<int32_t> vals = {
    INT32_MIN, INT32_MIN + 1, INT32_MIN + 2, -0x40000000, -0x3fffffff,
    -271181, -1309, -17, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 17, 26, 101,
    7919, 0x40000000, INT32_MAX - 2, INT32_MAX - 1, INT32_MAX};
  // clang-format on

  std::vector<int32_t> dividends, divisors, quotients, remainders;
  for (int32_t divisor : vals) {
    if (divisor != 0) {
      for (int32_t dividend : vals) {
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

  TestDivRem<int32_t>(dividends, divisors, quotients, remainders);
}

XLA_TEST_F(IntegerDivideOpTest, SignedOverflow) {
  std::vector<int32_t> dividends = {5, INT32_MIN}, divisors = {0, -1},
                       quotients = {-1, INT32_MIN}, remainders = {5, 0};

  TestDivRem<int32_t>(dividends, divisors, quotients, remainders);
}

XLA_TEST_F(IntegerDivideOpTest, DivU32s) {
  // clang-format off
  // Some interesting values to test.
  std::vector<uint32_t> vals = {
    0, 1, 2, 17, 101, 3333, 0x7FFFFFFF, 0xABCDEF12, 0xCAFEBEEF, 0x80000000,
    0x80000001, UINT32_MAX - 2, UINT32_MAX - 1, UINT32_MAX};
  // clang-format on

  std::vector<uint32_t> dividends, divisors, quotients, remainders;
  for (uint32_t divisor : vals) {
    if (divisor != 0) {
      for (uint32_t dividend : vals) {
        dividends.push_back(dividend);
        divisors.push_back(divisor);
        quotients.push_back(dividend / divisor);
        remainders.push_back(dividend % divisor);
      }
    }
  }

  TestDivRem<uint32_t>(dividends, divisors, quotients, remainders);
}

XLA_TEST_F(IntegerDivideOpTest, UnsignedOverflow) {
  std::vector<int32_t> dividends = {5}, divisors = {0}, quotients = {-1},
                       remainders = {5};

  TestDivRem<int32_t>(dividends, divisors, quotients, remainders);
}

XLA_TEST_F(ArrayElementwiseOpTest, DivTwoConstantC64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<complex64>(
      &builder, {{-2.5f, 1.0f}, {-25.5f, 0.0f}, {2.0f, -1.0f}});
  auto b = ConstantR1<complex64>(&builder,
                                 {{10.0f, 0.0f}, {0.0f, 1.0f}, {2.0f, -1.0f}});
  Div(a, b);

  ComputeAndCompareR1<complex64>(
      &builder, {{-0.25f, 0.1f}, {0.0f, 25.5f}, {1.0f, 0.0f}}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, DivTwoConstantZeroElementC64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<complex64>(&builder, {});
  auto b = ConstantR1<complex64>(&builder, {});
  Div(a, b);

  ComputeAndCompareR1<complex64>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, RemF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(
      &builder, {-2.5f, 25.5f, 2.25f, -10.0f, 6.0f, 3.0f, 3.0f, -1.0f, -8.0f});
  auto b = ConstantR1<float>(
      &builder, {10.0f, 5.1f, 1.0f, 10.0f, -6.0f, 2.0f, -2.0f, 7.0f, -4.0f});
  Rem(a, b);

  ComputeAndCompareR1<float>(
      &builder, {-2.5f, 0.0f, 0.25f, 0.0f, -0.0f, 1.0f, 1.0f, -1.0f, -0.0f}, {},
      error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, RemZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {});
  auto b = ConstantR1<float>(&builder, {});
  Rem(a, b);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, RemF64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<double>(
      &builder, {-2.5, 25.5, 2.25, -10.0, 6.0, 3.0, 3.0, -1.0, -8.0});
  auto b = ConstantR1<double>(
      &builder, {10.0, 5.1, 1.0, 10.0, -6.0, 2.0, -2.0, 7.0, -4.0});
  Rem(a, b);

  ComputeAndCompareR1<double>(
      &builder, {-2.5, 0.0, 0.25, 0.0, -0.0, 1.0, 1.0, -1.0, -0.0}, {},
      strict_error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {-2.5f, 25.5f, 2.25f, -10.0f, 6.0f});
  auto b = ConstantR1<float>(&builder, {10.0f, 5.0f, 1.0f, 10.0f, -6.0f});
  Mul(a, b);

  ComputeAndCompareR1<float>(&builder, {-25.0f, 127.5f, 2.25f, -100.0f, -36.0f},
                             {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {});
  auto b = ConstantR1<float>(&builder, {});
  Mul(a, b);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantS32s) {
  std::vector<int32_t> data = {0,
                               1,
                               -1,
                               1234,
                               0x1a243514,
                               std::numeric_limits<int32_t>::max(),
                               std::numeric_limits<int32_t>::min()};
  // Form the test data set using all products of 'data' with itself.
  std::vector<int32_t> a_data, b_data, expected;
  for (int32_t a : data) {
    for (int32_t b : data) {
      a_data.push_back(a);
      b_data.push_back(b);
      expected.push_back(static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
    }
  }

  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, a_data);
  auto b = ConstantR1<int32_t>(&builder, b_data);
  Mul(a, b);

  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantZeroElementS32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {});
  auto b = ConstantR1<int32_t>(&builder, {});
  Mul(a, b);

  ComputeAndCompareR1<int32_t>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantU32s) {
  std::vector<uint32_t> data = {0,          1,          0xDEADBEEF, 1234,
                                0x1a243514, 0xFFFFFFFF, 0x80808080};

  // Form the test data set using all products of 'data' with itself.
  std::vector<uint32_t> a_data, b_data, expected;
  for (uint32_t a : data) {
    for (uint32_t b : data) {
      a_data.push_back(a);
      b_data.push_back(b);
      expected.push_back(a * b);
    }
  }

  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder, a_data);
  auto b = ConstantR1<uint32_t>(&builder, b_data);
  Mul(a, b);

  ComputeAndCompareR1<uint32_t>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantC64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<complex64>(
      &builder, {{-2.5f, 0.0f}, {0.0f, 25.5f}, {2.0f, -10.0f}});
  auto b = ConstantR1<complex64>(&builder,
                                 {{0.0f, 10.0f}, {5.0f, 1.0f}, {10.0f, -6.0f}});
  Mul(a, b);

  ComputeAndCompareR1<complex64>(
      &builder, {{0.0f, -25.0f}, {-25.5f, 127.5f}, {-40.0f, -112.0}}, {},
      error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MulTwoConstantZeroElementC64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<complex64>(&builder, {});
  auto b = ConstantR1<complex64>(&builder, {});
  Mul(a, b);

  ComputeAndCompareR1<complex64>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AndPredR1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {false, false, true, true});
  auto b = ConstantR1<bool>(&builder, {false, true, false, true});
  And(a, b);

  ComputeAndCompareR1<bool>(&builder, {false, false, false, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndPredR2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<bool>(&builder, {{false, false}, {true, true}});
  auto b = ConstantR2<bool>(&builder, {{false, true}, {false, true}});
  And(a, b);

  Array2D<bool> expected_array({{false, false}, {false, true}});
  ComputeAndCompareR2<bool>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndZeroElementPredR1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {});
  auto b = ConstantR1<bool>(&builder, {});
  And(a, b);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndS32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {0, -1, -8});
  auto b = ConstantR1<int32_t>(&builder, {5, -7, 12});
  And(a, b);

  ComputeAndCompareR1<int32_t>(&builder, {0, -7, 8}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndS32R2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<int32_t>(&builder, {{0, -5}, {-1, 5}});
  auto b = ConstantR2<int32_t>(&builder, {{1, -6}, {4, 5}});
  And(a, b);

  Array2D<int32_t> expected_array({{0, -6}, {4, 5}});
  ComputeAndCompareR2<int32_t>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndZeroElementS32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {});
  auto b = ConstantR1<int32_t>(&builder, {});
  And(a, b);

  ComputeAndCompareR1<int32_t>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndU32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {0, 1, 8});
  auto b = ConstantR1<int32_t>(&builder, {5, 7, 12});
  And(a, b);

  ComputeAndCompareR1<int32_t>(&builder, {0, 1, 8}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndU32R2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<uint32_t>(&builder, {{0, 1}, {3, 8}});
  auto b = ConstantR2<uint32_t>(&builder, {{1, 0}, {7, 6}});
  And(a, b);

  Array2D<uint32_t> expected_array({{0, 0}, {3, 0}});
  ComputeAndCompareR2<uint32_t>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AndZeroElementU32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder, {});
  auto b = ConstantR1<uint32_t>(&builder, {});
  And(a, b);

  ComputeAndCompareR1<uint32_t>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrPredR1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {false, false, true, true});
  auto b = ConstantR1<bool>(&builder, {false, true, false, true});
  Or(a, b);

  ComputeAndCompareR1<bool>(&builder, {false, true, true, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrPredR2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<bool>(&builder, {{false, false}, {true, true}});
  auto b = ConstantR2<bool>(&builder, {{false, true}, {false, true}});
  Or(a, b);

  Array2D<bool> expected_array({{false, true}, {true, true}});
  ComputeAndCompareR2<bool>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrZeroElementPredR1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {});
  auto b = ConstantR1<bool>(&builder, {});
  Or(a, b);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrS32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {0, -1, 8});
  auto b = ConstantR1<int32_t>(&builder, {5, -7, 4});
  Or(a, b);

  ComputeAndCompareR1<int32_t>(&builder, {5, -1, 12}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrS32R2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<int32_t>(&builder, {{0, -1}, {8, 8}});
  auto b = ConstantR2<int32_t>(&builder, {{5, -7}, {4, 1}});
  Or(a, b);

  Array2D<int32_t> expected_array({{5, -1}, {12, 9}});
  ComputeAndCompareR2<int32_t>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrZeroElementS32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {});
  auto b = ConstantR1<int32_t>(&builder, {});
  Or(a, b);

  ComputeAndCompareR1<int32_t>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrU32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder, {0, 1, 8});
  auto b = ConstantR1<uint32_t>(&builder, {5, 7, 4});
  Or(a, b);

  ComputeAndCompareR1<uint32_t>(&builder, {5, 7, 12}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrU32R2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<uint32_t>(&builder, {{0, 1}, {8, 8}});
  auto b = ConstantR2<uint32_t>(&builder, {{5, 7}, {4, 1}});
  Or(a, b);

  Array2D<uint32_t> expected_array({{5, 7}, {12, 9}});
  ComputeAndCompareR2<uint32_t>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, OrZeroElementU32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder, {});
  auto b = ConstantR1<uint32_t>(&builder, {});
  Or(a, b);

  ComputeAndCompareR1<uint32_t>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, XorPredR1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {false, false, true, true});
  auto b = ConstantR1<bool>(&builder, {false, true, false, true});
  Xor(a, b);

  ComputeAndCompareR1<bool>(&builder, {false, true, true, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, XorPredR2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<bool>(&builder, {{false, false}, {true, true}});
  auto b = ConstantR2<bool>(&builder, {{false, true}, {false, true}});
  Xor(a, b);

  Array2D<bool> expected_array({{false, true}, {true, false}});
  ComputeAndCompareR2<bool>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, XorZeroElementPredR1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {});
  auto b = ConstantR1<bool>(&builder, {});
  Xor(a, b);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, XorS32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {0, -1, 8});
  auto b = ConstantR1<int32_t>(&builder, {5, -7, 4});
  Xor(a, b);

  ComputeAndCompareR1<int32_t>(&builder, {5, 6, 12}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, XorS32R2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<int32_t>(&builder, {{0, -1}, {8, 8}});
  auto b = ConstantR2<int32_t>(&builder, {{5, -7}, {4, 1}});
  Xor(a, b);

  Array2D<int32_t> expected_array({{5, 6}, {12, 9}});
  ComputeAndCompareR2<int32_t>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, XorZeroElementS32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {});
  auto b = ConstantR1<int32_t>(&builder, {});
  Xor(a, b);

  ComputeAndCompareR1<int32_t>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, XorU32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder, {0, 1, 8});
  auto b = ConstantR1<uint32_t>(&builder, {5, 7, 4});
  Xor(a, b);

  ComputeAndCompareR1<uint32_t>(&builder, {5, 6, 12}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, XorU32R2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<uint32_t>(&builder, {{0, 1}, {8, 8}});
  auto b = ConstantR2<uint32_t>(&builder, {{5, 7}, {4, 1}});
  Xor(a, b);

  Array2D<uint32_t> expected_array({{5, 6}, {12, 9}});
  ComputeAndCompareR2<uint32_t>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, XorZeroElementU32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder, {});
  auto b = ConstantR1<uint32_t>(&builder, {});
  Xor(a, b);

  ComputeAndCompareR1<uint32_t>(&builder, {}, {});
}
XLA_TEST_F(ArrayElementwiseOpTest, NotPredR1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {false, true, true, false});
  Not(a);

  ComputeAndCompareR1<bool>(&builder, {true, false, false, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotPredR2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<bool>(&builder, {{false, true}, {true, false}});
  Not(a);

  Array2D<bool> expected_array({{true, false}, {false, true}});
  ComputeAndCompareR2<bool>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotZeroElementPredR1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {});
  Not(a);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotS32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {-1, 0, 1});
  Not(a);

  ComputeAndCompareR1<int32_t>(&builder, {0, -1, -2}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotS32R2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<int32_t>(&builder, {{-1, 0}, {1, 8}});
  Not(a);

  Array2D<int32_t> expected_array({{0, -1}, {-2, -9}});
  ComputeAndCompareR2<int32_t>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotZeroElementS32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {});
  Not(a);

  ComputeAndCompareR1<int32_t>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotU32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder, {0, 4294967295});
  Not(a);

  ComputeAndCompareR1<uint32_t>(&builder, {4294967295, 0}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotU32R2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<uint32_t>(&builder, {{0, 4294967295}, {1, 4294967294}});
  Not(a);

  Array2D<uint32_t> expected_array({{4294967295, 0}, {4294967294, 1}});
  ComputeAndCompareR2<uint32_t>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NotZeroElementU32R1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder, {});
  Not(a);

  ComputeAndCompareR1<uint32_t>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, PopcntR1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {0, 1, -15, 341});
  PopulationCount(a);
  ComputeAndCompareR1<int32_t>(&builder, {0, 1, 29, 5}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, PopcntR2) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<int32_t>(&builder, {{0, 1}, {-15, 341}});
  PopulationCount(a);
  Array2D<int32_t> expected_array({{0, 1}, {29, 5}});
  ComputeAndCompareR2<int32_t>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, PopcntS64) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<int64_t>(&builder, {{0, -1}, {INT64_MAX, INT64_MAX - 1}});
  PopulationCount(a);
  Array2D<int64_t> expected_array({{0, 64}, {63, 62}});
  ComputeAndCompareR2<int64_t>(&builder, expected_array, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftLeftS32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(
      &builder, {static_cast<int32_t>(0x12345678),
                 static_cast<int32_t>(0xF0001000), 1, 3, 77, 1, -3, 77});
  auto b = ConstantR1<int32_t>(&builder, {4, 8, 2, 7, 15, 32, 100, -1});
  ShiftLeft(a, b);

  ComputeAndCompareR1<int32_t>(&builder,
                               {static_cast<int32_t>(0x23456780), 0x00100000,
                                0x4, 0x180, 2523136, 0, 0, 0},
                               {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftRightArithmeticS32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(
      &builder, {static_cast<int32_t>(0x92345678),
                 static_cast<int32_t>(0x10001000), 1, 3, 77, 1, -3, 77});
  auto b = ConstantR1<int32_t>(&builder, {4, 8, 2, 7, 2, 32, 100, -1});
  ShiftRightArithmetic(a, b);

  ComputeAndCompareR1<int32_t>(
      &builder,
      {static_cast<int32_t>(0xF9234567), static_cast<int32_t>(0x00100010), 0, 0,
       19, 0, -1, 0},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftRightLogicalS32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(
      &builder, {static_cast<int32_t>(0x92345678),
                 static_cast<int32_t>(0x10001000), 1, 3, 77, 1, -3, 77});
  auto b = ConstantR1<int32_t>(&builder, {4, 8, 2, 7, 5, 32, 100, -1});
  ShiftRightLogical(a, b);

  ComputeAndCompareR1<int32_t>(&builder,
                               {0x09234567, 0x00100010, 0, 0, 2, 0, 0, 0}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftLeftU32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder,
                                {0x12345678, 0xF0001000, 1, 3, 77, 1, ~3u, 77});
  auto b = ConstantR1<uint32_t>(&builder, {4, 8, 2, 7, 15, 32, 100, ~0u});
  ShiftLeft(a, b);

  ComputeAndCompareR1<uint32_t>(
      &builder, {0x23456780, 0x00100000, 0x4, 0x180, 2523136, 0, 0, 0}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftRightArithmeticU32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder,
                                {0x92345678, 0x10001000, 1, 3, 77, 1, ~3u, 77});
  auto b = ConstantR1<uint32_t>(&builder, {4, 8, 2, 7, 2, 32, 100, ~0u});
  ShiftRightArithmetic(a, b);

  ComputeAndCompareR1<uint32_t>(
      &builder, {0xF9234567, 0x00100010, 0, 0, 19, 0, ~0u, 0}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ShiftRightLogicalU32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder,
                                {0x92345678, 0x10001000, 1, 3, 77, 1, ~3u, 77});
  auto b = ConstantR1<uint32_t>(&builder, {4, 8, 2, 7, 5, 32, 100, ~0u});
  ShiftRightLogical(a, b);

  ComputeAndCompareR1<uint32_t>(&builder,
                                {0x09234567, 0x00100010, 0, 0, 2, 0, 0, 0}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqF32s) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
  auto rhs = ConstantR1<float>(&builder, {10.0f, 5.0f, 2.25f, 10.0f, NAN});
  Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {false, false, true, false, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqF32sTO) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
  auto rhs = ConstantR1<float>(&builder, {10.0f, 5.0f, 2.25f, NAN, NAN});
  EqTotalOrder(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {false, false, true, true, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {});
  auto rhs = ConstantR1<float>(&builder, {});
  Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGeF32s) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
  auto rhs = ConstantR1<float>(&builder, {10.0f, 5.0f, 1.0f, 10.0f, NAN});
  Ge(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {false, true, true, false, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGeF32sTO) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  // For portability, need to represent NAN using the following call.
  // The C++ standard does not specify if quiet_NaN() sets the sign bit of
  // its result. The call to std::fabs will ensure that it is not set.
  auto kNaN = std::fabs(std::numeric_limits<float>::quiet_NaN());
  auto lhs =
      ConstantR1<float>(&builder, {-2.5f, 25.5f, 2.25f, kNaN, 6.0f, 6.0f});
  auto rhs =
      ConstantR1<float>(&builder, {10.0f, 5.0f, 1.0f, 10.0f, kNaN, -kNaN});
  GeTotalOrder(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {false, true, true, true, false, true},
                            {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGtF32s) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
  auto rhs = ConstantR1<float>(&builder, {10.0f, 5.0f, 1.0f, 10.0f, NAN});
  Gt(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {false, true, true, false, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLeF32s) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {-2.5f, 5.0f, 2.25f, NAN, 6.0f});
  auto rhs = ConstantR1<float>(&builder, {10.0f, 5.0f, 1.0f, 10.0f, NAN});
  Le(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {true, true, false, false, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLtF32s) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
  auto rhs = ConstantR1<float>(&builder, {10.0f, 5.0f, 1.0f, 10.0f, NAN});
  Lt(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {true, false, false, false, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs =
      ConstantR1<int32_t>(&builder, {min, min, min, 0, 0, 0, max, max, max});
  auto rhs =
      ConstantR1<int32_t>(&builder, {min, 0, max, -1, 0, 1, min, 0, max});
  Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, false, false, false, true, false, false, false, true},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqZeroElementS32s) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<int32_t>(&builder, {});
  auto rhs = ConstantR1<int32_t>(&builder, {});
  Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqC64s) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<complex64>(&builder, {{-2.5f, 10.0f},
                                              {1.0f, 25.5f},
                                              {2.25f, -3.0f},
                                              {NAN, 0.0f},
                                              {1.0f, 6.0f}});
  auto rhs = ConstantR1<complex64>(&builder, {{0.0f, 10.0f},
                                              {1.0f, 5.0f},
                                              {2.25f, -3.0f},
                                              {10.0f, 0.0f},
                                              {1.0f, NAN}});
  Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {false, false, true, false, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqZeroElementC64s) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<complex64>(&builder, {});
  auto rhs = ConstantR1<complex64>(&builder, {});
  Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareNeC64s) {
  // Disable fast-math because we're operating on NaNs.
  SetFastMathDisabled(true);

  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<complex64>(&builder, {{-2.5f, 10.0f},
                                              {1.0f, 25.5f},
                                              {2.25f, -3.0f},
                                              {NAN, 0.0f},
                                              {1.0f, 6.0f}});
  auto rhs = ConstantR1<complex64>(&builder, {{0.0f, 10.0f},
                                              {1.0f, 5.0f},
                                              {2.25f, -3.0f},
                                              {10.0f, 0.0f},
                                              {1.0f, NAN}});
  Ne(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {true, true, false, true, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareNeF32s) {
  // Disable fast-math because we're operating on NaNs.
  SetFastMathDisabled(true);

  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
  auto rhs = ConstantR1<float>(&builder, {10.0f, 25.5f, 1.0f, 10.0f, NAN});
  Ne(lhs, rhs);

  ComputeAndCompareR1<bool>(&builder, {true, false, true, true, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareNeS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs =
      ConstantR1<int32_t>(&builder, {min, min, min, 0, 0, 0, max, max, max});
  auto rhs =
      ConstantR1<int32_t>(&builder, {min, 0, max, -1, 0, 1, min, 0, max});
  Ne(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, true, true, true, false, true, true, true, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGeS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs =
      ConstantR1<int32_t>(&builder, {min, min, min, 0, 0, 0, max, max, max});
  auto rhs =
      ConstantR1<int32_t>(&builder, {min, 0, max, -1, 0, 1, min, 0, max});
  Ge(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, false, false, true, true, false, true, true, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGtS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs =
      ConstantR1<int32_t>(&builder, {min, min, min, 0, 0, 0, max, max, max});
  auto rhs =
      ConstantR1<int32_t>(&builder, {min, 0, max, -1, 0, 1, min, 0, max});
  Gt(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, false, false, true, false, false, true, true, false},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLeS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs =
      ConstantR1<int32_t>(&builder, {min, min, min, 0, 0, 0, max, max, max});
  auto rhs =
      ConstantR1<int32_t>(&builder, {min, 0, max, -1, 0, 1, min, 0, max});
  Le(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, true, true, false, true, true, false, false, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLtS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs =
      ConstantR1<int32_t>(&builder, {min, min, min, 0, 0, 0, max, max, max});
  auto rhs =
      ConstantR1<int32_t>(&builder, {min, 0, max, -1, 0, 1, min, 0, max});
  Lt(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, true, true, false, false, true, false, false, false},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareEqU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<uint32_t>(&builder, {0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = ConstantR1<uint32_t>(&builder, {0, 1, max, 4, 5, 6, 0, 1, max});
  Eq(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, false, false, false, true, false, false, false, true},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareNeU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<uint32_t>(&builder, {0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = ConstantR1<uint32_t>(&builder, {0, 1, max, 4, 5, 6, 0, 1, max});
  Ne(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, true, true, true, false, true, true, true, false}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGeU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<uint32_t>(&builder, {0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = ConstantR1<uint32_t>(&builder, {0, 1, max, 4, 5, 6, 0, 1, max});
  Ge(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, false, false, true, true, false, true, true, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareGtU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<uint32_t>(&builder, {0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = ConstantR1<uint32_t>(&builder, {0, 1, max, 4, 5, 6, 0, 1, max});
  Gt(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, false, false, true, false, false, true, true, false},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLeU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<uint32_t>(&builder, {0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = ConstantR1<uint32_t>(&builder, {0, 1, max, 4, 5, 6, 0, 1, max});
  Le(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {true, true, true, false, true, true, false, false, true}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, CompareLtU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<uint32_t>(&builder, {0, 0, 0, 5, 5, 5, max, max, max});
  auto rhs = ConstantR1<uint32_t>(&builder, {0, 1, max, 4, 5, 6, 0, 1, max});
  Lt(lhs, rhs);

  ComputeAndCompareR1<bool>(
      &builder, {false, true, true, false, false, true, false, false, false},
      {});
}

XLA_TEST_F(ArrayElementwiseOpTest, PowF32s) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto eps = std::numeric_limits<float>::epsilon();

  const std::vector<float> kTestValues = {
      0.1f, 1.0f / 3.0f, 2.0f / 3.0f, 0.5f,      1.0f + eps,
      2.0f, 3.0f,        M_PI,        1e6 + 0.1, 1e2};
  std::vector<float> xs;
  std::vector<float> ys;
  std::tie(xs, ys) =
      AllSignedPairs<float>(AppendSpecialPositiveValues(kTestValues));
  auto lhs = ConstantR1<float>(&builder, xs);
  auto rhs = ConstantR1<float>(&builder, ys);
  Pow(lhs, rhs);

  ComputeAndCompare(&builder, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, PowNonIntegerF32s) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {-2.0f, -0.6f, -0.6f, 0.0f});
  auto rhs = ConstantR1<float>(&builder, {0.5f, 0.6f, -0.6f, -0.6f});
  Pow(lhs, rhs);

  ComputeAndCompareR1<float>(&builder, {NAN, NAN, NAN, INFINITY}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, PowC64s) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto lhs =
      ConstantR1<complex64>(&builder, {-2.0f, -0.6f, -0.6f, 0.0f, 0.0f, 0.0f});
  auto rhs =
      ConstantR1<complex64>(&builder, {0.5f, 0.6f, -0.6f, 0.5f, 0.6f, 0.0f});
  Pow(lhs, rhs);

  ComputeAndCompareR1<complex64>(&builder,
                                 {
                                     {0, 1.41421356},
                                     {-2.27443288e-01, 0.69999846},
                                     {-4.19847531e-01, -1.29215783},
                                     {0, 0},
                                     {0, 0},
                                     {1, 0},
                                 },
                                 {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, PowZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {});
  auto rhs = ConstantR1<float>(&builder, {});
  Pow(lhs, rhs);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

// Some Pow cases that can be implemented more efficiently.
XLA_TEST_F(ArrayElementwiseOpTest, PowSpecialF32) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kQNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kOneThird = 1.0f / 3.0f;
  std::vector<float> values = {-0.0f, 0.0f,  1.0f,  2.0f, -M_PI,
                               M_PI,  -4.0f, -kInf, kInf, kQNaN};
  std::vector<float> exponents = {-0.0f,
                                  0.0f,
                                  -0.25f,
                                  0.25f,
                                  -kOneThird,
                                  kOneThird,
                                  -0.5f,
                                  0.5f,
                                  -2 * kOneThird,
                                  2 * kOneThird,
                                  -1.0f,
                                  1.0f,
                                  -1.25f,
                                  1.25f,
                                  -1.0f - kOneThird,
                                  1.0f + kOneThird,
                                  -1.5f,
                                  1.5f,
                                  -1.75f,
                                  1.75f,
                                  -2.0f,
                                  2.0f,
                                  42 + 0.5 - kOneThird + 0.25,
                                  -kInf,
                                  kInf,
                                  kQNaN};

  for (float exponent : exponents) {
    XlaBuilder b(TestName());
    Pow(ConstantR1<float>(&b, values), ConstantR0<float>(&b, exponent));
    ComputeAndCompare(&b, {}, error_spec_);
  }
}

XLA_TEST_F(ArrayElementwiseOpTest, PowOfExpF32) {
  XlaBuilder b(TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.0f, 5.7f};
  std::vector<float> values1 = {0.0f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};

  Literal literal0 = LiteralUtil::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(literal0).value();
  Literal literal1 = LiteralUtil::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(literal1).value();
  auto param0 = Parameter(&b, 0, literal0.shape(), "param0");
  auto param1 = Parameter(&b, 1, literal1.shape(), "param1");
  Pow(Exp(param0), param1);

  std::vector<float> expected(values0.size());
  for (int64_t i = 0; i < values0.size(); ++i) {
    expected[i] = std::pow(std::exp(values0[i]), values1[i]);
  }

  ComputeAndCompareR1<float>(&b, expected, {data0.get(), data1.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, LogOfPowerF32) {
  XlaBuilder b(TestName());

  std::vector<float> values0 = {1.0f, -10.0f, -2.0f, 2.0f, 3.2f,
                                4.0f, 0.5f,   5.7f,  0.0f};
  std::vector<float> values1 = {0.0f, 10.0f, -4.0f, 1.0f, 2.0f,
                                0.5f, -1.0f, -0.5f, 0.0f};

  Literal literal0 = LiteralUtil::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(literal0).value();
  Literal literal1 = LiteralUtil::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(literal1).value();
  auto param0 = Parameter(&b, 0, literal0.shape(), "param0");
  auto param1 = Parameter(&b, 1, literal1.shape(), "param1");
  Log(Pow(param0, param1));

  std::vector<float> expected(values0.size());
  for (int64_t i = 0; i < values0.size(); ++i) {
    expected[i] = std::log(std::pow(values0[i], values1[i]));
  }

  // Log2 is very inaccurate onon some platforms.
  ErrorSpec error_spec(1000 * kEpsF32, 1000 * kEpsF32);
  ComputeAndCompareR1<float>(&b, expected, {data0.get(), data1.get()},
                             error_spec);
}

XLA_TEST_F(ArrayElementwiseOpTest, MulOfExpF32) {
  XlaBuilder b(TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.0f, 5.7f};
  std::vector<float> values1 = {0.0f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};

  Literal literal0 = LiteralUtil::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(literal0).value();
  Literal literal1 = LiteralUtil::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(literal1).value();
  auto param0 = Parameter(&b, 0, literal0.shape(), "param0");
  auto param1 = Parameter(&b, 1, literal1.shape(), "param1");
  Mul(Exp(param0), Exp(param1));

  std::vector<float> expected(values0.size());
  for (int64_t i = 0; i < values0.size(); ++i) {
    expected[i] = std::exp(values0[i]) * std::exp(values1[i]);
  }

  ComputeAndCompareR1<float>(&b, expected, {data0.get(), data1.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, DivOfExpF32) {
  XlaBuilder b(TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.0f, 5.7f};
  std::vector<float> values1 = {0.0f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};

  Literal literal0 = LiteralUtil::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(literal0).value();
  Literal literal1 = LiteralUtil::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(literal1).value();
  auto param0 = Parameter(&b, 0, literal0.shape(), "param0");
  auto param1 = Parameter(&b, 1, literal1.shape(), "param1");
  Div(param0, Exp(param1));

  std::vector<float> expected(values0.size());
  for (int64_t i = 0; i < values0.size(); ++i) {
    expected[i] = values0[i] / std::exp(values1[i]);
  }

  ComputeAndCompareR1<float>(&b, expected, {data0.get(), data1.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Div3_lhs_F32) {
  XlaBuilder b(TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.45f, 5.7f};
  std::vector<float> values1 = {0.1f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};
  std::vector<float> values2 = {0.1f, 1.1f, 6.9f, 12.5f, -15.0f, -0.5f};

  Literal literal0 = LiteralUtil::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(literal0).value();

  Literal literal1 = LiteralUtil::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(literal1).value();

  Literal literal2 = LiteralUtil::CreateR1<float>(values2);
  std::unique_ptr<GlobalData> data2 =
      client_->TransferToServer(literal2).value();
  auto param0 = Parameter(&b, 0, literal0.shape(), "param0");
  auto param1 = Parameter(&b, 1, literal1.shape(), "param1");
  auto param2 = Parameter(&b, 2, literal2.shape(), "param2");
  Div(Div(param0, param1), param2);

  std::vector<float> expected(values0.size());
  for (int64_t i = 0; i < values0.size(); ++i) {
    expected[i] = (values0[i] / values1[i]) / values2[i];
  }

  ComputeAndCompareR1<float>(
      &b, expected, {data0.get(), data1.get(), data2.get()}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Div3_rhs_F32) {
  XlaBuilder b(TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.45f, 5.7f};
  std::vector<float> values1 = {0.1f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};
  std::vector<float> values2 = {0.1f, 1.1f, 6.9f, 12.5f, -15.0f, -0.5f};

  Literal literal0 = LiteralUtil::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(literal0).value();

  Literal literal1 = LiteralUtil::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(literal1).value();

  Literal literal2 = LiteralUtil::CreateR1<float>(values2);
  std::unique_ptr<GlobalData> data2 =
      client_->TransferToServer(literal2).value();

  auto param0 = Parameter(&b, 0, literal0.shape(), "param0");
  auto param1 = Parameter(&b, 1, literal1.shape(), "param1");
  auto param2 = Parameter(&b, 2, literal2.shape(), "param2");
  Div(param0, Div(param1, param2));

  std::vector<float> expected(values0.size());
  for (int64_t i = 0; i < values0.size(); ++i) {
    expected[i] = values0[i] / (values1[i] / values2[i]);
  }

  ComputeAndCompareR1<float>(
      &b, expected, {data0.get(), data1.get(), data2.get()}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, DivOfPowerF32) {
  XlaBuilder b(TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.45f, 5.7f};
  std::vector<float> values1 = {0.1f, 1.0f, 2.0f, 0.5f, 1.0f, 0.5f};
  std::vector<float> values2 = {0.1f, 1.1f, 6.9f, 9.5f, -11.0f, -0.5f};

  Literal literal0 = LiteralUtil::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(literal0).value();

  Literal literal1 = LiteralUtil::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(literal1).value();

  Literal literal2 = LiteralUtil::CreateR1<float>(values2);
  std::unique_ptr<GlobalData> data2 =
      client_->TransferToServer(literal2).value();

  auto param0 = Parameter(&b, 0, literal0.shape(), "param0");
  auto param1 = Parameter(&b, 1, literal1.shape(), "param1");
  auto param2 = Parameter(&b, 2, literal2.shape(), "param2");
  Div(param0, Pow(param1, param2));

  std::vector<float> expected(values0.size());
  for (int64_t i = 0; i < values0.size(); ++i) {
    expected[i] = values0[i] / std::pow(values1[i], values2[i]);
  }

  ComputeAndCompareR1<float>(
      &b, expected, {data0.get(), data1.get(), data2.get()}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Div4F32) {
  XlaBuilder b(TestName());

  std::vector<float> values0 = {1.0f, 2.0f, 3.2f, -4.0f, 0.45f, 5.7f};
  std::vector<float> values1 = {0.1f, 1.0f, 2.0f, 0.5f, -1.0f, -0.5f};
  std::vector<float> values2 = {0.1f, 1.1f, 6.9f, 12.5f, -15.0f, -0.5f};
  std::vector<float> values3 = {2.1f, 3.1f, 9.9f, -4.5f, -11.0f, -21.5f};

  Literal literal0 = LiteralUtil::CreateR1<float>(values0);
  std::unique_ptr<GlobalData> data0 =
      client_->TransferToServer(literal0).value();

  Literal literal1 = LiteralUtil::CreateR1<float>(values1);
  std::unique_ptr<GlobalData> data1 =
      client_->TransferToServer(literal1).value();

  Literal literal2 = LiteralUtil::CreateR1<float>(values2);
  std::unique_ptr<GlobalData> data2 =
      client_->TransferToServer(literal2).value();

  Literal literal3 = LiteralUtil::CreateR1<float>(values3);
  std::unique_ptr<GlobalData> data3 =
      client_->TransferToServer(literal3).value();

  auto param0 = Parameter(&b, 0, literal0.shape(), "param0");
  auto param1 = Parameter(&b, 1, literal1.shape(), "param1");
  auto param2 = Parameter(&b, 2, literal2.shape(), "param2");
  auto param3 = Parameter(&b, 3, literal3.shape(), "param2");
  Div(Div(param0, param1), Div(param2, param3));

  std::vector<float> expected(values0.size());
  for (int64_t i = 0; i < values0.size(); ++i) {
    expected[i] = (values0[i] / values1[i]) / (values2[i] / values3[i]);
  }

  ComputeAndCompareR1<float>(
      &b, expected, {data0.get(), data1.get(), data2.get(), data3.get()},
      error_spec_);
}

TEST_P(ArrayElementwiseOpTestParamCount, SquareManyValues) {
  const int count = GetParam();
  XlaBuilder builder(TestName());
  std::vector<float> values;
  values.reserve(count);
  for (int i = 0; i < count; ++i) {
    values.push_back(i / static_cast<float>(count));
  }
  auto x = ConstantR1<float>(&builder, values);
  Pow(x, ConstantR0<float>(&builder, 2.0f));

  std::vector<float> expected;
  expected.reserve(values.size());
  for (float value : values) {
    expected.push_back(value * value);
  }

  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SquareIn4D) {
  XlaBuilder builder(TestName());
  Array4D<float> values(2, 2, 2, 2);

  std::vector<float> values_vector;
  std::vector<float> expected_vector;
  const auto num_elements = values.num_elements();
  values_vector.reserve(num_elements);
  expected_vector.reserve(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    values_vector.push_back(static_cast<float>(i) / values.num_elements());
    expected_vector.push_back(values_vector.back() * values_vector.back());
  }
  values.SetValues(values_vector);

  Array4D<float> expected(2, 2, 2, 2, expected_vector);

  auto x = ConstantR4FromArray4D<float>(&builder, values);
  Pow(x, ConstantR0<float>(&builder, 2.0f));

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, SquareIn4DZeroElements) {
  XlaBuilder builder(TestName());
  Array4D<float> values(2, 2, 0, 2);
  Array4D<float> expected(2, 2, 0, 2);

  auto x = ConstantR4FromArray4D<float>(&builder, values);
  Pow(x, ConstantR0<float>(&builder, 2.0f));

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MinF32s) {
  XlaBuilder builder(TestName());
  SetFastMathDisabled(true);
  auto lhs = ConstantR1<float>(&builder, {1.0f, 1.0f, 2.25f, NAN, 6.0f});
  auto rhs = ConstantR1<float>(&builder, {2.0f, -5.0f, 1.0f, 10.0f, NAN});
  Min(lhs, rhs);

  ComputeAndCompareR1<float>(&builder, {1.0f, -5.0f, 1.0f, NAN, NAN}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MinZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {});
  auto rhs = ConstantR1<float>(&builder, {});
  Min(lhs, rhs);
  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MinF64s) {
  XlaBuilder builder(TestName());
  SetFastMathDisabled(true);
  auto lhs = ConstantR1<double>(&builder, {1.0, 1.0, 2.25, NAN, 6.0});
  auto rhs = ConstantR1<double>(&builder, {2.0, -5.0, 1.0, 10.0, NAN});
  Min(lhs, rhs);

  ComputeAndCompareR1<double>(&builder, {1.0, -5.0, 1.0, NAN, NAN}, {},
                              strict_error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxF32s) {
  XlaBuilder builder(TestName());
  SetFastMathDisabled(true);
  auto lhs = ConstantR1<float>(&builder, {1.0f, 1.0f, 2.25f, NAN, 6.0f});
  auto rhs = ConstantR1<float>(&builder, {2.0f, -5.0f, 1.0f, 10.0f, NAN});
  Max(lhs, rhs);

  ComputeAndCompareR1<float>(&builder, {2.0f, 1.0f, 2.25f, NAN, NAN}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest,
           DISABLED_ON_CPU(DefaultMaxF32sNaNPropagation)) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {1.0f, 1.0f, 2.25f, NAN, 6.0f});
  auto rhs = ConstantR1<float>(&builder, {2.0f, -5.0f, 1.0f, 10.0f, NAN});
  Max(lhs, rhs);

  ComputeAndCompareR1<float>(&builder, {2.0f, 1.0f, 2.25f, NAN, NAN}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<float>(&builder, {});
  auto rhs = ConstantR1<float>(&builder, {});
  Max(lhs, rhs);
  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxF64s) {
  XlaBuilder builder(TestName());
  SetFastMathDisabled(true);
  auto lhs = ConstantR1<double>(&builder, {1.0, 1.0, 2.25, NAN, 6.0});
  auto rhs = ConstantR1<double>(&builder, {2.0, -5.0, 1.0, 10.0, NAN});
  Max(lhs, rhs);

  ComputeAndCompareR1<double>(&builder, {2.0, 1.0, 2.25, NAN, NAN}, {},
                              strict_error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  XlaBuilder builder(TestName());
  auto x = ConstantR1<int32_t>(
      &builder, {min, min, min, -1, -1, 0, 0, 0, 1, 1, max, max, max});
  auto y = ConstantR1<int32_t>(
      &builder, {min, max, 0, -10, 0, -1, 0, 1, 0, 10, 0, max, min});
  Max(x, y);

  std::vector<int32_t> expected = {min, max, 0,  -1,  0,   0,  0,
                                   1,   1,   10, max, max, max};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MinS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  XlaBuilder builder(TestName());
  auto x = ConstantR1<int32_t>(
      &builder, {min, min, min, -1, -1, 0, 0, 0, 1, 1, max, max, max});
  auto y = ConstantR1<int32_t>(
      &builder, {min, max, 0, -10, 0, -1, 0, 1, 0, 10, 0, max, min});
  Min(x, y);

  std::vector<int32_t> expected = {min, min, min, -10, -1,  -1, 0,
                                   0,   0,   1,   0,   max, min};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  XlaBuilder builder(TestName());
  auto x = ConstantR1<uint32_t>(&builder, {0, 0, 1, 1, 1, max, max, max});
  auto y = ConstantR1<uint32_t>(&builder, {0, 1, 0, 1, 10, 0, 234234, max});
  Max(x, y);

  std::vector<uint32_t> expected = {0, 1, 1, 1, 10, max, max, max};
  ComputeAndCompareR1<uint32_t>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MinU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  XlaBuilder builder(TestName());
  auto x = ConstantR1<uint32_t>(&builder, {0, 0, 1, 1, 1, max, max, max});
  auto y = ConstantR1<uint32_t>(&builder, {0, 1, 0, 1, 10, 0, 234234, max});
  Min(x, y);

  std::vector<uint32_t> expected = {0, 0, 0, 1, 1, 0, 234234, max};
  ComputeAndCompareR1<uint32_t>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxTenF32s) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {-0.0, 1.0, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0, 9.0});
  auto y = ConstantR1<float>(
      &builder, {-0.0, -1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0, -9.0});
  Max(x, y);

  std::vector<float> expected = {-0.0, 1.0, 2.0, 3.0, 4.0,
                                 5.0,  6.0, 7.0, 8.0, 9.0};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxR1S1AndR1S0F32s) {
  XlaBuilder builder(TestName());
  auto u = ConstantR1<float>(&builder, {3.5});
  auto v = ConstantR1<float>(&builder, {});
  Max(u, v);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxR1S0AndR2S0x2F32s) {
  for (int broadcast_dim : {0, 1}) {
    XlaBuilder builder(TestName());
    auto u = ConstantR1<float>(&builder, {3.5});
    auto v = ConstantR2FromArray2D<float>(&builder, Array2D<float>(0, 2));
    Max(u, v, /*broadcast_dimensions=*/{broadcast_dim});

    ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 2), {}, error_spec_);
  }
}

XLA_TEST_F(ArrayElementwiseOpTest, Max1DAnd2DF32s) {
  XlaBuilder builder(TestName());
  auto v = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  auto m = ConstantR2<float>(&builder,
                             {{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  Max(v, m, /*broadcast_dimensions=*/{1});

  Array2D<float> expected({{2.0f, 3.14f, 4.0f}, {2.25f, 3.0f, 4.0f}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Max1DAnd2DZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto v = ConstantR1<float>(&builder, {});
  auto m = ConstantR2<float>(&builder, {{}, {}});
  Max(v, m, /*broadcast_dimensions=*/{1});

  Array2D<float> expected({{}, {}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Max3DAndScalarS32s) {
  XlaBuilder builder(TestName());
  auto scalar = ConstantR0<int32_t>(&builder, 2);
  Array3D<int32_t> a_3d({{{3, 9, -1}, {2, -10, 3}}, {{-2, 2, 8}, {12, 10, 4}}});
  auto array = ConstantR3FromArray3D<int32_t>(&builder, a_3d);
  Max(array, scalar, /*broadcast_dimensions=*/{});

  Array3D<int32_t> expected({{{3, 9, 2}, {2, 2, 3}}, {{2, 2, 8}, {12, 10, 4}}});
  ComputeAndCompareR3<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, Max3DAndScalarZeroElementS32s) {
  XlaBuilder builder(TestName());
  auto scalar = ConstantR0<int32_t>(&builder, 2);
  Array3D<int32_t> a_3d(2, 0, 3);
  auto array = ConstantR3FromArray3D<int32_t>(&builder, a_3d);
  Max(array, scalar, /*broadcast_dimensions=*/{});

  Array3D<int32_t> expected(2, 0, 3);
  ComputeAndCompareR3<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, Min2DTo1DF32s) {
  XlaBuilder builder(TestName());
  auto m = ConstantR2<float>(&builder,
                             {{-10.4f, 64.0f, 6.0f}, {0.1f, 32.0f, 16.1f}});
  auto v = ConstantR1<float>(&builder, {-10.2f, 16.4f});
  Min(m, v, /*broadcast_dimensions=*/{0});

  Array2D<float> expected({{-10.4f, -10.2f, -10.2f}, {0.1f, 16.4f, 16.1f}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Min2DTo1DZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto m = ConstantR2<float>(&builder, {{}, {}});
  auto v = ConstantR1<float>(&builder, {-10.2f, 16.4f});
  Min(m, v, /*broadcast_dimensions=*/{0});

  Array2D<float> expected({{}, {}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Min2DTo4DF32s) {
  XlaBuilder builder(TestName());
  auto array2d =
      ConstantR2<float>(&builder, {{-12.2f, 64.3f, 6.1f}, {0.0f, 32.2f, 2.5f}});
  auto array4d = ConstantR4FromArray4D<float>(
      &builder, {{{{-12.1f, 32.3f, 6.2f}}, {{0.0f, 32.5f, 3.0f}}},
                 {{{-2.5f, 64.29f, 6.5f}}, {{-0.01f, 32.25f, 2.6f}}}});
  Min(array2d, array4d, /*broadcast_dimensions=*/{1, 3});

  Array4D<float> expected(
      {{{{-12.2f, 32.3f, 6.1f}}, {{0.0f, 32.2f, 2.5f}}},
       {{{-12.2f, 64.29f, 6.1f}}, {{-0.01f, 32.2f, 2.5f}}}});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Min2DTo4DZeroElementF32s) {
  XlaBuilder builder(TestName());
  auto array2d =
      ConstantR2<float>(&builder, {{-12.2f, 64.3f, 6.1f}, {0.0f, 32.2f, 2.5f}});
  Array4D<float> arg(2, 2, 0, 3);
  auto array4d = ConstantR4FromArray4D<float>(&builder, arg);
  Min(array2d, array4d, /*broadcast_dimensions=*/{1, 3});

  Array4D<float> expected(2, 2, 0, 3);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, MinTenS32s) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<int32_t>(&builder, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto y = ConstantR1<int32_t>(&builder, {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
  Min(x, y);

  std::vector<int32_t> expected = {0, 1, 2, 3, 4, 4, 3, 2, 1, 0};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, MaxTenS32s) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<int32_t>(&builder, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto y = ConstantR1<int32_t>(&builder, {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
  Max(x, y);

  std::vector<int32_t> expected = {9, 8, 7, 6, 5, 5, 6, 7, 8, 9};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, RemTwoConstantS32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {-3, 26, 2, -1, 1});
  auto b = ConstantR1<int32_t>(&builder, {10, 5, 1, 10, -10});
  Rem(a, b);

  ComputeAndCompareR1<int32_t>(&builder, {-3, 1, 0, -1, 1}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, NonNanClampF32) {
  XlaBuilder builder(TestName());
  auto minimum = ConstantR1<float>(&builder, {1.0f, -6.5f, 1.0f, 2.25f, 0.0f});
  auto argument =
      ConstantR1<float>(&builder, {2.0f, 10.0f, -5.0f, 1.0f, 10.0f});
  auto maximum = ConstantR1<float>(&builder, {3.0f, 0.5f, 25.5f, 5.0f, 123.0});
  Clamp(minimum, argument, maximum);

  ComputeAndCompareR1<float>(&builder, {2.0f, 0.5f, 1.0f, 2.25f, 10.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, ClampF32) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto minimum = ConstantR1<float>(&builder, {1.0f, -6.5f, 1.0f, 2.25f, NAN});
  auto argument =
      ConstantR1<float>(&builder, {2.0f, 10.0f, -5.0f, 1.0f, 10.0f});
  auto maximum = ConstantR1<float>(&builder, {3.0f, 0.5f, 25.5f, NAN, 123.0f});
  Clamp(minimum, argument, maximum);

  ComputeAndCompareR1<float>(&builder, {2.0f, 0.5f, 1.0f, NAN, NAN}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, ClampF32Scalar) {
  XlaBuilder builder(TestName());
  auto minimum = ConstantR0<float>(&builder, 0.0f);
  auto argument = ConstantR1<float>(&builder, {2.0f, 10.0f, -5.0f, 1.0f, 4.0f});
  auto maximum = ConstantR0<float>(&builder, 5.0f);
  Clamp(minimum, argument, maximum);

  ComputeAndCompareR1<float>(&builder, {2.0f, 5.0f, 0.0f, 1.0f, 4.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, ClampF32ScalarVector) {
  XlaBuilder builder(TestName());
  auto min_scalar = ConstantR0<float>(&builder, 0.0f);
  auto min_vector =
      ConstantR1<float>(&builder, {1.0f, -6.5f, 1.0f, 2.25f, 0.0f});
  auto arg_vector =
      ConstantR1<float>(&builder, {2.0f, 10.0f, -5.0f, 1.0f, 4.0f});
  auto max_scalar = ConstantR0<float>(&builder, 3.0f);
  auto max_vector =
      ConstantR1<float>(&builder, {3.0f, 0.5f, 25.5f, 5.0f, 123.0});
  // Perform clamp with broadcasted scalar and vector.
  Add(Add(Clamp(min_vector, arg_vector, max_scalar),
          Clamp(min_scalar, arg_vector, max_vector)),
      Add(Clamp(min_vector, arg_vector, max_vector),
          Clamp(min_scalar, arg_vector, max_scalar)));

  ComputeAndCompareR1<float>(&builder, {8.0f, 7.0f, 2.0f, 6.5f, 14.0f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, ClampS32Vector) {
  XlaBuilder builder(TestName());
  auto min_vector = ConstantR1<int32_t>(&builder, {1, -6, 1, 2, 0, -5});
  auto arg_vector = ConstantR1<int32_t>(&builder, {2, 10, -5, 1, 4, 10});
  auto max_vector = ConstantR1<int32_t>(&builder, {3, 0, 25, 5, 123, -1});
  Clamp(min_vector, arg_vector, max_vector);

  ComputeAndCompareR1<int32_t>(&builder, {2, 0, 1, 2, 4, -1}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ClampS32ScalarVector) {
  XlaBuilder builder(TestName());
  auto min_scalar = ConstantR0<int32_t>(&builder, 0);
  auto min_vector = ConstantR1<int32_t>(&builder, {1, -6, 1, 2, 0});
  auto arg_vector = ConstantR1<int32_t>(&builder, {2, 10, -5, 1, 4});
  auto max_scalar = ConstantR0<int32_t>(&builder, 3);
  auto max_vector = ConstantR1<int32_t>(&builder, {3, 1, 25, 5, 123});
  // Perform clamp with broadcasted scalar and vector.
  Add(Add(Clamp(min_vector, arg_vector, max_scalar),
          Clamp(min_scalar, arg_vector, max_vector)),
      Add(Clamp(min_vector, arg_vector, max_vector),
          Clamp(min_scalar, arg_vector, max_scalar)));

  ComputeAndCompareR1<int32_t>(&builder, {8, 8, 2, 6, 14}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ClampU32Vector) {
  XlaBuilder builder(TestName());
  auto min_vector = ConstantR1<uint32_t>(&builder, {1, 2, 1, 2, 0, ~0u - 4});
  auto arg_vector = ConstantR1<uint32_t>(&builder, {2, 10, 5, 1, 4, 10});
  auto max_vector = ConstantR1<uint32_t>(&builder, {3, 5, 25, 5, 123, ~0u});
  Clamp(min_vector, arg_vector, max_vector);

  ComputeAndCompareR1<uint32_t>(&builder, {2, 5, 5, 2, 4, ~0u - 4}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ClampU32ScalarVector) {
  XlaBuilder builder(TestName());
  auto min_scalar = ConstantR0<uint32_t>(&builder, 0);
  auto min_vector = ConstantR1<uint32_t>(&builder, {1, 0, 1, 2, 0});
  auto arg_vector = ConstantR1<uint32_t>(&builder, {2, 10, 0, 1, 4});
  auto max_scalar = ConstantR0<uint32_t>(&builder, 3);
  auto max_vector = ConstantR1<uint32_t>(&builder, {3, 1, 25, 5, 123});
  // Perform clamp with broadcasted scalar and vector.
  Add(Add(Clamp(min_vector, arg_vector, max_scalar),
          Clamp(min_scalar, arg_vector, max_vector)),
      Add(Clamp(min_vector, arg_vector, max_vector),
          Clamp(min_scalar, arg_vector, max_scalar)));

  ComputeAndCompareR1<uint32_t>(&builder, {8, 8, 2, 6, 14}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoParametersF32s) {
  XlaBuilder builder(TestName());

  Literal param0_literal =
      LiteralUtil::CreateR1<float>({1.1f, 2.2f, 3.3f, 5.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(param0_literal).value();

  Literal param1_literal =
      LiteralUtil::CreateR1<float>({7.2f, 2.3f, 3.4f, 5.6f});
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(param1_literal).value();

  auto p0 = Parameter(&builder, 0, param0_literal.shape(), "param0");
  auto p1 = Parameter(&builder, 1, param1_literal.shape(), "param1");
  Add(p0, p1);

  ComputeAndCompareR1<float>(&builder, {8.3f, 4.5f, 6.7f, 11.1f},
                             {param0_data.get(), param1_data.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoParametersZeroElementF32s) {
  XlaBuilder builder(TestName());

  Literal param0_literal =
      LiteralUtil::CreateR3FromArray3D<float>(Array3D<float>(0, 7, 0));
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(param0_literal).value();

  Literal param1_literal =
      LiteralUtil::CreateR3FromArray3D<float>(Array3D<float>(0, 7, 0));
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(param1_literal).value();

  auto p0 = Parameter(&builder, 0, param0_literal.shape(), "param0");
  auto p1 = Parameter(&builder, 1, param1_literal.shape(), "param1");
  Add(p0, p1);

  Array3D<float> expected(0, 7, 0);
  ComputeAndCompareR3<float>(
      &builder, expected, {param0_data.get(), param1_data.get()}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddParameterToConstantF32s) {
  XlaBuilder builder(TestName());

  Literal param0_literal =
      LiteralUtil::CreateR1<float>({1.1f, 2.2f, 3.3f, 5.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(param0_literal).value();

  auto a = ConstantR1<float>(&builder, {1.1f, 2.2f, 3.3f, 4.4f});
  auto p = Parameter(&builder, 0, param0_literal.shape(), "param0");
  Add(a, p);

  ComputeAndCompareR1<float>(&builder, {2.2f, 4.4f, 6.6f, 9.9f},
                             {param0_data.get()}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, CosF32s) {
  XlaBuilder builder(TestName());
  // Test a variety of values of both signs that stress trigonometric range
  // reduction, as well as numbers that fall in different quadrants.
  // -2.19993846e+10 is a hard case because this number is so close to a
  // multiple of pi/2 that the leading 31 bits cancel in the Payne-Hanek
  // algorithm, leading to catastrophic loss of (relative) accuracy unless
  // 64-bit fixed pont arithmeic is used.
  //
  // Also test IEEE special values {+/-0, +/-Inf, NaN}; for the latter two
  // Cos(x) should return NaN.
  auto kInf = std::numeric_limits<float>::infinity();
  auto kQNaN = std::numeric_limits<float>::quiet_NaN();
  auto a = ConstantR1<float>(
      &builder,
      {-1.9938988e-28, 1.9938988e-28, -1e20f, 1e20f, -2.3564024f, -3.14159f,
       3.14159f, -0.0f, 0.0f, -1.570796f, 1.570796f, -0.78539f, 0.78539f,
       -2.19993846e+10, -1.70141183e+38, -kInf, kInf, kQNaN});
  Cos(a);

  // This error spec corresponds to 1 ULP max relative error.
  ComputeAndCompare(&builder, {},
                    ErrorSpec(0, std::numeric_limits<float>::epsilon()));
}

XLA_TEST_F(ArrayElementwiseOpTest, SinF32s) {
  XlaBuilder builder(TestName());
  // Test a variety of values of both signs that stress trigonometric range
  // reduction, as well as numbers that fall in different quadrants.
  // -2.19993846e+10 is a hard case because this number is so close to a
  // multiple of pi/2 that the leading 31 bits cancel in the Payne-Hanek
  // algorithm, leading to catastrophic loss of (relative) accuracy unless
  // 64-bit fixed pont arithmeic is used.
  //
  // Also test IEEE special values {+/-0, +/-Inf, NaN}; for the latter two
  // Sin(x) should return NaN.
  auto kInf = std::numeric_limits<float>::infinity();
  auto kQNaN = std::numeric_limits<float>::quiet_NaN();
  auto a = ConstantR1<float>(
      &builder,
      {-1.9938988e-28, 1.9938988e-28, -1e20f, 1e20f, -2.3564024f, -3.14159f,
       3.14159f, -0.0f, 0.0f, -1.570796f, 1.570796f, -0.78539f, 0.78539f,
       -2.19993846e+10, -1.70141183e+38, -kInf, kInf, kQNaN});
  Sin(a);

  // This error spec corresponds to 1 ULP max relative error.
  ComputeAndCompare(&builder, {},
                    ErrorSpec(0, std::numeric_limits<float>::epsilon()));
}

XLA_TEST_F(ArrayElementwiseOpTest, RealF64s) {
  XlaBuilder builder(TestName());
  std::vector<double> xs = {3.14159f, 0.0f, 1.570796f, -0.78539f};
  auto a = ConstantR1<double>(&builder, xs);
  Real(a);
  ComputeAndCompareR1<double>(&builder, xs, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ImagF64s) {
  XlaBuilder builder(TestName());
  std::vector<double> xs = {3.14159, 0.0, 1.570796, -0.78539};
  auto a = ConstantR1<double>(&builder, xs);
  Imag(a);
  ComputeAndCompareR1<double>(&builder, {0., 0., 0., 0.}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, Atan2F32s) {
  XlaBuilder builder(TestName());
  auto kInf = std::numeric_limits<float>::infinity();
  std::vector<float> ys;
  std::vector<float> xs;
  const auto _ys = {+0.0f, -0.0f, kInf, -kInf, 5.0f, -3.0f, 2.0f, -8.0f, 1.0f};
  const auto _xs = {+0.0f, -0.0f, kInf, -kInf, 6.0f, -4.0f, 2.0f, 8.0f};
  const auto n = _ys.size() * _xs.size();
  ys.reserve(n);
  xs.reserve(n);
  for (auto y : _ys) {
    for (auto x : _xs) {
      ys.push_back(y);
      xs.push_back(x);
    }
  }
  auto y = ConstantR1<float>(&builder, ys);
  auto x = ConstantR1<float>(&builder, xs);
  Atan2(y, x);

  ComputeAndCompare(&builder, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Atan2F64s) {
  XlaBuilder builder(TestName());
  auto kInf = std::numeric_limits<double>::infinity();
  auto qnan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> ys;
  std::vector<double> xs;
  std::tie(xs, ys) =
      AllSignedPairs<double>({0.0, 0.1, 0.9, 1.0, 1.1, M_PI, 1e6, qnan, kInf});
  auto y = ConstantR1<double>(&builder, ys);
  auto x = ConstantR1<double>(&builder, xs);
  Atan2(y, x);

  ComputeAndCompare(&builder, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Atan2C64s) {
  XlaBuilder builder(TestName());
  auto kInf = std::numeric_limits<float>::infinity();
  std::vector<std::complex<float>> ys;
  std::vector<std::complex<float>> xs;
  const auto _ys = {+0.0f, -0.0f, kInf, -kInf, 5.0f, -3.0f, 2.0f, -8.0f, 1.0f};
  const auto _xs = {+0.0f, -0.0f, kInf, -kInf, 6.0f, -4.0f, 2.0f, 8.0f};
  const auto n = _ys.size() * _xs.size();
  ys.reserve(n);
  xs.reserve(n);
  for (auto y : _ys) {
    for (auto x : _xs) {
      ys.push_back(y);
      xs.push_back(x);
    }
  }
  auto y = ConstantR1<std::complex<float>>(&builder, ys);
  auto x = ConstantR1<std::complex<float>>(&builder, xs);
  Atan2(y, x);

  ComputeAndCompare(&builder, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, TanhF32s) {
  XlaBuilder builder(TestName());
  auto kInf = std::numeric_limits<float>::infinity();
  auto kNaN = std::numeric_limits<float>::quiet_NaN();
  auto a = ConstantR1<float>(
      &builder, {-kInf, -2.5f, 3.14f, -0.0f, 0.0f, 2.25f, kInf, kNaN});

  Tanh(a);

  // Tanh is relatively inaccurate because we use a rational approximation.
  ErrorSpec error_spec{165 * kEpsF32, 165 * kEpsF32};
  ComputeAndCompare(&builder, {}, error_spec);
}

XLA_TEST_F(ArrayElementwiseOpTest, TanhF64s) {
  XlaBuilder builder(TestName());
  auto kInf = std::numeric_limits<double>::infinity();
  auto kNaN = std::numeric_limits<double>::quiet_NaN();
  auto a = ConstantR1<double>(&builder,
                              {-kInf, -2.5, 3.14, -0.0, 0.0, 2.25, kInf, kNaN});

  Tanh(a);

  // The error spec is unusually high here to account for the fact that we
  // use a rational interpolant to approximate tanh.
  ErrorSpec error_spec{165 * kEpsF64, 165 * kEpsF64};
  ComputeAndCompare(&builder, {}, error_spec);
}

XLA_TEST_F(ArrayElementwiseOpTest, TanhF32sVector) {
  // This is like the test ArrayElementwiseOpTest.TanhF32s above, except that
  // the input tensor is large enough to exercise the vectorized tanh
  // implementation on XLA CPU.
  XlaBuilder builder(TestName());
  auto input_literal = ConstantR1<float>(
      &builder,
      {1.02,  -0.32, 0.85,  0.90,  1.23,  -0.91, -0.49, 0.80,  -0.67, 0.16,
       -0.07, 0.39,  -0.41, 0.04,  1.36,  1.25,  0.41,  0.65,  -1.08, 0.32,
       -1.45, -0.77, -1.09, 0.91,  -1.03, -0.30, -1.11, -1.17, 1.50,  -0.85,
       0.04,  1.02,  0.34,  -0.61, 0.41,  0.07,  -0.02, 1.42,  -0.62, 0.81,
       0.08,  0.81,  -0.30, 1.17,  -0.65, -0.44, 0.92,  1.26,  -1.29, 1.35,
       0.08,  -1.24, -0.92, 0.49,  1.17,  -0.45, -1.31, -1.44, -0.13, -1.31,
       -0.79, 1.41,  1.21,  1.05});

  Tanh(input_literal);

  // The error spec is unusually high here to account for the fact that we
  // use a rational interpolant to approximate tanh.
  ErrorSpec error_spec{440 * kEpsF32, 440 * kEpsF32};
  ComputeAndCompare(&builder, {}, error_spec);
}

XLA_TEST_F(ArrayElementwiseOpTest, ExpF32sVector) {
  // The input tensor is large enough to exercise the vectorized exp
  // implementation on XLA CPU.
  XlaBuilder builder(TestName());

  // Just to help make sense of the scales here -- exp(89) saturates float32 and
  // exp(-10) is smaller than our error spec.
  Literal input_literal = LiteralUtil::CreateR1<float>(
      {1.02,   -0.32,  0.85,   0.9,    1.23,   -0.91,  -0.49, 0.8,    -1.31,
       -1.44,  -0.13,  -1.31,  -0.79,  1.41,   1.21,   1.05,  -195.6, -194.5,
       -193.4, -192.3, -191.2, -190.1, -189.0, -187.9, -19.6, -18.5,  -17.4,
       -16.3,  -15.2,  -14.1,  -13.0,  -11.9,  -10.8,  -9.7,  -8.6,   -7.5,
       -6.4,   -5.3,   -4.2,   -3.1,   -2.0,   -0.9,   0.2,   1.3,    2.4,
       3.5,    4.6,    5.7,    6.8,    7.9,    9.0,    10.1,  11.2,   12.3,
       13.4,   14.5,   15.6,   16.7,   17.8,   18.9,   20.0,  21.1,   22.2,
       23.3,   24.4,   25.5,   26.6,   27.7,   28.8,   29.9,  31.0,   32.1,
       68.4,   69.5,   70.6,   71.7,   72.8,   73.9,   75.0,  76.1,   77.2,
       78.3,   79.4,   80.5,   81.6,   82.7,   83.8,   84.9,  85.2,   86.3,
       86.4,   86.5,   87.6,   87.7,   87.8,   87.9});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client_->TransferToServer(input_literal));

  auto input = Parameter(&builder, 0, input_literal.shape(), "input");
  Exp(input);

  std::vector<float> expected_result;
  int64_t input_size = input_literal.shape().dimensions(0);
  expected_result.reserve(input_size);
  for (int64_t i = 0; i < input_size; i++) {
    expected_result.push_back(std::exp(input_literal.Get<float>({i})));
  }

  ComputeAndCompareR1<float>(&builder, expected_result, {input_data.get()},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, LogF32sVector) {
  // The input tensor is large enough to exercise the vectorized exp
  // implementation on XLA CPU.
  XlaBuilder builder(TestName());

  Literal input_literal = LiteralUtil::CreateR1<float>(
      {-1.29,    -1.41,    -1.25,    -13.5,    -11.7,    -17.9,    -198,
       -167,     1.29,     1.41,     1.25,     13.5,     11.7,     17.9,
       198,      167,      1.27e+03, 1.33e+03, 1.74e+03, 1.6e+04,  1.84e+04,
       1.74e+04, 1.89e+05, 1.9e+05,  1.93e+06, 1.98e+06, 1.65e+06, 1.97e+07,
       1.66e+07, 1e+07,    1.98e+08, 1.96e+08, 1.64e+09, 1.58e+09, 1.64e+09,
       1.44e+10, 1.5e+10,  1.99e+10, 1.17e+11, 1.08e+11, 1.08e+12, 1.38e+12,
       1.4e+12,  1.03e+13, 1.6e+13,  1.99e+13, 1.26e+14, 1.51e+14, 1.33e+15,
       1.41e+15, 1.63e+15, 1.39e+16, 1.21e+16, 1.27e+16, 1.28e+17, 1.62e+17,
       2e+18,    1.96e+18, 1.81e+18, 1.99e+19, 1.86e+19, 1.61e+19, 1.71e+20,
       1.47e+20, 1.83e+21, 1.33e+21, 1.3e+21,  1.35e+22, 1.84e+22, 1.02e+22,
       1.81e+23, 1.02e+23, 1.89e+24, 1.49e+24, 1.08e+24, 1.95e+25, 1.1e+25,
       1.62e+25, 1.2e+26,  1.41e+26, 1.93e+27, 1.66e+27, 1.62e+27, 1.05e+28,
       1.5e+28,  1.79e+28, 1.36e+29, 1.95e+29, 1.5e+30,  1.81e+30, 1.34e+30,
       1.7e+31,  1.44e+31, 1.1e+31,  1.4e+32,  1.67e+32, 1.96e+33, 1.11e+33,
       1.19e+33, 1.61e+34, 1.05e+34, 1.88e+34, 1.67e+35, 1.7e+35});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client_->TransferToServer(input_literal));

  auto input = Parameter(&builder, 0, input_literal.shape(), "input");
  Log(input);

  std::vector<float> expected_result;
  int64_t input_size = input_literal.shape().dimensions(0);
  expected_result.reserve(input_size);
  for (int64_t i = 0; i < input_size; i++) {
    expected_result.push_back(std::log(input_literal.Get<float>({i})));
  }

  // Log2 is very inaccurate onon some platforms.
  ErrorSpec error_spec(1000 * kEpsF32, 1000 * kEpsF32);
  ComputeAndCompareR1<float>(&builder, expected_result, {input_data.get()},
                             error_spec);
}

XLA_TEST_F(ArrayElementwiseOpTest, ClzU32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(
      &builder, {0, 1, 0x10, 0x10000, 0x700000, 0x12345678, 0xF2345678});
  Clz(a);

  ComputeAndCompareR1<uint32_t>(&builder, {32, 31, 27, 15, 9, 3, 0}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, ClzS64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int64_t>(&builder,
                               {0, 1, 0x80000000, 0x7FFFFFFFF2345678ul, -1});
  Clz(a);

  ComputeAndCompareR1<int64_t>(&builder, {64, 63, 32, 1, 0}, {});
}

XLA_TEST_F(ArrayElementwiseOpTest, AddChainFoldLeft) {
  // a ------ (add) --------- (add)
  //         /               /
  // b -----/               /
  // c---------------------/
  XlaBuilder builder(TestName());

  auto a = ConstantR1<float>(&builder, {1.1f, 2.2f, 3.3f, 4.4f});
  auto b = ConstantR1<float>(&builder, {2.1f, 3.2f, 4.3f, 5.4f});
  auto c = ConstantR1<float>(&builder, {-3.3f, -15.5f, -7.7f, -29.9f});

  auto add = Add(a, b);
  Add(add, c);

  ComputeAndCompareR1<float>(&builder, {-0.1f, -10.1f, -0.1f, -20.1f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddChainFoldRight) {
  // b ------ (add) --------- (add)
  //         /               /
  // c -----/               /
  // a---------------------/
  XlaBuilder builder(TestName());

  auto a = ConstantR1<float>(&builder, {91.1f, 2.2f, 3.3f, 4.4f});
  auto b = ConstantR1<float>(&builder, {2.1f, 3.2f, 4.3f, 5.4f});
  auto c = ConstantR1<float>(&builder, {-3.3f, -15.5f, -7.7f, -29.9f});

  auto add = Add(b, c);
  Add(a, add);

  ComputeAndCompareR1<float>(&builder, {89.9f, -10.1f, -0.1f, -20.1f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddWithNeg) {
  // a ----- (neg) ----- (add)
  //                    /
  // b ----- (neg) ----/
  XlaBuilder builder(TestName());

  auto a = ConstantR1<float>(&builder, {91.1f, 2.2f, 3.3f, 4.4f});
  auto b = ConstantR1<float>(&builder, {2.1f, 3.2f, 4.3f, 5.4f});

  auto neg_a = Neg(a);
  auto neg_b = Neg(b);
  Add(neg_a, neg_b);

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
  XlaBuilder builder(TestName());

  auto a = ConstantR1<float>(&builder, {91.1f, 2.2f, 3.3f, 4.4f});
  auto b = ConstantR1<float>(&builder, {2.1f, 3.2f, 4.3f, 5.4f});
  auto c = ConstantR1<float>(&builder, {-3.3f, -15.5f, -7.7f, -29.9f});
  auto d = ConstantR1<float>(&builder, {-19.0f, 10.0f, -40.0f, 20.2f});

  auto add_ab = Add(a, b);
  auto add_cd = Add(c, d);
  Add(add_ab, add_cd);

  ComputeAndCompareR1<float>(&builder, {70.9f, -0.1f, -40.1f, 0.1f}, {},
                             error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, 2DBinaryOpF32s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<float>(&builder,
                             {{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto b = ConstantR2<float>(&builder,
                             {{-1.5f, 8.14f, 42.0}, {-1.0f, -4.0f, 5.55f}});
  Add(a, b);

  Array2D<float> expected_array(
      {{-4.0f, 11.28f, 43.0f}, {1.25f, -14.0f, 8.88f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, ScalarPlus2DF32) {
  // Add a scalar + matrix.
  XlaBuilder builder(TestName());
  auto a = ConstantR2<float>(&builder,
                             {{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto scalar = ConstantR0<float>(&builder, 3.0f);
  Add(scalar, a);

  Array2D<float> expected_array({{0.5f, 6.14f, 4.0f}, {5.25f, -7.0f, 6.33f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, 2DPlusScalarF32) {
  // Add a matrix + scalar.
  XlaBuilder builder(TestName());
  auto a = ConstantR2<float>(&builder,
                             {{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto scalar = ConstantR0<float>(&builder, 3.0f);
  Add(a, scalar);

  Array2D<float> expected_array({{0.5f, 6.14f, 4.0f}, {5.25f, -7.0f, 6.33f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add1DTo2DF32) {
  // Test simple broadcasting of a R1F32 over R2F32. The vector's size matches
  // only dim 0 of the matrix.
  XlaBuilder builder(TestName());
  auto v = ConstantR1<float>(&builder, {20.0f, 40.0f, 60.0f});
  // clang-format off
  auto m = ConstantR2<float>(&builder, {
    {-2.5f, 3.14f, 1.0f},
    {2.25f, -10.0f, 3.33f}});
  // clang-format on
  Add(v, m, /*broadcast_dimensions=*/{1});
  Array2D<float> expected_array(
      {{17.5f, 43.14f, 61.0f}, {22.25f, 30.0f, 63.33f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Eq) {
  // Test broadcasting in Eq comparison.
  XlaBuilder builder(TestName());
  auto v = ConstantR1<int32_t>(&builder, {42, 73});
  auto m = ConstantR2<int32_t>(&builder, {{42, 73}, {42, 52}});

  // This test exercises both possible broadcast dimensions for a vector/matrix
  // comparison.
  auto cmp_dim_0 = Eq(v, m, /*broadcast_dimensions=*/{1});
  auto cmp_dim_1 = Eq(v, m, /*broadcast_dimensions=*/{0});
  Tuple(&builder, {cmp_dim_0, cmp_dim_1});

  auto expected = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR2<bool>({{true, true}, {true, false}}),
       LiteralUtil::CreateR2<bool>({{true, false}, {false, false}})});
  ComputeAndCompareTuple(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Ne) {
  // Test broadcasting in Ne comparison.
  XlaBuilder builder(TestName());
  auto v = ConstantR1<int32_t>(&builder, {42, 73});
  auto m = ConstantR2<int32_t>(&builder, {{42, 73}, {42, 52}});
  Ne(v, m, /*broadcast_dimensions=*/{1});

  const std::string expected = R"(pred[2,2] {
  { 0, 0 },
  { 0, 1 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Ge) {
  // Test broadcasting in Ge comparison.
  XlaBuilder builder(TestName());
  auto v = ConstantR1<int32_t>(&builder, {1, 2, 3, 4});
  auto m = ConstantR2<int32_t>(&builder, {{1, 0, 5, 6}, {42, 52, 10, 4}});
  Ge(v, m, /*broadcast_dimensions=*/{1});

  const std::string expected = R"(pred[2,4] {
  { 1, 1, 0, 0 },
  { 0, 0, 0, 1 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Gt) {
  // Test broadcasting in Gt comparison.
  XlaBuilder builder(TestName());
  auto v = ConstantR1<int32_t>(&builder, {1, 2, 3, 4});
  auto m = ConstantR2<int32_t>(&builder, {{1, 0, 5, 6}, {42, 52, 10, 4}});
  Gt(v, m, /*broadcast_dimensions=*/{1});

  const std::string expected = R"(pred[2,4] {
  { 0, 1, 0, 0 },
  { 0, 0, 0, 0 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Le) {
  // Test broadcasting in Le comparison.
  XlaBuilder builder(TestName());
  auto v = ConstantR1<int32_t>(&builder, {1, 2, 3, 4});
  auto m = ConstantR2<int32_t>(&builder, {{1, 0, 5, 6}, {42, 52, 10, 4}});
  Le(v, m, /*broadcast_dimensions=*/{1});

  const std::string expected = R"(pred[2,4] {
  { 1, 0, 1, 1 },
  { 1, 1, 1, 1 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, Compare1DTo2DS32Lt) {
  // Test broadcasting in Lt comparison.
  XlaBuilder builder(TestName());
  auto v = ConstantR1<int32_t>(&builder, {1, 2, 3, 4});
  auto m = ConstantR2<int32_t>(&builder, {{1, 0, 5, 6}, {42, 52, 10, 4}});
  Lt(v, m, /*broadcast_dimensions=*/{1});

  const std::string expected = R"(pred[2,4] {
  { 0, 0, 1, 1 },
  { 1, 1, 1, 0 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, Mul2Dby1DF32) {
  // Test simple broadcasting of a R1F32 over R2F32 when the order of binary op
  // arguments is reversed.
  XlaBuilder builder(TestName());
  auto m =
      ConstantR2<float>(&builder, {{1.5f, 2.5f, 3.5f}, {4.5f, 5.5f, 6.5f}});
  auto v = ConstantR1<float>(&builder, {2.0f, 4.0f, 6.0f});
  Mul(m, v, /*broadcast_dimensions=*/{1});
  Array2D<float> expected_array({{3.0f, 10.0f, 21.0f}, {9.0f, 22.0f, 39.0f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add2DTo2DWithDegenerateDim1) {
  // Tests broadcasting for arrays with degenerate (size == 1) dimensions.
  XlaBuilder builder(TestName());
  // m's shape in XLA notation is {3, 2}
  // md's shape in XLA notation is {3, 1}
  // The result has shape {3, 2}, where md is broadcast over m
  auto m = ConstantR2<float>(&builder,
                             {{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto md = ConstantR2<float>(&builder, {{10.0f, 20.0f, 30.0f}});
  Add(m, md);
  Array2D<float> expected_array(
      {{7.5f, 23.14f, 31.0f}, {12.25f, 10.0f, 33.33f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add2DTo2DWithDegenerateDim0) {
  // Tests broadcasting for arrays with degenerate (size == 1) dimensions.
  XlaBuilder builder(TestName());
  // m's shape in XLA notation is {3, 2}
  // md's shape in XLA notation is {1, 2}
  // The result has shape {3, 2}, where md is broadcast over m
  auto m = ConstantR2<float>(&builder,
                             {{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto md = ConstantR2<float>(&builder, {{10.0f}, {20.0f}});
  Add(m, md);
  Array2D<float> expected_array(
      {{7.5f, 13.14f, 11.0f}, {22.25f, 10.0f, 23.33f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add2DsWithDegenerateDimsOuterProduct) {
  // Tests broadcasting for two degenerate arrays. This kind of broadcasting
  // effectively creates an "outer product" operation.
  // This is taken from the Numpy docs example at:
  // http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html
  XlaBuilder builder(TestName());
  // a's shape in XLA notation is {1, 4}
  // b's shape in XLA notation is {3, 1}
  // The result has shape {3, 4}.
  auto a = ConstantR2<float>(&builder, {{0.0f}, {10.0f}, {20.0f}, {30.0f}});
  auto b = ConstantR2<float>(&builder, {{1.0f, 2.0f, 3.0f}});
  Add(a, b);
  Array2D<float> expected_array({{1.0f, 2.0f, 3.0f},
                                 {11.0f, 12.0f, 13.0f},
                                 {21.0f, 22.0f, 23.0f},
                                 {31.0f, 32.0f, 33.0f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add1DTo2DF32TwoWaysOver1) {
  // Add together a (2,2) array and a (2) array, using dimension 0 for
  // broadcasting (though there are two ways to broadcast these shapes).
  XlaBuilder builder(TestName());
  auto v = ConstantR1<float>(&builder, {20.0f, 40.0f});
  auto m = ConstantR2<float>(&builder, {{10.0f, 50.0f}, {77.0f, 88.0f}});
  Add(v, m, /*broadcast_dimensions=*/{1});
  Array2D<float> expected_array({{30.0f, 90.0f}, {97.0f, 128.0f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add1DTo2DF32TwoWaysOver0) {
  // Add together a (2,2) array and a (2) array, using dimension 1 for
  // broadcasting (though there are two ways to broadcast these shapes).
  XlaBuilder builder(TestName());
  auto v = ConstantR1<float>(&builder, {20.0f, 40.0f});
  auto m = ConstantR2<float>(&builder, {{10.0f, 50.0f}, {77.0f, 88.0f}});
  Add(v, m, /*broadcast_dimensions=*/{0});
  Array2D<float> expected_array({{30.0f, 70.0f}, {117.0f, 128.0f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, 3DBinaryOpF32s) {
  // Binary add of two R3s together
  XlaBuilder builder(TestName());
  Array3D<float> a_3d({{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                       {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}});
  auto a = ConstantR3FromArray3D<float>(&builder, a_3d);

  Array3D<float> b_3d({{{2.0f, 4.0f}, {6.0f, 8.0f}, {10.0f, 12.0f}},
                       {{14.0f, 16.0f}, {18.0f, 20.0f}, {22.0f, 24.0f}}});
  auto b = ConstantR3FromArray3D<float>(&builder, b_3d);
  Add(a, b);

  Array3D<float> expected_3d(
      {{{3.0f, 6.0f}, {9.0f, 12.0f}, {15.0f, 18.0f}},
       {{21.0f, 24.0f}, {27.0f, 30.0f}, {33.0f, 36.0f}}});
  ComputeAndCompareR3<float>(&builder, expected_3d, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add1DTo3DTwoWaysOver2) {
  // Add together a (2, 3, 2) array with a (2) array, using dimension 0 for
  // broadcasting (though there are two ways to broadcast these shapes).
  XlaBuilder builder(TestName());
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
  auto a = ConstantR3FromArray3D<float>(&builder, a_3d);
  auto v = ConstantR1<float>(&builder, {10.0f, 20.0f});
  Add(a, v, /*broadcast_dimensions=*/{2});

  Array3D<float> expected_3d(
      {{{11.0f, 22.0f}, {13.0f, 24.0f}, {15.0f, 26.0f}},
       {{17.0f, 28.0f}, {19.0f, 30.0f}, {21.0f, 32.0f}}});
  ComputeAndCompareR3<float>(&builder, expected_3d, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, Add1DTo3DTwoWaysOver0) {
  // Add together a (2, 3, 2) array with a (2) array, using dimension 2 for
  // broadcasting (though there are two ways to broadcast these shapes).
  XlaBuilder builder(TestName());
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
  auto a = ConstantR3FromArray3D<float>(&builder, a_3d);
  auto v = ConstantR1<float>(&builder, {10.0f, 20.0f});
  Add(a, v, /*broadcast_dimensions=*/{0});

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
  XlaBuilder builder(TestName());
  // clang-format off
  Array3D<float> a_3d({
    {{1.0f, 2.0f},
     {3.0f, 4.0f},
     {5.0f, 6.0f}},
    {{7.0f, 8.0f},
     {9.0f, 10.0f},
     {11.0f, 12.0f}},
  });
  auto a = ConstantR3FromArray3D<float>(&builder, a_3d);
  auto m = ConstantR2<float>(&builder, {
    {10.0f, 20.0f, 30.0f},
    {40.0f, 50.0f, 60.0f},
  });
  Add(a, m, /*broadcast_dimensions=*/{0, 1});

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
  XlaBuilder builder(TestName());
  Array3D<float> a_3d({{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
                       {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}});
  auto a = ConstantR3FromArray3D<float>(&builder, a_3d);

  Array3D<float> b_3d({{{7.0f, 1.0f}, {3.0f, 10.0f}, {15.0f, 6.0f}}});
  auto b = ConstantR3FromArray3D<float>(&builder, b_3d);

  Gt(a, b);

  Array3D<int> expected_3d(
      {{{0, 1}, {0, 0}, {0, 0}}, {{0, 1}, {1, 0}, {0, 1}}});
  const std::string expected = R"(pred[2,3,2] {
{
  { 0, 1 },
  { 0, 0 },
  { 0, 0 }
},
{
  { 0, 1 },
  { 1, 0 },
  { 0, 1 }
}
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

XLA_TEST_F(ArrayElementwiseOpTest, 4DBinaryOpF32s) {
  XlaBuilder builder(TestName());

  std::unique_ptr<Array4D<float>> operand_a_4d(new Array4D<float>(2, 3, 4, 5));
  std::unique_ptr<Array4D<float>> operand_b_4d(new Array4D<float>(2, 3, 4, 5));
  std::unique_ptr<Array4D<float>> expected_4d(new Array4D<float>(2, 3, 4, 5));
  float value = 0.0;
  for (int64_t p = 0; p < 2; ++p) {
    for (int64_t z = 0; z < 3; ++z) {
      for (int64_t y = 0; y < 4; ++y) {
        for (int64_t x = 0; x < 5; ++x) {
          (*operand_a_4d)(p, z, y, x) = value;
          (*operand_b_4d)(p, z, y, x) = 2.0 * value;
          (*expected_4d)(p, z, y, x) = 3.0 * value;
          value += 0.1;
        }
      }
    }
  }

  auto a = ConstantR4FromArray4D<float>(&builder, *operand_a_4d);
  auto b = ConstantR4FromArray4D<float>(&builder, *operand_b_4d);
  Add(a, b);

  ComputeAndCompareR4<float>(&builder, *expected_4d, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, R4PlusR1InDim1) {
  XlaBuilder builder(TestName());

  std::unique_ptr<Array4D<float>> operand_a_4d(new Array4D<float>(2, 3, 4, 5));
  std::unique_ptr<Array4D<float>> expected_4d(new Array4D<float>(2, 3, 4, 5));
  std::vector<float> operand_b_1d(3);
  std::iota(operand_b_1d.begin(), operand_b_1d.end(), 1.0);

  float value = 0.0;
  for (int64_t p = 0; p < 2; ++p) {
    for (int64_t z = 0; z < 3; ++z) {
      for (int64_t y = 0; y < 4; ++y) {
        for (int64_t x = 0; x < 5; ++x) {
          (*operand_a_4d)(p, z, y, x) = value;
          (*expected_4d)(p, z, y, x) = value + operand_b_1d[z];
          value += 0.1;
        }
      }
    }
  }

  auto a = ConstantR4FromArray4D<float>(&builder, *operand_a_4d);
  auto b = ConstantR1<float>(&builder, operand_b_1d);
  Add(a, b, {1});

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

  XlaBuilder builder(TestName());
  Literal a_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      r4, LayoutUtil::MakeLayout({0, 1, 2, 3}));
  auto a = ConstantLiteral(&builder, a_literal);
  auto b = ConstantR1<float>(&builder, r1);
  Add(a, b, {1});

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
  XlaBuilder builder(TestName());
  auto shape = ShapeUtil::MakeOpaqueShape();
  auto x = Parameter(&builder, 0, shape, "x");
  Add(x, x);
  auto computation_status = builder.Build();
  ASSERT_FALSE(computation_status.ok());
  EXPECT_THAT(computation_status.status().ToString(),
              ::testing::ContainsRegex(
                  "Expected array argument for lhs of binary operation"));
}

XLA_TEST_F(ArrayElementwiseOpTest, IdentityBroadcastOfSameRankIsAllowed) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<float>(&builder,
                             {{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto b = ConstantR2<float>(&builder,
                             {{-1.5f, 8.14f, 42.0}, {-1.0f, -4.0f, 5.55f}});
  Add(a, b, /*broadcast_dimensions=*/{0, 1});

  Array2D<float> expected_array(
      {{-4.0f, 11.28f, 43.0f}, {1.25f, -14.0f, 8.88f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, NonIdentityBroadcastOfSameRankIsDisallowed) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<float>(&builder,
                             {{-2.5f, 3.14f, 1.0f}, {2.25f, -10.0f, 3.33f}});
  auto b = ConstantR2<float>(&builder,
                             {{-1.5f, 8.14f, 42.0}, {-1.0f, -4.0f, 5.55f}});
  Add(a, b, /*broadcast_dimensions=*/{1, 0});

  auto computation_status = builder.Build();
  ASSERT_FALSE(computation_status.ok());
  EXPECT_THAT(computation_status.status().error_message(),
              ::testing::ContainsRegex("must.*be the identity"));
}

// Regression test for b/31927799. "slice - y" is fused and requires implicit
// broadcast.
XLA_TEST_F(ArrayElementwiseOpTest, ImplicitBroadcastInFusedExpressions) {
  XlaBuilder builder(TestName());
  auto x_literal = LiteralUtil::CreateR1<float>({1, 2, 3});
  auto y_literal = LiteralUtil::CreateR1<float>({4, 5});
  auto x_data = client_->TransferToServer(x_literal).value();
  auto y_data = client_->TransferToServer(y_literal).value();

  auto x = Parameter(&builder, 0, x_literal.shape(), "x");
  auto y = Parameter(&builder, 1, y_literal.shape(), "y");
  auto slice = Slice(x, {1}, {2}, {1});
  Sub(slice, y);

  ComputeAndCompareR1<float>(&builder, {-2, -3}, {x_data.get(), y_data.get()},
                             error_spec_);
}

INSTANTIATE_TEST_CASE_P(ArrayElementwiseOpTestParamCount,
                        ArrayElementwiseOpTestParamCount,
                        ::testing::Values(127, 128, 129, 17 * 4096));

}  // namespace
}  // namespace xla
