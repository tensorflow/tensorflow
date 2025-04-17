/* Copyright 2018 The OpenXLA Authors.

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
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"

// Tests the handling of the basic mathematics operations with F16 operands.

namespace xla {
namespace {

using UnaryBuildFuncTy = std::function<void(const xla::XlaOp& src)>;

// Number of elements in the input buffers.
constexpr int kNumElements = 4;
constexpr ErrorSpec kErrorSpec{0.001, 0.001};

using HalfTestBase = ClientLibraryTestRunnerMixin<
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>;

struct UnaryOpTestParam {
  std::function<half(half)> compute_func;
  UnaryBuildFuncTy build_func;
};

class UnaryOpTest : public HalfTestBase,
                    public ::testing::WithParamInterface<UnaryOpTestParam> {};

XLA_TEST_P(UnaryOpTest, Ops) {
  std::vector<half> x({half(1.4), half(-2.3), half(3.2), half(-4.1), half(9.0),
                       half(42.0), half(-9.0), half(-100.0)});
  XlaBuilder builder(TestName());
  XlaOp x_opnd;
  auto x_data = CreateR1Parameter<half>(x, /*parameter_number=*/0, "x",
                                        &builder, &x_opnd);

  std::function<half(half)> compute_func = GetParam().compute_func;
  std::vector<half> expected;
  const int64_t n = x.size();
  expected.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    expected.push_back(compute_func(x[i]));
  }

  UnaryBuildFuncTy build_func = GetParam().build_func;
  build_func(x_opnd);

  ComputeAndCompareR1<half>(&builder, expected, {&x_data}, kErrorSpec);
}

half sign_imp(half value) {
  const float x(std::move(value));
  return half((x < .0) ? -1 : (x > .0));
}

half round_imp(half value) {
  return half(std::round(static_cast<float>(std::move(value))));
}

INSTANTIATE_TEST_SUITE_P(
    half, UnaryOpTest,
    ::testing::Values(
        UnaryOpTestParam{[](half x) { return abs(x); }, &Abs},
        UnaryOpTestParam{[](half x) { return round_imp(x); }, &Round},
        UnaryOpTestParam{[](half x) { return ceil(x); }, &Ceil},
        UnaryOpTestParam{[](half x) { return cos(x); },
                         [](XlaOp x) { return xla::Cos(x); }},
        UnaryOpTestParam{[](half x) { return exp(x); },
                         [](XlaOp x) { return xla::Exp(x); }},
        UnaryOpTestParam{[](half x) { return floor(x); }, &Floor},
        UnaryOpTestParam{[](half x) { return log(x); },
                         [](XlaOp x) { return xla::Log(x); }},
        UnaryOpTestParam{[](half x) { return -x; }, &Neg},
        UnaryOpTestParam{[](half x) { return sign_imp(x); }, &Sign},
        UnaryOpTestParam{[](half x) { return sin(x); },
                         [](XlaOp x) { return xla::Sin(x); }},
        UnaryOpTestParam{[](half x) { return tanh(x); },
                         [](XlaOp x) { return xla::Tanh(x); }}));

struct UnaryPredTestParam {
  std::function<bool(half)> compute_func;
  UnaryBuildFuncTy build_func;
};

class UnaryPredTest : public HalfTestBase,
                      public ::testing::WithParamInterface<UnaryPredTestParam> {
};

XLA_TEST_P(UnaryPredTest, Ops) {
  std::vector<half> x({half(1.4), half(-2.3), half(3.2), half(-4.1)});
  XlaBuilder builder(TestName());
  XlaOp x_opnd;
  auto x_data = CreateR1Parameter<half>(x, /*parameter_number=*/0, "x",
                                        &builder, &x_opnd);

  std::function<bool(half)> compute_func = GetParam().compute_func;
  CHECK_EQ(kNumElements, x.size());
  bool expected[kNumElements];
  for (int64_t i = 0; i < x.size(); ++i) {
    expected[i] = compute_func(x[i]);
  }

  UnaryBuildFuncTy build_func = GetParam().build_func;
  build_func(x_opnd);

  ComputeAndCompareR1<bool>(&builder, expected, {&x_data});
}

INSTANTIATE_TEST_SUITE_P(half, UnaryPredTest,
                         ::testing::Values(UnaryPredTestParam{
                             [](half x) { return isfinite(x); }, &IsFinite}));

using BinaryBuildFuncTy = std::function<void(
    const xla::XlaOp& x, const xla::XlaOp& y, absl::Span<const int64_t>)>;

struct BinaryOpTestParam {
  std::function<half(half, half)> compute_func;
  BinaryBuildFuncTy build_func;
};

class BinaryOpTest : public HalfTestBase,
                     public ::testing::WithParamInterface<BinaryOpTestParam> {};

XLA_TEST_P(BinaryOpTest, Ops) {
  std::vector<half> x({half(1.0), half(2.0), half(3.0), half(-4.0)});
  std::vector<half> y({half(0.4), half(-0.3), half(0.2), half(0.1)});
  XlaBuilder builder(TestName());
  XlaOp x_opnd;
  auto x_data = CreateR1Parameter<half>(x, /*parameter_number=*/0, "x",
                                        &builder, &x_opnd);

  XlaOp y_opnd;
  auto y_data = CreateR1Parameter<half>(y, /*parameter_number=*/1, "y",
                                        &builder, &y_opnd);

  std::function<half(half, half)> compute_func = GetParam().compute_func;
  std::vector<half> expected;
  const int64_t n = x.size();
  expected.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    expected.push_back(compute_func(x[i], y[i]));
  }

  BinaryBuildFuncTy build_func = GetParam().build_func;
  build_func(x_opnd, y_opnd, {});

  ComputeAndCompareR1<half>(&builder, expected, {&x_data, &y_data}, kErrorSpec);
}

half atan2_imp(half x, half y) {
  return half(std::atan2(static_cast<float>(std::move(x)),
                         static_cast<float>(std::move(y))));
}

INSTANTIATE_TEST_SUITE_P(
    half, BinaryOpTest,
    ::testing::Values(
        BinaryOpTestParam{[](half x, half y) { return x + y; }, &Add},
        BinaryOpTestParam{[](half x, half y) { return atan2_imp(x, y); },
                          &Atan2},
        BinaryOpTestParam{[](half x, half y) { return x / y; }, &Div},
        BinaryOpTestParam{[](half x, half y) { return max(x, y); }, &Max},
        BinaryOpTestParam{[](half x, half y) { return min(x, y); }, &Min},
        BinaryOpTestParam{[](half x, half y) { return x * y; }, &Mul},
        BinaryOpTestParam{[](half x, half y) { return pow(x, y); }, &Pow},
        BinaryOpTestParam{[](half x, half y) { return x - y; }, &Sub}

        ));

struct BinaryPredTestParam {
  std::function<bool(half, half)> compute_func;
  BinaryBuildFuncTy build_func;
};

class BinaryPredTest
    : public HalfTestBase,
      public ::testing::WithParamInterface<BinaryPredTestParam> {};

XLA_TEST_P(BinaryPredTest, Ops) {
  std::vector<half> x({half(1.0), half(2.0), half(0.2), half(-4.0)});
  std::vector<half> y({half(0.4), half(-0.3), half(0.2), half(0.1)});
  XlaBuilder builder(TestName());
  XlaOp x_opnd;
  auto x_data = CreateR1Parameter<half>(x, /*parameter_number=*/0, "x",
                                        &builder, &x_opnd);

  XlaOp y_opnd;
  auto y_data = CreateR1Parameter<half>(y, /*parameter_number=*/1, "y",
                                        &builder, &y_opnd);

  std::function<bool(half, half)> compute_func = GetParam().compute_func;
  CHECK_EQ(kNumElements, x.size());
  bool expected[kNumElements];
  for (int64_t i = 0; i < x.size(); ++i) {
    expected[i] = compute_func(x[i], y[i]);
  }

  BinaryBuildFuncTy build_func = GetParam().build_func;
  build_func(x_opnd, y_opnd, {});

  ComputeAndCompareR1<bool>(&builder, expected, {&x_data, &y_data});
}

INSTANTIATE_TEST_SUITE_P(
    half, BinaryPredTest,
    ::testing::Values(
        BinaryPredTestParam{[](half x, half y) { return x == y; }, &Eq},
        BinaryPredTestParam{[](half x, half y) { return x != y; }, &Ne},
        BinaryPredTestParam{[](half x, half y) { return x >= y; }, &Ge},
        BinaryPredTestParam{[](half x, half y) { return x > y; }, &Gt},
        BinaryPredTestParam{[](half x, half y) { return x <= y; }, &Le},
        BinaryPredTestParam{[](half x, half y) { return x < y; }, &Lt}

        ));

}  // namespace
}  // namespace xla
