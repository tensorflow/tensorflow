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

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "xla/tests/xla_test_backend_predicates.h"
#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla {
namespace {

using Atan2Test =
    ClientLibraryTestRunnerMixin<HloInterpreterReferenceMixin<HloTestBase>>;

TEST_F(Atan2Test, atan2) {
  XlaBuilder builder("atan2 with special and non-special float values");
  float infinity = std::numeric_limits<float>::infinity();
  float qNaN = std::numeric_limits<float>::quiet_NaN();
  float sNaN = std::numeric_limits<float>::signaling_NaN();
  auto x = ConstantR1<float>(&builder,
                             {infinity, infinity, infinity, qNaN, qNaN, sNaN});
  auto y =
      ConstantR1<float>(&builder, {1.0, -1.0, qNaN, 1.0, infinity, infinity});
  Atan2(x, y);
  float pi_over_2 = 1.5708;
  std::vector<float> expected = {pi_over_2, pi_over_2, qNaN, qNaN, qNaN, qNaN};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// atan2 only depends on the ratio of its operands, so it should be accurate
// even when the operands are very large, as long as y/x is representable.
TEST_F(Atan2Test, LargeMagnitudeOperands) {
  XlaBuilder builder("atan2 with large-magnitude operands");
  float max = std::numeric_limits<float>::max();
  std::vector<float> ys = {0x1p127f, -0x1p127f, 0x1p127f, 0x1.8p125f,
                           max,      0x1p110f,  0x1p110f, -0x1.8p120f};
  std::vector<float> xs = {0x1p127f, 0x1p127f,   -0x1p127f, 0x1p127f,
                           max / 2,  0x1.8p110f, 0x1p100f,  -0x1p121f};
  std::vector<float> expected;
  for (size_t i = 0; i < ys.size(); ++i) {
    expected.push_back(static_cast<float>(
        std::atan2(static_cast<double>(ys[i]), static_cast<double>(xs[i]))));
  }
  Literal y_literal = LiteralUtil::CreateR1<float>(ys);
  Literal x_literal = LiteralUtil::CreateR1<float>(xs);
  // Use parameters rather than constants so the computation isn't
  // constant-folded on the host.
  auto y = Parameter(&builder, 0, y_literal.shape(), "y");
  auto x = Parameter(&builder, 1, x_literal.shape(), "x");
  Atan2(y, x);
  ComputeAndCompareR1<float>(&builder, expected, {&y_literal, &x_literal},
                             ErrorSpec(1e-6, 1e-6));
}

TEST_F(Atan2Test, F64SubnormalOperands) {
  // Only XLA:CPU flushes subnormal inputs; GPU preserves them, so
  // atan2(tiny, tiny) is pi/4 there rather than +0.
  if (!test::DeviceTypeIs(test::kCpu)) {
    GTEST_SKIP() << "Subnormal flushing is specific to the CPU backend.";
  }
  XlaBuilder builder("atan2 with subnormal F64 operands");
  const double tiny = std::numeric_limits<double>::denorm_min();
  const double min_normal = std::numeric_limits<double>::min();
  const std::vector<double> ys = {tiny,     -tiny, tiny,           -tiny,
                                  2 * tiny, tiny,  min_normal / 2, min_normal};
  const std::vector<double> xs = {tiny, tiny,     -tiny,      -tiny,
                                  tiny, 2 * tiny, min_normal, min_normal / 2};
  // The CPU builtin flushes subnormal inputs to signed zero, so each pair
  // above reduces to atan2 on signed zeros or on one normal operand: the
  // results are +/-0, +/-pi, or +pi/2.
  constexpr double kPi = 3.141592653589793;
  constexpr double kHalfPi = 1.5707963267948966;
  const std::vector<double> expected = {0.0, -0.0, kPi, -kPi,
                                        0.0, 0.0,  0.0, kHalfPi};
  Literal y_literal = LiteralUtil::CreateR1<double>(ys);
  Literal x_literal = LiteralUtil::CreateR1<double>(xs);
  // Parameters ensure the subnormal inputs reach compiled CPU code instead of
  // being constant-folded on the host.
  auto y = Parameter(&builder, 0, y_literal.shape(), "y");
  auto x = Parameter(&builder, 1, x_literal.shape(), "x");
  Atan2(y, x);
  ComputeAndCompareR1<double>(&builder, expected, {&y_literal, &x_literal},
                              ErrorSpec(1e-15, 1e-15));
}

TEST_F(Atan2Test, F32SubnormalOperands) {
  // Only XLA:CPU flushes subnormal inputs; GPU preserves them, so
  // atan2(tiny, tiny) is pi/4 there rather than +0.
  if (!test::DeviceTypeIs(test::kCpu)) {
    GTEST_SKIP() << "Subnormal flushing is specific to the CPU backend.";
  }
  XlaBuilder builder("atan2 with subnormal F32 operands");
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float min_normal = std::numeric_limits<float>::min();
  const std::vector<float> ys = {tiny,       tiny, min_normal / 2, -tiny,
                                 min_normal, tiny, 1.0f,           -1.0f};
  const std::vector<float> xs = {tiny, 1.0f,       -1.0f, -tiny,
                                 tiny, min_normal, -tiny, tiny};
  // Flushed to signed zero, each pair above gives +/-0, +/-pi, or +/-pi/2.
  constexpr float kPi = 3.1415927f;
  constexpr float kHalfPi = 1.5707964f;
  const std::vector<float> expected = {0.0f,    0.0f, kPi,     -kPi,
                                       kHalfPi, 0.0f, kHalfPi, -kHalfPi};
  Literal y_literal = LiteralUtil::CreateR1<float>(ys);
  Literal x_literal = LiteralUtil::CreateR1<float>(xs);
  // Parameters ensure the subnormal inputs reach compiled CPU code instead of
  // being constant-folded on the host.
  auto y = Parameter(&builder, 0, y_literal.shape(), "y");
  auto x = Parameter(&builder, 1, x_literal.shape(), "x");
  Atan2(y, x);
  ComputeAndCompareR1<float>(&builder, expected, {&y_literal, &x_literal},
                             ErrorSpec(1e-6, 1e-6));
}

}  // namespace
}  // namespace xla
