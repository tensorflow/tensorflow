/* Copyright 2023 The OpenXLA Authors.

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

#include <limits>
#include <memory>
#include <utility>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/statusor.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/types.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using NumericsTest = HloTestBase;

XLA_TEST_F(NumericsTest, AbsOfLargeComplexNumber) {
  const char* hlo = R"(
HloModule module

ENTRY entry {
  x = c64[] parameter(0)
  ROOT power = f32[] abs(x)
}
)";

  auto abs_of_complex_x = [&hlo, this](float x) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    auto x_lit = LiteralUtil::CreateR0<complex64>(x);
    return RunAndCompare(std::move(module), {&x_lit}, ErrorSpec{1e-5, 1e-5});
  };

  EXPECT_TRUE(abs_of_complex_x(1e19));
  EXPECT_TRUE(abs_of_complex_x(1e25));
  EXPECT_TRUE(abs_of_complex_x(1e30));
}

XLA_TEST_F(NumericsTest, PowerOfLargeComplexNumber) {
  const char* hlo = R"(
HloModule module

ENTRY entry {
  large = c64[] parameter(0)
  x = c64[] parameter(1)
  ROOT power = c64[] power(large, x)
}
)";

  auto complex_a_raised_to_complex_b = [&hlo, this](float num, float exp) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    auto num_lit = LiteralUtil::CreateR0<complex64>(num);
    auto exp_lit = LiteralUtil::CreateR0<complex64>(exp);
    return RunAndCompare(std::move(module), {&num_lit, &exp_lit},
                         ErrorSpec{1e-5, 1e-5});
  };

  EXPECT_TRUE(complex_a_raised_to_complex_b(1e19, 0));
  EXPECT_TRUE(complex_a_raised_to_complex_b(1e19, 1));
  EXPECT_TRUE(complex_a_raised_to_complex_b(1e19, 1.2));
  EXPECT_TRUE(complex_a_raised_to_complex_b(1e19, 2));
  EXPECT_TRUE(complex_a_raised_to_complex_b(1e30, 0));
  EXPECT_TRUE(complex_a_raised_to_complex_b(1e30, 1));
  EXPECT_TRUE(complex_a_raised_to_complex_b(1e30, 1.2));
  EXPECT_TRUE(
      complex_a_raised_to_complex_b(std::numeric_limits<float>::infinity(), 0));
  EXPECT_TRUE(complex_a_raised_to_complex_b(
      std::numeric_limits<float>::quiet_NaN(), 0));
}

}  // namespace
}  // namespace xla
