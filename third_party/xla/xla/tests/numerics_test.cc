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

#include "xla/tests/xla_test_backend_predicates.h"
#include "absl/status/statusor.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"

namespace xla {
namespace {

using NumericsTest = HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

TEST_F(NumericsTest, AbsOfLargeComplexNumber) {
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

TEST_F(NumericsTest, PowerOfLargeComplexNumber) {
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

// Case from one of XLA users, the following code produced incorrect results on
// CPU thunks backend (due to incorrect LLVM IR generated).
// This is an HLO module optimized for CPU backend, it may be invalid for other
// backends.
TEST_F(NumericsTest, DISABLED_ON_TPU(MultiplySubtractConcatTest)) {
  if (test::DeviceIs(test::kGpu)) {
    GTEST_SKIP();
  }
  const char* test_hlo = R"(
    HloModule jit_step, is_scheduled=true

    fused_computation {
      param_0.2 = f32[1,5] parameter(0)
      slice.11 = f32[1,1] slice(param_0.2), slice={[0:1], [1:2]}
      slice.10 = f32[1,1] slice(param_0.2), slice={[0:1], [4:5]}
      multiply.11 = f32[1,1] multiply(slice.11, slice.10)
      slice.9 = f32[1,1] slice(param_0.2), slice={[0:1], [2:3]}
      slice.8 = f32[1,1] slice(param_0.2), slice={[0:1], [3:4]}
      multiply.10 = f32[1,1] multiply(slice.9, slice.8)
      subtract.5 = f32[1,1] subtract(multiply.11, multiply.10)
      slice.6 = f32[1,1] slice(param_0.2), slice={[0:1], [0:1]}
      multiply.8 = f32[1,1] multiply(slice.6, slice.10)
      subtract.4 = f32[1,1] subtract(slice.9, multiply.8)
      ROOT concatenate.1 = f32[1,3] concatenate(
        subtract.5, subtract.4, subtract.4), dimensions={1}
    } // fused_computation

    ENTRY main {
      Arg_0.0 = f32[1,5] parameter(0)
      ROOT fusion = f32[1,3] fusion(Arg_0.0), kind=kLoop,
        calls=fused_computation
    } // main
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnVerifiedModule(test_hlo));
  auto argument = LiteralUtil::CreateR2<float>(
      {{0.261473775, -0.642940283, -0.719902277, 0.712947428, 0.543724537}});

  TF_ASSERT_OK_AND_ASSIGN(auto test_result,
                          Execute(std::move(test_module), {&argument},
                                  /*run_hlo_passes=*/false));

  // Reference HLO module. It's a subgraph of the test module, it performs only
  // the calculations needed for the first output element from the test module.
  const char* reference_hlo = R"(
    HloModule jit_step, is_scheduled=true

    fused_computation {
      param_0.2 = f32[1,5] parameter(0)
      slice.11 = f32[1,1] slice(param_0.2), slice={[0:1], [1:2]}
      slice.10 = f32[1,1] slice(param_0.2), slice={[0:1], [4:5]}
      multiply.11 = f32[1,1] multiply(slice.11, slice.10)
      slice.9 = f32[1,1] slice(param_0.2), slice={[0:1], [2:3]}
      slice.8 = f32[1,1] slice(param_0.2), slice={[0:1], [3:4]}
      multiply.10 = f32[1,1] multiply(slice.9, slice.8)
      ROOT subtract.5 = f32[1,1] subtract(multiply.11, multiply.10)
    } // fused_computation

    ENTRY main {
      Arg_0.0 = f32[1,5] parameter(0)
      ROOT fusion = f32[1,1] fusion(Arg_0.0), kind=kLoop,
        calls=fused_computation
    } // main
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto reference_module,
                          ParseAndReturnVerifiedModule(reference_hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto reference_result,
                          Execute(std::move(reference_module), {&argument},
                                  /*run_hlo_passes=*/false));

  // Only compare the first element.
  EXPECT_FLOAT_EQ(reference_result.data<float>()[0],
                  test_result.data<float>()[0]);
}

}  // namespace
}  // namespace xla
