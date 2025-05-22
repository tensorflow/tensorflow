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

#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using StochasticConvertTest = HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

const char* const kModuleStr = R"(
  HloModule stochastic-convert

  ENTRY entry {
    %arg_param.1 = f32[65536]{0} parameter(0)
    %random_param.2 = u32[65536]{0} parameter(1)
    ROOT %stochastic-convert.3 = s32[65536]{0} stochastic-convert(
      f32[65536]{0} %arg_param.1, u32[65536]{0} %random_param.2)
  }
)";

XLA_TEST_F(StochasticConvertTest, CorrectComputation) {
  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{0.001}));
}

TEST_F(StochasticConvertTest, ReturnsErrorWhenHloPassesDisabled) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  auto arg0_shape = ShapeUtil::MakeShape(F32, {65536});
  auto arg0 = MakeFakeLiteral(arg0_shape).value();

  auto arg1_shape = ShapeUtil::MakeShape(U32, {65536});
  auto arg1 = MakeFakeLiteral(arg1_shape).value();

  auto status_or_result =
      Execute(std::move(module), {&arg0, &arg1}, /*run_hlo_passes=*/false);
  EXPECT_EQ(status_or_result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(
      status_or_result.status().message(),
      ::testing::HasSubstr("StochasticConvert should be decomposed for CPU"));
}

}  // namespace
}  // namespace xla
