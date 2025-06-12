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

#include <cstdint>
#include <utility>

#include "xla/tests/xla_test_backend_predicates.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

const char* const kModuleStr = R"(
    HloModule m

    ENTRY %test {
      %arg0 = f32[1,8] parameter(0)
      %arg1 = s32[] parameter(1)
      ROOT %set-dimension-size.0 = f32[1,<=8] set-dimension-size(%arg0, %arg1),
        dimensions={1}
    }
  )";

void DisableAllHloPasses(HloModule& module) {
  auto debug_options = module.config().debug_options();
  debug_options.set_xla_disable_all_hlo_passes(true);
  module.mutable_config().set_debug_options(debug_options);
}

class SetDimensionSizeTest : public HloPjRtTestBase {};

TEST_F(SetDimensionSizeTest, CorrectComputation) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 =
      LiteralUtil::CreateR2<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}});
  Literal arg1 = LiteralUtil::CreateR0<int32_t>(5);

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          Execute(std::move(module), {&arg0, &arg1}));

  Literal expected = LiteralUtil::CreateR2<float>({{0.0, 1.0, 2.0, 3.0, 4.0}});
  EXPECT_EQ(result, expected);
}

TEST_F(SetDimensionSizeTest, ReturnsErrorWhenHloPassesDisabled) {
  if (test::DeviceIsOneOf({test::kGpu, test::kInterpreter}) ||
      test::DeviceTypeIs(test::kTpu)) {
    GTEST_SKIP();
  }
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  DisableAllHloPasses(*module);

  Literal arg0 =
      LiteralUtil::CreateR1<float>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
  Literal arg1 = LiteralUtil::CreateR0<int32_t>(5);

  auto status_or_result = Execute(std::move(module), {&arg0, &arg1});
  EXPECT_EQ(status_or_result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(
      status_or_result.status().message(),
      ::testing::HasSubstr("SetDimensionSize should be rewritten for CPU"));
}

}  // namespace
}  // namespace xla
