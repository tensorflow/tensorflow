/* Copyright 2020 The OpenXLA Authors.

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

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

void DisableAllHloPasses(HloModule& module) {
  auto debug_options = module.config().debug_options();
  debug_options.set_xla_disable_all_hlo_passes(true);
  module.mutable_config().set_debug_options(debug_options);
}

class GetDimensionSizeTest : public HloTestBase {};

// Test that the interpreter can correctly compute get_dimension_size.
TEST_F(GetDimensionSizeTest, CorrectComputation) {
  const char* const kModuleStr = R"(
HloModule a_inference_call_110__.55

ENTRY %a_inference_call_110__.55 (arg0.1: f32[1,8], arg1.2: f32[8], arg2.3: f32[8]) -> s32[] {
  %constant.37 = f32[] constant(1e-12)
  %broadcast.38 = f32[1,1]{1,0} broadcast(f32[] %constant.37), dimensions={}
  %arg0.1 = f32[1,8]{1,0} parameter(0), parameter_replication={false}
  %reshape.4 = f32[1,8]{1,0} reshape(f32[1,8]{1,0} %arg0.1)
  %convert.5 = f32[1,8]{1,0} convert(f32[1,8]{1,0} %reshape.4)
  %constant.6 = f32[] constant(0)
  %convert.7 = f32[] convert(f32[] %constant.6)
  ROOT %get-dimension-size.13 = s32[] get-dimension-size(f32[1,8]{1,0} %convert.5), dimensions={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01, 0.01}));
}

TEST_F(GetDimensionSizeTest,
       DISABLED_ON_INTERPRETER(DISABLED_ON_GPU(
           DISABLED_ON_TPU(ReturnsErrorWhenHloPassesDisabled)))) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY %test {
      %arg0 = f32[1,8] parameter(0)
      ROOT %get-dimension-size.0 = s32[] get-dimension-size(%arg0),
        dimensions={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  DisableAllHloPasses(*module);

  Literal arg0 =
      LiteralUtil::CreateR1<float>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});

  auto status_or_result = Execute(std::move(module), {&arg0});
  EXPECT_EQ(status_or_result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(
      status_or_result.status().message(),
      ::testing::HasSubstr("GetDimensionSize should be rewritten for CPU"));
}

}  // anonymous namespace
}  // namespace xla
