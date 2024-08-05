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

#include "xla/service/gpu/async_wrapper.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

class AsyncWrapperTest : public HloTestBase {};

int CountAsyncInstructions(HloComputation* computation) {
  int count = 0;
  for (const HloInstruction* instruction : computation->instructions()) {
    if (instruction->IsAsynchronous()) ++count;
  }
  return count;
}

TEST_F(AsyncWrapperTest, BasicFusion) {
  const char* hlo_text = R"(
  HloModule m

  double1 {
    p0 = f32[1] parameter(0)
    ROOT add = f32[1] add(p0, p0)
  }

  double2 {
    p0 = f32[1] parameter(0)
    ROOT add = f32[1] add(p0, p0)
  }

  ENTRY main {
    p0 = f32[1] parameter(0)
    agg1 = f32[1] fusion(p0), kind=kLoop, calls=double1
    agg2 = f32[1] fusion(p0), kind=kLoop, calls=double2
    ROOT done = f32[1] add(agg1, agg2)
  })";

  std::unique_ptr<VerifiedHloModule> module =
      ParseAndReturnVerifiedModule(hlo_text).value();

  AsyncWrapper wrapper([](const HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kFusion;
  });
  EXPECT_THAT(wrapper.HloModulePass::Run(module.get()), IsOkAndHolds(true));
  EXPECT_EQ(CountAsyncInstructions(module->entry_computation()), 4);

  Literal argument = LiteralUtil::CreateR1<float>({1.0});
  Literal expected = LiteralUtil::CreateR1<float>({4.0});

  Literal result = ExecuteNoHloPasses(std::move(module), {&argument});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

}  // namespace
}  // namespace xla::gpu
