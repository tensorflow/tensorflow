/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/nullary_function_wrap_inliner.h"

#include <gmock/gmock.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"

namespace xla {
namespace {

using NullaryFunctionWrapInlinerTest = HloHardwareIndependentTestBase;

TEST_F(NullaryFunctionWrapInlinerTest, InlineTokenCallAndAddControlDep) {
  const char* const hlo_string = R"(
HloModule TokenCallModule

wrapped_iota {
  token_param = token[] parameter(0)
  ROOT iota = s32[4] iota(), iota_dimension=0
}

ENTRY entry {
  tok = token[] after-all()
  ROOT call = s32[4] call(tok), to_apply=wrapped_iota
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       NullaryFunctionWrapInliner{}.Run(module.get()));
  ASSERT_TRUE(changed);

  EXPECT_EQ(FindInstruction(module.get(), HloOpcode::kCall), nullptr);
  HloInstruction* tok = FindInstruction(module.get(), HloOpcode::kAfterAll);
  HloInstruction* iota = module->entry_computation()->root_instruction();
  ASSERT_NE(tok, nullptr);
  ASSERT_NE(iota, nullptr);
  EXPECT_THAT(tok->control_successors(), ::testing::ElementsAre(iota));
  EXPECT_THAT(iota->control_predecessors(), ::testing::ElementsAre(tok));
}

TEST_F(NullaryFunctionWrapInlinerTest, DoesNotInlineCallWithExtraOperands) {
  const char* const hlo_string = R"(
HloModule NonNullaryCallModule

callee {
  token_param = token[] parameter(0)
  x_param = s32[4] parameter(1)
  ROOT add = s32[4] add(x_param, x_param)
}

ENTRY entry {
  tok = token[] after-all()
  x = s32[4] parameter(0)
  ROOT call = s32[4] call(tok, x), to_apply=callee
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       NullaryFunctionWrapInliner{}.Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_NE(FindInstruction(module.get(), HloOpcode::kCall), nullptr);
}

}  // namespace
}  // namespace xla
