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

#include "xla/hlo/transforms/expanders/rng_expander.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using RngExpanderTest = HloHardwareIndependentTestBase;

TEST_F(RngExpanderTest, ReplacesRngWithGetRngSeedCustomCall) {
  const char* const hlo_string = R"(
    HloModule m
    ENTRY entry {
      min = f32[] constant(0)
      max = f32[] constant(1)
      ROOT result = f32[2,4,8] rng(min, max), distribution=rng_uniform
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  RngExpander expander;

  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&expander, module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* rng_seed = nullptr;
  int rng_count = 0;

  // Traverse entry computation instructions.
  for (HloInstruction* inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kRng) {
      rng_count++;
    }
  }

  // Find the called computation (rng_computation).
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kCall);
  HloComputation* rng_comp = root->to_apply();
  ASSERT_NE(rng_comp, nullptr);

  for (HloInstruction* inst : rng_comp->instructions()) {
    if (inst->opcode() == HloOpcode::kCustomCall &&
        inst->custom_call_target() == "GetRngSeed") {
      rng_seed = inst;
    }
  }

  // Confirm explicit replacement requirements:
  EXPECT_EQ(rng_count, 0);
  ASSERT_NE(rng_seed, nullptr) << "Failed to locate custom call GetRngSeed.";
  EXPECT_EQ(rng_seed->shape().element_type(), U64);

  // Confirm correct placement of anti-folding attribute for constant folding /
  // CSE checks.
  EXPECT_TRUE(rng_seed->frontend_attributes().map().contains(
      kXlaCseSafeZeroOperandAttr));
  EXPECT_EQ(
      rng_seed->frontend_attributes().map().at(kXlaCseSafeZeroOperandAttr),
      "true");
}

}  // namespace
}  // namespace xla
