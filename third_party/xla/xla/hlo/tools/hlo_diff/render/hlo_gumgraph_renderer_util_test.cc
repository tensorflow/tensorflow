// Copyright 2025 The OpenXLA Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_renderer_util.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class HloDiffRendererUtilTest : public HloHardwareIndependentTestBase {};

TEST_F(HloDiffRendererUtilTest, GetChangedInstructionDiffTypesSuccess) {
  const char* const hlo_string = R"(
HloModule test_module

ENTRY test_computation {
  // For opcode, shape, and operand shape comparison
  p_s32_10_a = s32[10] parameter(0)
  p_s32_10_b = s32[10] parameter(1)
  add_s32_10 = s32[10] add(p_s32_10_a, p_s32_10_b)
  sub_s32_10 = s32[10] subtract(p_s32_10_a, p_s32_10_b)

  // For shape comparison
  p_f32_5_a = f32[5] parameter(2)
  p_f32_5_b = f32[5] parameter(3)
  add_f32_5 = f32[5] add(p_f32_5_a, p_f32_5_b)

  // For constant literal comparison
  const_42 = s32[] constant(42)
  const_99 = s32[] constant(99)
  
  // For operand number comparison
  p_s32_10_c = s32[10] parameter(4)
  clamp_s32_10 = s32[10] clamp(p_s32_10_a, p_s32_10_b, p_s32_10_c)

  // For operand shape comparison
  p_f32_10_a = f32[10] parameter(5)
  p_f32_10_b = f32[10] parameter(6)
  add_with_f32_operands = f32[10] add(p_f32_10_a, p_f32_10_b)

  ROOT tuple = (s32[10], s32[10], f32[5], s32[], s32[], s32[10], f32[10]) tuple(add_s32_10, sub_s32_10, add_f32_5, const_42, const_99, clamp_s32_10, add_with_f32_operands)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* computation = module->entry_computation();

  EXPECT_THAT(GetChangedInstructionDiffTypes(
                  *computation->GetInstructionWithName("add_s32_10"),
                  *computation->GetInstructionWithName("sub_s32_10")),
              UnorderedElementsAre(ChangedInstructionDiffType::kOpCodeChanged));

  EXPECT_THAT(
      GetChangedInstructionDiffTypes(
          *computation->GetInstructionWithName("add_s32_10"),
          *computation->GetInstructionWithName("add_f32_5")),
      UnorderedElementsAre(ChangedInstructionDiffType::kShapeChange,
                           ChangedInstructionDiffType::kChangedOperandsShape));

  EXPECT_THAT(GetChangedInstructionDiffTypes(
                  *computation->GetInstructionWithName("const_42"),
                  *computation->GetInstructionWithName("const_99")),
              ElementsAre(ChangedInstructionDiffType::kConstantLiteralChanged));

  EXPECT_THAT(
      GetChangedInstructionDiffTypes(
          *computation->GetInstructionWithName("add_s32_10"),
          *computation->GetInstructionWithName("clamp_s32_10")),
      UnorderedElementsAre(ChangedInstructionDiffType::kChangedOperandsNumber,
                           ChangedInstructionDiffType::kOpCodeChanged));

  EXPECT_THAT(
      GetChangedInstructionDiffTypes(
          *computation->GetInstructionWithName("add_s32_10"),
          *computation->GetInstructionWithName("add_with_f32_operands")),
      UnorderedElementsAre(ChangedInstructionDiffType::kShapeChange,
                           ChangedInstructionDiffType::kChangedOperandsShape));
}

TEST_F(HloDiffRendererUtilTest, GroupInstructionsByOpcode) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule test_module

ENTRY test_computation {
  param1 = s32[10] parameter(0)
  param2 = s32[10] parameter(1)
  add = s32[10] add(param1, param2)
  ROOT sub = s32[10] subtract(add, param2)
}
  )"));
  absl::flat_hash_set<const HloInstruction*> instructions;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      instructions.insert(instruction);
    }
  }

  EXPECT_THAT(GroupInstructionsByOpcode(instructions),
              UnorderedElementsAre(Pair(HloOpcode::kParameter, SizeIs(2)),
                                   Pair(HloOpcode::kAdd, SizeIs(1)),
                                   Pair(HloOpcode::kSubtract, SizeIs(1))));
}

TEST_F(HloDiffRendererUtilTest, GroupInstructionPairsByOpcode) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule test_module

ENTRY test_computation {
  param1 = s32[10] parameter(0)
  param2 = s32[10] parameter(1)
  add = s32[10] add(param1, param2)
  ROOT sub = s32[10] subtract(add, param2)
}
  )"));
  absl::flat_hash_map<const HloInstruction*, const HloInstruction*>
      instruction_map;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      instruction_map[instruction] = instruction;
    }
  }

  EXPECT_THAT(GroupInstructionPairsByOpcode(instruction_map),
              UnorderedElementsAre(Pair(HloOpcode::kParameter, SizeIs(2)),
                                   Pair(HloOpcode::kAdd, SizeIs(1)),
                                   Pair(HloOpcode::kSubtract, SizeIs(1))));
}

TEST_F(HloDiffRendererUtilTest, FilterDiffResultByOpcode) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule test_module

ENTRY test_computation {
  param1 = s32[10] parameter(0)
  param2 = s32[10] parameter(1)
  add = s32[10] add(param1, param2)
  ROOT sub = s32[10] subtract(add, param2)
}
  )"));
  DiffResult diff_result;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      diff_result.left_module_unmatched_instructions.insert(instruction);
      diff_result.right_module_unmatched_instructions.insert(instruction);
      diff_result.changed_instructions[instruction] = instruction;
      diff_result.unchanged_instructions[instruction] = instruction;
    }
  }
  absl::flat_hash_set<HloOpcode> ignored_opcodes = {HloOpcode::kParameter};
  auto filtered_diff_result =
      FilterDiffResultByOpcode(diff_result, ignored_opcodes);
  EXPECT_THAT(filtered_diff_result.left_module_unmatched_instructions,
              SizeIs(2));
  for (const HloInstruction* instruction :
       filtered_diff_result.left_module_unmatched_instructions) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kParameter);
  }
  EXPECT_THAT(filtered_diff_result.right_module_unmatched_instructions,
              SizeIs(2));
  for (const HloInstruction* instruction :
       filtered_diff_result.right_module_unmatched_instructions) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kParameter);
  }
  EXPECT_THAT(filtered_diff_result.changed_instructions, SizeIs(2));
  for (const auto& [left, right] : filtered_diff_result.changed_instructions) {
    EXPECT_NE(left->opcode(), HloOpcode::kParameter);
    EXPECT_NE(right->opcode(), HloOpcode::kParameter);
  }
  EXPECT_THAT(filtered_diff_result.unchanged_instructions, SizeIs(2));
  for (const auto& [left, right] :
       filtered_diff_result.unchanged_instructions) {
    EXPECT_NE(left->opcode(), HloOpcode::kParameter);
    EXPECT_NE(right->opcode(), HloOpcode::kParameter);
  }
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
