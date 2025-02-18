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
#include "xla/hlo/pass/hlo_pass_fix.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// A module pass that decrements positive scalar signed integer constants.
class DecrementPositiveConstants : public HloModulePass {
 public:
  absl::string_view name() const override { return "decrement-constants"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    bool changed = false;
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kConstant &&
            ShapeUtil::IsScalar(instruction->shape()) &&
            instruction->shape().element_type() == S32) {
          int32_t value = instruction->literal().GetFirstElement<int32_t>();
          if (value > 0) {
            TF_RETURN_IF_ERROR(instruction->parent()->ReplaceWithNewInstruction(
                instruction, HloInstruction::CreateConstant(
                                 LiteralUtil::CreateR0<int32_t>(value - 1))));
            changed = true;
          }
        }
      }
    }
    return changed;
  }
};

// A module pass that changes addition operations to subtract operations and
// vice versa.
class FlipAddSubtract : public HloModulePass {
 public:
  absl::string_view name() const override { return "flip-add-subtract"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    bool changed = false;
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kAdd) {
          HloInstruction* lhs;
          HloInstruction* rhs;
          CHECK(
              Match(instruction, match::Add(match::Op(&lhs), match::Op(&rhs))));
          TF_RETURN_IF_ERROR(instruction->parent()->ReplaceWithNewInstruction(
              instruction,
              HloInstruction::CreateBinary(instruction->shape(),
                                           HloOpcode::kSubtract, lhs, rhs)));
          changed = true;
        }
        if (instruction->opcode() == HloOpcode::kSubtract) {
          HloInstruction* lhs;
          HloInstruction* rhs;
          CHECK(Match(instruction,
                      match::Subtract(match::Op(&lhs), match::Op(&rhs))));
          TF_RETURN_IF_ERROR(instruction->parent()->ReplaceWithNewInstruction(
              instruction,
              HloInstruction::CreateBinary(instruction->shape(),
                                           HloOpcode::kAdd, lhs, rhs)));
          changed = true;
        }
      }
    }
    return changed;
  }
};

class HloPassFixTest : public HloHardwareIndependentTestBase {};

TEST_F(HloPassFixTest, RunModuleToFixedPoint) {
  constexpr absl::string_view kModule = R"(
    HloModule Converges

    ENTRY main {
      ROOT c = s32[] constant(5)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kModule));

  HloPassFix<DecrementPositiveConstants> pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(root->literal().GetFirstElement<int32_t>(), 0);
}

TEST_F(HloPassFixTest, RunModuleGroupToFixedPoint) {
  constexpr absl::string_view kModule0 = R"(
    HloModule First

    ENTRY main {
      ROOT c = s32[] constant(5)
    }
  )";

  constexpr absl::string_view kModule1 = R"(
    HloModule Second

    ENTRY main {
      ROOT c = s32[] constant(3)
    }
  )";

  constexpr absl::string_view kModule2 = R"(
    HloModule Second

    ENTRY main {
      ROOT c = s32[] constant(0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module0,
                          ParseAndReturnVerifiedModule(kModule0));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module1,
                          ParseAndReturnVerifiedModule(kModule1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module2,
                          ParseAndReturnVerifiedModule(kModule2));
  HloModuleGroup module_group("group");
  module_group.push_back(std::move(module0));
  module_group.push_back(std::move(module1));
  module_group.push_back(std::move(module2));

  HloPassFix<DecrementPositiveConstants> pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.RunOnModuleGroup(&module_group));
  EXPECT_TRUE(changed);
  HloInstruction* root0 =
      module_group.module(0).entry_computation()->root_instruction();
  ASSERT_EQ(root0->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(root0->literal().GetFirstElement<int32_t>(), 0);
  HloInstruction* root1 =
      module_group.module(1).entry_computation()->root_instruction();
  ASSERT_EQ(root1->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(root1->literal().GetFirstElement<int32_t>(), 0);
  HloInstruction* root2 =
      module_group.module(2).entry_computation()->root_instruction();
  ASSERT_EQ(root2->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(root2->literal().GetFirstElement<int32_t>(), 0);
}

TEST_F(HloPassFixTest, OscillationsStillTerminate) {
  constexpr absl::string_view kModule = R"(
    HloModule Oscillating

    ENTRY main {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      ROOT c = f32[4] add(a, b)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kModule));
  HloPassFix<FlipAddSubtract> pass;

  // We expect this to terminate and report that the module did not change.
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
