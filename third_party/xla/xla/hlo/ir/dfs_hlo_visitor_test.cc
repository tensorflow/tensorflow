/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/hlo/ir/dfs_hlo_visitor.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using DfsHloVisitorWithDefaultTest = HloHardwareIndependentTestBase;

TEST_F(DfsHloVisitorWithDefaultTest, DefaultElementwiseTest) {
  // Verify that HandleElementwiseBinary and HandleElementwiseUnary are called
  // on the appropriate HLO ops (elementwise binary/unary ops).

  class ElementwiseTestVisitor : public DfsHloVisitorWithDefault {
   public:
    absl::Status DefaultAction(HloInstruction* hlo) override {
      // The HLO should be neither an elementwise unary nor binary op. These
      // cases are handled in HandleElementwiseBinary/Unary.
      TF_RET_CHECK(!(hlo->IsElementwise() && hlo->operand_count() == 2))
          << hlo->ToString();
      TF_RET_CHECK(!(hlo->IsElementwise() && hlo->operand_count() == 1))
          << hlo->ToString();
      return absl::OkStatus();
    }

    absl::Status HandleElementwiseBinary(HloInstruction* hlo) override {
      // HLO should be elementwise binary.
      TF_RET_CHECK(hlo->IsElementwise() && hlo->operand_count() == 2)
          << hlo->ToString();
      return absl::OkStatus();
    }
    absl::Status HandleElementwiseUnary(HloInstruction* hlo) override {
      // HLO should be elementwise unary.
      TF_RET_CHECK(hlo->IsElementwise() && hlo->operand_count() == 1)
          << hlo->ToString();
      return absl::OkStatus();
    }
  };

  // HLO module contains are arbitrary mix of elementwise and non-elementwise
  // operations.
  const std::string& hlo_string = R"(
HloModule TestModule

ENTRY TestComputation {
  arg = f32[] parameter(0)
  tuple = (f32[]) tuple(arg)
  gte = f32[] get-tuple-element(tuple), index=0
  abs = f32[] abs(arg)
  add = f32[] add(arg, gte)
  broadcast = f32[42] broadcast(add), dimensions={}
  slice = f32[1] slice(broadcast), slice={[1:2]}
  copy = f32[] copy(arg)
  eq = pred[] compare(arg, gte), direction=EQ
  neg = f32[] negate(arg)
  ROOT convert = f64[] convert(f32[] arg)
})";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).value();
  ElementwiseTestVisitor visitor;
  EXPECT_OK(module->entry_computation()->Accept(&visitor));
}

TEST(FilteredDfsHloVisitorTest, FiltersInstructions) {
  // Create a module with a few instructions.
  auto builder = HloComputation::Builder("test");
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));
  builder.AddInstruction(
      HloInstruction::CreateUnary(add->shape(), HloOpcode::kNegate, add));

  auto module = std::make_unique<HloModule>("test", HloModuleConfig());
  auto computation = module->AddEntryComputation(builder.Build());

  std::vector<HloInstruction*> visited_instructions;
  auto action = [&visited_instructions](HloInstruction* instruction) {
    visited_instructions.push_back(instruction);
    return absl::OkStatus();
  };

  // Create a filtered visitor that only visits Add instructions.
  FilteredDfsHloVisitor filtered_visitor(
      std::move(action), [](const HloInstruction* instruction) {
        return instruction->opcode() == HloOpcode::kAdd;
      });

  // Run the filtered visitor on the computation.
  EXPECT_OK(computation->Accept(&filtered_visitor));

  // Check that the recording visitor only visited the Add instruction.
  EXPECT_THAT(visited_instructions, ElementsAre(add));
}

}  // namespace
}  // namespace xla
