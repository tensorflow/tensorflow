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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

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
  TF_EXPECT_OK(computation->Accept(&filtered_visitor));

  // Check that the recording visitor only visited the Add instruction.
  EXPECT_THAT(visited_instructions, ElementsAre(add));
}

}  // namespace
}  // namespace xla
