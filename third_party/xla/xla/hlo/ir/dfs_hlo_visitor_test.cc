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

#include <cstdint>
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
using DfsHloVisitorTest = HloHardwareIndependentTestBase;

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
  TF_EXPECT_OK(module->entry_computation()->Accept(&visitor));
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
  TF_EXPECT_OK(computation->Accept(&filtered_visitor));

  // Check that the recording visitor only visited the Add instruction.
  EXPECT_THAT(visited_instructions, ElementsAre(add));
}

// Records the names of the instructions it is called on.
class RecordingVisitor : public DfsHloVisitorWithDefault {
 public:
  absl::Status DefaultAction(HloInstruction* hlo) override {
    visited_.emplace_back(hlo->name());
    return absl::OkStatus();
  }

  const std::vector<std::string>& visited() const { return visited_; }
  void ClearVisited() { visited_.clear(); }

 private:
  std::vector<std::string> visited_;
};

TEST(DfsHloVisitorVisitStateTest, ResetVisitStatesClearsEveryState) {
  RecordingVisitor visitor;
  visitor.SetVisitState(1, DfsHloVisitor::kVisiting);
  visitor.SetVisitState(2, DfsHloVisitor::kVisited);
  EXPECT_EQ(visitor.GetVisitState(1), DfsHloVisitor::kVisiting);
  EXPECT_EQ(visitor.GetVisitState(2), DfsHloVisitor::kVisited);
  EXPECT_EQ(visitor.GetVisitState(3), DfsHloVisitor::kNotVisited);

  visitor.ResetVisitStates();
  EXPECT_EQ(visitor.GetVisitState(1), DfsHloVisitor::kNotVisited);
  EXPECT_EQ(visitor.GetVisitState(2), DfsHloVisitor::kNotVisited);
  EXPECT_EQ(visitor.GetVisitState(3), DfsHloVisitor::kNotVisited);

  // States set after a reset are visible and stay independent of ids that
  // were only set before the reset.
  visitor.SetVisitState(2, DfsHloVisitor::kVisiting);
  EXPECT_EQ(visitor.GetVisitState(1), DfsHloVisitor::kNotVisited);
  EXPECT_EQ(visitor.GetVisitState(2), DfsHloVisitor::kVisiting);
  visitor.SetVisitState(2, DfsHloVisitor::kVisited);
  EXPECT_EQ(visitor.GetVisitState(2), DfsHloVisitor::kVisited);

  // Back to back resets keep clearing, including states set in between.
  for (int i = 0; i < 3; ++i) {
    visitor.SetVisitState(1, DfsHloVisitor::kVisited);
    visitor.ResetVisitStates();
    EXPECT_EQ(visitor.GetVisitState(1), DfsHloVisitor::kNotVisited);
    EXPECT_EQ(visitor.GetVisitState(2), DfsHloVisitor::kNotVisited);
  }
}

TEST(DfsHloVisitorVisitStateTest, DestroyVisitStateReleasesStorage) {
  constexpr int64_t kNumIds = 1000;
  RecordingVisitor visitor;
  for (int64_t id = 0; id < kNumIds; ++id) {
    visitor.SetVisitState(id, DfsHloVisitor::kVisited);
  }
  EXPECT_GE(visitor.VisitStateCapacity(), kNumIds);

  visitor.DestroyVisitState();
  // An empty flat_hash_map may keep a small inline capacity, so only check
  // that the storage shrank.
  EXPECT_LT(visitor.VisitStateCapacity(), kNumIds);
  EXPECT_EQ(visitor.GetVisitState(1), DfsHloVisitor::kNotVisited);
  visitor.SetVisitState(1, DfsHloVisitor::kVisiting);
  EXPECT_EQ(visitor.GetVisitState(1), DfsHloVisitor::kVisiting);
}

TEST_F(DfsHloVisitorTest, ResetVisitStatesDropsStaleEntriesOnceTheyDominate) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule TestModule

ENTRY TestComputation {
  arg = f32[] parameter(0)
  neg = f32[] negate(arg)
  ROOT abs = f32[] abs(neg)
})"));
  HloComputation* computation = module->entry_computation();
  // Ids that no instruction of the module has, standing in for instructions
  // that a pass deleted after visiting them.
  constexpr int64_t kNumStaleIds = 1000;
  constexpr int64_t kFirstStaleId = int64_t{1} << 40;

  RecordingVisitor visitor;
  for (int64_t i = 0; i < kNumStaleIds; ++i) {
    visitor.SetVisitState(kFirstStaleId + i, DfsHloVisitor::kVisited);
  }
  visitor.ResetVisitStates();
  EXPECT_GE(visitor.VisitStateCapacity(), kNumStaleIds);

  // Writing the same ids again in the next generation is not growth, so the
  // storage is kept.
  for (int64_t i = 0; i < kNumStaleIds; ++i) {
    visitor.SetVisitState(kFirstStaleId + i, DfsHloVisitor::kVisited);
  }
  visitor.ResetVisitStates();
  EXPECT_GE(visitor.VisitStateCapacity(), kNumStaleIds);

  // A visit that touches far fewer ids leaves the stale entries in the
  // majority, so the next reset releases them. Every id still reads as after
  // any other reset, and the visitor keeps working.
  ASSERT_OK(computation->Accept(&visitor));
  EXPECT_THAT(visitor.visited(), ElementsAre("arg", "neg", "abs"));
  visitor.ResetVisitStates();
  EXPECT_LT(visitor.VisitStateCapacity(), kNumStaleIds);
  EXPECT_EQ(visitor.GetVisitState(kFirstStaleId), DfsHloVisitor::kNotVisited);
  EXPECT_EQ(visitor.GetVisitState(kFirstStaleId + kNumStaleIds - 1),
            DfsHloVisitor::kNotVisited);
  EXPECT_TRUE(visitor.NotVisited(*computation->root_instruction()));

  visitor.ClearVisited();
  ASSERT_OK(computation->Accept(&visitor));
  EXPECT_THAT(visitor.visited(), ElementsAre("arg", "neg", "abs"));
  EXPECT_TRUE(visitor.DidVisit(*computation->root_instruction()));
  visitor.SetVisitState(kFirstStaleId, DfsHloVisitor::kVisiting);
  EXPECT_EQ(visitor.GetVisitState(kFirstStaleId), DfsHloVisitor::kVisiting);
  EXPECT_EQ(visitor.GetVisitState(kFirstStaleId + 1),
            DfsHloVisitor::kNotVisited);
}

TEST_F(DfsHloVisitorTest, ResetVisitStatesLetsTheSameVisitorTraverseAgain) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule TestModule

ENTRY TestComputation {
  arg = f32[] parameter(0)
  neg = f32[] negate(arg)
  ROOT abs = f32[] abs(neg)
})"));
  HloComputation* computation = module->entry_computation();

  RecordingVisitor visitor;
  ASSERT_OK(computation->Accept(&visitor));
  EXPECT_THAT(visitor.visited(), ElementsAre("arg", "neg", "abs"));

  // Without a reset every instruction is still marked visited.
  visitor.ClearVisited();
  ASSERT_OK(computation->Accept(&visitor));
  EXPECT_THAT(visitor.visited(), ElementsAre());

  visitor.ResetVisitStates();
  ASSERT_OK(computation->Accept(&visitor));
  EXPECT_THAT(visitor.visited(), ElementsAre("arg", "neg", "abs"));
}

TEST_F(DfsHloVisitorTest, SetVisitedSkipsInstructionUntilReset) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule TestModule

ENTRY TestComputation {
  arg = f32[] parameter(0)
  neg = f32[] negate(arg)
  ROOT abs = f32[] abs(neg)
})"));
  HloComputation* computation = module->entry_computation();
  HloInstruction* neg = FindInstruction(module.get(), "neg");

  RecordingVisitor visitor;
  visitor.SetVisited(*neg);
  EXPECT_TRUE(visitor.DidVisit(*neg));
  // Neither neg nor arg, which is only reachable through neg, is visited.
  ASSERT_OK(computation->Accept(&visitor));
  EXPECT_THAT(visitor.visited(), ElementsAre("abs"));

  visitor.ResetVisitStates();
  EXPECT_TRUE(visitor.NotVisited(*neg));
  visitor.ClearVisited();
  ASSERT_OK(computation->Accept(&visitor));
  EXPECT_THAT(visitor.visited(), ElementsAre("arg", "neg", "abs"));

  // States set after a reset are marked with a later generation and must
  // still be visible to the traversal.
  visitor.ResetVisitStates();
  visitor.SetVisited(*neg);
  EXPECT_TRUE(visitor.DidVisit(*neg));
  visitor.ClearVisited();
  ASSERT_OK(computation->Accept(&visitor));
  EXPECT_THAT(visitor.visited(), ElementsAre("abs"));

  visitor.ResetVisitStates();
  visitor.SetVisiting(*neg);
  EXPECT_TRUE(visitor.IsVisiting(*neg));
  EXPECT_FALSE(visitor.DidVisit(*neg));
  EXPECT_FALSE(visitor.NotVisited(*neg));
}

}  // namespace
}  // namespace xla
