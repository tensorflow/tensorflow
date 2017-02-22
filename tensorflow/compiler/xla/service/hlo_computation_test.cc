/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_computation.h"

#include <set>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {

namespace {

class HloComputationTest : public HloTestBase {
 protected:
  HloComputationTest() {}

  // Create a computation which takes a scalar and returns its negation.
  std::unique_ptr<HloComputation> CreateNegateComputation() {
    auto builder = HloComputation::Builder("Negate");
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, r0f32_, "param0"));
    builder.AddInstruction(
        HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, param));
    return builder.Build();
  }

  // Creates a computation which calls map with the given computation.
  std::unique_ptr<HloComputation> CreateMapComputation(
      HloComputation* map_computation) {
    auto builder = HloComputation::Builder("Map");
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, r0f32_, "param0"));
    builder.AddInstruction(
        HloInstruction::CreateMap(r0f32_, {param}, map_computation));
    return builder.Build();
  }

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
};

TEST_F(HloComputationTest, GetEmbeddedComputationsEmpty) {
  auto negate_computation = CreateNegateComputation();
  EXPECT_TRUE(negate_computation->MakeEmbeddedComputationsList().empty());
}

TEST_F(HloComputationTest, GetEmbeddedComputationsOneComputation) {
  // Create computation which calls one other computation.
  auto negate_computation = CreateNegateComputation();
  auto map_computation = CreateMapComputation(negate_computation.get());
  EXPECT_TRUE(negate_computation->MakeEmbeddedComputationsList().empty());
  EXPECT_EQ(map_computation->MakeEmbeddedComputationsList().front(),
            negate_computation.get());
}

TEST_F(HloComputationTest, GetEmbeddedComputationsDiamond) {
  // Create computations with a diamond-shaped callgraph.
  auto negate_computation = CreateNegateComputation();
  auto map1_computation = CreateMapComputation(negate_computation.get());
  auto map2_computation = CreateMapComputation(negate_computation.get());

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32_, "param0"));
  auto map1 = builder.AddInstruction(
      HloInstruction::CreateMap(r0f32_, {param}, map1_computation.get()));
  auto map2 = builder.AddInstruction(
      HloInstruction::CreateMap(r0f32_, {param}, map2_computation.get()));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, map1, map2));
  auto computation = builder.Build();

  auto embedded_computations = computation->MakeEmbeddedComputationsList();
  EXPECT_EQ(3, embedded_computations.size());
  // GetEmbeddedComputations returns a post order of the embedded computations,
  // so the negate computation must come first.
  EXPECT_EQ(negate_computation.get(), *embedded_computations.begin());
  EXPECT_MATCH(testing::ListToVec<HloComputation*>(embedded_computations),
               testing::UnorderedMatcher<HloComputation*>(
                   negate_computation.get(), map1_computation.get(),
                   map2_computation.get()));
}

TEST_F(HloComputationTest, PostOrderSingleton) {
  // Test GetInstructionPostOrder for a computation with one instruction.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto computation = builder.Build();

  EXPECT_EQ(computation->MakeInstructionPostOrder().front(), constant);
}

TEST_F(HloComputationTest, PostOrderSimple) {
  // Test GetInstructionPostOrder for a computation with a chain of
  // instructions.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, negate1));
  auto computation = builder.Build();

  EXPECT_MATCH(
      testing::ListToVec<HloInstruction*>(
          computation->MakeInstructionPostOrder()),
      testing::OrderedMatcher<HloInstruction*>(constant, negate1, negate2));
}

TEST_F(HloComputationTest, PostOrderTrace) {
  // Test GetInstructionPostOrder for a computation with a trace instruction.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto trace =
      builder.AddInstruction(HloInstruction::CreateTrace("foobar", negate1));
  auto negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, negate1));
  auto computation = builder.Build();

  // Trace instructions should be at the end of the sort.
  EXPECT_MATCH(testing::ListToVec<HloInstruction*>(
                   computation->MakeInstructionPostOrder()),
               testing::OrderedMatcher<HloInstruction*>(constant, negate1,
                                                        negate2, trace));
}

TEST_F(HloComputationTest, PostOrderDisconnectedInstructions) {
  // Test GetInstructionPostOrder for a computation with multiple instructions
  // which are not connected.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant4 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto computation = builder.Build();

  EXPECT_MATCH(testing::ListToVec<HloInstruction*>(
                   computation->MakeInstructionPostOrder()),
               testing::UnorderedMatcher<HloInstruction*>(
                   constant1, constant2, constant3, constant4));
}

TEST_F(HloComputationTest, PostOrderWithMultipleRoots) {
  // Test GetInstructionPostOrder for a computation with multiple instructions
  // which are not connected.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant2));
  auto add2 = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant2, constant3));
  auto add3 = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant3));
  auto computation = builder.Build();

  auto post_order = computation->MakeInstructionPostOrder();
  EXPECT_EQ(6, post_order.size());
  EXPECT_MATCH(testing::ListToVec<HloInstruction*>(post_order),
               testing::UnorderedMatcher<HloInstruction*>(
                   constant1, constant2, constant3, add1, add2, add3));
}

TEST_F(HloComputationTest, VisitWithMultipleRoots) {
  // Test that Accept visits all instructions in the computation even if the
  // computation has multiple roots (dead code).
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  // Add three disconnected add expressions.
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                      constant1, constant2));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                      constant2, constant3));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                      constant1, constant3));
  auto computation = builder.Build();

  // Visitor which keeps track of which instructions have been visited.
  class TestVisitor : public DfsHloVisitorWithDefault {
   public:
    explicit TestVisitor(HloComputation* computation)
        : computation_(computation) {}

    Status DefaultAction(HloInstruction* hlo_instruction) override {
      EXPECT_EQ(0, visited_set_.count(hlo_instruction));
      visited_set_.insert(hlo_instruction);
      last_visited_ = hlo_instruction;
      return Status::OK();
    }

    Status FinishVisit(HloInstruction* root) override {
      EXPECT_EQ(computation_->root_instruction(), root);
      ++finish_visit_calls_;
      return Status::OK();
    }

    HloComputation* computation_;
    std::set<HloInstruction*> visited_set_;
    int64 finish_visit_calls_ = 0;
    HloInstruction* last_visited_ = nullptr;
  };

  TestVisitor visitor(computation.get());
  EXPECT_IS_OK(computation->Accept(&visitor));

  EXPECT_EQ(6, visitor.visited_set_.size());
  EXPECT_EQ(1, visitor.finish_visit_calls_);
  EXPECT_EQ(computation->root_instruction(), visitor.last_visited_);
}

TEST_F(HloComputationTest, DeepCopyArray) {
  // Test that DeepCopyInstruction properly copies an array.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0})));
  auto computation = builder.Build();

  auto copy = computation->DeepCopyInstruction(constant).ValueOrDie();

  EXPECT_EQ(HloOpcode::kCopy, copy->opcode());
  EXPECT_EQ(constant, copy->operand(0));
}

TEST_F(HloComputationTest, DeepCopyTuple) {
  // Test that DeepCopyInstruction properly copies a tuple.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0})));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));

  auto computation = builder.Build();

  auto tuple_copy = computation->DeepCopyInstruction(tuple).ValueOrDie();

  EXPECT_EQ(HloOpcode::kTuple, tuple_copy->opcode());
  EXPECT_EQ(HloOpcode::kCopy, tuple_copy->operand(0)->opcode());
  const HloInstruction* gte0 = tuple_copy->operand(0)->operand(0);
  EXPECT_EQ(HloOpcode::kGetTupleElement, gte0->opcode());
  EXPECT_EQ(0, gte0->tuple_index());
  EXPECT_EQ(tuple, gte0->operand(0));

  EXPECT_EQ(HloOpcode::kCopy, tuple_copy->operand(1)->opcode());
  const HloInstruction* gte1 = tuple_copy->operand(1)->operand(0);
  EXPECT_EQ(HloOpcode::kGetTupleElement, gte1->opcode());
  EXPECT_EQ(1, gte1->tuple_index());
  EXPECT_EQ(tuple, gte1->operand(0));
}

TEST_F(HloComputationTest, CycleDetection) {
  // Test whether the visitor can detect cycles in the graph.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, negate, negate));
  auto computation = builder.Build();

  // Add a control dependency to create a cycle.
  ASSERT_IS_OK(computation->AddControlDependency(add, negate));

  const auto visitor = [](HloInstruction* instruction) { return Status::OK(); };
  auto visit_status = computation->Accept(visitor);
  ASSERT_FALSE(visit_status.ok());
  ASSERT_MATCH(visit_status.error_message(),
               testing::ContainsRegex("cycle is detecte"));
}

TEST_F(HloComputationTest, RemoveInstructionWithDuplicateOperand) {
  // Test RemoveInstructionAndUnusedOperands with an instruction which has a
  // duplicated (dead) operand. This verifies that the operand is not deleted
  // twice.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto dead_negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto dead_add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, dead_negate, dead_negate));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto computation = builder.Build();

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_EQ(negate, computation->root_instruction());

  ASSERT_IS_OK(computation->RemoveInstructionAndUnusedOperands(dead_add));

  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_EQ(negate, computation->root_instruction());
}

}  // namespace

}  // namespace xla
