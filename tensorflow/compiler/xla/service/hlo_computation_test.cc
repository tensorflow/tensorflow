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
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {

namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

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
  auto module = CreateNewModule();
  auto negate_computation =
      module->AddEntryComputation(CreateNegateComputation());
  EXPECT_TRUE(negate_computation->MakeEmbeddedComputationsList().empty());
}

TEST_F(HloComputationTest, GetEmbeddedComputationsOneComputation) {
  // Create computation which calls one other computation.
  auto module = CreateNewModule();
  auto negate_computation =
      module->AddEmbeddedComputation(CreateNegateComputation());
  auto map_computation =
      module->AddEntryComputation(CreateMapComputation(negate_computation));
  EXPECT_TRUE(negate_computation->MakeEmbeddedComputationsList().empty());
  EXPECT_THAT(map_computation->MakeEmbeddedComputationsList(),
              ElementsAre(negate_computation));
}

TEST_F(HloComputationTest, GetEmbeddedComputationsDiamond) {
  // Create computations with a diamond-shaped callgraph.
  auto module = CreateNewModule();
  auto negate_computation =
      module->AddEmbeddedComputation(CreateNegateComputation());
  auto map1_computation =
      module->AddEmbeddedComputation(CreateMapComputation(negate_computation));
  auto map2_computation =
      module->AddEmbeddedComputation(CreateMapComputation(negate_computation));

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32_, "param0"));
  auto map1 = builder.AddInstruction(
      HloInstruction::CreateMap(r0f32_, {param}, map1_computation));
  auto map2 = builder.AddInstruction(
      HloInstruction::CreateMap(r0f32_, {param}, map2_computation));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, map1, map2));
  auto computation = module->AddEntryComputation(builder.Build());

  auto embedded_computations = computation->MakeEmbeddedComputationsList();
  EXPECT_EQ(3, embedded_computations.size());
  // GetEmbeddedComputations returns a post order of the embedded computations,
  // so the negate computation must come first.
  EXPECT_EQ(negate_computation, *embedded_computations.begin());
  EXPECT_THAT(embedded_computations,
              UnorderedElementsAre(negate_computation, map1_computation,
                                   map2_computation));
}

TEST_F(HloComputationTest, PostOrderSingleton) {
  // Test GetInstructionPostOrder for a computation with one instruction.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->MakeInstructionPostOrder(), ElementsAre(constant));
}

TEST_F(HloComputationTest, PostOrderSimple) {
  // Test GetInstructionPostOrder for a computation with a chain of
  // instructions.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, negate1));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->MakeInstructionPostOrder(),
              ElementsAre(constant, negate1, negate2));
}

TEST_F(HloComputationTest, PostOrderTrace) {
  // Test GetInstructionPostOrder for a computation with a trace instruction.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto trace =
      builder.AddInstruction(HloInstruction::CreateTrace("foobar", negate1));
  auto negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, negate1));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  // Trace instructions should be at the end of the sort.
  EXPECT_THAT(computation->MakeInstructionPostOrder(),
              ElementsAre(constant, negate1, negate2, trace));
}

TEST_F(HloComputationTest, PostOrderDisconnectedInstructions) {
  // Test GetInstructionPostOrder for a computation with multiple instructions
  // which are not connected.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto constant4 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->MakeInstructionPostOrder(),
              UnorderedElementsAre(constant1, constant2, constant3, constant4));
}

TEST_F(HloComputationTest, PostOrderWithMultipleRoots) {
  // Test GetInstructionPostOrder for a computation with multiple instructions
  // which are not connected.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant2));
  auto add2 = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant2, constant3));
  auto add3 = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant3));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto post_order = computation->MakeInstructionPostOrder();
  EXPECT_EQ(6, post_order.size());
  EXPECT_THAT(post_order, UnorderedElementsAre(constant1, constant2, constant3,
                                               add1, add2, add3));
}

TEST_F(HloComputationTest, VisitWithMultipleRoots) {
  // Test that Accept visits all instructions in the computation even if the
  // computation has multiple roots (dead code).
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  // Add three disconnected add expressions.
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                      constant1, constant2));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                      constant2, constant3));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                      constant1, constant3));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
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

  TestVisitor visitor(computation);
  EXPECT_IS_OK(computation->Accept(&visitor));

  EXPECT_EQ(6, visitor.visited_set_.size());
  EXPECT_EQ(1, visitor.finish_visit_calls_);
  EXPECT_EQ(computation->root_instruction(), visitor.last_visited_);
}

TEST_F(HloComputationTest, DeepCopyArray) {
  // Test that DeepCopyInstruction properly copies an array.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR1<float>({1.0, 2.0, 3.0})));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto copy = computation->DeepCopyInstruction(constant).ValueOrDie();

  EXPECT_THAT(copy, op::Copy(constant));
}

TEST_F(HloComputationTest, DeepCopyTuple) {
  // Test that DeepCopyInstruction properly copies a tuple.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR1<float>({1.0, 2.0, 3.0})));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto tuple_copy = computation->DeepCopyInstruction(tuple).ValueOrDie();

  EXPECT_THAT(tuple_copy, op::Tuple(op::Copy(op::GetTupleElement(tuple)),
                                    op::Copy(op::GetTupleElement(tuple))));
  EXPECT_EQ(0, tuple_copy->operand(0)->operand(0)->tuple_index());
  EXPECT_EQ(1, tuple_copy->operand(1)->operand(0)->tuple_index());
}

TEST_F(HloComputationTest, DeepCopyArrayAtIndices) {
  // Test that DeepCopyInstruction properly handles an array when the indices to
  // copy are specified.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR1<float>({1.0, 2.0, 3.0})));
  auto computation = builder.Build();

  {
    // If the index is true, then a copy should be made.
    ShapeTree<bool> indices_to_copy(constant->shape(), /*init_value=*/true);
    EXPECT_THAT(computation->DeepCopyInstruction(constant, &indices_to_copy)
                    .ValueOrDie(),
                op::Copy(constant));
  }

  {
    // If the index is false, then no copy should be made.
    ShapeTree<bool> indices_to_copy(constant->shape(), /*init_value=*/false);
    EXPECT_EQ(computation->DeepCopyInstruction(constant, &indices_to_copy)
                  .ValueOrDie(),
              constant);
  }
}

TEST_F(HloComputationTest, DeepCopyTupleAtIndices) {
  // Test that DeepCopyInstruction properly copies elements of a tuple as
  // specified by the given indices.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR1<float>({1.0, 2.0, 3.0})));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto computation = builder.Build();

  {
    // All true values should copy all array elements.
    ShapeTree<bool> indices_to_copy(tuple->shape(), /*init_value=*/true);
    ShapeTree<HloInstruction*> copies_added(tuple->shape(),
                                            /*init_value=*/nullptr);
    HloInstruction* deep_copy =
        computation->DeepCopyInstruction(tuple, &indices_to_copy, &copies_added)
            .ValueOrDie();

    EXPECT_THAT(deep_copy, op::Tuple(op::Copy(op::GetTupleElement(tuple)),
                                     op::Copy(op::GetTupleElement(tuple))));
    EXPECT_THAT(deep_copy, op::Tuple(copies_added.element({0}),
                                     copies_added.element({1})));
  }

  {
    // All false elements should copy no array elements, but the GTE and tuple
    // instruction scaffolding should be built.
    ShapeTree<bool> indices_to_copy(tuple->shape(), /*init_value=*/false);
    ShapeTree<HloInstruction*> copies_added(tuple->shape(),
                                            /*init_value=*/nullptr);
    HloInstruction* deep_copy =
        computation->DeepCopyInstruction(tuple, &indices_to_copy, &copies_added)
            .ValueOrDie();

    EXPECT_THAT(deep_copy, op::Tuple(op::GetTupleElement(tuple),
                                     op::GetTupleElement(tuple)));
    EXPECT_TRUE(copies_added.element({}) == nullptr);
    EXPECT_TRUE(copies_added.element({0}) == nullptr);
    EXPECT_TRUE(copies_added.element({1}) == nullptr);
  }

  {
    // Verify one element copied, the other not.
    ShapeTree<bool> indices_to_copy(tuple->shape(), /*init_value=*/false);
    *indices_to_copy.mutable_element({0}) = true;
    ShapeTree<HloInstruction*> copies_added(tuple->shape(),
                                            /*init_value=*/nullptr);
    HloInstruction* deep_copy =
        computation->DeepCopyInstruction(tuple, &indices_to_copy, &copies_added)
            .ValueOrDie();

    EXPECT_THAT(deep_copy, op::Tuple(op::Copy(op::GetTupleElement(tuple)),
                                     op::GetTupleElement(tuple)));
    EXPECT_TRUE(copies_added.element({}) == nullptr);
    EXPECT_TRUE(copies_added.element({0}) != nullptr);
    EXPECT_TRUE(copies_added.element({1}) == nullptr);
  }
}

TEST_F(HloComputationTest, CycleDetection) {
  // Test whether the visitor can detect cycles in the graph.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, negate, negate));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  // Add a control dependency to create a cycle.
  ASSERT_IS_OK(add->AddControlDependencyTo(negate));

  const auto visitor = [](HloInstruction* instruction) { return Status::OK(); };
  auto visit_status = computation->Accept(visitor);
  ASSERT_FALSE(visit_status.ok());
  ASSERT_THAT(visit_status.error_message(),
              ::testing::ContainsRegex("cycle is detecte"));
}

TEST_F(HloComputationTest, RemoveInstructionWithDuplicateOperand) {
  // Test RemoveInstructionAndUnusedOperands with an instruction which has a
  // duplicated (dead) operand. This verifies that the operand is not deleted
  // twice.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto dead_negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto dead_add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, dead_negate, dead_negate));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(computation->root_instruction(), op::Negate(constant));
  EXPECT_EQ(negate, computation->root_instruction());

  ASSERT_IS_OK(computation->RemoveInstructionAndUnusedOperands(dead_add));

  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_THAT(computation->root_instruction(), op::Negate(constant));
  EXPECT_EQ(negate, computation->root_instruction());
}

TEST_F(HloComputationTest, CloneWithControlDependency) {
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0f)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant2));

  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32_, "param0"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, param));
  auto module = CreateNewModule();
  auto computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/add));

  TF_CHECK_OK(negate->AddControlDependencyTo(add));

  auto clone = computation->Clone();

  auto cloned_add = clone->root_instruction();
  EXPECT_EQ(cloned_add->opcode(), HloOpcode::kAdd);

  auto predecessors = cloned_add->control_predecessors();
  EXPECT_EQ(1, predecessors.size());
  EXPECT_EQ(HloOpcode::kNegate, predecessors[0]->opcode());
  auto successors = predecessors[0]->control_successors();
  EXPECT_THAT(successors, ::testing::ElementsAre(cloned_add));
}

TEST_F(HloComputationTest, Reachability) {
  // Test reachability of a non-trivial computation:
  //
  // const1    const2
  //    |         |
  //    | +-------+
  //    | |       |
  //    add ..   negate
  //     |   .     |
  //     |   .... exp
  //     |         |
  //     +---+   +-+---+
  //         |   |     |
  //       multiply   copy
  //
  // There is a control dependency from 'add' to 'exp'.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0f)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant2));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant2));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, negate));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kMultiply, add, exp));
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kCopy, exp));

  auto module = CreateNewModule();
  auto computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/mul));

  TF_CHECK_OK(add->AddControlDependencyTo(exp));
  auto reachability = computation->ComputeReachability();

  EXPECT_TRUE(reachability->IsReachable(constant1, constant1));
  EXPECT_FALSE(reachability->IsReachable(constant1, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant1, add));
  EXPECT_FALSE(reachability->IsReachable(constant1, negate));
  EXPECT_TRUE(reachability->IsReachable(constant1, exp));
  EXPECT_TRUE(reachability->IsReachable(constant1, mul));
  EXPECT_TRUE(reachability->IsReachable(constant1, copy));

  EXPECT_FALSE(reachability->IsReachable(constant2, constant1));
  EXPECT_TRUE(reachability->IsReachable(constant2, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant2, add));
  EXPECT_TRUE(reachability->IsReachable(constant2, negate));
  EXPECT_TRUE(reachability->IsReachable(constant2, exp));
  EXPECT_TRUE(reachability->IsReachable(constant2, mul));
  EXPECT_TRUE(reachability->IsReachable(constant2, copy));

  EXPECT_FALSE(reachability->IsReachable(exp, constant1));
  EXPECT_FALSE(reachability->IsReachable(exp, constant2));
  EXPECT_FALSE(reachability->IsReachable(exp, add));
  EXPECT_FALSE(reachability->IsReachable(exp, negate));
  EXPECT_TRUE(reachability->IsReachable(exp, exp));
  EXPECT_TRUE(reachability->IsReachable(exp, mul));
  EXPECT_TRUE(reachability->IsReachable(exp, copy));

  EXPECT_FALSE(reachability->IsReachable(mul, constant1));
  EXPECT_FALSE(reachability->IsReachable(mul, constant2));
  EXPECT_FALSE(reachability->IsReachable(mul, add));
  EXPECT_FALSE(reachability->IsReachable(mul, negate));
  EXPECT_FALSE(reachability->IsReachable(mul, exp));
  EXPECT_TRUE(reachability->IsReachable(mul, mul));
  EXPECT_FALSE(reachability->IsReachable(mul, copy));

  EXPECT_TRUE(reachability->IsConnected(constant1, copy));
  EXPECT_TRUE(reachability->IsConnected(copy, constant1));
  EXPECT_FALSE(reachability->IsConnected(negate, add));
  EXPECT_FALSE(reachability->IsConnected(add, negate));

  // Remove the control dependency then update and verify the reachability map
  ASSERT_IS_OK(add->RemoveControlDependencyTo(exp));
  computation->UpdateReachabilityThroughInstruction(exp, reachability.get());

  EXPECT_TRUE(reachability->IsReachable(constant1, constant1));
  EXPECT_FALSE(reachability->IsReachable(constant1, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant1, add));
  EXPECT_FALSE(reachability->IsReachable(constant1, negate));
  EXPECT_FALSE(reachability->IsReachable(constant1, exp));
  EXPECT_TRUE(reachability->IsReachable(constant1, mul));
  EXPECT_FALSE(reachability->IsReachable(constant1, copy));

  // Change a use within the graph then update and verify the reachability map
  ASSERT_IS_OK(constant2->ReplaceUseWith(negate, constant1));
  computation->UpdateReachabilityThroughInstruction(negate, reachability.get());

  EXPECT_FALSE(reachability->IsReachable(constant2, constant1));
  EXPECT_TRUE(reachability->IsReachable(constant2, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant2, add));
  EXPECT_FALSE(reachability->IsReachable(constant2, negate));
  EXPECT_FALSE(reachability->IsReachable(constant2, exp));
  EXPECT_TRUE(reachability->IsReachable(constant2, mul));
  EXPECT_FALSE(reachability->IsReachable(constant2, copy));
}

}  // namespace

}  // namespace xla
