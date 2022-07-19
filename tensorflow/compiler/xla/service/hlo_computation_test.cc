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

#include <memory>
#include <set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {

namespace {

namespace m = match;
namespace op = xla::testing::opcode_matchers;
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
  auto module = CreateNewVerifiedModule();
  auto negate_computation =
      module->AddEntryComputation(CreateNegateComputation());
  EXPECT_TRUE(negate_computation->MakeEmbeddedComputationsList().empty());
}

TEST_F(HloComputationTest, GetEmbeddedComputationsOneComputation) {
  // Create computation which calls one other computation.
  auto module = CreateNewVerifiedModule();
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
  auto module = CreateNewVerifiedModule();
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->MakeInstructionPostOrder(), ElementsAre(constant));
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
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->MakeInstructionPostOrder(),
              ElementsAre(constant, negate1, negate2));
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
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->MakeInstructionPostOrder(),
              UnorderedElementsAre(constant1, constant2, constant3, constant4));
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
  auto module = CreateNewVerifiedModule();
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
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  // Visitor which keeps track of which instructions have been visited.
  class TestVisitor : public DfsHloVisitorWithDefault {
   public:
    explicit TestVisitor(HloComputation* computation)
        : computation_(computation) {}

    Status DefaultAction(HloInstruction* hlo_instruction) override {
      EXPECT_FALSE(visited_set_.contains(hlo_instruction));
      visited_set_.insert(hlo_instruction);
      last_visited_ = hlo_instruction;
      return OkStatus();
    }

    Status FinishVisit(HloInstruction* root) override {
      EXPECT_EQ(computation_->root_instruction(), root);
      ++finish_visit_calls_;
      return OkStatus();
    }

    HloComputation* computation_;
    absl::flat_hash_set<HloInstruction*> visited_set_;
    int64_t finish_visit_calls_ = 0;
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
      LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0})));
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto copy = computation->DeepCopyInstruction(constant).ValueOrDie();

  EXPECT_THAT(copy, GmockMatch(m::Copy(m::Op().Is(constant))));
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

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto tuple_copy = computation->DeepCopyInstruction(tuple).ValueOrDie();

  EXPECT_THAT(tuple_copy, GmockMatch(m::Tuple(
                              m::Copy(m::GetTupleElement(m::Op().Is(tuple))),
                              m::Copy(m::GetTupleElement(m::Op().Is(tuple))))));
  EXPECT_EQ(0, tuple_copy->operand(0)->operand(0)->tuple_index());
  EXPECT_EQ(1, tuple_copy->operand(1)->operand(0)->tuple_index());
}

TEST_F(HloComputationTest, DeepCopyArrayAtIndices) {
  // Test that DeepCopyInstruction properly handles an array when the indices to
  // copy are specified.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0})));
  auto computation = builder.Build();

  {
    // If the index is true, then a copy should be made.
    ShapeTree<bool> indices_to_copy(constant->shape(), /*init_value=*/true);
    EXPECT_THAT(computation->DeepCopyInstruction(constant, &indices_to_copy)
                    .ValueOrDie(),
                GmockMatch(m::Copy(m::Op().Is(constant))));
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
      LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0})));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
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

    EXPECT_THAT(deep_copy, GmockMatch(m::Tuple(
                               m::Copy(m::GetTupleElement(m::Op().Is(tuple)))
                                   .Is(copies_added.element({0})),
                               m::Copy(m::GetTupleElement(m::Op().Is(tuple)))
                                   .Is(copies_added.element({1})))));
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

    EXPECT_THAT(deep_copy,
                GmockMatch(m::Tuple(m::GetTupleElement(m::Op().Is(tuple)),
                                    m::GetTupleElement(m::Op().Is(tuple)))));
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

    EXPECT_THAT(deep_copy, GmockMatch(m::Tuple(
                               m::Copy(m::GetTupleElement(m::Op().Is(tuple))),
                               m::GetTupleElement(m::Op().Is(tuple)))));
    EXPECT_TRUE(copies_added.element({}) == nullptr);
    EXPECT_TRUE(copies_added.element({0}) != nullptr);
    EXPECT_TRUE(copies_added.element({1}) == nullptr);
  }
}

TEST_F(HloComputationTest, DeepCopyToken) {
  // Test that DeepCopyInstruction properly handles tokens which should not be
  // copied.
  auto builder = HloComputation::Builder(TestName());
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto copy = computation->DeepCopyInstruction(token).ValueOrDie();

  // No copy should be added.
  EXPECT_THAT(copy, GmockMatch(m::AfterAll()));
}

TEST_F(HloComputationTest, DeepCopyTokenTuple) {
  // Test that DeepCopyInstruction properly handles tokens which should not be
  // copied.
  auto builder = HloComputation::Builder(TestName());
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({token, constant}));
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto copy = computation->DeepCopyInstruction(tuple).ValueOrDie();

  // Only the array (second tuple element) should be copied. The token is passed
  // through transparently.
  EXPECT_THAT(copy, GmockMatch(m::Tuple(
                        m::GetTupleElement(m::Op().Is(tuple)),
                        m::Copy(m::GetTupleElement(m::Op().Is(tuple))))));
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
  auto module = CreateNewUnverifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  // Add a control dependency to create a cycle.
  ASSERT_IS_OK(add->AddControlDependencyTo(negate));

  auto instructions = computation->MakeInstructionPostOrder();
  EXPECT_EQ(3, instructions.size());

  FunctionVisitor visitor(
      [](HloInstruction* instruction) { return OkStatus(); });
  auto visit_status = computation->Accept(&visitor);
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto dead_negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto dead_add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, dead_negate, dead_negate));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, constant));
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Negate(m::Op().Is(constant))));
  EXPECT_EQ(negate, computation->root_instruction());

  ASSERT_IS_OK(computation->RemoveInstructionAndUnusedOperands(dead_add));

  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Negate(m::Op().Is(constant))));
  EXPECT_EQ(negate, computation->root_instruction());
}

TEST_F(HloComputationTest, CloneWithControlDependency) {
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant2));

  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32_, "param0"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, param));
  auto module = CreateNewVerifiedModule();
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

TEST_F(HloComputationTest, CloneWithReplacements) {
  auto builder = HloComputation::Builder(TestName());
  Shape r0s64 = ShapeUtil::MakeShape(S64, {});
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  Shape r0u32 = ShapeUtil::MakeShape(U32, {});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32_, "p.0.lhs"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32_, "p.0.rhs"));
  auto param2 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, r0s64, "p.1"));
  auto lt = builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param0,
                                    param1, ComparisonDirection::kLt));
  auto module = CreateNewVerifiedModule();
  auto computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/lt));
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  replacements.emplace(param2,
                       HloInstruction::CreateParameter(2, r0s32, "p.1"));
  auto param3 = HloInstruction::CreateParameter(3, r0u32, "p.2");
  std::vector<const HloInstruction*> extra_parameters{param3.get()};
  auto clone =
      computation->CloneWithReplacements(&replacements, extra_parameters);
  ASSERT_EQ(clone->num_parameters(), 4);
  EXPECT_TRUE(
      ShapeUtil::Equal(clone->parameter_instruction(0)->shape(), r0f32_));
  EXPECT_TRUE(
      ShapeUtil::Equal(clone->parameter_instruction(1)->shape(), r0f32_));
  EXPECT_TRUE(
      ShapeUtil::Equal(clone->parameter_instruction(2)->shape(), r0s32));
  EXPECT_TRUE(
      ShapeUtil::Equal(clone->parameter_instruction(3)->shape(), r0u32));
}

TEST_F(HloComputationTest, Stringification) {
  const Shape s1 = ShapeUtil::MakeShape(F32, {5, 10});
  const Shape s2 = ShapeUtil::MakeShape(F32, {20, 10});
  const Shape s2t = ShapeUtil::MakeShape(F32, {10, 20});
  const Shape sout = ShapeUtil::MakeShape(F32, {5, 20});

  HloComputation::Builder builder("TransposeDot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "y"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateTranspose(s2t, y, {1, 0}));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  builder.AddInstruction(
      HloInstruction::CreateDot(sout, x, reshape, dot_dnums, precision_config));
  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  computation->SetExecutionThread("MainThread");

  auto options = HloPrintOptions().set_print_metadata(false);
  const std::string expected_computation =
      R"(%TransposeDot (x: f32[5,10], y: f32[20,10]) -> f32[5,20] {
  %x = f32[5,10]{1,0} parameter(0)
  %y = f32[20,10]{1,0} parameter(1)
  %transpose = f32[10,20]{1,0} transpose(f32[20,10]{1,0} %y), dimensions={1,0}
  ROOT %dot = f32[5,20]{1,0} dot(f32[5,10]{1,0} %x, f32[10,20]{1,0} %transpose), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}, execution_thread="MainThread")";
  EXPECT_EQ(computation->ToString(options), expected_computation);
}

TEST_F(HloComputationTest, StringificationIndent) {
  const Shape s1 = ShapeUtil::MakeShape(F32, {5, 10});
  const Shape s2 = ShapeUtil::MakeShape(F32, {20, 10});
  const Shape s2t = ShapeUtil::MakeShape(F32, {10, 20});
  const Shape sout = ShapeUtil::MakeShape(F32, {5, 20});

  HloComputation::Builder builder("TransposeDot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "y"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateTranspose(s2t, y, {1, 0}));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  builder.AddInstruction(
      HloInstruction::CreateDot(sout, x, reshape, dot_dnums, precision_config));
  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  computation->SetExecutionThread("MainThread");

  auto options =
      HloPrintOptions().set_print_metadata(false).set_indent_amount(2);
  const std::string expected_computation =
      R"(    %TransposeDot (x: f32[5,10], y: f32[20,10]) -> f32[5,20] {
      %x = f32[5,10]{1,0} parameter(0)
      %y = f32[20,10]{1,0} parameter(1)
      %transpose = f32[10,20]{1,0} transpose(f32[20,10]{1,0} %y), dimensions={1,0}
      ROOT %dot = f32[5,20]{1,0} dot(f32[5,10]{1,0} %x, f32[10,20]{1,0} %transpose), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }, execution_thread="MainThread")";
  EXPECT_EQ(computation->ToString(options), expected_computation);
}

TEST_F(HloComputationTest, StringificationCanonical) {
  const Shape s1 = ShapeUtil::MakeShape(F32, {5, 10});
  const Shape s2 = ShapeUtil::MakeShape(F32, {20, 10});
  const Shape s2t = ShapeUtil::MakeShape(F32, {10, 20});
  const Shape sout = ShapeUtil::MakeShape(F32, {5, 20});

  HloComputation::Builder builder("TransposeDot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "y"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateTranspose(s2t, y, {1, 0}));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  builder.AddInstruction(
      HloInstruction::CreateDot(sout, x, reshape, dot_dnums, precision_config));
  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  computation->SetExecutionThread("MainThread");

  auto options = HloPrintOptions().set_print_metadata(false);
  const std::string expected_computation1 =
      R"(%TransposeDot (x: f32[5,10], y: f32[20,10]) -> f32[5,20] {
  %x = f32[5,10]{1,0} parameter(0)
  %y = f32[20,10]{1,0} parameter(1)
  %transpose = f32[10,20]{1,0} transpose(f32[20,10]{1,0} %y), dimensions={1,0}
  ROOT %dot = f32[5,20]{1,0} dot(f32[5,10]{1,0} %x, f32[10,20]{1,0} %transpose), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}, execution_thread="MainThread")";
  EXPECT_EQ(computation->ToString(options), expected_computation1);

  options = HloPrintOptions().Canonical();
  const std::string expected_computation2 = R"(TransposeDot {
  tmp_0 = f32[5,10]{1,0} parameter(0)
  tmp_1 = f32[20,10]{1,0} parameter(1)
  tmp_2 = f32[10,20]{1,0} transpose(f32[20,10]{1,0} tmp_1), dimensions={1,0}
  ROOT tmp_3 = f32[5,20]{1,0} dot(f32[5,10]{1,0} tmp_0, f32[10,20]{1,0} tmp_2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}, execution_thread="MainThread")";
  EXPECT_EQ(computation->ToString(options), expected_computation2);
}

std::unique_ptr<HloComputation> MakeAddNComputation(int n) {
  auto builder = HloComputation::Builder("add_n");
  auto result = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "x_value"));
  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  for (int i = 0; i < n; ++i) {
    result = builder.AddInstruction(HloInstruction::CreateBinary(
        one->shape(), HloOpcode::kAdd, result, one));
  }
  return builder.Build();
}

TEST_F(HloComputationTest, DeepEquality) {
  auto computation_a = MakeAddNComputation(200000);
  auto computation_b = MakeAddNComputation(200000);
  EXPECT_TRUE(*computation_a == *computation_b);

  auto computation_c = MakeAddNComputation(199999);
  EXPECT_FALSE(*computation_a == *computation_c);
  EXPECT_FALSE(*computation_c == *computation_b);
}

// Tests that cross-module AllReduce instructions are ordered before all their
// predecessors and after all their successors.
TEST_F(HloComputationTest, InstructionPostOrderWithAllReduce) {
  const char* const hlo_string = R"(
HloModule Module

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  param = f32[128] parameter(0), sharding={maximal device=0}
  crs0 = f32[128] all-reduce(param),
    replica_groups={{0}}, channel_id=1, to_apply=add,
    sharding={maximal device=0}
  crs1 = f32[128] all-reduce(param),
    replica_groups={{0}}, channel_id=1, to_apply=add,
    sharding={maximal device=1}
  add = f32[128] add(crs0, crs0), sharding={maximal device=0}
  ROOT t = (f32[128], f32[128]) tuple(add, crs1)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(module->entry_computation()->MakeInstructionPostOrder(),
              ElementsAre(op::Parameter(), op::AllReduce(), op::AllReduce(),
                          op::Add(), op::Tuple()));
}

}  // namespace
}  // namespace xla
