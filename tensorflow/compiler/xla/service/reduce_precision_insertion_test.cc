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

#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {

using ::testing::UnorderedElementsAre;

class ReducePrecisionInsertionTest : public HloTestBase {
 protected:
  bool InsertOps(HloModule* module,
                 const HloReducePrecisionOptions::Location location,
                 const std::function<bool(const HloInstruction*)>& filter) {
    ReducePrecisionInsertion op_insertion(5, 10, location, filter);
    StatusOr<bool> result = op_insertion.Run(module);
    EXPECT_IS_OK(result.status());
    return result.ValueOrDie();
  }
};

TEST_F(ReducePrecisionInsertionTest, BeforeUnaryInstruction) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  // Create a simple graph with a parameter feeding a unary cosine function.
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, a));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected state before adding ops.
  EXPECT_EQ(computation->root_instruction(), b);
  EXPECT_EQ(b->operand(0), a);

  EXPECT_TRUE(InsertOps(module.get(), HloReducePrecisionOptions::OP_INPUTS,
                        [](const HloInstruction* instruction) {
                          return instruction->opcode() == HloOpcode::kCos;
                        }));

  // Confirm expected graph after adding ops.
  EXPECT_EQ(computation->root_instruction(), b);
  EXPECT_THAT(b->operand(0), op::ReducePrecision(a));
}

TEST_F(ReducePrecisionInsertionTest, BeforeBinaryInstruction) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  // Create a simple graph with parameter feeding a binary add function.

  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a, b));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected state before adding ops.
  EXPECT_EQ(computation->root_instruction(), c);
  EXPECT_EQ(c->operand(0), a);
  EXPECT_EQ(c->operand(1), b);

  EXPECT_TRUE(InsertOps(module.get(), HloReducePrecisionOptions::OP_INPUTS,
                        [](const HloInstruction* instruction) {
                          return instruction->opcode() == HloOpcode::kAdd;
                        }));

  // Confirm expected graph after adding ops.
  EXPECT_EQ(computation->root_instruction(), c);
  EXPECT_THAT(c->operand(0), op::ReducePrecision(a));
  EXPECT_THAT(c->operand(1), op::ReducePrecision(b));
}

TEST_F(ReducePrecisionInsertionTest, BeforeZeroInputInstruction) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  // Create a simple graph with a parameter feeding a unary cosine function.
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, a));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected state before adding ops.
  EXPECT_EQ(computation->root_instruction(), b);
  EXPECT_EQ(b->operand(0), a);

  EXPECT_FALSE(InsertOps(module.get(), HloReducePrecisionOptions::OP_INPUTS,
                         [](const HloInstruction* instruction) {
                           return instruction->opcode() ==
                                  HloOpcode::kParameter;
                         }));

  // Confirm that graph has not changed.
  EXPECT_EQ(computation->root_instruction(), b);
  EXPECT_EQ(b->operand(0), a);
}

TEST_F(ReducePrecisionInsertionTest, AvoidAddingDuplicateInstructions) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  // Create a simple graph with parameter feeding a binary add function.

  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, a));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kSin, a));
  HloInstruction* d = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, b, c));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected state before adding ops.
  EXPECT_EQ(computation->root_instruction(), d);
  EXPECT_EQ(b->operand(0), a);
  EXPECT_EQ(c->operand(0), a);

  EXPECT_TRUE(InsertOps(module.get(), HloReducePrecisionOptions::OP_INPUTS,
                        [](const HloInstruction* instruction) {
                          return instruction->opcode() == HloOpcode::kCos ||
                                 instruction->opcode() == HloOpcode::kSin;
                        }));

  // Confirm expected graph after adding ops.  In particular, we want to confirm
  // that the reduced-precision operation added for the input to b is re-used
  // for the input to c.
  EXPECT_THAT(b->operand(0), op::ReducePrecision(a));
  EXPECT_THAT(c->operand(0), op::ReducePrecision(a));
  EXPECT_EQ(b->operand(0), c->operand(0));
}

TEST_F(ReducePrecisionInsertionTest, AfterRootInstruction) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  // Create a simple graph with a parameter feeding a unary cosine function.
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, a));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected state before adding ops.
  EXPECT_EQ(computation->root_instruction(), b);

  EXPECT_TRUE(InsertOps(module.get(), HloReducePrecisionOptions::OP_OUTPUTS,
                        [](const HloInstruction* instruction) {
                          return instruction->opcode() == HloOpcode::kCos;
                        }));

  // Confirm expected graph after adding ops.
  EXPECT_THAT(computation->root_instruction(), op::ReducePrecision(b));
}

TEST_F(ReducePrecisionInsertionTest, AfterNonRootInstruction) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  // Create a graph with two parameters feeding into unary cosine functions,
  // and the output of those feeds into an add function.  Feeding the outputs
  // from the suffixed cosine functions into a binary add function allows us to
  // confirm that the separate operand streams are not crossed when the new
  // instructions are inserted.
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* a_cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, a));

  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* b_cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, b));

  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a_cos, b_cos));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  // Confirm expected graph before adding ops.
  EXPECT_EQ(c->operand(0), a_cos);
  EXPECT_EQ(c->operand(1), b_cos);

  EXPECT_TRUE(InsertOps(module.get(), HloReducePrecisionOptions::OP_OUTPUTS,
                        [](const HloInstruction* instruction) {
                          return instruction->opcode() == HloOpcode::kCos;
                        }));

  // Confirm expected graph after adding ops.
  EXPECT_THAT(c->operand(0), op::ReducePrecision());
  EXPECT_EQ(c->operand(0)->operand(0), a_cos);
  EXPECT_THAT(c->operand(1), op::ReducePrecision());
  EXPECT_EQ(c->operand(1)->operand(0), b_cos);
}

TEST_F(ReducePrecisionInsertionTest, OutputIsNotFloat) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, x));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected graph before adding ops.
  EXPECT_THAT(x->users(), UnorderedElementsAre(y));
  EXPECT_EQ(computation->root_instruction(), y);

  // Since none of the instructions produce F32 data, this should not change
  // the graph.
  EXPECT_FALSE(
      InsertOps(module.get(), HloReducePrecisionOptions::OP_OUTPUTS,
                [](const HloInstruction* instruction) { return true; }));

  // Confirm that graph has not changed.
  EXPECT_THAT(x->users(), UnorderedElementsAre(y));
  EXPECT_EQ(computation->root_instruction(), y);
}

TEST_F(ReducePrecisionInsertionTest, ShouldReduceOutputPrecisionIsFalse) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, x));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected graph before adding ops.
  EXPECT_THAT(x->users(), UnorderedElementsAre(y));
  EXPECT_EQ(computation->root_instruction(), y);

  // Since none of the instructions match the should_reduce_output_precision
  // function, this should not change the graph.
  EXPECT_FALSE(
      InsertOps(module.get(), HloReducePrecisionOptions::OP_OUTPUTS,
                [](const HloInstruction* instruction) { return false; }));

  // Confirm that graph has not changed.
  EXPECT_THAT(x->users(), UnorderedElementsAre(y));
  EXPECT_EQ(computation->root_instruction(), y);
}

TEST_F(ReducePrecisionInsertionTest, InsertionIsNotRecursive) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateReducePrecision(shape, a, 8, 23));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected state before adding ops.
  EXPECT_EQ(computation->root_instruction(), b);

  // This should insert a new ReducePrecision after the existing one, but
  // should not then recurse by adding another after the just-inserted one.
  EXPECT_TRUE(InsertOps(module.get(), HloReducePrecisionOptions::OP_OUTPUTS,
                        [](const HloInstruction* instruction) {
                          return instruction->opcode() ==
                                 HloOpcode::kReducePrecision;
                        }));

  // Confirm expected graph after adding ops.
  EXPECT_THAT(computation->root_instruction(), op::ReducePrecision());
  EXPECT_EQ(computation->root_instruction()->operand(0), b);
}

TEST_F(ReducePrecisionInsertionTest, SkipRedundantReducePrecisionAfter) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateReducePrecision(shape, x, 5, 10));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected graph before adding ops.
  EXPECT_THAT(x->users(), UnorderedElementsAre(y));
  EXPECT_EQ(computation->root_instruction(), y);

  // Since the new reduce-precision operation would be redundant, this
  // should not change the graph.
  EXPECT_FALSE(InsertOps(module.get(), HloReducePrecisionOptions::OP_OUTPUTS,
                         [](const HloInstruction* instruction) {
                           return instruction->opcode() ==
                                  HloOpcode::kParameter;
                         }));

  // Confirm that graph has not changed.
  EXPECT_THAT(x->users(), UnorderedElementsAre(y));
  EXPECT_EQ(computation->root_instruction(), y);
}

TEST_F(ReducePrecisionInsertionTest, AddNonRedundantReducePrecision) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateReducePrecision(shape, x, 8, 23));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected graph before adding ops.
  EXPECT_THAT(x->users(), UnorderedElementsAre(y));
  EXPECT_EQ(computation->root_instruction(), y);

  // Since the new reduce-precision operation is not the same as the existing
  // one, this should add a new one.
  EXPECT_TRUE(InsertOps(module.get(), HloReducePrecisionOptions::OP_OUTPUTS,
                        [](const HloInstruction* instruction) {
                          return instruction->opcode() == HloOpcode::kParameter;
                        }));

  // Confirm that graph is as expected.
  EXPECT_EQ(computation->root_instruction(), y);
  EXPECT_THAT(y->operand(0), op::ReducePrecision(x));
}

TEST_F(ReducePrecisionInsertionTest, IgnoreOpsInsideFusionNode) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, x));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Manually fuse the kCos operation into a fusion operation.
  HloInstruction* z = computation->AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, y));
  EXPECT_IS_OK(y->ReplaceAllUsesWith(z));
  EXPECT_IS_OK(computation->RemoveInstruction(y));

  // Confirm expected graph before adding reduce-precision ops.
  EXPECT_THAT(x->users(), UnorderedElementsAre(z));
  EXPECT_EQ(computation->root_instruction(), z);
  HloInstruction* y_fused = z->fused_expression_root();
  EXPECT_EQ(y_fused->opcode(), HloOpcode::kCos);

  // The ReducePrecisionInsertion pass should not see inside the fusion
  // operation, so this should not change the graph.
  EXPECT_FALSE(InsertOps(module.get(),
                         HloReducePrecisionOptions::UNFUSED_OP_OUTPUTS,
                         [](const HloInstruction* instruction) {
                           return instruction->opcode() == HloOpcode::kCos;
                         }));

  // Confirm that graph has not changed.
  EXPECT_THAT(x->users(), UnorderedElementsAre(z));
  EXPECT_EQ(computation->root_instruction(), z);
  EXPECT_EQ(z->fused_expression_root(), y_fused);
}

TEST_F(ReducePrecisionInsertionTest, OpGetsInsertedInHeadOfFusionNode) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, x));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Manually fuse the kCos operation into a fusion operation.
  HloInstruction* z = computation->AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, y));
  EXPECT_IS_OK(y->ReplaceAllUsesWith(z));
  EXPECT_IS_OK(computation->RemoveInstruction(y));

  // Confirm expected graph before adding reduce-precision ops.
  EXPECT_THAT(x->users(), UnorderedElementsAre(z));
  EXPECT_EQ(computation->root_instruction(), z);
  HloInstruction* y_fused = z->fused_expression_root();
  EXPECT_EQ(y_fused->opcode(), HloOpcode::kCos);

  // This should see that the fusion computation contains a kCos operation,
  // and insert a new reduce-precision node at its input.
  EXPECT_TRUE(InsertOps(module.get(),
                        HloReducePrecisionOptions::FUSION_INPUTS_BY_CONTENT,
                        [](const HloInstruction* instruction) {
                          return instruction->opcode() == HloOpcode::kCos;
                        }));

  // This should refuse to insert a second reduce-precision operation, as
  // it would be redundant with the first.
  EXPECT_FALSE(InsertOps(module.get(),
                         HloReducePrecisionOptions::FUSION_INPUTS_BY_CONTENT,
                         [](const HloInstruction* instruction) {
                           return instruction->opcode() == HloOpcode::kCos;
                         }));

  // Confirm that the top-level computation still only contains the fusion
  // instruction, but that the fused computation now has a reduce-precision
  // instruction inserted after its parameter instruction.
  EXPECT_THAT(x->users(), UnorderedElementsAre(z));
  EXPECT_EQ(computation->root_instruction(), z);
  EXPECT_THAT(z->fused_expression_root(), y_fused);
  EXPECT_THAT(y_fused->operand(0), op::ReducePrecision(op::Parameter()));
}

TEST_F(ReducePrecisionInsertionTest, OpGetsInsertedInTailOfFusionNode) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, x));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Manually fuse the kCos operation into a fusion operation.
  HloInstruction* z = computation->AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, y));
  EXPECT_IS_OK(y->ReplaceAllUsesWith(z));
  EXPECT_IS_OK(computation->RemoveInstruction(y));

  // Confirm expected graph before adding reduce-precision ops.
  EXPECT_THAT(x->users(), UnorderedElementsAre(z));
  EXPECT_EQ(computation->root_instruction(), z);
  HloInstruction* y_fused = z->fused_expression_root();
  EXPECT_EQ(y_fused->opcode(), HloOpcode::kCos);

  // This should see that the fusion computation contains a kCos operation,
  // and insert a new reduce-precision node at its root.
  EXPECT_TRUE(InsertOps(module.get(),
                        HloReducePrecisionOptions::FUSION_OUTPUTS_BY_CONTENT,
                        [](const HloInstruction* instruction) {
                          return instruction->opcode() == HloOpcode::kCos;
                        }));

  // This should refuse to insert a second reduce-precision operation, as
  // it would be redundant with the first.
  EXPECT_FALSE(InsertOps(module.get(),
                         HloReducePrecisionOptions::FUSION_OUTPUTS_BY_CONTENT,
                         [](const HloInstruction* instruction) {
                           return instruction->opcode() == HloOpcode::kCos;
                         }));

  // Confirm that the top-level computation still only contains the fusion
  // instruction, but that the fused computation now has a reduce-precision
  // instruction inserted as its root.
  EXPECT_THAT(x->users(), UnorderedElementsAre(z));
  EXPECT_EQ(computation->root_instruction(), z);
  EXPECT_THAT(z->fused_expression_root(), op::ReducePrecision(y_fused));
}

TEST_F(ReducePrecisionInsertionTest, MakeFilterFunctionNoSubstrings) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, a));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kSin, a));

  auto options_proto = ReducePrecisionInsertion::make_options_proto(
      HloReducePrecisionOptions::OP_OUTPUTS, 5, 10,
      [](const HloOpcode opcode) { return opcode == HloOpcode::kCos; });

  auto filter_function =
      ReducePrecisionInsertion::make_filter_function(options_proto);

  EXPECT_TRUE(filter_function(b));
  EXPECT_FALSE(filter_function(c));
}

TEST_F(ReducePrecisionInsertionTest, MakeFilterFunctionWithSubstrings) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));

  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, a));
  OpMetadata b_metadata;
  b_metadata.set_op_name("FlowTensor/foom");
  b->set_metadata(b_metadata);

  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, a));
  OpMetadata c_metadata;
  c_metadata.set_op_name("FlowTensor/barn");
  c->set_metadata(c_metadata);

  auto options_proto = ReducePrecisionInsertion::make_options_proto(
      HloReducePrecisionOptions::OP_OUTPUTS, 5, 10,
      [](const HloOpcode opcode) { return opcode == HloOpcode::kCos; },
      {"foo", "baz"});

  auto filter_function =
      ReducePrecisionInsertion::make_filter_function(options_proto);

  EXPECT_TRUE(filter_function(b));
  EXPECT_FALSE(filter_function(c));
}

}  // namespace xla
