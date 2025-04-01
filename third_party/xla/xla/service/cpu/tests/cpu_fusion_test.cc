/* Copyright 2018 The OpenXLA Authors.

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

#include <memory>
#include <utility>

#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/cpu_instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {
namespace {

class CpuFusionTest : public HloTestBase {
 protected:
  CpuFusionTest() {}

  ErrorSpec error_spec_{0.0001, 1e-5};

 private:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.add_xla_disable_hlo_passes("layout-assignment");
    return debug_options;
  }
};

TEST_F(CpuFusionTest, FuseTwoElementwiseOps) {
  auto builder = HloComputation::Builder(TestName());
  auto input_literal1 = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0});
  auto input_literal2 = LiteralUtil::CreateR1<float>({-2.0, -42.0, 2.0});
  Shape vshape = input_literal1.shape();

  auto input1 = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal1)));
  auto input2 = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal2)));

  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(vshape, HloOpcode::kAdd, input1, input2));
  builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kNegate, add1));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  CpuInstructionFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());

  // The computation root instruction was fused. Verify the fusion instruction
  // is now the root.
  auto computation = module->entry_computation();
  auto fusion_instruction = computation->root_instruction();
  EXPECT_EQ(HloOpcode::kFusion, fusion_instruction->opcode());
  EXPECT_EQ(HloOpcode::kNegate,
            fusion_instruction->fused_expression_root()->opcode());
  // There should be four fused instructions: 2 parameters, the add, and the
  // negate.
  EXPECT_EQ(4, fusion_instruction->fused_instruction_count());

  // Compile and execute the computation.
  auto result = ExecuteAndTransfer(module->Clone(), {});

  // Check the output correctness.
  LiteralTestUtil::ExpectR1Near<float>({1.0, 40.0, -5.0}, result, error_spec_);
}

TEST_F(CpuFusionTest, FuseElementwiseOpChain) {
  auto builder = HloComputation::Builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({-1.5, -2.5, -3.0});
  Shape vshape = input_literal.shape();

  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kNegate, input));
  auto ceil = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kCeil, negate));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kExp, ceil));
  auto floor = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kFloor, exp));
  auto two = builder.AddInstruction(HloInstruction::CreateBroadcast(
      vshape,
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0))),
      {}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(vshape, HloOpcode::kMultiply, two, floor));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  CpuInstructionFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());

  // The computation root instruction was fused. Verify the fusion instruction
  // is now the root.
  auto computation = module->entry_computation();
  auto fusion_instruction = computation->root_instruction();
  EXPECT_EQ(HloOpcode::kFusion, fusion_instruction->opcode());
  EXPECT_EQ(HloOpcode::kMultiply,
            fusion_instruction->fused_expression_root()->opcode());
  // There should be 8 fused instructions: 2 parameters and the fused
  // operations.
  EXPECT_EQ(8, fusion_instruction->fused_instruction_count());

  // Compile and execute the computation.
  auto result = ExecuteAndTransfer(module->Clone(), {});

  // Check the output correctness.
  LiteralTestUtil::ExpectR1Near<float>({14.0, 40.0, 40.0}, result, error_spec_);
}

TEST_F(CpuFusionTest, ElementwiseOpChainWithNonfusibleInstruction) {
  // Test a chain of fusible ops with a non-fusible op (a reduce) thrown in the
  // middle.
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({-1.5, -2.5, -3.0});
  Shape vshape = input_literal.shape();

  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kNegate, input));
  auto ceil = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kCeil, negate));

  auto cshape = ShapeUtil::MakeShape(F32, {6});
  auto concatenate = builder.AddInstruction(
      HloInstruction::CreateConcatenate(cshape, {ceil, ceil}, /*dimension=*/0));

  // Build an x+y computation to use in a reduce.
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto embedded_builder = HloComputation::Builder("f32+f32");
  embedded_builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kAdd,
      embedded_builder.AddInstruction(
          HloInstruction::CreateParameter(0, r0f32, "x")),
      embedded_builder.AddInstruction(
          HloInstruction::CreateParameter(1, r0f32, "y"))));
  auto add_f32 = module->AddEmbeddedComputation(embedded_builder.Build());

  Window window;
  WindowDimension* window_dimension = window.add_dimensions();
  window_dimension->set_size(1);
  window_dimension->set_stride(1);
  window_dimension->set_base_dilation(1);
  window_dimension->set_window_dilation(1);
  // This is a nop reduction.
  auto reduce = builder.AddInstruction(HloInstruction::CreateReduceWindow(
      cshape, concatenate,
      /*init_value=*/
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0))),
      window, add_f32));

  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(cshape, HloOpcode::kExp, reduce));
  auto floor = builder.AddInstruction(
      HloInstruction::CreateUnary(cshape, HloOpcode::kFloor, exp));
  auto two = builder.AddInstruction(HloInstruction::CreateBroadcast(
      cshape,
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0))),
      {}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(cshape, HloOpcode::kMultiply, two, floor));

  module->AddEntryComputation(builder.Build());

  CpuInstructionFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());

  // The computation root instruction was fused. Verify the fusion instruction
  // is now the root.
  auto computation = module->entry_computation();

  auto fusion_instruction1 = computation->root_instruction();
  EXPECT_EQ(HloOpcode::kFusion, fusion_instruction1->opcode());
  EXPECT_EQ(HloOpcode::kMultiply,
            fusion_instruction1->fused_expression_root()->opcode());
  // There should be 6 fused instructions in the root fusion instruction: 2
  // parameters, multiply, floor, and exp.
  EXPECT_EQ(6, fusion_instruction1->fused_instruction_count())
      << fusion_instruction1->fused_instructions_computation()->ToString();

  auto fusion_instruction2 = reduce->operand(0);
  EXPECT_EQ(HloOpcode::kFusion, fusion_instruction1->opcode());
  EXPECT_EQ(HloOpcode::kConcatenate,
            fusion_instruction2->fused_expression_root()->opcode());
  // There should be 5 fused instructions in the second fusion instruction: 1
  // parameter, negate, ceil and concat.
  EXPECT_EQ(4, fusion_instruction2->fused_instruction_count())
      << fusion_instruction2->fused_instructions_computation()->ToString();

  // Compile and execute the computation.
  auto result = ExecuteAndTransfer(module->Clone(), {});

  // Check the output correctness.
  LiteralTestUtil::ExpectR1Near<float>({14.0, 40.0, 40.0, 14.0, 40.0, 40.0},
                                       result, error_spec_);
}

TEST_F(CpuFusionTest, TestOperandOrderToAvoidDuplication) {
  // Test that the operands of an instruction to be fused are considered in the
  // proper order to avoid duplication. Test input:
  //
  //   constant = {...}
  //   negate    = neg(constant)
  //   ceil      = ceil(negate)
  //   add1      = add(negate, ceil)
  //   add2      = add(ceil, negate)
  //
  // In this example, the operands of both add1 and add2 should be fused in the
  // order {ceil, negate} even though they have different orders in their
  // operand vectors. Test for this problem by counting the number of nodes in
  // each fusion instruction to ensure that negate is not duplicated.
  auto builder = HloComputation::Builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0});
  Shape vshape = input_literal.shape();

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kNegate, constant));
  auto ceil = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kCeil, negate));

  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(vshape, HloOpcode::kMultiply, negate, ceil));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(vshape, HloOpcode::kMultiply, ceil, negate));

  // Tie together the two adds with a tuple to create a single root.
  auto result =
      builder.AddInstruction(HloInstruction::CreateTuple({add1, add2}));

  // Create computation and module.
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  // Run fusion.
  CpuInstructionFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());

  auto fusion1 = result->operand(0);
  auto fusion2 = result->operand(1);
  EXPECT_EQ(HloOpcode::kFusion, fusion1->opcode());
  EXPECT_EQ(HloOpcode::kFusion, fusion2->opcode());

  // Each fusion instruction should have 4 fused instruction inside: add, ceil,
  // negate, and the fused parameter.
  EXPECT_EQ(4, fusion1->fused_instruction_count());
  EXPECT_EQ(4, fusion2->fused_instruction_count());

  // The fusion has no parameters, everything is fused including constants.
  EXPECT_EQ(0, fusion1->operand_count());
  EXPECT_EQ(0, fusion2->operand_count());
}

TEST_F(CpuFusionTest, DoNotDuplicateExpensiveOps) {
  // Verify that expensive operations will not be fused if the fusion results in
  // duplication. Test code:
  //
  //   constant = 42.0
  //   exp1 = exp(constant)
  //   negate1 = negate(exp1)
  //   exp2 = exp(constant)
  //   negate2 = negate(exp2)
  //   tuple = tuple(negate1, negate2, exp2)
  //
  // exp1 should be fused down into negate1, but exp2 will not be fused into
  // negate2 because this will result in duplication of the expensive exp
  // computation. The duplication is caused by the other use of exp2 in the
  // tuple.
  auto builder = HloComputation::Builder(TestName());
  auto input_literal1 = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0});
  auto input_literal2 = LiteralUtil::CreateR1<float>({-2.0, -42.0, 2.0});
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  Shape shape = constant->shape();

  auto exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, constant));
  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, exp1));

  auto exp2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, constant));
  auto negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, exp2));

  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({negate1, negate2, exp2}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  CpuInstructionFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());

  // The only fusion instruction should be operand 0 of the tuple (formerly
  // negate1).
  EXPECT_EQ(HloOpcode::kFusion, tuple->operand(0)->opcode());
  EXPECT_EQ(HloOpcode::kNegate, tuple->operand(1)->opcode());
  EXPECT_EQ(HloOpcode::kExp, tuple->operand(2)->opcode());

  auto fusion_inst = tuple->operand(0);
  // There should be three fused instructions: negate2, exp2, and the fused
  // constant.
  EXPECT_EQ(3, fusion_inst->fused_instruction_count());
  EXPECT_EQ(0, fusion_inst->operand_count());
}

}  // namespace
}  // namespace cpu
}  // namespace xla
