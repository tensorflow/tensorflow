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

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/cpu_instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {
namespace {

class CpuFusionTest : public HloTestBase {
 protected:
  ErrorSpec error_spec_{0.0001, 1e-5};
  AliasInfo alias_info_;

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

  CpuInstructionFusion fusion(&alias_info_);
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
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));

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

  CpuInstructionFusion fusion(&alias_info_);
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
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));

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

  CpuInstructionFusion fusion(&alias_info_);
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
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));

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
  CpuInstructionFusion fusion(&alias_info_);
  EXPECT_TRUE(fusion.Run(module.get()).value());

  auto fusion1 = result->operand(0);
  auto fusion2 = result->operand(1);
  EXPECT_EQ(HloOpcode::kFusion, fusion1->opcode());
  EXPECT_EQ(HloOpcode::kFusion, fusion2->opcode());

  // Each fusion instruction should have 4 fused instruction inside: add, ceil,
  // negate, and the fused parameter.
  EXPECT_EQ(4, fusion1->fused_instruction_count());
  EXPECT_EQ(4, fusion2->fused_instruction_count());

  // The fusion has a single constant parameter.
  EXPECT_EQ(1, fusion1->operand_count());
  EXPECT_EQ(1, fusion2->operand_count());
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

  CpuInstructionFusion fusion(&alias_info_);
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

TEST_F(CpuFusionTest, DynamicUpdateSliceOutOfPlace) {
  const char* const hlo_string = R"(
HloModule OutOfPlaceDus

dus_fusion {
  base = f32[4,4] parameter(0)
  update = f32[2,2] parameter(1)
  idx0 = s32[] parameter(2)
  idx1 = s32[] parameter(3)
  ROOT dus = f32[4,4] dynamic-update-slice(base, update, idx0, idx1)
}

ENTRY main {
  p0 = f32[4,4] parameter(0)
  p1 = f32[2,2] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  fusion = f32[4,4] fusion(p0, p1, p2, p3), kind=kLoop, calls=dus_fusion
  neg = f32[4,4] negate(p0)
  ROOT tuple = (f32[4,4], f32[4,4]) tuple(fusion, neg)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  auto p0_literal = LiteralUtil::CreateR2<float>({
      {1.0f, 2.0f, 3.0f, 4.0f},
      {5.0f, 6.0f, 7.0f, 8.0f},
      {9.0f, 10.0f, 11.0f, 12.0f},
      {13.0f, 14.0f, 15.0f, 16.0f},
  });
  auto p1_literal = LiteralUtil::CreateR2<float>({
      {100.0f, 200.0f},
      {300.0f, 400.0f},
  });
  auto p2_literal = LiteralUtil::CreateR0<int32_t>(1);
  auto p3_literal = LiteralUtil::CreateR0<int32_t>(1);

  ASSERT_OK_AND_ASSIGN(const Literal result,
                       Execute(std::move(module), {&p0_literal, &p1_literal,
                                                   &p2_literal, &p3_literal}));

  LiteralTestUtil::ExpectR2Near<float>({{1.0f, 2.0f, 3.0f, 4.0f},
                                        {5.0f, 100.0f, 200.0f, 8.0f},
                                        {9.0f, 300.0f, 400.0f, 12.0f},
                                        {13.0f, 14.0f, 15.0f, 16.0f}},
                                       LiteralSlice(result, {0}), error_spec_);

  LiteralTestUtil::ExpectR2Near<float>({{-1.0f, -2.0f, -3.0f, -4.0f},
                                        {-5.0f, -6.0f, -7.0f, -8.0f},
                                        {-9.0f, -10.0f, -11.0f, -12.0f},
                                        {-13.0f, -14.0f, -15.0f, -16.0f}},
                                       LiteralSlice(result, {1}), error_spec_);
}

TEST_F(CpuFusionTest, DynamicUpdateSliceOutOfPlaceWithBitcast) {
  const char* const hlo_string = R"(
HloModule OutOfPlaceDusWithBitcast

dus_fusion {
  base = f32[16] parameter(0)
  bitcast_base = f32[4,4] bitcast(base)
  update = f32[2,2] parameter(1)
  idx0 = s32[] parameter(2)
  idx1 = s32[] parameter(3)
  dus = f32[4,4] dynamic-update-slice(bitcast_base, update, idx0, idx1)
  ROOT bitcast_dus = f32[16] bitcast(dus)
}

ENTRY main {
  p0 = f32[16] parameter(0)
  p1 = f32[2,2] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  fusion = f32[16] fusion(p0, p1, p2, p3), kind=kLoop, calls=dus_fusion
  neg = f32[16] negate(p0)
  ROOT tuple = (f32[16], f32[16]) tuple(fusion, neg)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  auto p0_literal = LiteralUtil::CreateR1<float>({
      1.0f,
      2.0f,
      3.0f,
      4.0f,
      5.0f,
      6.0f,
      7.0f,
      8.0f,
      9.0f,
      10.0f,
      11.0f,
      12.0f,
      13.0f,
      14.0f,
      15.0f,
      16.0f,
  });
  auto p1_literal = LiteralUtil::CreateR2<float>({
      {100.0f, 200.0f},
      {300.0f, 400.0f},
  });
  auto p2_literal = LiteralUtil::CreateR0<int32_t>(1);
  auto p3_literal = LiteralUtil::CreateR0<int32_t>(1);

  ASSERT_OK_AND_ASSIGN(const Literal result,
                       Execute(std::move(module), {&p0_literal, &p1_literal,
                                                   &p2_literal, &p3_literal}));

  LiteralTestUtil::ExpectR1Near<float>(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 100.0f, 200.0f, 8.0f, 9.0f, 300.0f, 400.0f,
       12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
      LiteralSlice(result, {0}), error_spec_);

  LiteralTestUtil::ExpectR1Near<float>(
      {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -9.0f, -10.0f,
       -11.0f, -12.0f, -13.0f, -14.0f, -15.0f, -16.0f},
      LiteralSlice(result, {1}), error_spec_);
}

TEST_F(CpuFusionTest, ScatterFusionZeroUpdates) {
  // Run an unfused variadic zero update scatter through the full CPU HLO
  // pipeline with algebraic simplification disabled so ScatterSimplifier,
  // FusionWrapper, CopyInsertion, and CpuScatterFusion all handle it.
  constexpr absl::string_view kUnfusedVariadicZeroUpdatesHlo = R"hlo(
    HloModule UnfusedVariadicZeroUpdates

    reducer {
      lhs0 = s32[] parameter(0)
      lhs1 = f64[] parameter(1)
      rhs0 = s32[] parameter(2)
      rhs1 = f64[] parameter(3)
      sum0 = s32[] add(lhs0, rhs0)
      sum1 = f64[] add(lhs1, rhs1)
      ROOT out = (s32[], f64[]) tuple(sum0, sum1)
    }

    ENTRY main {
      op0 = s32[3,2]{1,0} parameter(0)
      op1 = f64[3,2]{1,0} parameter(1)
      idx = s32[0]{0} parameter(2)
      upd0 = s32[0,2]{1,0} parameter(3)
      upd1 = f64[0,2]{1,0} parameter(4)
      ROOT scatter = (s32[3,2]{1,0}, f64[3,2]{1,0}) scatter(
          op0, op1, idx, upd0, upd1),
        update_window_dims={1},
        inserted_window_dims={0},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=reducer
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> variadic_module,
      ParseAndReturnVerifiedModule(kUnfusedVariadicZeroUpdatesHlo));
  variadic_module->mutable_config()
      .mutable_debug_options()
      .add_xla_disable_hlo_passes("algsimp");
  Literal operand = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}, {5, 6}});
  Literal op1 =
      LiteralUtil::CreateR2<double>({{1.5, 2.5}, {3.5, 4.5}, {5.5, 6.5}});
  Literal idx_1d = LiteralUtil::CreateR1<int32_t>({});
  Literal upd0(ShapeUtil::MakeShape(S32, {0, 2}));
  Literal upd1(ShapeUtil::MakeShape(F64, {0, 2}));
  ASSERT_OK_AND_ASSIGN(const Literal variadic_result,
                       Execute(std::move(variadic_module),
                               {&operand, &op1, &idx_1d, &upd0, &upd1}));
  EXPECT_EQ(LiteralSlice(variadic_result, {0}), operand);
  EXPECT_EQ(LiteralSlice(variadic_result, {1}), op1);
}

TEST_F(CpuFusionTest, ScatterFusionZeroSizedUpdateSlice) {
  // Wrapped fusion on a zero sized operand (distinct from the unfused case
  // below because ZeroSizedHloElimination folds zero sized outputs).
  constexpr absl::string_view kZeroSizedOperandHlo = R"hlo(
    HloModule ScatterFusionZeroSliceZeroOperand

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    scatter_computation {
      operand = s32[3,0]{1,0} parameter(0)
      indices = s32[2,1]{1,0} parameter(1)
      updates = s32[2,1,0]{2,1,0} parameter(2)
      ROOT scatter = s32[3,0]{1,0} scatter(operand, indices, updates),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add_s32
    }

    ENTRY main {
      p0 = s32[3,0]{1,0} parameter(0)
      operand_copy = s32[3,0]{1,0} copy(p0)
      p1 = s32[2,1]{1,0} parameter(1)
      p2 = s32[2,1,0]{2,1,0} parameter(2)
      ROOT wrapped_scatter = s32[3,0]{1,0} fusion(operand_copy, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> empty_operand_module,
                       ParseAndReturnVerifiedModule(kZeroSizedOperandHlo));
  Literal empty_operand(ShapeUtil::MakeShape(S32, {3, 0}));
  Literal empty_op_indices = LiteralUtil::CreateR2<int32_t>({{0}, {2}});
  Literal empty_op_updates(ShapeUtil::MakeShape(S32, {2, 1, 0}));
  ASSERT_OK_AND_ASSIGN(
      const Literal empty_result,
      Execute(std::move(empty_operand_module),
              {&empty_operand, &empty_op_indices, &empty_op_updates},
              /*run_hlo_passes=*/false));
  EXPECT_EQ(empty_result, empty_operand);

  // Full CPU HLO pipeline with algebraic simplification disabled on a zero
  // sized update slice into a populated operand.
  constexpr absl::string_view kUnfusedZeroSliceHlo = R"hlo(
    HloModule UnfusedZeroSliceScatter

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    ENTRY main {
      operand = s32[3,4]{1,0} parameter(0)
      indices = s32[2,2]{1,0} parameter(1)
      updates = s32[2,0]{1,0} parameter(2)
      ROOT scatter = s32[3,4]{1,0} scatter(operand, indices, updates),
        update_window_dims={1},
        inserted_window_dims={0},
        scatter_dims_to_operand_dims={0,1},
        index_vector_dim=1,
        to_apply=add_s32
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> unfused_module,
                       ParseAndReturnVerifiedModule(kUnfusedZeroSliceHlo));
  unfused_module->mutable_config()
      .mutable_debug_options()
      .add_xla_disable_hlo_passes("algsimp");
  Literal operand = LiteralUtil::CreateR2<int32_t>(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  Literal indices = LiteralUtil::CreateR2<int32_t>({{1, 2}, {9, 9}});
  Literal updates_2d(ShapeUtil::MakeShape(S32, {2, 0}));
  ASSERT_OK_AND_ASSIGN(
      const Literal unfused_result,
      Execute(std::move(unfused_module), {&operand, &indices, &updates_2d}));
  EXPECT_EQ(unfused_result, operand);
}

TEST_F(CpuFusionTest, ScatterFusionRankZeroScalarOperands) {
  // Unfused variadic rank 0 scatter going through the full CPU HLO pipeline
  // (ScatterSimplifier + FusionWrapper + CpuScatterFusion).
  constexpr absl::string_view kUnfusedVariadicRankZeroHlo = R"hlo(
    HloModule UnfusedVariadicRankZeroScatter

    reducer {
      lhs0 = s32[] parameter(0)
      lhs1 = f64[] parameter(1)
      rhs0 = s32[] parameter(2)
      rhs1 = f64[] parameter(3)
      sum0 = s32[] add(lhs0, rhs0)
      sum1 = f64[] add(lhs1, rhs1)
      ROOT out = (s32[], f64[]) tuple(sum0, sum1)
    }

    ENTRY main {
      op0 = s32[] parameter(0)
      op1 = f64[] parameter(1)
      idx = s32[2,2,0]{2,1,0} parameter(2)
      upd0 = s32[2,2]{1,0} parameter(3)
      upd1 = f64[2,2]{1,0} parameter(4)
      ROOT scatter = (s32[], f64[]) scatter(op0, op1, idx, upd0, upd1),
        update_window_dims={},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={},
        index_vector_dim=2,
        to_apply=reducer
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> variadic_module,
      ParseAndReturnVerifiedModule(kUnfusedVariadicRankZeroHlo));
  Literal op0 = LiteralUtil::CreateR0<int32_t>(100);
  Literal op1 = LiteralUtil::CreateR0<double>(1.5);
  Literal idx_2x2(ShapeUtil::MakeShape(S32, {2, 2, 0}));
  Literal upd0 = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}});
  Literal upd1 = LiteralUtil::CreateR2<double>({{0.5, 1.0}, {2.0, 3.0}});
  ASSERT_OK_AND_ASSIGN(const Literal variadic_result,
                       Execute(std::move(variadic_module),
                               {&op0, &op1, &idx_2x2, &upd0, &upd1}));
  EXPECT_EQ(LiteralSlice(variadic_result, {0}),
            LiteralUtil::CreateR0<int32_t>(110));
  LiteralTestUtil::ExpectR0Near<double>(8.0, LiteralSlice(variadic_result, {1}),
                                        error_spec_);

  // Wrapped fusion with zero updates into a rank 0 scalar operand.
  constexpr absl::string_view kZeroUpdatesScalarHlo = R"hlo(
    HloModule ScatterFusionRankZeroZeroUpdates

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    scatter_computation {
      operand = s32[] parameter(0)
      indices = s32[0,0]{1,0} parameter(1)
      updates = s32[0]{0} parameter(2)
      ROOT scatter = s32[] scatter(operand, indices, updates),
        update_window_dims={},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={},
        index_vector_dim=1,
        to_apply=add_s32
    }

    ENTRY main {
      p0 = s32[] parameter(0)
      operand_copy = s32[] copy(p0)
      p1 = s32[0,0]{1,0} parameter(1)
      p2 = s32[0]{0} parameter(2)
      ROOT wrapped_scatter = s32[] fusion(operand_copy, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> zero_upd_scalar_module,
                       ParseAndReturnVerifiedModule(kZeroUpdatesScalarHlo));
  Literal operand = LiteralUtil::CreateR0<int32_t>(10);
  Literal empty_scalar_idx(ShapeUtil::MakeShape(S32, {0, 0}));
  Literal empty_scalar_upd = LiteralUtil::CreateR1<int32_t>({});
  ASSERT_OK_AND_ASSIGN(const Literal zero_upd_scalar_result,
                       Execute(std::move(zero_upd_scalar_module),
                               {&operand, &empty_scalar_idx, &empty_scalar_upd},
                               /*run_hlo_passes=*/false));
  EXPECT_EQ(zero_upd_scalar_result, operand);
}

TEST_F(CpuFusionTest, ScatterFusionMixedWidthVariadic) {
  constexpr absl::string_view kMixedWidthVariadicHlo = R"hlo(
    HloModule ScatterFusionMixedWidthVariadic

    reducer {
      lhs0 = s8[] parameter(0)
      lhs1 = c128[] parameter(1)
      rhs0 = s8[] parameter(2)
      rhs1 = c128[] parameter(3)
      sum0 = s8[] add(lhs0, rhs0)
      sum1 = c128[] add(lhs1, rhs1)
      ROOT out = (s8[], c128[]) tuple(sum0, sum1)
    }

    ENTRY main {
      op0 = s8[3,8]{1,0} parameter(0)
      op1 = c128[3,8]{1,0} parameter(1)
      indices = s32[2,1]{1,0} parameter(2)
      upd0 = s8[2,1,8]{2,1,0} parameter(3)
      upd1 = c128[2,1,8]{2,1,0} parameter(4)
      ROOT scatter = (s8[3,8]{1,0}, c128[3,8]{1,0}) scatter(
          op0, op1, indices, upd0, upd1),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=reducer
    }
  )hlo";

  Literal op0 =
      LiteralUtil::CreateR2<int8_t>({{1, 2, 3, 4, 5, 6, 7, 8},
                                     {10, 20, 30, 40, 50, 60, 70, 80},
                                     {-1, -2, -3, -4, -5, -6, -7, -8}});
  Literal op1 = LiteralUtil::CreateR2<complex128>(
      {{{1, -1}, {2, -2}, {3, -3}, {4, -4}, {5, -5}, {6, -6}, {7, -7}, {8, -8}},
       {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}},
       {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8}}});
  Literal indices = LiteralUtil::CreateR2<int32_t>({{0}, {2}});
  Literal upd0 = LiteralUtil::CreateR3<int8_t>(
      {{{10, 10, 10, 10, 20, 20, 20, 20}}, {{1, 2, 3, 4, 5, 6, 7, 8}}});
  Literal upd1 = LiteralUtil::CreateR3<complex128>({{{{0.5, 1},
                                                      {0.5, 2},
                                                      {0.5, 3},
                                                      {0.5, 4},
                                                      {1.5, 5},
                                                      {1.5, 6},
                                                      {1.5, 7},
                                                      {1.5, 8}}},
                                                    {{{1, -1},
                                                      {2, -2},
                                                      {3, -3},
                                                      {4, -4},
                                                      {5, -5},
                                                      {6, -6},
                                                      {7, -7},
                                                      {8, -8}}}});

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMixedWidthVariadicHlo));
  ASSERT_OK_AND_ASSIGN(
      const Literal result,
      Execute(std::move(module), {&op0, &op1, &indices, &upd0, &upd1}));
  EXPECT_EQ(LiteralSlice(result, {0}),
            LiteralUtil::CreateR2<int8_t>({{11, 12, 13, 14, 25, 26, 27, 28},
                                           {10, 20, 30, 40, 50, 60, 70, 80},
                                           {0, 0, 0, 0, 0, 0, 0, 0}}));
  EXPECT_EQ(
      LiteralSlice(result, {1}),
      LiteralUtil::CreateR2<complex128>(
          {{{1.5, 0},
            {2.5, 0},
            {3.5, 0},
            {4.5, 0},
            {6.5, 0},
            {7.5, 0},
            {8.5, 0},
            {9.5, 0}},
           {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}},
           {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}}}));
}

}  // namespace
}  // namespace cpu
}  // namespace xla
