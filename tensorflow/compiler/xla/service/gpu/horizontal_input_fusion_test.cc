/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/horizontal_input_fusion.h"

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;

class HorizontalInputFusionTest : public GpuCodegenTest {};

TEST_F(HorizontalInputFusionTest, BasicTest) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule BasicTest

  %add_f16 {
    %x = f16[] parameter(0)
    %y = f16[] parameter(1)
    ROOT %add = f16[] add(%x, %y)
  }

 fused_computation.1 {
   arg.1 = f16[1024]{0} parameter(0)
   constant0 = f16[] constant(0)
   ROOT reduce1 = f16[] reduce(arg.1, constant0), dimensions={0}, to_apply=%add_f16
 }

 fused_computation.2 {
   arg.1 = f16[1024]{0} parameter(0)
   constant0 = f16[] constant(0)
   ROOT reduce1 = f16[] reduce(arg.1, constant0), dimensions={0}, to_apply=%add_f16
 }

 ENTRY entry_computation {
   arg.1 = f16[1024]{0} parameter(0)
   arg.2 = f16[1024]{0} parameter(1)
   fusion.1 = f16[] fusion(arg.1), kind=kInput, calls=fused_computation.1
   fusion.2 = f16[] fusion(arg.2), kind=kInput, calls=fused_computation.2
   ROOT tuple.1 = (f16[], f16[]) tuple(fusion.1, fusion.2)
 }
)")
                    .ValueOrDie();

  EXPECT_TRUE(GpuHorizontalInputFusion().Run(module.get()).ValueOrDie());

  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(entry_root, op::Tuple((op::GetTupleElement(op::Fusion())),
                                    (op::GetTupleElement(op::Fusion()))));

  const HloInstruction* fusion = entry_root->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Reduce()));
}

TEST_F(HorizontalInputFusionTest, ManyInputFusions) {
  auto module = CreateNewVerifiedModule();

  HloComputation* reduce_computation;
  {
    auto embedded_builder = HloComputation::Builder("add");
    auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {}), "lhs"));
    auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(F32, {}), "rhs"));
    embedded_builder.AddInstruction(
        HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
    reduce_computation =
        module->AddEmbeddedComputation(embedded_builder.Build());
  }

  HloComputation::Builder builder(TestName());
  std::vector<HloInstruction*> var_outs;
  auto input_shape = ShapeUtil::MakeShape(F32, {1024, 1024});
  auto output_shape = ShapeUtil::MakeShape(F32, {1024});
  for (int64_t i = 0; i < 130; ++i) {
    // %fused_computation.3 (param_0: f32[1024,1024], param_1: f32[]) ->
    // f32[1024] {
    //  %param_0 = f32[1024,1024]{1,0} parameter(0)
    //  %param_1 = f32[] parameter(1)
    //  %broadcast = f32[1024,1024]{1,0} broadcast(f32[] %param_1),
    //  dimensions={}
    //  %multiply = f32[1024,1024]{1,0}
    //      multiply(f32[1024,1024]{1,0} %param_0, f32[1024,1024]{1,0}
    //      %broadcast)
    //  %constant0 = f32[] constant(0)
    //  ROOT %reduce = f32[1024]{0}
    //      reduce(f32[1024,1024]{1,0} %multiply, f32[] %constant0),
    //          dimensions={1}, to_apply=%add
    // }
    HloInstruction* param_var_in = builder.AddInstruction(
        HloInstruction::CreateParameter(i * 2 + 0, input_shape, "var.in"));
    HloInstruction* param_alpha =
        builder.AddInstruction(HloInstruction::CreateParameter(
            i * 2 + 1, ShapeUtil::MakeShape(F32, {}), "alpha"));
    auto alpha_broadcasted = builder.AddInstruction(
        HloInstruction::CreateBroadcast(input_shape, param_alpha, {}));
    auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
        input_shape, HloOpcode::kMultiply, param_var_in, alpha_broadcasted));
    HloInstruction* const0 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
    auto reduce = builder.AddInstruction(HloInstruction::CreateReduce(
        output_shape, mul, const0, {1}, reduce_computation));
    var_outs.push_back(reduce);
  }
  builder.AddInstruction(HloInstruction::CreateTuple(var_outs));
  module->AddEntryComputation(builder.Build());

  // Verify that horizontal fusion is kicked in. Check that there are multiple
  // `reduce` instructions fused into the same fusion. 6 is just a randomly
  // picked number as we don't exactly know how large the fusion will be
  // created due to the `FusionFitsInBudget` constraint.
  CompileAndVerifyIr(module->Clone(), R"(CHECK: reduce-group-6)",
                     /*match_optimized_ir=*/false);

  // Testing with the entire gpu optimization pipeline.
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-5, 1e-5}));
}

TEST_F(HorizontalInputFusionTest, MultiOutputFusionTest) {
  // This tests the below pattern. One known issue is that gtes (to fusions) can
  // be removed after their producer fusions are merged. In the below case, gte2
  // and gte6 will be gone if Fusion2 is fused into Fusion1.
  //
  // Fusion1   Fusion2
  //  |   |    |     |
  //  |  gte1 gte2   |
  //  |   |    |     |
  //  |   Fusion3    |
  //  |    |   |     |
  // gte3 gte4 gte5 gte6
  //  \  |     |    /
  //  =====ROOT=====
  //
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule MultiOutputFusionTest

  %add_f16 {
    %x = f16[] parameter(0)
    %y = f16[] parameter(1)
    ROOT %add = f16[] add(%x, %y)
  }

 fused_computation.1 {
   arg.1 = f16[1024]{0} parameter(0)
   constant0 = f16[] constant(0)
   reduce.1 = f16[] reduce(arg.1, constant0), dimensions={0}, to_apply=%add_f16
   add.0 = f16[1024] add(arg.1, arg.1)
   ROOT tuple.1 = (f16[], f16[1024]) tuple(reduce.1, add.0)
 }

 fused_computation.2 {
   arg.1 = f16[1024]{0} parameter(0)
   constant0 = f16[] constant(0)
   reduce.1 = f16[] reduce(arg.1, constant0), dimensions={0}, to_apply=%add_f16
   add.0 = f16[1024] add(arg.1, arg.1)
   ROOT tuple.1 = (f16[], f16[1024]) tuple(reduce.1, add.0)
 }

 fused_computation.3 {
   arg.0 = f16[1024]{0} parameter(0)
   arg.1 = f16[1024]{0} parameter(1)
   add.0 = f16[1024] add(arg.0, arg.1)
   mul.0 = f16[1024] multiply(arg.0, arg.1)
   ROOT tuple.1 = (f16[1024], f16[1024]) tuple(add.0, mul.0)
 }

 ENTRY entry_computation {
   arg.1 = f16[1024]{0} parameter(0)
   arg.2 = f16[1024]{0} parameter(1)
   fusion.1 = (f16[],f16[1024]) fusion(arg.1), kind=kInput, calls=fused_computation.1
   fusion.2 = (f16[],f16[1024]) fusion(arg.2), kind=kInput, calls=fused_computation.2
   gte.3 = f16[] get-tuple-element(fusion.1), index=0
   gte.1 = f16[1024]{0} get-tuple-element(fusion.1), index=1
   gte.2 = f16[1024]{0} get-tuple-element(fusion.2), index=1
   gte.6 = f16[] get-tuple-element(fusion.2), index=0
   fusion.3 = (f16[1024],f16[1024]) fusion(gte.1, gte.2),
       kind=kLoop, calls=fused_computation.3
   gte.4 = f16[1024] get-tuple-element(fusion.3), index=0
   gte.5 = f16[1024]{0} get-tuple-element(fusion.3), index=1
   ROOT tuple.1 = (f16[], f16[1024], f16[1024]{0}, f16[])
       tuple(gte.3, gte.4, gte.5, gte.6)
 }
)")
                    .ValueOrDie();

  EXPECT_TRUE(GpuHorizontalInputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(HorizontalInputFusionTest, NonfusionInstrs) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule NonfusionInstrs

 %add_f16 {
   %x = f16[] parameter(0)
   %y = f16[] parameter(1)
   ROOT %add = f16[] add(%x, %y)
 }

 ENTRY entry_computation {
   arg.0 = f16[1024]{0} parameter(0)
   arg.1 = f16[1024]{0} parameter(1)
   constant0 = f16[] constant(0)
   reduce.0 = f16[] reduce(arg.0, constant0), dimensions={0}, to_apply=%add_f16
   reduce.1 = f16[] reduce(arg.1, constant0), dimensions={0}, to_apply=%add_f16
   ROOT tuple.0 = (f16[], f16[]) tuple(reduce.0, reduce.1)
 }
)")
                    .ValueOrDie();

  EXPECT_TRUE(GpuHorizontalInputFusion().Run(module.get()).ValueOrDie());

  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(entry_root, op::Tuple((op::GetTupleElement(op::Fusion())),
                                    (op::GetTupleElement(op::Fusion()))));

  const HloInstruction* fusion = entry_root->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Reduce()));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
