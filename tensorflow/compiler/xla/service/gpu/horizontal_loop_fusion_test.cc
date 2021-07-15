/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/horizontal_loop_fusion.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;

class HorizontalLoopFusionTest : public HloTestBase {};

TEST_F(HorizontalLoopFusionTest, BasicTest) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule BasicTest

 fused_computation.1 {
   arg.1 = f16[1024]{0} parameter(0)
   arg.2 = f16[1024]{0} parameter(1)
   ROOT mul.1 = f16[1024]{0} multiply(arg.1, arg.2)
 }

 fused_computation.2 {
   arg.1 = f16[123]{0} parameter(0)
   arg.2 = f16[123]{0} parameter(1)
   ROOT add.1 = f16[123]{0} add(arg.1, arg.2)
 }

 ENTRY entry_computation {
   arg.1 = f16[1024]{0} parameter(0)
   arg.2 = f16[1024]{0} parameter(1)
   arg.3 = f16[123]{0} parameter(2)
   arg.4 = f16[123]{0} parameter(3)
   fusion.1 = f16[1024]{0}
       fusion(arg.1, arg.2), kind=kLoop, calls=fused_computation.1
   fusion.2 = f16[123]{0}
       fusion(arg.3, arg.4), kind=kLoop, calls=fused_computation.2
   ROOT tuple.1 = (f16[1024]{0}, f16[123]{0})
       tuple(fusion.1, fusion.2)
 }
)")
                    .ValueOrDie();

  EXPECT_TRUE(GpuHorizontalLoopFusion().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());

  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(entry_root,
              op::Tuple(op::Bitcast(op::GetTupleElement(op::Fusion())),
                        op::Bitcast(op::GetTupleElement(op::Fusion()))));

  const HloInstruction* fusion = entry_root->operand(0)->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(
      fusion->fused_expression_root(),
      op::Tuple(op::Slice(op::Concatenate(op::Reshape(), op::Reshape())),
                op::Slice(op::Concatenate(op::Reshape(), op::Reshape()))));
}

// Horizontal fusion should not be triggered as fusion will create cycles.
TEST_F(HorizontalLoopFusionTest, NegativeTestForCycle) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule NegativeTestForCycle

 fused_computation.1 {
   arg.1 = f16[123]{0} parameter(0)
   arg.2 = f16[123]{0} parameter(1)
   ROOT mul.1 = f16[123]{0} multiply(arg.1, arg.2)
 }

 fused_computation.2 {
   arg.1 = f16[123]{0} parameter(0)
   arg.2 = f16[123]{0} parameter(1)
   ROOT add.1 = f16[123]{0} add(arg.1, arg.2)
 }

 ENTRY entry_computation {
   arg.1 = f16[123]{0} parameter(0)
   arg.2 = f16[123]{0} parameter(1)
   arg.3 = f16[123]{0} parameter(2)
   arg.4 = f16[123]{0} parameter(3)
   // fusion.1 and fusion.2 will not be horizontally fused as it will create
   // a cycle through fusion.1 -> add.2 -> fusion.2
   fusion.1 = f16[123]{0}
       fusion(arg.1, arg.2), kind=kLoop, calls=fused_computation.1
   add.2 = f16[123]{0} add(fusion.1, arg.4)
   fusion.2 = f16[123]{0}
       fusion(add.2, arg.3), kind=kLoop, calls=fused_computation.2
   ROOT tuple.1 = (f16[123]{0}, f16[123]{0}, f16[123]{0})
       tuple(fusion.1, fusion.2, add.2)
 }
)")
                    .ValueOrDie();

  EXPECT_FALSE(GpuHorizontalLoopFusion().Run(module.get()).ValueOrDie());
}

TEST_F(HorizontalLoopFusionTest, NegativeTestForIncompatibleTypes) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule NegativeTestForIncompatibleTypes

 fused_computation.1 {
   arg.1 = f16[1024]{0} parameter(0)
   arg.2 = f16[1024]{0} parameter(1)
   ROOT mul.1 = f16[1024]{0} multiply(arg.1, arg.2)
 }

 fused_computation.2 {
   arg.1 = s32[123]{0} parameter(0)
   arg.2 = s32[123]{0} parameter(1)
   ROOT add.1 = s32[123]{0} add(arg.1, arg.2)
 }

 ENTRY entry_computation {
   arg.1 = f16[1024]{0} parameter(0)
   arg.2 = f16[1024]{0} parameter(1)
   arg.3 = s32[123]{0} parameter(2)
   arg.4 = s32[123]{0} parameter(3)
   // fusion.1 and fusion.2 will not be horizontally fused because their output
   // types are different.
   fusion.1 = f16[1024]{0}
       fusion(arg.1, arg.2), kind=kLoop, calls=fused_computation.1
   fusion.2 = s32[123]{0}
       fusion(arg.3, arg.4), kind=kLoop, calls=fused_computation.2
   ROOT tuple.1 = (f16[1024]{0}, s32[123]{0})
       tuple(fusion.1, fusion.2)
 }
)")
                    .ValueOrDie();

  EXPECT_FALSE(GpuHorizontalLoopFusion().Run(module.get()).ValueOrDie());
}

TEST_F(HorizontalLoopFusionTest, HorizontalLoopFusionAfterVerticalFusion) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule MergeSharedFusionInstruction

 ENTRY MergeSharedFusionInstruction.Computation0 {
  param.1.1   = f32[4,1024]{1,0} parameter(0)
  param.1.2   = f32[4,1024]{1,0} parameter(1)
  param.1.3   = f32[4,1024]{1,0} parameter(2)
  param.2.1   = f32[321,5]{1,0} parameter(3)
  param.2.2   = f32[321,5]{1,0} parameter(4)
  param.2.3   = f32[321,5]{1,0} parameter(5)
  const.1     = f32[] constant(3)
  const.2     = f32[] constant(3)
  broadcast.1 = f32[4,1024]{1,0} broadcast(const.1), dimensions={}
  broadcast.2 = f32[321,5]{1,0} broadcast(const.2), dimensions={}
  mul.1.1     = f32[4,1024]{1,0} multiply(param.1.1, param.1.2)
  mul.1.2     = f32[4,1024]{1,0} multiply(param.1.3, broadcast.1)
  add.1       = f32[4,1024]{1,0} add(mul.1.1, mul.1.2)
  mul.2.1     = f32[321,5]{1,0} multiply(param.2.1, param.2.2)
  mul.2.2     = f32[321,5]{1,0} multiply(param.2.3, broadcast.2)
  add.2       = f32[321,5]{1,0} add(mul.2.1, mul.2.2)
  ROOT tuple = (f32[4,1024]{1,0}, f32[321,5]{1,0}) tuple(add.1, add.2)
})")
                    .ValueOrDie();

  HloPassPipeline fusion("fusion");
  fusion.AddPass<xla::gpu::GpuInstructionFusion>(/*may_duplicate=*/false);
  fusion.AddPass<xla::gpu::GpuInstructionFusion>(/*may_duplicate=*/true);
  EXPECT_TRUE(fusion.Run(module.get()).ValueOrDie());
  EXPECT_TRUE(GpuHorizontalLoopFusion().Run(module.get()).ValueOrDie());

  VLOG(2) << "Dump after horizontal fusion:";
  VLOG(2) << module->ToString();

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec{0, 0}));
}

TEST_F(HorizontalLoopFusionTest, GradientDescentOptimizerLike) {
  HloComputation::Builder builder(TestName());

  std::vector<HloInstruction*> var_outs;
  for (int64_t i = 0; i < 128; ++i) {
    // For shapes {1, 1024}, {2, 1024}, ..., {128, 1024}
    Shape shape = ShapeUtil::MakeShape(F32, {i + 1, 1024});
    HloInstruction* param_var_in = builder.AddInstruction(
        HloInstruction::CreateParameter(i * 3 + 0, shape, "var.in"));
    HloInstruction* param_alpha =
        builder.AddInstruction(HloInstruction::CreateParameter(
            i * 3 + 1, ShapeUtil::MakeShape(F32, {}), "alpha"));
    HloInstruction* param_delta = builder.AddInstruction(
        HloInstruction::CreateParameter(i * 3 + 2, shape, "delta"));
    HloInstruction* alpha_broadcasted = builder.AddInstruction(
        HloInstruction::CreateBroadcast(shape, param_alpha, {}));
    HloInstruction* alpha_delta =
        builder.AddInstruction(HloInstruction::CreateBinary(
            shape, HloOpcode::kMultiply, alpha_broadcasted, param_delta));
    HloInstruction* var_out =
        builder.AddInstruction(HloInstruction::CreateBinary(
            shape, HloOpcode::kSubtract, param_var_in, alpha_delta));
    var_outs.push_back(var_out);
  }
  builder.AddInstruction(HloInstruction::CreateTuple(var_outs));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  // Testing with the entire gpu optimization pipeline.
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0, 0}));
}

TEST_F(HorizontalLoopFusionTest, FusingDifferentOutputs) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule HeterogeneousMultiOutputFusions

 fused_computation.1 {
   arg.1 = f16[1024]{0} parameter(0)
   arg.2 = f16[1024]{0} parameter(1)
   arg.3 = f16[1024]{0} parameter(2)
   arg.4 = f16[1024]{0} parameter(3)
   mul.1 = f16[1024]{0} multiply(arg.1, arg.2)
   mul.2 = f16[1024]{0} multiply(arg.3, arg.4)
   add.1 = f16[1024]{0} add(mul.1, mul.2)
   ROOT tuple.1 = (f16[1024]{0}, f16[1024]{0}) tuple(add.1, mul.1)
 }

 fused_computation.2 {
   arg.1 = f16[123]{0} parameter(0)
   arg.2 = f16[123]{0} parameter(1)
   arg.3 = f16[123]{0} parameter(2)
   arg.4 = f16[123]{0} parameter(3)
   add.1 = f16[123]{0} add(arg.1, arg.2)
   add.2 = f16[123]{0} add(arg.3, arg.4)
   mul.1 = f16[123]{0} multiply(add.1, add.2)
   ROOT tuple.1 = (f16[123]{0}, f16[123]{0}) tuple(mul.1, add.1)
 }

 ENTRY entry_computation {
   arg.1 = f16[1024]{0} parameter(0)
   arg.2 = f16[1024]{0} parameter(1)
   arg.3 = f16[1024]{0} parameter(2)
   arg.4 = f16[1024]{0} parameter(3)
   arg.5 = f16[123]{0} parameter(4)
   arg.6 = f16[123]{0} parameter(5)
   arg.7 = f16[123]{0} parameter(6)
   arg.8 = f16[123]{0} parameter(7)
   fusion.1 = (f16[1024]{0}, f16[1024]{0})
       fusion(arg.1, arg.2, arg.3, arg.4),
           kind=kLoop, calls=fused_computation.1
   fusion.2 = (f16[123]{0}, f16[123]{0})
       fusion(arg.5, arg.6, arg.7, arg.8),
           kind=kLoop, calls=fused_computation.2
   gte.1 = f16[1024]{0} get-tuple-element(fusion.1), index=0
   gte.2 = f16[1024]{0} get-tuple-element(fusion.1), index=1
   gte.3 = f16[123]{0} get-tuple-element(fusion.2), index=0
   gte.4 = f16[123]{0} get-tuple-element(fusion.2), index=1
   ROOT tuple.1 = (f16[1024]{0}, f16[1024]{0}, f16[123]{0}, f16[123]{0})
       tuple(gte.1, gte.2, gte.3, gte.4)
 }
)")
                    .ValueOrDie();

  EXPECT_TRUE(GpuHorizontalLoopFusion().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());

  VLOG(2) << "Dump after horizontal fusion:";
  VLOG(2) << module->ToString();

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec{0, 0}));
}

TEST_F(HorizontalLoopFusionTest, RMSPropLike) {
  HloComputation::Builder builder(TestName());

  std::vector<HloInstruction*> all_outputs;
  for (int64_t i = 0; i < 48; ++i) {
    Shape shape = ShapeUtil::MakeShape(F32, {2, 1024 + i});
    // ms <- grad**2 (1 - rho) + ms * rho
    HloInstruction* grad = builder.AddInstruction(
        HloInstruction::CreateParameter(i * 9 + 0, shape, "grad"));
    HloInstruction* ms = builder.AddInstruction(
        HloInstruction::CreateParameter(i * 9 + 1, shape, "ms"));
    HloInstruction* rho =
        builder.AddInstruction(HloInstruction::CreateParameter(
            i * 9 + 2, ShapeUtil::MakeShape(F32, {}), "rho"));
    HloInstruction* one_minus_rho =
        builder.AddInstruction(HloInstruction::CreateParameter(
            i * 9 + 3, ShapeUtil::MakeShape(F32, {}), "one_minus_rho"));
    HloInstruction* rho_broadcasted =
        builder.AddInstruction(HloInstruction::CreateBroadcast(shape, rho, {}));
    HloInstruction* one_mins_rho_broadcasted = builder.AddInstruction(
        HloInstruction::CreateBroadcast(shape, one_minus_rho, {}));
    HloInstruction* grad_squared = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, grad, grad));
    HloInstruction* ms_1st_term = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, grad_squared,
                                     one_mins_rho_broadcasted));
    HloInstruction* ms_2nd_term =
        builder.AddInstruction(HloInstruction::CreateBinary(
            shape, HloOpcode::kMultiply, ms, rho_broadcasted));
    HloInstruction* ms_out =
        builder.AddInstruction(HloInstruction::CreateBinary(
            shape, HloOpcode::kAdd, ms_1st_term, ms_2nd_term));

    // mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
    HloInstruction* momentum = builder.AddInstruction(
        HloInstruction::CreateParameter(i * 9 + 4, shape, "momemtum"));
    HloInstruction* mom = builder.AddInstruction(
        HloInstruction::CreateParameter(i * 9 + 5, shape, "mom"));
    HloInstruction* lr = builder.AddInstruction(HloInstruction::CreateParameter(
        i * 9 + 6, ShapeUtil::MakeShape(F32, {}), "lr"));
    HloInstruction* epsilon =
        builder.AddInstruction(HloInstruction::CreateParameter(
            i * 9 + 7, ShapeUtil::MakeShape(F32, {}), "epsilon"));
    HloInstruction* lr_broadcasted =
        builder.AddInstruction(HloInstruction::CreateBroadcast(shape, lr, {}));
    HloInstruction* epsilon_broadcasted = builder.AddInstruction(
        HloInstruction::CreateBroadcast(shape, epsilon, {}));
    HloInstruction* mom_1st_term =
        builder.AddInstruction(HloInstruction::CreateBinary(
            shape, HloOpcode::kMultiply, momentum, mom));
    HloInstruction* ms_eps =
        builder.AddInstruction(HloInstruction::CreateBinary(
            shape, HloOpcode::kAdd, ms_out, epsilon_broadcasted));
    HloInstruction* ms_eps_rsq = builder.AddInstruction(
        HloInstruction::CreateUnary(shape, HloOpcode::kRsqrt, ms_eps));
    HloInstruction* grad_ms_eps_rsq =
        builder.AddInstruction(HloInstruction::CreateBinary(
            shape, HloOpcode::kMultiply, grad, ms_eps_rsq));
    HloInstruction* mom_2nd_term =
        builder.AddInstruction(HloInstruction::CreateBinary(
            shape, HloOpcode::kMultiply, lr_broadcasted, grad_ms_eps_rsq));
    HloInstruction* mom_out =
        builder.AddInstruction(HloInstruction::CreateBinary(
            shape, HloOpcode::kAdd, mom_1st_term, mom_2nd_term));

    // var <- var - mom
    HloInstruction* var = builder.AddInstruction(
        HloInstruction::CreateParameter(i * 9 + 8, shape, "var"));
    HloInstruction* var_out =
        builder.AddInstruction(HloInstruction::CreateBinary(
            shape, HloOpcode::kSubtract, var, mom_out));

    all_outputs.push_back(ms_out);
    all_outputs.push_back(mom_out);
    all_outputs.push_back(var_out);
  }
  builder.AddInstruction(HloInstruction::CreateTuple(all_outputs));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1.0e-5, 1.0e-5}));
}

TEST_F(HorizontalLoopFusionTest, DynamicUpdateSlice) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule NegativeTestForDynamicUpdateSlice

  fusion.1 {
    p.0 = f16[5,9,10]{2,1,0} parameter(0)
    p.1 = s32[] parameter(1)
    p.2 = f16[1,9,10]{2,1,0} parameter(2)
    c.0 = s32[] constant(0)
    ROOT %dynamic-update-slice =
        f16[5,9,10]{2,1,0} dynamic-update-slice(p.0, p.2, p.1, c.0, c.0)
  }

  fusion.2 {
    p.0 = f16[5,9,10]{2,1,0} parameter(0)
    p.1 = s32[] parameter(1)
    p.2 = f16[1,9,10]{2,1,0} parameter(2)
    c.0 = s32[] constant(0)
    ROOT %dynamic-update-slice =
        f16[5,9,10]{2,1,0} dynamic-update-slice(p.0, p.2, p.1, c.0, c.0)
  }

  ENTRY entry {
    p.00 = f16[5,9,10]{2,1,0} parameter(0)
    p.01 = f16[5,9,10]{2,1,0} parameter(1)
    p.10 = s32[] parameter(2)
    p.11 = s32[] parameter(3)
    p.20 = f16[1,9,10]{2,1,0} parameter(4)
    p.21 = f16[1,9,10]{2,1,0} parameter(5)

    f1 = f16[5,9,10] fusion(p.00, p.10, p.20), kind=kLoop, calls=fusion.1
    f2 = f16[5,9,10] fusion(p.01, p.11, p.21), kind=kLoop, calls=fusion.2
    ROOT tuple = (f16[5,9,10],f16[5,9,10]) tuple(f1, f2)
  })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuHorizontalLoopFusion().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());

  VLOG(2) << "Dump after horizontal fusion:";
  VLOG(2) << module->ToString();

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec{0, 0}));
}

TEST_F(HorizontalLoopFusionTest, NegativeTestForSharedParam) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule BasicTest

 fused_computation.1 {
   arg.1 = f16[123]{0} parameter(0)
   arg.2 = f16[123]{0} parameter(1)
   ROOT mul.1 = f16[123]{0} multiply(arg.1, arg.2)
 }

 fused_computation.2 {
   arg.1 = f16[123]{0} parameter(0)
   arg.2 = f16[123]{0} parameter(1)
   ROOT add.1 = f16[123]{0} add(arg.1, arg.2)
 }

 ENTRY entry_computation {
   arg.1 = f16[123]{0} parameter(0)
   // arg.2 is shared by fusion.1 and fusion.2
   arg.2 = f16[123]{0} parameter(1)
   arg.3 = f16[123]{0} parameter(2)
   fusion.1 = f16[123]{0}
       fusion(arg.1, arg.2), kind=kLoop, calls=fused_computation.1
   fusion.2 = f16[123]{0}
       fusion(arg.3, arg.2), kind=kLoop, calls=fused_computation.2
   ROOT tuple.1 = (f16[123]{0}, f16[123]{0})
       tuple(fusion.1, fusion.2)
 }
)")
                    .ValueOrDie();

  EXPECT_FALSE(GpuHorizontalLoopFusion().Run(module.get()).ValueOrDie());
}

TEST_F(HorizontalLoopFusionTest, IterativeHorizontalFusion) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule NonfusionInstrs

 fused_computation.0 {
   arg.0 = f16[] parameter(0)
   arg.1 = f16[123]{0} parameter(1)
   broadcast.0 = f16[123]{0} broadcast(arg.0), dimensions={}
   ROOT mul.1 = f16[123]{0} multiply(broadcast.0, arg.1)
 }

 fused_computation.1 {
   arg.0 = f16[] parameter(0)
   arg.1 = f16[456]{0} parameter(1)
   broadcast.0 = f16[456]{0} broadcast(arg.0), dimensions={}
   ROOT add.1 = f16[456]{0} add(broadcast.0, arg.1)
 }

 ENTRY entry_computation {
   arg.0 = f16[] parameter(0)
   arg.1 = f16[] parameter(1)
   arg.2 = f16[123]{0} parameter(2)
   arg.3 = f16[456]{0} parameter(3)
   // Test fusion of non-fusion instructions. sqrt.0 and sqrt.1 are to be
   // fused.
   sqrt.0 = f16[] sqrt(arg.0)
   sqrt.1 = f16[] sqrt(arg.1)
   // fusion.0 and fusion.1 are to be fused.
   fusion.0 = f16[123]{0}
       fusion(sqrt.0, arg.2), kind=kLoop, calls=fused_computation.0
   fusion.1 = f16[456]{0}
       fusion(sqrt.1, arg.3), kind=kLoop, calls=fused_computation.1
   ROOT tuple.1 = (f16[123]{0}, f16[456]{0}) tuple(fusion.0, fusion.1)
 }
)")
                    .ValueOrDie();

  HloPassFix<HloPassPipeline> iterative_h_fusion("iterative_h_fusion");
  iterative_h_fusion.AddPass<GpuHorizontalLoopFusion>();
  iterative_h_fusion.AddPass<HloDCE>();
  EXPECT_TRUE(iterative_h_fusion.Run(module.get()).ValueOrDie());

  // Verify that fusion.0 and fusion.1 are fused.
  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(entry_root,
              op::Tuple(op::Bitcast(op::GetTupleElement(op::Fusion())),
                        op::Bitcast(op::GetTupleElement(op::Fusion()))));
  const HloInstruction* fusion = entry_root->operand(0)->operand(0)->operand(0);
  EXPECT_TRUE(fusion->IsMultiOutputFusion());

  // Verify that the total number of fusion instructions is 2 so that we
  // know sqrt.0 and sqrt.1 are fused.
  size_t total_fusion_instrs = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kFusion) {
      ++total_fusion_instrs;
    }
  }
  EXPECT_EQ(total_fusion_instrs, 2);
}

TEST_F(HorizontalLoopFusionTest, TraversalOrder) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule cluster

 %fused_computation (param_0: f32[256,256], param_1: f32[], param_2: f32[])
     -> f32[256,256] {
   %param_0 = f32[256,256]{1,0} parameter(0)
   %param_1 = f32[] parameter(1)
   %param_2 = f32[] parameter(2)
   %multiply.0 = f32[] multiply(f32[] %param_1, f32[] %param_2)
   %broadcast.0 = f32[256,256]{1,0} broadcast(f32[] %multiply.0), dimensions={}
   ROOT %multiply.1 = f32[256,256]{1,0}
       multiply(f32[256,256]{1,0} %param_0, f32[256,256]{1,0} %broadcast.0)
 }

 %fused_computation.1 (param_0: f32[256,256], param_1: f32[], param_2: f32[])
     -> f32[256,256] {
   %param_0 = f32[256,256]{1,0} parameter(0)
   %param_1 = f32[] parameter(1)
   %param_2 = f32[] parameter(2)
   %multiply.0 = f32[] multiply(f32[] %param_1, f32[] %param_2)
   %broadcast.0 = f32[256,256]{1,0} broadcast(f32[] %multiply.0), dimensions={}
   ROOT %multiply.1 = f32[256,256]{1,0}
       multiply(f32[256,256]{1,0} %param_0, f32[256,256]{1,0} %broadcast.0)
 }

 ENTRY %entry_computation (arg0: f32[256,256], arg1: f32[256,256], arg2: f32[],
                           arg3: f32[], arg4: f32[], arg5: f32[])
                               -> (f32[256,256], f32[256,256]) {
   %arg0 = f32[256,256]{1,0} parameter(0), parameter_replication={false}
   %arg1 = f32[256,256]{1,0} parameter(1), parameter_replication={false}
   %arg2 = f32[] parameter(2), parameter_replication={false}
   %arg3 = f32[] parameter(3), parameter_replication={false}
   %arg4 = f32[] parameter(4), parameter_replication={false}
   %arg5 = f32[] parameter(5), parameter_replication={false}
   %sqrt = f32[] sqrt(f32[] %arg2)
   %sqrt.1 = f32[] sqrt(f32[] %arg3)
   %fusion = f32[256,256]{1,0}
       fusion(f32[256,256]{1,0} %arg0, f32[] %sqrt, f32[] %sqrt.1),
       kind=kLoop, calls=%fused_computation
   %sqrt.2 = f32[] sqrt(f32[] %arg4)
   %sqrt.3 = f32[] sqrt(f32[] %arg5)
   %fusion.1 = f32[256,256]{1,0}
       fusion(f32[256,256]{1,0} %arg1, f32[] %sqrt.2, f32[] %sqrt.3),
       kind=kLoop, calls=%fused_computation.1
   ROOT %tuple.163 = (f32[256,256]{1,0}, f32[256,256]{1,0})
       tuple(f32[256,256]{1,0} %fusion.1, f32[256,256]{1,0} %fusion)
 }
)")
                    .ValueOrDie();

  HloPassFix<HloPassPipeline> iterative_h_fusion("iterative_h_fusion");
  iterative_h_fusion.AddPass<GpuHorizontalLoopFusion>();
  EXPECT_TRUE(iterative_h_fusion.Run(module.get()).ValueOrDie());

  // Verify that the total number of fusion instructions is 2 so that we
  // know all the sqrt instructions are fused into a kernel. Note that if we
  // traverse from def-to-use (i.e., top-to-down) instead of use-to-def, we
  // will end up having 3 fusions instead of 2.
  size_t total_fusion_instrs = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kFusion) {
      ++total_fusion_instrs;
    }
  }
  EXPECT_EQ(total_fusion_instrs, 2);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
