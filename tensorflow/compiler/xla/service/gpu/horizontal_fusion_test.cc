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

#include "tensorflow/compiler/xla/service/gpu/horizontal_fusion.h"

#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

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
#include "tensorflow/compiler/xla/tests/filecheck.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;

class HorizontalFusionTest : public HloTestBase {};

TEST_F(HorizontalFusionTest, BasicTest) {
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
)").ValueOrDie();

  EXPECT_TRUE(GpuHorizontalFusion().Run(module.get()).ValueOrDie());
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
TEST_F(HorizontalFusionTest, NegativeTestForCycle) {
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
)").ValueOrDie();

  EXPECT_FALSE(GpuHorizontalFusion().Run(module.get()).ValueOrDie());
}

TEST_F(HorizontalFusionTest, NegativeTestForIncompatibleTypes) {
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
)").ValueOrDie();

  EXPECT_FALSE(GpuHorizontalFusion().Run(module.get()).ValueOrDie());
}

TEST_F(HorizontalFusionTest, HorizontalFusionAfterVerticalFusion) {
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
})").ValueOrDie();

  HloPassPipeline fusion("fusion");
  fusion.AddPass<xla::gpu::GpuInstructionFusion>(/*may_duplicate=*/false);
  fusion.AddPass<xla::gpu::GpuInstructionFusion>(/*may_duplicate=*/true);
  EXPECT_TRUE(fusion.Run(module.get()).ValueOrDie());
  EXPECT_TRUE(GpuHorizontalFusion().Run(module.get()).ValueOrDie());

  VLOG(2) << "Dump after horizontal fusion:";
  VLOG(2) << module->ToString();

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec{0, 0}));
}

TEST_F(HorizontalFusionTest, GradientDescentOptimizerLike) {
  HloComputation::Builder builder(TestName());

  std::vector<HloInstruction*> var_outs;
  int64 dim = 1;
  for (size_t i = 0; i < 128; ++i) {
    // For shapes {1, 1024}, {2, 1024}, ..., {128, 1024}
    auto shape = ShapeUtil::MakeShape(F32, {i + 1, 1024});
    dim *= 2;
    HloInstruction* param_var_in = builder.AddInstruction(
        HloInstruction::CreateParameter(i * 3 + 0, shape, "var.in"));
    HloInstruction* param_alpha =
        builder.AddInstruction(HloInstruction::CreateParameter(
            i * 3 + 1, ShapeUtil::MakeShape(F32, {}), "alpha"));
    HloInstruction* param_delta = builder.AddInstruction(
        HloInstruction::CreateParameter(i * 3 + 2, shape, "delta"));
    auto alpha_broadcasted = builder.AddInstruction(
        HloInstruction::CreateBroadcast(shape, param_alpha, {}));
    auto alpha_delta = builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kMultiply, alpha_broadcasted, param_delta));
    auto var_out = builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kSubtract, param_var_in, alpha_delta));
    var_outs.push_back(var_out);
  }
  builder.AddInstruction(HloInstruction::CreateTuple(var_outs));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  // Testing with the entire gpu optimization pipeline.
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0, 0}));
}

TEST_F(HorizontalFusionTest, FusingDifferentOutputs) {
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
)").ValueOrDie();

  EXPECT_TRUE(GpuHorizontalFusion().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());

  VLOG(2) << "Dump after horizontal fusion:";
  VLOG(2) << module->ToString();

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec{0, 0}));
}

TEST_F(HorizontalFusionTest, RMSPropLike) {
  HloComputation::Builder builder(TestName());

  std::vector<HloInstruction*> all_outputs;
  for (size_t i = 0; i < 48; ++i) {
    auto shape = ShapeUtil::MakeShape(F32, {2, 1024 + i});
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
    auto rho_broadcasted =
        builder.AddInstruction(HloInstruction::CreateBroadcast(shape, rho, {}));
    auto one_mins_rho_broadcasted = builder.AddInstruction(
        HloInstruction::CreateBroadcast(shape, one_minus_rho, {}));
    auto grad_squared = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, grad, grad));
    auto ms_1st_term = builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kMultiply, grad_squared, one_mins_rho_broadcasted));
    auto ms_2nd_term = builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kMultiply, ms, rho_broadcasted));
    auto ms_out = builder.AddInstruction(HloInstruction::CreateBinary(
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
    auto lr_broadcasted =
        builder.AddInstruction(HloInstruction::CreateBroadcast(shape, lr, {}));
    auto epsilon_broadcasted = builder.AddInstruction(
        HloInstruction::CreateBroadcast(shape, epsilon, {}));
    auto mom_1st_term = builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kMultiply, momentum, mom));
    auto ms_eps = builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, ms_out, epsilon_broadcasted));
    auto ms_eps_rsq = builder.AddInstruction(
        HloInstruction::CreateUnary(shape, HloOpcode::kRsqrt, ms_eps));
    auto grad_ms_eps_rsq = builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kMultiply, grad, ms_eps_rsq));
    auto mom_2nd_term = builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kMultiply, lr_broadcasted, grad_ms_eps_rsq));
    auto mom_out = builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, mom_1st_term, mom_2nd_term));

    // var <- var - mom
    HloInstruction* var = builder.AddInstruction(
        HloInstruction::CreateParameter(i * 9 + 8, shape, "var"));
    auto var_out = builder.AddInstruction(HloInstruction::CreateBinary(
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

}  // namespace
}  // namespace gpu
}  // namespace xla
