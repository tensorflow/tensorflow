/* Copyright 2016 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/horizontal_loop_fusion.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

auto MakeDeviceDescription() {
  stream_executor::DeviceDescription device_description{
      stream_executor::GpuDeviceInfoProto{}};
  device_description.set_threads_per_warp(32);
  return device_description;
}

class HorizontalLoopFusionTest : public HloTestBase {
 public:
  static bool IsFusion(const HloInstruction* instr) {
    return HloPredicateIsOp<HloOpcode::kFusion>(instr);
  }
  const se::DeviceDescription device_description_{MakeDeviceDescription()};
};

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
                    .value();

  EXPECT_TRUE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_FALSE(HloDCE().Run(module.get()).value());

  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(entry_root,
              GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                  m::GetTupleElement(m::Fusion()))));
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Slice(m::Concatenate()),
                                  m::Slice(m::Concatenate()))));
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
                    .value();

  EXPECT_FALSE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
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
                    .value();

  EXPECT_FALSE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
}

TEST_F(HorizontalLoopFusionTest, NegativeTestForDifferentMemorySpace) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule NegativeTestForIncompatibleSpaces
 ENTRY main {
   arg0 = f32[1]{0} parameter(0)
   arg1 = f32[1]{0:S(5)} parameter(1)
   cp1 = f32[1]{0} copy(arg0)
   cp2 = f32[1]{0:S(5)} copy(arg1)
   ROOT tuple_out = (f32[1]{0}, f32[1]{0:S(5)}) tuple(cp1, cp2)
 }
)")
                    .value();

  EXPECT_FALSE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
}

TEST_F(HorizontalLoopFusionTest, FusingIntoKLoopAndKInputTogether) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule FusingIntoKLoopAndKInputTogether

 fused_computation.1 {
   arg.1 = f16[129, 2048]{1, 0} parameter(0)
   arg.2 = f16[129, 2048]{1, 0} parameter(1)
   ROOT mul.1 = f16[129,2048]{1, 0} multiply(arg.1, arg.2)
 }

 fused_computation.2 {
   arg.1 = f16[129, 2048]{1, 0} parameter(0)
   arg.2 = f16[129, 2048]{1, 0} parameter(1)
   ROOT mul.1 = f16[129,2048]{1, 0} multiply(arg.1, arg.2)
 }

 fused_computation.3 {
   arg.1 = f16[130, 2048]{1, 0} parameter(0)
   arg.2 = f16[130, 2048]{1, 0} parameter(1)
   ROOT mul.1 = f16[130,2048]{1, 0} multiply(arg.1, arg.2)
 }

 fused_computation.4 {
   arg.1 = f16[130, 2048]{1, 0} parameter(0)
   arg.2 = f16[130, 2048]{1, 0} parameter(1)
   ROOT mul.1 = f16[130,2048]{1, 0} multiply(arg.1, arg.2)
 }

 fused_computation.5 {
   arg.1 = f16[123]{0} parameter(0)
   arg.2 = f16[123]{0} parameter(1)
   ROOT add.1 = f16[123]{0} add(arg.1, arg.2)
 }

 fused_computation.6 {
   arg.1 = f16[128]{0} parameter(0)
   arg.2 = f16[128]{0} parameter(1)
   ROOT add.1 = f16[128]{0} add(arg.1, arg.2)
 }

 ENTRY entry_computation {
   arg.1 = f16[129, 2048]{1, 0} parameter(0)
   arg.2 = f16[129, 2048]{1, 0} parameter(1)
   arg.3 = f16[129, 2048]{1, 0} parameter(2)
   arg.4 = f16[129, 2048]{1, 0} parameter(3)
   arg.5 = f16[130, 2048]{1, 0} parameter(4)
   arg.6 = f16[130, 2048]{1, 0} parameter(5)
   arg.7 = f16[130, 2048]{1, 0} parameter(6)
   arg.8 = f16[130, 2048]{1, 0} parameter(7)
   arg.9 = f16[123]{0} parameter(8)
   arg.10 = f16[123]{0} parameter(9)
   arg.11 = f16[128]{0} parameter(10)
   arg.12 = f16[128]{0} parameter(11)

   // fusion.1 and fusion.2 will be fused into kLoop fusion
   // fusion.3 and fusion.4 will be fused into another kLoop fusion
   // fusion.5 and fusion.6 will be fused into kInput fusion

   fusion.1 = f16[129,2048]{1, 0}
      fusion(arg.1, arg.2), kind=kLoop, calls=fused_computation.1

   fusion.2 = f16[129,2048]{1, 0}
      fusion(arg.3, arg.4), kind=kLoop, calls=fused_computation.2

   fusion.3 = f16[130,2048]{1, 0}
      fusion(arg.5, arg.6), kind=kLoop, calls=fused_computation.3

   fusion.4 = f16[130,2048]{1, 0}
      fusion(arg.7, arg.8), kind=kLoop, calls=fused_computation.4

   fusion.5 = f16[123]{0}
      fusion(arg.9, arg.10), kind=kLoop, calls=fused_computation.5

   fusion.6 = f16[128]{0}
      fusion(arg.11, arg.12), kind=kLoop, calls=fused_computation.6

   ROOT tuple.1 = (f16[129,2048]{1, 0}, f16[129,2048]{1, 0},
                   f16[130,2048]{1, 0}, f16[130,2048]{1, 0},
                   f16[123]{0}, f16[128]{0})
      tuple(fusion.1, fusion.2, fusion.3, fusion.4, fusion.5, fusion.6)
 }
)")
                    .value();

  EXPECT_TRUE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());

  int input_fusion_count = 0;
  int loop_fusion_count = 0;
  for (auto inst : module->entry_computation()->MakeInstructionPostOrder()) {
    if (HloPredicateIsOp<HloOpcode::kFusion>(inst)) {
      input_fusion_count +=
          (inst->fusion_kind() == HloInstruction::FusionKind::kInput) ? 1 : 0;
      loop_fusion_count +=
          (inst->fusion_kind() == HloInstruction::FusionKind::kLoop) ? 1 : 0;
    }
  }
  EXPECT_EQ(input_fusion_count, 1);
  EXPECT_EQ(loop_fusion_count, 2);
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
                    .value();

  HloPassPipeline fusion("fusion");
  const se::DeviceDescription device_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  GpuHloCostAnalysis::Options cost_analysis_options{
      HloCostAnalysis::DefaultShapeSize,
      /*per_second_rates=*/{},
      /*min_latencies_seconds=*/{},
      /*count_multiple_input_accesses=*/true};
  fusion.AddPass<xla::gpu::PriorityFusion>(/*thread_pool=*/nullptr, device_info,
                                           cost_analysis_options);
  EXPECT_TRUE(fusion.Run(module.get()).value());
  EXPECT_TRUE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
  TF_ASSERT_OK(verifier().Run(module.get()).status());

  VLOG(2) << "Dump after horizontal fusion:";
  VLOG(2) << module->ToString();

  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  const HloInstruction* fusion_instr = nullptr;
  // Check that we add bitcast when needed.
  ASSERT_THAT(entry_root,
              GmockMatch(m::Tuple(
                  m::Bitcast(m::GetTupleElement(m::Fusion(&fusion_instr))),
                  m::Bitcast(m::GetTupleElement(m::Fusion())))));
  ASSERT_TRUE(fusion_instr->IsMultiOutputFusion());
  EXPECT_THAT(fusion_instr->fused_expression_root(),
              GmockMatch(m::Tuple(
                  m::Slice(m::Concatenate(m::Reshape(), m::Reshape())),
                  m::Slice(m::Concatenate(m::Reshape(), m::Reshape())))));

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
                    .value();

  EXPECT_TRUE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_FALSE(HloDCE().Run(module.get()).value());

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
  HloModule DynamicUpdateSlice

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
                    .value();

  EXPECT_TRUE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_FALSE(HloDCE().Run(module.get()).value());

  VLOG(2) << "Dump after horizontal fusion:";
  VLOG(2) << module->ToString();

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec{0, 0}));
}

TEST_F(HorizontalLoopFusionTest, DontFuseDynamicUpdateSlice) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule DynamicUpdateSliceFusionsShareParameter

fused_dynamic_update_slice {
  p0 = s32[3,3]{1,0} parameter(0)
  p1 = pred[3,2]{1,0} parameter(1)
  convert = s32[3,2]{1,0} convert(p1)
  zero = s32[] constant(0)
  ROOT dynamic-update-slice = s32[3,3]{1,0} dynamic-update-slice(p0, convert, zero, zero)
}

fused_dynamic_update_slice.1 {
  p0 = s32[3,3]{1,0} parameter(0)
  p1 = pred[2,3]{1,0} parameter(1)
  convert = s32[2,3]{1,0} convert(p1)
  zero = s32[] constant(0)
  ROOT dynamic-update-slice = s32[3,3]{1,0} dynamic-update-slice(p0, convert, zero, zero)
}

ENTRY main {
  param_0 = s32[3,3]{1,0} parameter(0)
  param_1 = pred[2,3]{1,0} parameter(1)
  param_2 = pred[3,2]{1,0} parameter(2)
  loop_dynamic_update_slice_fusion = s32[3,3]{1,0} fusion(param_0, param_2), kind=kLoop, calls=fused_dynamic_update_slice
  loop_dynamic_update_slice_fusion.1 = s32[3,3]{1,0} fusion(param_0, param_1), kind=kLoop, calls=fused_dynamic_update_slice.1
  ROOT tuple.11.0 = (s32[3,3]{1,0}, s32[3,3]{1,0}) tuple(loop_dynamic_update_slice_fusion.1, loop_dynamic_update_slice_fusion)
}
)"));
  EXPECT_FALSE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
}

TEST_F(HorizontalLoopFusionTest,
       AllowSharedParametersWhenNotUsingConcatenation) {
  auto module = ParseAndReturnVerifiedModule(R"(
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
 })")
                    .value();

  EXPECT_TRUE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()
                  ->parameter_instruction(0)
                  ->users()[0]
                  ->fused_instructions_computation()
                  ->root_instruction(),
              GmockMatch(m::Tuple(m::Multiply(), m::Add())));
}

TEST_F(HorizontalLoopFusionTest, ForbidSharedParametersWhenUsingConcatenation) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
f {
  p = f16[] parameter(0)
}

g {
  p = f16[] parameter(0)
  b = f16[1] bitcast(p)
}

e {
  p = f16[] parameter(0)
  a = f16[] fusion(p), kind=kLoop, calls=f
  b = f16[1] fusion(p), kind=kLoop, calls=g
  t = tuple(a, b)
})"));

  // As fusions f and g have different output shapes, the horizontal fusion
  // algorithm would only consider merging them using concatenation/slicing.
  // The horizontal fusion is not supposed to happen in this
  // example though because f and g share an input parameter.
  EXPECT_FALSE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
}

TEST_F(HorizontalLoopFusionTest, FuseSmallConcatenationInputFusions) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
a {
  p = s4[1] parameter(0)
  q = s4[2] parameter(1)
  c = s4[3] concatenate(p, q), dimensions={0}
}

b {
  p = s4[4] parameter(0)
  q = s4[5] parameter(1)
  c = s4[9] concatenate(p, q), dimensions={0}
}

e {
  p = s4[1] constant({...})
  q = s4[2] constant({...})
  x = s4[3] fusion(p, q), kind=kInput, calls=a
  r = s4[4] constant({...})
  s = s4[5] constant({...})
  y = s4[9] fusion(r, s), kind=kInput, calls=b
  t = tuple(x, y)
})"));

  EXPECT_TRUE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
}

TEST_F(HorizontalLoopFusionTest, DoNotFuseLargerConcatenationInputFusions) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
a {
  p = s4[100000] parameter(0)
  q = s4[200000] parameter(1)
  c = s4[300000] concatenate(p, q), dimensions={0}
}

b {
  p = s4[200000] parameter(0)
  q = s4[100000] parameter(1)
  c = s4[300000] concatenate(p, q), dimensions={0}
}

e {
  p = s4[100000] constant({...})
  q = s4[200000] constant({...})
  x = s4[300000] fusion(p, q), kind=kInput, calls=a
  r = s4[200000] constant({...})
  s = s4[100000] constant({...})
  y = s4[300000] fusion(r, s), kind=kInput, calls=b
  t = tuple(x, y)
})"));

  EXPECT_FALSE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
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
                    .value();

  HloPassFix<HloPassPipeline> iterative_h_fusion("iterative_h_fusion");
  iterative_h_fusion.AddPass<HorizontalLoopFusion>(device_description_);
  iterative_h_fusion.AddPass<HloDCE>();
  EXPECT_TRUE(iterative_h_fusion.Run(module.get()).value());

  // Verify that fusion.0 and fusion.1 are fused.
  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(entry_root,
              GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                  m::GetTupleElement(m::Fusion()))));
  EXPECT_TRUE(fusion->IsMultiOutputFusion());

  // Verify that the total number of fusion instructions is 2 so that we
  // know sqrt.0 and sqrt.1 are fused.
  EXPECT_EQ(
      absl::c_count_if(module->entry_computation()->instructions(), IsFusion),
      2);
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
                    .value();

  HloPassFix<HloPassPipeline> iterative_h_fusion("iterative_h_fusion");
  iterative_h_fusion.AddPass<HorizontalLoopFusion>(device_description_);
  EXPECT_TRUE(iterative_h_fusion.Run(module.get()).value());

  // Verify that the total number of fusion instructions is 2 so that we
  // know all the sqrt instructions are fused into a kernel. Note that if we
  // traverse from def-to-use (i.e., top-to-down) instead of use-to-def, we
  // will end up having 3 fusions instead of 2.
  EXPECT_EQ(
      absl::c_count_if(module->entry_computation()->instructions(), IsFusion),
      2);
}

// Simplified reproducer for Google bug b/242287055.
// Things that happened:
//  - horizontal loop fusion joined addition a0 and multiplication m0
//  - the resulting fusion had 4 inputs: (gte1, gte0, gte1, gte0)
//  - buffer assignment aliased outputs of this fusion with its inputs
//  - some threads simultaneously did the addition, some - multiplication
//  - as a result some inputs were overwritten before being read
// Conditional operation is meaningless (branches are equivalent) and
// is there only to properly confuse the buffer assignment.
TEST_F(HorizontalLoopFusionTest, NoBufferAliasingOfDuplicateParameter) {
  const char* hlo_text = R"(
HloModule m

branch_a {
  p0 = s32[] parameter(0)
  c0 = s32[] constant(1)
  c1 = s32[] constant(2)
  b0 = s32[4096] broadcast(c0), dimensions={}
  b1 = s32[4096] broadcast(c1), dimensions={}
  ROOT r = (s32[4096], s32[4096]) tuple(b0, b1)
}

branch_b {
  p0 = s32[] parameter(0)
  c0 = s32[] constant(1)
  c1 = s32[] constant(2)
  b0 = s32[4096] broadcast(c0), dimensions={}
  b1 = s32[4096] broadcast(c1), dimensions={}
  ROOT r = (s32[4096], s32[4096]) tuple(b0, b1)
}

ENTRY e {
  p0 = s32[] parameter(0)
  c0 = s32[] constant(0)
  cond = (s32[4096], s32[4096]) conditional(p0, c0, c0), branch_computations={branch_a, branch_b}
  p1 = s32[4096] parameter(1)
  gte0 = s32[4096] get-tuple-element(cond), index=0
  gte1 = s32[4096] get-tuple-element(cond), index=1
  a0 = s32[4096] add(gte1, gte0)
  m0 = s32[4096] multiply(gte1, gte0)
  ROOT r = (s32[4096], s32[4096]) tuple(m0, a0)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

TEST_F(HorizontalLoopFusionTest, CopyInsertionFusionControlFlow) {
  const char* hlo_text = R"(
HloModule cluster

ENTRY main {
  cst = f32[1]{0} constant({0})
  cp1 = f32[1]{0} copy(cst)
  cp2 = f32[1]{0} copy(cst)
  cp3 = f32[1]{0} copy(cst)
  cp4 = f32[1]{0} copy(cst), control-predecessors={cp1}
  ROOT tuple_out = (f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}) tuple(cp1, cp2, cp3, cp4)
}
)";

  auto module = ParseAndReturnUnverifiedModule(hlo_text).value();
  EXPECT_TRUE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());

  VLOG(2) << module->ToString();

  // Verify that the total number of fusion instructions is 1.
  EXPECT_EQ(
      absl::c_count_if(module->entry_computation()->instructions(), IsFusion),
      1);

  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  // Check that we fuse when supported.
  EXPECT_THAT(entry_root,
              GmockMatch(m::Tuple(m::Copy(), m::GetTupleElement(m::Fusion()),
                                  m::GetTupleElement(m::Fusion()), m::Copy())));
}

TEST_F(HorizontalLoopFusionTest, DoNotMergeVariadicReductions) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule m

  fused_computation.94 {
    tmp_0 = f32[] parameter(0)
    tmp_1 = f32[] parameter(1)
    tmp_2 = pred[] compare(tmp_0, tmp_1), direction=GE
    tmp_3 = f32[] select(tmp_2, tmp_0, tmp_1)
    tmp_4 = pred[] compare(tmp_0, tmp_1), direction=EQ
    tmp_5 = s32[] parameter(2)
    tmp_6 = s32[] parameter(3)
    tmp_7 = s32[] minimum(tmp_5, tmp_6)
    tmp_8 = s32[] select(tmp_2, tmp_5, tmp_6)
    tmp_9 = s32[] select(tmp_4, tmp_7, tmp_8)
    ROOT tmp_10 = (f32[], s32[]) tuple(tmp_3, tmp_9)
  }

  minmax_func.1536 {
    tmp_0 = f32[] parameter(0)
    tmp_1 = f32[] parameter(2)
    tmp_2 = s32[] parameter(1)
    tmp_3 = s32[] parameter(3)
    ROOT tmp_4 = (f32[], s32[]) fusion(tmp_0, tmp_1, tmp_2, tmp_3), kind=kLoop, calls=fused_computation.94
  }

  fused_computation {
    tmp_0 = f32[554112,10]{1,0} parameter(0)
    tmp_1 = s32[554112,10]{1,0} iota(), iota_dimension=1
    tmp_2 = f32[] constant(-inf)
    tmp_3 = s32[] constant(0)
    ROOT tmp_4 = (f32[554112]{0}, s32[554112]{0}) reduce(tmp_0, tmp_1, tmp_2, tmp_3), dimensions={1}, to_apply=minmax_func.1536
  }

  fused_computation2 {
    tmp_0 = f32[554112,10]{1,0} parameter(0)
    tmp_1 = s32[554112,10]{1,0} iota(), iota_dimension=1
    tmp_2 = f32[] constant(inf)
    tmp_3 = s32[] constant(1)
    ROOT tmp_4 = (f32[554112]{0}, s32[554112]{0}) reduce(tmp_0, tmp_1, tmp_2, tmp_3), dimensions={1}, to_apply=minmax_func.1536
  }

  ENTRY e {
    tmp_0 = f32[554112,10]{1,0} parameter(0)
    tmp_1 = (f32[554112]{0}, s32[554112]{0}) fusion(tmp_0), kind=kLoop, calls=fused_computation
    tmp_2 = s32[554112]{0} get-tuple-element(tmp_1), index=1
    tmp_3 = f32[554112,10]{1,0} parameter(1)
    tmp_4 = (f32[554112]{0}, s32[554112]{0}) fusion(tmp_3), kind=kLoop, calls=fused_computation2
    tmp_5 = s32[554112]{0} get-tuple-element(tmp_4), index=1
    ROOT tmp_6 = s32[554112]{0} add(tmp_2, tmp_5)
  })")
                    .value();

  EXPECT_FALSE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
}

TEST_F(HorizontalLoopFusionTest, DoFusionInsideWhileLoop) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
b {
  a = (s8[]) parameter(0)
  b = s8[] get-tuple-element(a), index=0
  c = s8[] add(b, b)
  d = s8[] multiply(b, b)
  e = s8[] subtract(c, d)
  t = tuple(e)
}

c {
  p = (s8[]) parameter(0)
  r = pred[] constant(true)
}

e {
  p = (s8[]) parameter(0)
  r = (s8[]) while(p), condition=c, body=b
})"));

  EXPECT_TRUE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
}

TEST_F(HorizontalLoopFusionTest, FuseNonDefaultLayoutsUsingTuple) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
f {
  p = s8[2,2]{0,1} parameter(0)
  b = s8[2,2]{1,0} bitcast(p)
  n = s8[2,2]{1,0} negate(b)
 }

g {
  p = s8[2,2]{0,1} parameter(0)
  b = s8[2,2]{1,0} bitcast(p)
  a = s8[2,2]{1,0} add(b, b)
}

e {
  p0 = s8[2,2]{0,1} parameter(0)
  p1 = s8[2,2]{0,1} parameter(1)
  a = s8[2,2] fusion(p0), kind=kLoop, calls=f
  b = s8[2,2] fusion(p1), kind=kLoop, calls=g
  t = tuple(a, b)
})"));

  EXPECT_TRUE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()
                  ->parameter_instruction(0)
                  ->users()[0]
                  ->fused_instructions_computation()
                  ->root_instruction(),
              GmockMatch(m::Tuple(m::Negate(), m::Add())));
}

TEST_F(HorizontalLoopFusionTest, FuseNonDefaultLayoutsUsingConcatenation) {
  const std::string kHloText = R"(
HloModule m, entry_computation_layout={()->(s32[2,3]{0,1}, s32[2,3]{1,0})}

e {
  a = s32[2,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 } })
  b = s32[2,3]{1,0} constant({ { 10, 20, 30 }, { 40, 50, 60 } })
  t = tuple(a, b)
})";

  MatchOptimizedHlo(kHloText, R"(
CHECK: copy_horizontally_fused_computation
CHECK: %[[p0:.+]] = s32[2,3]{0,1} parameter(0)
CHECK: %[[c0:.+]] = s32[2,3]{0,1} copy(%[[p0]])
CHECK: %[[b0:.+]] = s32[3,2]{1,0} bitcast(%[[c0]])
CHECK: %[[r0:.+]] = s32[6]{0} reshape(%[[b0]])
CHECK: %[[p1:.+]] = s32[2,3]{1,0} parameter(1)
CHECK: %[[c1:.+]] = s32[2,3]{1,0} copy(%[[p1]])
CHECK: %[[r1:.+]] = s32[6]{0} reshape(%[[c1]])
CHECK: s32[12]{0} concatenate(%[[r0]], %[[r1]]), dimensions={0}
)");

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0}));
}

TEST_F(HorizontalLoopFusionTest, PassBitcastsLookingForFusionCandidates) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
a {
  p = s4[1] parameter(0)
}

b {
  p = s4[2] parameter(0)
}

e {
  p = s4[1] constant({...})
  x = s4[1] fusion(p), kind=kLoop, calls=a
  xb = s4[1,3] bitcast(x)
  q = s4[2] constant({...})
  y = s4[2] fusion(q), kind=kLoop, calls=b
  yb = s4[9,1] bitcast(y)
  t = tuple(xb, yb)
})"));

  EXPECT_TRUE(
      HorizontalLoopFusion{device_description_}.Run(module.get()).value());
}

TEST_F(HorizontalLoopFusionTest, DontFuseCopiesInsideWhileLoops) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module_main, entry_computation_layout={(f32[10]{0}, f32[20]{0})->(s32[], f32[10]{0}, f32[20]{0})}

f {
  param0 = f32[10]{0} parameter(0)
  reverse = f32[10]{0} reverse(param0), dimensions={0}
  param1 = f32[20]{0} parameter(1)
  param2 = s32[] parameter(2)
  dynamic_slice = f32[10]{0} dynamic-slice(param1, param2), dynamic_slice_sizes={10}
  ROOT res = f32[10]{0} add(reverse, dynamic_slice)
}

body {
  p0 = (s32[], f32[10]{0}, f32[20]{0}) parameter(0)
  iter = s32[] get-tuple-element(p0), index=0
  one = s32[] constant(1)
  next_iter = s32[] add(iter, one)
  a = f32[10]{0} get-tuple-element(p0), index=1
  b = f32[20]{0} get-tuple-element(p0), index=2
  next_a = f32[10]{0} fusion(a, b, iter), kind=kLoop, calls=f
  copy.0 = f32[10]{0} copy(next_a)
  next_b = f32[20]{0} reverse(b), dimensions={0}
  copy.1 = f32[20]{0} copy(next_b)
  ROOT r = (s32[], f32[10]{0}, f32[20]{0}) tuple(next_iter, copy.0, copy.1)
}

cond {
  p = (s32[], f32[10]{0}, f32[20]{0}) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  bound = s32[] constant(10)
  ROOT res.1 = pred[] compare(i, bound), direction=LT
}

ENTRY main {
  zero = s32[] constant(0)
  p0.1 = f32[10]{0} parameter(0)
  p1.0 = f32[20]{0} parameter(1)
  while_init = (s32[], f32[10]{0}, f32[20]{0}) tuple(zero, p0.1, p1.0)
  ROOT while = (s32[], f32[10]{0}, f32[20]{0}) while(while_init), condition=cond, body=body
})"));
  HorizontalLoopFusion loop_fusion(device_description_, /*prefix=*/"",
                                   /*only_entry_computation=*/true);
  EXPECT_FALSE(loop_fusion.Run(module.get()).value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
