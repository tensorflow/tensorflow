/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/transforms/softmax_rewriter_triton.h"

#include <memory>
#include <string>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

using ::testing::HasSubstr;

bool HasBlockLevelFusionConfig(const HloInstruction* fusion) {
  return HloPredicateIsOp<HloOpcode::kFusion>(fusion) &&
         fusion->has_backend_config() &&
         fusion->backend_config<GpuBackendConfig>().ok() &&
         fusion->backend_config<GpuBackendConfig>()
             ->fusion_backend_config()
             .has_block_level_fusion_config();
}

class SoftmaxRewriterTritonTest
    : public HloTestBase,
      public ::testing::WithParamInterface<PrimitiveType> {
 protected:
  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};
  SoftmaxRewriterTriton fusion_rewriter_{device_info_,
                                         HloCostAnalysis::DefaultShapeSize};
};

TEST_F(SoftmaxRewriterTritonTest, CanFuseSingleNormalizationF32) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig)));
}

TEST_F(SoftmaxRewriterTritonTest,
       CanFuseSignleNormalizationWithNonF32DataType) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f16[] parameter(0)
  arg_1 = f16[] parameter(1)
  ROOT maximum = f16[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0 = f16[] parameter(0)
  arg_1 = f16[] parameter(1)
  ROOT add = f16[] add(arg_0, arg_1)
}
ENTRY main {
  param_0 = f16[127,125]{1,0} parameter(0)
  constant_neg_inf = f16[] constant(-inf)
  reduce = f16[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f16[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f16[127,125]{1,0} subtract(param_0, broadcast)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig)));
}

TEST_F(SoftmaxRewriterTritonTest, CanFuseSingleNormalizationDiamond) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig)));
}

TEST_F(SoftmaxRewriterTritonTest,
       DoesNotFuseDiamondInvolvingUnsupportedTritonInstruction) {
  const std::string hlo_string = R"(
HloModule softmax
add_computation {
  arg_0.1 = bf16[] parameter(0)
  arg_1.1 = bf16[] parameter(1)
  ROOT add = bf16[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = bf16[127,125]{1,0} parameter(0)
  constant_zero = bf16[] constant(0)
  reduce = bf16[127]{0} reduce(param_0, constant_zero), dimensions={1}, to_apply=add_computation
  broadcast = bf16[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT divide = bf16[127,125]{1,0} divide(param_0, broadcast)
})";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  const HloInstruction* bf16_divide =
      module->entry_computation()->root_instruction();
  EXPECT_FALSE(IsTritonSupportedInstruction(
      *bf16_divide, device_info_.gpu_compute_capability()));
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       DoesNotFuseInstructionsUnsupportedByTritonIntoDiamonds) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = bf16[] parameter(0)
  arg_1 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = bf16[127,125]{1,0} parameter(0)
  constant_neg_inf = bf16[] constant(-inf)
  reduce = bf16[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = bf16[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = bf16[127,125]{1,0} subtract(param_0, broadcast)
  ROOT round = bf16[127,125]{1,0} round-nearest-even(subtract)
})";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  const HloInstruction* bf16_round_nearest_even =
      hlo_query::GetFirstInstructionWithOpcode(*module->entry_computation(),
                                               HloOpcode::kRoundNearestEven);
  EXPECT_FALSE(IsTritonSupportedInstruction(
      *bf16_round_nearest_even, device_info_.gpu_compute_capability()));
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::RoundNearestEven(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig))));
}

TEST_F(SoftmaxRewriterTritonTest, CanNotFuseSoftmaxDiamondWithWrongLayout) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{0,1} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithWrongReduceDimension) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[125]{0} reduce(param_0, constant_neg_inf), dimensions={0}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={1}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithWrongBroadcastDimension) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[125,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[125]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[125,125]{1,0} broadcast(reduce), dimensions={1}
  ROOT subtract = f32[125,125]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithExtraBroadcastUsage) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
  ROOT multiply = f32[127,125]{1,0} multiply(broadcast, subtract)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest, DoesNotFuseReductionOnNonMinorAxis) {
  const std::string hlo_string = R"(
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[8,16,16]{2,1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[8,16]{1,0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[8,16,16]{2,1,0} broadcast(reduce), dimensions={0,1}
  ROOT subtract = f32[8,16,16]{2,1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest, DoesNotFuseReductionOnMultipleReductionAxes) {
  const std::string hlo_string = R"(
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[8,16,16]{2,1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[8]{0} reduce(param_0, constant_neg_inf), dimensions={2,1}, to_apply=max_computation
  broadcast = f32[8,16,16]{2,1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[8,16,16]{2,1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest, CanFuseDiamondWithUnaryElementwisePrefix) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  abs = f32[127,125]{1,0} abs(param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(abs, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig)));
}

TEST_F(SoftmaxRewriterTritonTest,
       CanFuseDiamondWithMultipleBroadcastDimensions) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[1,3,125,125]{3,2,1,0} parameter(0)
  bitcast = f32[3,125,125]{2,1,0} bitcast(f32[1,3,125,125]{3,2,1,0} param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[3,125]{1,0} reduce(f32[3,125,125]{2,1,0} bitcast, f32[] constant_neg_inf), dimensions={2}, to_apply=max_computation
  broadcast = f32[1,3,125,125]{3,2,1,0} broadcast(f32[3,125]{1,0} reduce), dimensions={1,2}
  ROOT subtract = f32[1,3,125,125]{3,2,1,0} subtract(f32[1,3,125,125]{3,2,1,0} param_0, f32[1,3,125,125]{3,2,1,0} broadcast)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig)));
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithParameterReducerIdentity) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  identity = f32[] parameter(1)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, identity), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithTritonIncompatibleReducer) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  if_0 = pred[] is-finite(arg_0)
  c = f32[] convert(if_0)
  ROOT maximum = f32[] maximum(c, arg_1)
}

ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanFuseSoftmaxDiamondWithLastDimensionBitcastAfterReduce) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[3,127,125]{2,1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[3,127]{1,0} reduce(param_0, constant_neg_inf), dimensions={2}, to_apply=max_computation
  bitcasted_reduce = f32[381]{0} bitcast(reduce)
  broadcast = f32[381,125]{1,0} broadcast(bitcasted_reduce), dimensions={0}
  bitcasted_broadcast = f32[3,127,125]{2,1,0} bitcast(broadcast)
  ROOT subtract = f32[3,127,125]{2,1,0} subtract(param_0, bitcasted_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig)));
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithTransposeBitcast) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[1,127,125]{2,1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  bitcasted_param_0 = f32[127,1,125]{2,0,1} bitcast(param_0)
  reduce = f32[127,1]{0,1} reduce(bitcasted_param_0, constant_neg_inf), dimensions={2}, to_apply=max_computation
  broadcast = f32[127,1,125]{2,0,1} broadcast(reduce), dimensions={0,1}
  bitcasted_broadcast = f32[1,127,125]{2,1,0} bitcast(broadcast)
  ROOT subtract = f32[1,127,125]{2,1,0} subtract(param_0, bitcasted_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithNonFusibleBitcastBetweenReduceAndProducer) {
  const std::string hlo_string = R"(
HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[1,127,5,25]{3,2,1,0} parameter(0)
  bitcast_0 = f32[127,125] bitcast(param_0)
  bitcast_1 = f32[127,125] bitcast(param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(bitcast_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(bitcast_1, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest, CanFuseSoftmaxDiamondWithBitcastsOnEachUse) {
  const std::string hlo_string = R"(
HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  bitcast_0 = f32[127,125]{1,0} bitcast(param_0)
  bitcast_1 = f32[127,125]{1,0} bitcast(param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(bitcast_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(bitcast_1, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig)));
}

TEST_F(SoftmaxRewriterTritonTest, RewriterBailsOutOnPreAmpereCudaGpu) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = bf16[127,125]{1,0} parameter(0)
  param_0_f32 = f32[127,125]{1,0} convert(param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0_f32, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0_f32, broadcast)
})";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_THAT(
      SoftmaxRewriterTriton(
          TestGpuDeviceInfo::RTXA6000DeviceInfo(
              se::CudaComputeCapability{se::CudaComputeCapability::kVolta, 0}),
          HloCostAnalysis::DefaultShapeSize)
          .Run(module.get()),
      tsl::testing::StatusIs(
          tsl::error::FAILED_PRECONDITION,
          ::testing::HasSubstr("Triton support is only enabled for Ampere GPUs "
                               "(compute capability 8.0) and up, but got")));
}

TEST_F(SoftmaxRewriterTritonTest, RewriterSucceedsOnNonCudaGpu) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = bf16[127,125]{1,0} parameter(0)
  param_0_f32 = f32[127,125]{1,0} convert(param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0_f32, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0_f32, broadcast)
})";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(SoftmaxRewriterTriton(TestGpuDeviceInfo::AMDMI210DeviceInfo(),
                                    HloCostAnalysis::DefaultShapeSize)
                  .Run(module.get())
                  .ok());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    CanFuseIntermediateBinaryElementwiseWithinDiamondWhenBothOperandsAreTheSame) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule fusible_diamond
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  multiply =  f32[127]{0} multiply(reduce, reduce)
  broadcast = f32[127,125]{1,0} broadcast(multiply), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig)));
}

TEST_F(
    SoftmaxRewriterTritonTest,
    DoesNotFuseIntermediateBinaryElementwiseWithBothSplatOperandsIntoDiamond) {
  const std::string hlo_string = R"(
HloModule nonfusible_splat
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  constant_0 = f32[] constant(0.333333343)
  splat_0 = f32[127,125]{1,0} broadcast(constant_0), dimensions={}
  constant_1 = f32[] constant(0.66666)
  splat_1 = f32[127,125]{1,0} broadcast(constant_1), dimensions={}
  param_0 = f32[127,125]{1,0} parameter(0)
  multiply_splats = f32[127,125]{1,0} multiply(splat_0, splat_1)
  multiply_splat_param = f32[127,125]{1,0} multiply(multiply_splats, param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(multiply_splat_param, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    DoesNotFuseIntermediateBinaryElementwiseWithSameSplatOperandsIntoDiamond) {
  const std::string hlo_string = R"(
HloModule nonfusible_splat_diamond
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  constant_0 = f32[] constant(0.333333343)
  splat = f32[127,125]{1,0} broadcast(constant_0), dimensions={}
  param_0 = f32[127,125]{1,0} parameter(0)
  multiply = f32[127,125]{1,0} multiply(splat, splat)
  add = f32[127,125]{1,0} add(param_0, multiply)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(add, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(device_info_,
                                        HloCostAnalysis::DefaultShapeSize);
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest, CanFuseRMSNormDiamond) {
  const std::string hlo_string = R"(
HloModule rms_norm
add_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(arg_0, arg_1)
}
ENTRY main.30 {
  param_0 = f32[10,10,10,128]{3,2,1,0} parameter(0)
  multiply_param = f32[10,10,10,128]{3,2,1,0} multiply(param_0, param_0)
  constant_0 = f32[] constant(0)
  reduce = f32[10,10,10]{2,1,0} reduce(multiply_param, constant_0), dimensions={3}, to_apply=add_computation
  constant_1 = f32[] constant(0.333333343)
  splat = f32[10,10,10]{2,1,0} broadcast(constant_1), dimensions={}
  multiply_splat = f32[10,10,10]{2,1,0} multiply(reduce, splat)
  epsilon = f32[] constant(1e-06)
  splat_epsilon = f32[10,10,10]{2,1,0} broadcast(epsilon), dimensions={}
  add = f32[10,10,10]{2,1,0} add(multiply_splat, splat_epsilon)
  rsqrt = f32[10,10,10]{2,1,0} rsqrt(add)
  broadcast = f32[10,10,10,128]{3,2,1,0} broadcast(rsqrt), dimensions={0,1,2}
  ROOT multiply = f32[10,10,10,128]{3,2,1,0} multiply(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig)));
}

TEST_F(
    SoftmaxRewriterTritonTest,
    CanFuseBinaryElementwiseWhereTheFirstOperandIsASplatConstantWithinDiamond) {
  const std::string hlo_string = R"(
HloModule fusible_diamond
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  constant = f32[] constant(0.333333343)
  broadcast_splat = f32[127]{0} broadcast(constant), dimensions={}
  multiply = f32[127]{0} multiply(broadcast_splat, reduce)
  broadcast = f32[127,125]{1,0} broadcast(multiply), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig)));
}


TEST_F(SoftmaxRewriterTritonTest,
       CanFuseBinaryElementwiseOperationWhereOneOperandIsASharedSplatProducer) {
  const std::string hlo_string = R"(
HloModule nonfusible_diamond
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT max = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_2 = f32[] constant(0.333333343)
  broadcast_splat = f32[127,125]{1,0} broadcast(constant_2), dimensions={}
  param_1 = f32[127,125]{1,0} parameter(1)
  multiply_splat = f32[127,125]{1,0} multiply(broadcast_splat, param_1)
  multiply = f32[127,125]{1,0} multiply(param_0, broadcast_splat)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(multiply, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::Parameter()).WithPredicate(HasBlockLevelFusionConfig)));
}

TEST_F(
    SoftmaxRewriterTritonTest,
    DoesNotFuseBinaryElementwiseOperationWhereFirstOperandIsASplatAndSecondOperandIsASharedSplatProducer) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule nonfusible_diamond
add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_2 = f32[] constant(0.333333343)
  broadcast_splat_shared = f32[127,125]{1,0} broadcast(constant_2), dimensions={}
  param_1 = f32[127,125]{1,0} parameter(1)
  multiply_splat_shared = f32[127,125]{1,0} multiply(broadcast_splat_shared, param_1)
  constant_3 = f32[] constant(0.5)
  broadcast_splat = f32[127,125]{1,0} broadcast(constant_3), dimensions={}
  multiply_splat = f32[127,125]{1,0} multiply(broadcast_splat, broadcast_splat_shared)
  multiply = f32[127,125]{1,0} multiply(param_0, multiply_splat)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(multiply, constant_neg_inf), dimensions={1}, to_apply=add_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest, FusionDecisionIsCapturedExplicitly) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  identity_f8 = f8e5m2[] parameter(1)
  identity = f32[] convert(identity_f8)
  reduce = f32[127]{0} reduce(param_0, identity), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton softmax_rewriter_triton(
      device_info_, HloCostAnalysis::DefaultShapeSize);
  int unmatched = 0, matched = 0;
  for (HloInstruction* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    DiamondMatchingDecision decision =
        softmax_rewriter_triton.MatchesTritonCompatibleClosedReductionDiamond(
            instruction);
    if (std::holds_alternative<FusionDecision>(decision)) {
      std::string actual_decision =
          std::get<FusionDecision>(decision).Explain();
      EXPECT_THAT(
          actual_decision,
          AnyOf(
              HasSubstr("Root is not elementwise binary"),
              HasSubstr("identity is not a constant or a supported convert")));
      unmatched++;
    } else {
      matched++;
    }
  }
  EXPECT_EQ(unmatched, 6);
  EXPECT_EQ(matched, 0);
}

TEST_F(
    SoftmaxRewriterTritonTest,
    FusesBinaryElementwiseIfIntermediateDiamondOpWithBroadcastAlongReductionDimAsParameter) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[32]{0} parameter(0)
  p1 = f32[32,16]{1,0} parameter(1)
  c = f32[] constant(0)

  r0 = f32[32]{0} reduce(p1, c), dimensions={1}, to_apply=add_computation
  b0 = f32[32,16]{1,0} broadcast(r0), dimensions={0}
  b1 = f32[32,16]{1,0} broadcast(p0), dimensions={0}
  add0 = f32[32,16]{1,0} add(b1, p1)
  ROOT add1 = f32[32,16]{1,0} add(add0, b0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    FusesBinaryElementwiseIfIntermediateDiamondOpWithBroadcastAlongBatchDimAsParameter) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[16]{0} parameter(0)
  p1 = f32[32,16]{1,0} parameter(1)
  c = f32[] constant(0)

  r0 = f32[32]{0} reduce(p1, c), dimensions={1}, to_apply=add_computation
  b0 = f32[32,16]{1,0} broadcast(r0), dimensions={0}
  b1 = f32[32,16]{1,0} broadcast(p0), dimensions={1}
  add0 = f32[32,16]{1,0} add(b1, p1)
  ROOT add1 = f32[32,16]{1,0} add(add0, b0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    FusesBinaryElementwiseIfIntermediateDiamondOpWithMultiDimTensorBroadcastAlongBatchDimAsParameter) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[32,16]{1,0} parameter(0)
  p1 = f32[64,32,16]{2,1,0} parameter(1)
  c = f32[] constant(0)

  r0 = f32[64,32]{1,0} reduce(p1, c), dimensions={2}, to_apply=add_computation
  b0 = f32[64,32,16]{2,1,0} broadcast(r0), dimensions={0,1}
  b1 = f32[64,32,16]{2,1,0} broadcast(p0), dimensions={1,2}
  add0 = f32[64,32,16]{2,1,0} add(b1, p1)
  ROOT add1 = f32[64,32,16]{2,1,0} add(add0, b0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    FusesBinaryElementwiseIfIntermediateDiamondOpWithZeroDimTensorBroadcastAsParameter) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  parameter_0 = f32[] parameter(0)
  parameter_1 = f32[64,32,16]{2,1,0} parameter(1)
  c = f32[] constant(0)

  reduce_0 = f32[64,32]{1,0} reduce(parameter_1, c), dimensions={2}, to_apply=add_computation
  broadcast_0 = f32[64,32,16]{2,1,0} broadcast(reduce_0), dimensions={0,1}
  broadcast_1 = f32[64,32,16]{2,1,0} broadcast(parameter_0), dimensions={}
  add_0 = f32[64,32,16]{2,1,0} add(broadcast_1, parameter_1)
  ROOT add1 = f32[64,32,16]{2,1,0} add(add_0, broadcast_0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    FusesBinaryElementwiseIfIntermediateDiamondOpIsBroadcastOf1DParameterAlongNonReductionDimensions) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  parameter_0 = f32[16] parameter(0)
  parameter_1 = f32[64,32,16]{2,1,0} parameter(1)
  c = f32[] constant(0)

  reduce_0 = f32[64,32]{1,0} reduce(parameter_1, c), dimensions={2}, to_apply=add_computation
  broadcast_0 = f32[64,32,16]{2,1,0} broadcast(reduce_0), dimensions={0,1}
  broadcast_1 = f32[64,32,16]{2,1,0} broadcast(parameter_0), dimensions={2}
  add_0 = f32[64,32,16]{2,1,0} add(broadcast_1, parameter_1)
  ROOT add1 = f32[64,32,16]{2,1,0} add(add_0, broadcast_0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       FusesBinaryElementwiseIfIntermediateDiamondOpIsBroadcastOfParameter) {
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  parameter_0 = f32[64] parameter(0)
  parameter_1 = f32[64,32,16]{2,1,0} parameter(1)
  c = f32[] constant(0)

  reduce_0 = f32[64,32]{1,0} reduce(parameter_1, c), dimensions={2}, to_apply=add_computation
  broadcast_0 = f32[64,32,16]{2,1,0} broadcast(reduce_0), dimensions={0,1}
  broadcast_1 = f32[64,32,16]{2,1,0} broadcast(parameter_0), dimensions={0}
  add_0 = f32[64,32,16]{2,1,0} add(broadcast_1, parameter_1)
  ROOT add1 = f32[64,32,16]{2,1,0} add(add_0, broadcast_0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    FusesBinaryElementwiseIfIntermediateDiamondOpWithMultipleDimensionsAsParameter) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[32,16]{1,0} parameter(0)
  p1 = f32[128,64,32,16]{3,2,1,0} parameter(1)
  c = f32[] constant(0)

  r0 = f32[128,64,32]{2,1,0} reduce(p1, c), dimensions={3}, to_apply=add_computation
  b0 = f32[128,64,32,16]{3,2,1,0} broadcast(r0), dimensions={0,1,2}
  b1 = f32[128,64,32,16]{3,2,1,0} broadcast(p0), dimensions={2,3}
  add0 = f32[128,64,32,16]{3,2,1,0} add(b1, p1)
  ROOT add1 = f32[128,64,32,16]{3,2,1,0} add(add0, b0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
}

// Triton has a requirement that any tile in the program should not have more
// than 1048576 elements.
TEST_F(SoftmaxRewriterTritonTest, DoesNotFuseIfResultingFusionCannotBeTiled) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[8,2097152] parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[8]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[8,2097152] broadcast(reduce), dimensions={0}
  ROOT subtract = f32[8,2097152] subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(fusion_rewriter_.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       DoNotFuseNormalizationWithVeryLongRowsIfProfitabilityCheckIsEnabled) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[8,262144] parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[8]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[8,262144] broadcast(reduce), dimensions={0}
  ROOT subtract = f32[8,262144] subtract(param_0, broadcast)
})";

  {
    // Verify that SoftmaxRewriterTriton without Cost Model will fuse the
    // normalization diamond.
    SoftmaxRewriterTriton fusion_rewriter_without_cost_model{
        device_info_, HloCostAnalysis::DefaultShapeSize,
        /*only_fuse_if_profitable=*/false};

    auto module = ParseAndReturnVerifiedModule(hlo_string).value();
    EXPECT_TRUE(fusion_rewriter_without_cost_model.Run(module.get()).value());
    EXPECT_TRUE(verifier().Run(module.get()).status().ok());
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                GmockMatch(m::Fusion(m::Parameter())
                               .WithPredicate(HasBlockLevelFusionConfig)));
  }

  {
    // SoftmaxRewriterTriton with Cost Model will discard the normalization
    // diamond, because row size is too large.
    SoftmaxRewriterTriton fusion_rewriter_with_cost_model{
        device_info_, HloCostAnalysis::DefaultShapeSize,
        /*only_fuse_if_profitable=*/true};

    auto module = ParseAndReturnVerifiedModule(hlo_string).value();
    EXPECT_FALSE(fusion_rewriter_with_cost_model.Run(module.get()).value());
  }
}

TEST_F(SoftmaxRewriterTritonTest, DoesNotCrashOnScalarBroadcast) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  param_1 = f32[] parameter(1)
  broadcast_from_scalar = f32[127] broadcast(param_1), dimensions={}
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  add = f32[127]{0} add(broadcast_from_scalar, reduce)
  broadcast = f32[127,125]{1,0} broadcast(add), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(fusion_rewriter_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                             .WithPredicate(HasBlockLevelFusionConfig)));
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
