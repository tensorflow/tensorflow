/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
#include "tensorflow/compiler/xla/service/gpu/softmax_rewriter_triton.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class SoftmaxRewriterTritonTest : public HloTestBase {
 public:
  void SetUp() override {
    gpu_version_ = GpuVersion{
        se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  }

 protected:
  GpuVersion gpu_version_;
};

TEST_F(SoftmaxRewriterTritonTest, CanFuseExactSoftmaxF32) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
  exponential = f32[127,125]{1,0} exponential(subtract)
  constant_zero = f32[] constant(0)
  second_reduce = f32[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = f32[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(SoftmaxRewriterTritonTest, CanFuseFirstSoftmaxDiamondF16) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f16[] parameter(0)
  arg_1 = f16[] parameter(1)
  ROOT maximum = f16[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f16[127,125]{1,0} parameter(0)
  constant_neg_inf = f16[] constant(-inf)
  reduce = f16[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f16[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f16[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(SoftmaxRewriterTritonTest, CanFuseExactSoftmaxF64) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f64[] parameter(0)
  arg_1 = f64[] parameter(1)
  ROOT maximum = f64[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = f64[] parameter(0)
  arg_1.1 = f64[] parameter(1)
  ROOT add = f64[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f64[127,125]{1,0} parameter(0)
  constant_neg_inf = f64[] constant(-inf)
  reduce = f64[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f64[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = f64[127,125]{1,0} subtract(param_0, broadcast)
  exponential = f64[127,125]{1,0} exponential(subtract)
  constant_zero = f64[] constant(0)
  second_reduce = f64[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f64[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = f64[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(SoftmaxRewriterTritonTest, CanNotFuseExactSoftmaxBF16) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = bf16[] parameter(0)
  arg_1 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = bf16[] parameter(0)
  arg_1.1 = bf16[] parameter(1)
  ROOT add = bf16[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = bf16[127,125]{1,0} parameter(0)
  constant_neg_inf = bf16[] constant(-inf)
  reduce = bf16[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = bf16[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = bf16[127,125]{1,0} subtract(param_0, broadcast)
  exponential = bf16[127,125]{1,0} exponential(subtract)
  constant_zero = bf16[] constant(0)
  second_reduce = bf16[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = bf16[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = bf16[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanFuseSoftmaxWithBatchDimMergingAndSplittingBitcastsOnEveryEdge) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f32[130,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  bitcasted_param_0 = f32[65,2,125] bitcast(param_0)
  reduce = f32[65,2]{1,0} reduce(bitcasted_param_0, constant_neg_inf), dimensions={2}, to_apply=max_computation
  bitcasted_reduce = f32[130] bitcast(reduce)
  broadcast = f32[130,125]{1,0} broadcast(bitcasted_reduce), dimensions={0}
  bitcasted_broadcast = f32[65,2,125] bitcast(broadcast)
  subtract = f32[65,2,125]{2,1,0} subtract(bitcasted_param_0, bitcasted_broadcast)
  bitcasted_subtract = f32[130,125] bitcast(subtract)
  exponential = f32[130,125]{1,0} exponential(bitcasted_subtract)
  constant_zero = f32[] constant(0)
  bitcasted_exponential = f32[2,65,125] bitcast(exponential)
  second_reduce = f32[2,65]{1,0} reduce(bitcasted_exponential, constant_zero), dimensions={2}, to_apply=add_computation
  second_bitcasted_reduce = f32[130] bitcast(second_reduce)
  second_broadcast = f32[130,125]{1,0} broadcast(second_bitcasted_reduce), dimensions={0}
  second_bitcasted_broadcast = f32[2,65,125] bitcast(second_broadcast)
  divide = f32[2,65,125]{2,1,0} divide(bitcasted_exponential, second_bitcasted_broadcast)
  ROOT bitcasted_divide = f32[130,125] bitcast(divide)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
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
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
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
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
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
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
}

// TODO(bchetioui): expand so this can be supported?
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
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanFuseSoftmaxWithIntermediateUnaryElementwise) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
  abs = f32[127,125]{1,0} abs(subtract)
  exponential = f32[127,125]{1,0} exponential(abs)
  constant_zero = f32[] constant(0)
  second_reduce = f32[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = f32[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(SoftmaxRewriterTritonTest,
       CanFuseTwoDiamondsWithSecondDiamondProducerEqualToFirstDiamondRoot) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
  constant_zero = f32[] constant(0)
  second_reduce = f32[127]{0} reduce(subtract, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = f32[127,125]{1,0} divide(subtract, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(SoftmaxRewriterTritonTest,
       CanFuseDiamondWithTrailingUnaryElementwiseAtTheRoot) {
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
  ROOT abs = f32[127,125]{1,0} abs(subtract)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
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
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
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
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithNonConstantReducerIdentity) {
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
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithTritonIncompatibleRoot) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f16[] parameter(0)
  arg_1 = f16[] parameter(1)
  ROOT maximum = f16[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f16[127,125]{1,0} parameter(0)
  constant_neg_inf = f16[] constant(-inf)
  reduce = f16[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f16[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT divide = f16[127,125]{1,0} divide(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithTritonIncompatibleReducer) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  floor_0 = f32[] floor(arg_0)
  ROOT maximum = f32[] maximum(floor_0, arg_1)
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
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
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
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
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
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseTwoDiamondsWithDifferentReductionAxisSizeTogether) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f32[127,625]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,625]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[127,625]{1,0} subtract(param_0, broadcast)
  bitcasted_subtract = f32[127,5,125] bitcast(subtract)
  exponential = f32[127,5,125] exponential(bitcasted_subtract)
  constant_zero = f32[] constant(0)
  second_reduce = f32[127,5] reduce(exponential, constant_zero), dimensions={2}, to_apply=add_computation
  second_broadcast = f32[127,5,125] broadcast(second_reduce), dimensions={0,1}
  ROOT divide = f32[127,5,125] divide(exponential, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Bitcast(m::Fusion(m::Parameter())))));
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseTwoDiamondsWithExtraUsageForFirstDiamondRoot) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
  exponential = f32[127,125]{1,0} exponential(subtract)
  constant_zero = f32[] constant(0)
  second_reduce = f32[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  divide = f32[127,125]{1,0} divide(exponential, second_broadcast)
  ROOT tuple = (f32[127,125]{1,0}, f32[127,125]{1,0}) tuple(divide, subtract)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Fusion(m::Fusion()), m::Fusion(m::Parameter()))));
}

TEST_F(SoftmaxRewriterTritonTest,
       CanNotFuseTwoDiamondsWithExtraUsageForSecondDiamondProducer) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
  exponential = f32[127,125]{1,0} exponential(subtract)
  constant_zero = f32[] constant(0)
  second_reduce = f32[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  divide = f32[127,125]{1,0} divide(exponential, second_broadcast)
  ROOT tuple = (f32[127,125]{1,0}, f32[127,125]{1,0}) tuple(divide, exponential)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Fusion(m::Fusion()), m::Fusion(m::Parameter()))));
}

TEST_F(SoftmaxRewriterTritonTest,
       CanFuseSoftmaxDiamondWithTritonIncompatibleProducer) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  floor_0 = f32[127,125] floor(param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(floor_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(floor_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Floor(m::Parameter()))));
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
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest,
       CanFuseSoftmaxDiamondWithBitcastProducerFollowedByBitcastsOnEachUse) {
  const std::string hlo_string = R"(
HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[1,1,127,125]{3,2,1,0} parameter(0)
  bitcast_parent = f32[127,125]{1,0} bitcast(param_0)
  bitcast_0 = f32[127,125]{1,0} bitcast(bitcast_parent)
  bitcast_1 = f32[127,125]{1,0} bitcast(bitcast_parent)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(bitcast_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(bitcast_1, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(
    SoftmaxRewriterTritonTest,
    CanNotFuseSoftmaxDiamondWithBitcastProducerFollowedByThreeBitcastsOnTheLeftIncludingTwoNonFusibleOnes) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[1,1,127,125]{3,2,1,0} parameter(0)
  bitcast_parent = f32[127,125] bitcast(param_0)
  bitcast_0 = f32[127,5,25] bitcast(bitcast_parent)
  bitcast_1 = f32[1,127,125] bitcast(bitcast_0)
  bitcast_2 = f32[127,125] bitcast(bitcast_1)
  bitcast_3 = f32[127,125] bitcast(bitcast_parent)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(bitcast_3, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(bitcast_2, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest, DoNotFuseSoftmaxWithSmallRows) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,50]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,50]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,50]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    CanOnlyFuseConvertInvolvingBF16InputIntoSoftmaxDiamondWithAtLeastAmpereComputeCapability) {  // NOLINT(whitespace/line_length)
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
}
)";
  auto ampere_module = ParseAndReturnVerifiedModule(hlo_string).value();
  auto volta_module = ampere_module->Clone();

  // Ampere
  SoftmaxRewriterTriton fusion_rewriter_ampere(
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0});
  EXPECT_TRUE(fusion_rewriter_ampere.Run(ampere_module.get()).value());
  EXPECT_TRUE(verifier().Run(ampere_module.get()).status().ok());
  VLOG(2) << ampere_module->ToString();
  EXPECT_THAT(ampere_module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));

  // Volta (pre-Ampere)
  SoftmaxRewriterTriton fusion_rewriter_volta(
      se::CudaComputeCapability{se::CudaComputeCapability::VOLTA, 0});
  EXPECT_TRUE(fusion_rewriter_volta.Run(volta_module.get()).value());
  EXPECT_TRUE(verifier().Run(volta_module.get()).status().ok());
  VLOG(2) << volta_module->ToString();
  EXPECT_THAT(volta_module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Convert(m::Parameter()))));
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
