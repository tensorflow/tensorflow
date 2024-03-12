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

#include "xla/service/gpu/fusion_merger_triton.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/autotune_results.pb.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/status_matchers.h"

using ::tsl::testing::IsOk;
using ::tsl::testing::IsOkAndHolds;

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;
using FusionMergerTritonTest = HloTestBase;

TEST_F(FusionMergerTritonTest,
       CanMergeTritonFusionWithSingleParameterProducer) {
  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

auxiliary_computation {
  parameter_0 = f32[125]{0} parameter(0)
  ROOT broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125]{0} parameter(0)
  auxiliary_fusion = f32[125,127]{1,0} fusion(param_0), kind=kLoop, calls=auxiliary_computation
  ROOT triton_softmax = f32[125,127]{1,0} fusion(auxiliary_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
})";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger;
  EXPECT_THAT(fusion_merger.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOk());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(FusionMergerTritonTest, CanMergeWithTwoParameterConsumer) {
  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

consumer_computation {
  parameter_0 = f32[125]{0} parameter(0)
  parameter_1 = f32[125,127]{1,0} parameter(1)
  broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(parameter_1, broadcast)
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[125]{0} parameter(1)
  triton_softmax = f32[125,127]{1,0} fusion(param_0), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
  ROOT consumer_fusion = f32[125,127]{1,0} fusion(param_1, triton_softmax), kind=kLoop, calls=consumer_computation
})";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger{};
  EXPECT_TRUE(fusion_merger.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(
    FusionMergerTritonTest,
    CanMergeProducerFusionIntoTritonSoftmaxConsumerWhenTheConsumerIsNotRoot) {
  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

auxiliary_computation {
  parameter_0 = f32[125]{0} parameter(0)
  ROOT broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125]{0} parameter(0)
  auxiliary_fusion = f32[125,127]{1,0} fusion(param_0), kind=kLoop, calls=auxiliary_computation
  triton_softmax = f32[125,127]{1,0} fusion(auxiliary_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
  ROOT broadcast = f32[10,125,127]{2,1,0} broadcast(triton_softmax), dimensions={1,2}
})";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger;
  EXPECT_THAT(fusion_merger.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOk());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Fusion(m::Parameter()))));
}

TEST_F(FusionMergerTritonTest,
       CanMergeTritonFusionWithMultipleParameterProducer) {
  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

auxiliary_computation {
  parameter_0 = f32[125]{0} parameter(0)
  parameter_1 = f32[125,127]{1,0} parameter(1)
  broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(parameter_1, broadcast)
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125]{0} parameter(0)
  param_1 = f32[125,127]{1,0} parameter(1)
  auxiliary_fusion = f32[125,127]{1,0} fusion(param_0, param_1), kind=kLoop, calls=auxiliary_computation
  ROOT triton_softmax = f32[125,127]{1,0} fusion(auxiliary_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
})";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger;
  EXPECT_TRUE(fusion_merger.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(FusionMergerTritonTest, CanMergeTritonFusionWithTransposeProducer) {
  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

auxiliary_computation {
  parameter_0 = f32[125]{0} parameter(0)
  parameter_1 = f32[127,125]{1,0} parameter(1)
  transpose = f32[125,127]{1,0} transpose(parameter_1), dimensions={1,0}
  broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(transpose, broadcast)
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125]{0} parameter(0)
  param_1 = f32[127,125]{1,0} parameter(1)
  auxiliary_fusion = f32[125,127]{1,0} fusion(param_0, param_1), kind=kLoop, calls=auxiliary_computation
  ROOT triton_softmax = f32[125,127]{1,0} fusion(auxiliary_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
})";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger;
  EXPECT_TRUE(fusion_merger.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(FusionMergerTritonTest,
       DoesNotMergeTritonFusionWithProducerContainingUntileableOp) {
  // Right now, concatenate is not tileable.
  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

auxiliary_computation {
  parameter_0 = f32[125,63]{1,0} parameter(0)
  parameter_1 = f32[125,64]{1,0} parameter(1)
  ROOT concatenate = f32[125,127]{1,0} concatenate(parameter_0, parameter_1), dimensions={1}
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125,63]{1,0} parameter(0)
  param_1 = f32[125,64]{1,0} parameter(1)
  auxiliary_fusion = f32[125,127]{1,0} fusion(param_0, param_1), kind=kLoop, calls=auxiliary_computation
  ROOT triton_softmax = f32[125,127]{1,0} fusion(auxiliary_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
})";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger;
  EXPECT_FALSE(fusion_merger.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Fusion(m::Parameter(), m::Parameter()))));
}

TEST_F(FusionMergerTritonTest, CanMergeTritonFusionWithElementwiseProducer) {
  const std::string kHloText = R"(
HloModule layernorm

add_f32 {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add_6 = f32[] add(Arg_0, Arg_1)
}

auxiliary_fusion {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  parameter_1 = f32[125,127]{1,0} parameter(1)
  ROOT multiply_1 = f32[125,127]{1,0} multiply(parameter_0, parameter_1)
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  constant_0 = f32[] constant(0)
  reduce = f32[125]{0} reduce(parameter_0, constant_0), dimensions={1}, to_apply=add_f32
  broadcast = f32[125,127]{1,0} broadcast(reduce), dimensions={0}
  ROOT multiply_result = f32[125,127]{1,0} multiply(parameter_0, broadcast)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[125,127]{1,0} parameter(1)
  auxiliary_fusion = f32[125,127]{1,0} fusion(param_0, param_1), kind=kCustom, calls=auxiliary_fusion
  ROOT triton_softmax = f32[125,127]{1,0} fusion(auxiliary_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
}

)";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger;
  EXPECT_TRUE(fusion_merger.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(FusionMergerTritonTest,
       DoesNotMergeSoftmaxWithParamBroadcastedAlongBatchAndReduceDimensions) {
  const std::string kHloText = R"(
HloModule t

add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

auxiliary_computation {
  param_0 = f32[10,125,127]{2,1,0} parameter(0)
  param_1 = f32[10]{0} parameter(1)
  broadcast_0 = f32[10,125,127]{2,1,0} broadcast(param_1), dimensions={0}
  ROOT multiply_0 = f32[10,125,127]{2,1,0} multiply(param_0, broadcast_0)
}

triton_softmax_computation {
  param_0 = f32[10,125,127]{2,1,0} parameter(0)
  multiply = f32[10,125,127]{2,1,0} multiply(param_0, param_0)
  constant = f32[] constant(0)
  reduce = f32[10,125]{1,0} reduce(multiply, constant), dimensions={2}, to_apply=add
  broadcast = f32[10,125,127]{2,1,0} broadcast(reduce), dimensions={0,1}
  ROOT multiply_out = f32[10,125,127]{2,1,0} multiply(param_0, broadcast)
}

ENTRY main {
  param_0 = f32[10,125,127]{2,1,0} parameter(0)
  param_1 = f32[10]{0} parameter(1)
  auxiliary_fusion = f32[10,125,127]{2,1,0} fusion(param_0, param_1), kind=kCustom, calls=auxiliary_computation
  ROOT triton_softmax = f32[10,125,127]{2,1,0} fusion(auxiliary_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
}
)";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger;
  EXPECT_FALSE(fusion_merger.Run(module.get()).value());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Fusion())));
}

TEST_F(FusionMergerTritonTest, CanMergeWithBothProducerAndConsumerFusions) {
  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

producer_computation {
  parameter_0 = f32[125]{0} parameter(0)
  ROOT broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
}

consumer_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  parameter_1 = f32[125,127]{1,0} parameter(1)
  ROOT multiply = f32[125,127]{1,0} multiply(parameter_1, parameter_0)
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125]{0} parameter(0)
  param_1 = f32[125,127]{1,0} parameter(1)
  producer_fusion = f32[125,127]{1,0} fusion(param_0), kind=kLoop, calls=producer_computation
  triton_softmax = f32[125,127]{1,0} fusion(producer_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
  ROOT consumer_fusion = f32[125,127]{1,0} fusion(param_1, triton_softmax), kind=kLoop, calls=consumer_computation
})";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger{};
  EXPECT_TRUE(fusion_merger.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(FusionMergerTritonTest,
       CanMergeWithMultiInputProducerAndConsumerFusions) {
  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

producer_computation {
  parameter_0 = f32[125]{0} parameter(0)
  parameter_1 = f32[125,127]{1,0} parameter(1)
  broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
  ROOT add = f32[125,127]{1,0} add(parameter_1, broadcast)
}

consumer_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  parameter_1 = f32[125,127]{1,0} parameter(1)
  ROOT multiply = f32[125,127]{1,0} multiply(parameter_1, parameter_0)
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125]{0} parameter(0)
  param_1 = f32[125,127]{1,0} parameter(1)
  param_2 = f32[125,127]{1,0} parameter(2)
  producer_fusion = f32[125,127]{1,0} fusion(param_0, param_1), kind=kLoop, calls=producer_computation
  triton_softmax = f32[125,127]{1,0} fusion(producer_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
  ROOT consumer_fusion = f32[125,127]{1,0} fusion(param_2, triton_softmax), kind=kLoop, calls=consumer_computation
})";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger{};
  EXPECT_TRUE(fusion_merger.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter())));
}

TEST_F(FusionMergerTritonTest,
       CanMergeWithBothProducerAndConsumerFusionsSharingParameter) {
  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

producer_computation {
  parameter_0 = f32[125]{0} parameter(0)
  ROOT broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
}

consumer_computation {
  parameter_0 = f32[125]{0} parameter(0)
  parameter_1 = f32[125,127]{1,0} parameter(1)
  broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(parameter_1, broadcast)
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125]{0} parameter(0)
  producer_fusion = f32[125,127]{1,0} fusion(param_0), kind=kLoop, calls=producer_computation
  triton_softmax = f32[125,127]{1,0} fusion(producer_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
  ROOT consumer_fusion = f32[125,127]{1,0} fusion(param_0, triton_softmax), kind=kLoop, calls=consumer_computation
})";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger{};
  EXPECT_TRUE(fusion_merger.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(FusionMergerTritonTest, DoesNotMergeSoftmaxWithMultiOutputConsumer) {
  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

producer_computation {
  parameter_0 = f32[125]{0} parameter(0)
  ROOT broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
}

consumer_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  parameter_1 = f32[125,127]{1,0} parameter(1)
  add = f32[125,127]{1,0} add(parameter_1, parameter_0)
  multiply = f32[125,127]{1,0} multiply(parameter_1, parameter_0)
  ROOT tuple = (f32[125,127]{1,0}, f32[125,127]{1,0}) tuple(add, multiply)
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  triton_softmax = f32[125,127]{1,0} fusion(param_0), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
  ROOT consumer_fusion = (f32[125,127]{1,0}, f32[125,127]{1,0}) fusion(param_0, triton_softmax), kind=kLoop, calls=consumer_computation
})";
  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  FusionMergerTriton fusion_merger;
  EXPECT_FALSE(fusion_merger.Run(module.get()).value());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Fusion())));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
