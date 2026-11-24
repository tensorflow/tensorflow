/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/hlo/transforms/fusion/horizontal_fusion.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

class HorizontalFusionTest : public HloHardwareIndependentTestBase {
 protected:
  const se::DeviceDescription device_info_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo()};
};

TEST_F(HorizontalFusionTest, FusesConsecutiveIndependentLoopFusions) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_add {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  ROOT add = f32[1024]{0} add(p0, p1)
}

fused_mul {
  p0 = f32[512]{0} parameter(0)
  ROOT mul = f32[512]{0} multiply(p0, p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  p2 = f32[512]{0} parameter(2)
  f0 = f32[1024]{0} fusion(p0, p1), kind=kLoop, calls=fused_add
  f1 = f32[512]{0} fusion(p2), kind=kLoop, calls=fused_mul
  ROOT out = (f32[1024]{0}, f32[512]{0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: horizontal_fusion_computation
// CHECK-DAG: %[[P0:.*]] = f32[1024]{0} parameter(0)
// CHECK-DAG: %[[P1:.*]] = f32[1024]{0} parameter(1)
// CHECK-DAG: %[[P2:.*]] = f32[512]{0} parameter(2)
// CHECK-DAG: %[[NESTED0:.*]] = f32[1024]{0} fusion(%[[P0]], %[[P1]]), kind=kLoop, calls=%fused_add
// CHECK-DAG: %[[NESTED1:.*]] = f32[512]{0} fusion(%[[P2]]), kind=kLoop, calls=%fused_mul
// CHECK: ROOT %{{.*}} = (f32[1024]{0}, f32[512]{0}) tuple(%[[NESTED0]], %[[NESTED1]])
// CHECK: ENTRY %main
// CHECK: %[[HF:.*]] = (f32[1024]{0}, f32[512]{0}) fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation, backend_config={{.*}}"kind":"__thread_block_specialized"
// CHECK: ROOT %{{.*}} = (f32[1024]{0}, f32[512]{0}) tuple(%[[HF]]#0, %[[HF]]#1)
)"));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(HorizontalFusionTest, DeduplicatesSharedExternalOperands) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg
  f1 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_exp
  ROOT out = (f32[1024]{0}, f32[1024]{0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  const HloInstruction* gte0 = root->operand(0);
  ASSERT_EQ(gte0->opcode(), HloOpcode::kGetTupleElement);
  const HloInstruction* outer_fusion = gte0->operand(0);
  ASSERT_EQ(outer_fusion->opcode(), HloOpcode::kFusion);
  // Shared operand `p0` should be deduplicated to a single parameter.
  EXPECT_EQ(outer_fusion->operand_count(), 1);
}

TEST_F(HorizontalFusionTest, SkipsInterveningIndependentBitcast) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg
  bc = f32[32,32]{1,0} bitcast(f0)
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_exp
  ROOT out = (f32[32,32]{1,0}, f32[1024]{0}) tuple(bc, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HorizontalFusionTest, DoesNotFuseDependentFusionsThroughBitcast) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p0 = f32[32,32]{1,0} parameter(0)
  ROOT exp = f32[32,32]{1,0} exponential(p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg
  bc = f32[32,32]{1,0} bitcast(f0)
  ROOT f1 = f32[32,32]{1,0} fusion(bc), kind=kLoop, calls=fused_exp
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, DoesNotFuseControlDependentFusions) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_exp, control-predecessors={f0}
  ROOT out = (f32[1024]{0}, f32[1024]{0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, FlushesBatchAtNonSkippableInstruction) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg
  cc = f32[1024]{0} custom-call(p0), custom_call_target="foo"
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_exp
  ROOT out = (f32[1024]{0}, f32[1024]{0}, f32[1024]{0}) tuple(f0, cc, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, SkipsLargeGridFusionsExceedingSmThreshold) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_large {
  p0 = f32[16777216]{0} parameter(0)
  ROOT neg = f32[16777216]{0} negate(p0)
}

fused_small {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[16777216]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[16777216]{0} fusion(p0), kind=kLoop, calls=fused_large
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_small
  ROOT out = (f32[16777216]{0}, f32[1024]{0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, FusesLoopAndReductionWithWarpAlignedThreads) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

fused_loop {
  p0 = f32[512]{0} parameter(0)
  ROOT neg = f32[512]{0} negate(p0)
}

fused_reduce {
  p1 = f32[8,512]{1,0} parameter(0)
  c0 = f32[] constant(0)
  ROOT red = f32[8]{0} reduce(p1, c0), dimensions={1}, to_apply=add_f32
}

ENTRY main {
  p0 = f32[512]{0} parameter(0)
  p1 = f32[8,512]{1,0} parameter(1)
  f0 = f32[512]{0} fusion(p0), kind=kLoop, calls=fused_loop
  f1 = f32[8]{0} fusion(p1), kind=kInput, calls=fused_reduce
  ROOT out = (f32[512]{0}, f32[8]{0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HorizontalFusionTest, FusesMultiOutputNestedFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_mof {
  p0 = f32[1024]{0} parameter(0)
  n0 = f32[1024]{0} negate(p0)
  e0 = f32[1024]{0} exponential(p0)
  ROOT t = (f32[1024]{0}, f32[1024]{0}) tuple(n0, e0)
}

fused_single {
  p1 = f32[512]{0} parameter(0)
  ROOT c0 = f32[512]{0} cosine(p1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = (f32[1024]{0}, f32[1024]{0}) fusion(p0), kind=kLoop, calls=fused_mof
  gte0 = f32[1024]{0} get-tuple-element(f0), index=0
  gte1 = f32[1024]{0} get-tuple-element(f0), index=1
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_single
  ROOT out = (f32[1024]{0}, f32[1024]{0}, f32[512]{0}) tuple(gte0, gte1, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: horizontal_fusion_computation
// CHECK-DAG: %[[MOF:.*]] = (f32[1024]{0}, f32[1024]{0}) fusion({{.*}}), kind=kLoop, calls=%fused_mof
// CHECK-DAG: %[[SINGLE:.*]] = f32[512]{0} fusion({{.*}}), kind=kLoop, calls=%fused_single
// CHECK: ROOT %{{.*}} = (f32[1024]{0}, f32[1024]{0}, f32[512]{0}) tuple(%[[MOF]]#0, %[[MOF]]#1, %[[SINGLE]])
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest,
       FusesMultiOutputNestedFusionWithControlDependentGte) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_mof {
  p0 = f32[1024]{0} parameter(0)
  n0 = f32[1024]{0} negate(p0)
  e0 = f32[1024]{0} exponential(p0)
  ROOT t = (f32[1024]{0}, f32[1024]{0}) tuple(n0, e0)
}

fused_single {
  p1 = f32[512]{0} parameter(0)
  ROOT c0 = f32[512]{0} cosine(p1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = (f32[1024]{0}, f32[1024]{0}) fusion(p0), kind=kLoop, calls=fused_mof
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_single
  gte0 = f32[1024]{0} get-tuple-element(f0), index=0
  gte1 = f32[1024]{0} get-tuple-element(f0), index=1, control-predecessors={f1}
  ROOT out = (f32[1024]{0}, f32[1024]{0}, f32[512]{0}) tuple(gte0, gte1, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* gte1 = root->operand(1);
  ASSERT_EQ(gte1->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_THAT(gte1->control_predecessors(),
              ::testing::ElementsAre(gte1->operand(0)));
}

TEST_F(HorizontalFusionTest, FusesMultiOutputNestedFusionUsedAsWholeTuple) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_mof {
  p0 = f32[1024]{0} parameter(0)
  n0 = f32[1024]{0} negate(p0)
  e0 = f32[1024]{0} exponential(p0)
  ROOT t = (f32[1024]{0}, f32[1024]{0}) tuple(n0, e0)
}

fused_single {
  p1 = f32[512]{0} parameter(0)
  ROOT c0 = f32[512]{0} cosine(p1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = (f32[1024]{0}, f32[1024]{0}) fusion(p0), kind=kLoop, calls=fused_mof
  gte1 = f32[1024]{0} get-tuple-element(f0), index=1
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_single
  ROOT out = ((f32[1024]{0}, f32[1024]{0}), f32[1024]{0}, f32[512]{0}) tuple(f0, gte1, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: ENTRY %main
// CHECK: %[[HF:.*]] = (f32[1024]{0}, f32[1024]{0}, f32[512]{0}) fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation
// CHECK: %[[TUPLE:.*]] = (f32[1024]{0}, f32[1024]{0}) tuple(%[[HF]]#0, %[[HF]]#1)
// CHECK: ROOT %{{.*}} = ((f32[1024]{0}, f32[1024]{0}), f32[1024]{0}, f32[512]{0}) tuple(%[[TUPLE]], %[[HF]]#1, %[[HF]]#2)
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest, FusesLoopAndTritonFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_loop {
  p0 = f32[256]{0} parameter(0)
  ROOT neg = f32[256]{0} negate(p0)
}

fused_triton {
  p1 = f32[64,64]{1,0} parameter(0)
  ROOT exp = f32[64,64]{1,0} exponential(p1)
}

ENTRY main {
  p0 = f32[256]{0} parameter(0)
  p1 = f32[64,64]{1,0} parameter(1)
  f0 = f32[256]{0} fusion(p0), kind=kLoop, calls=fused_loop
  f1 = f32[64,64]{1,0} fusion(p1), kind=kCustom, calls=fused_triton,
    backend_config={"fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16","64"]}],
        "num_warps":"2",
        "num_ctas":1,
        "num_stages":1
      }
    }}
  ROOT out = (f32[256]{0}, f32[64,64]{1,0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: horizontal_fusion_computation
// CHECK-DAG: %[[LOOP:.*]] = f32[256]{0} fusion({{.*}}), kind=kLoop, calls=%fused_loop
// CHECK-DAG: %[[TRITON:.*]] = f32[64,64]{1,0} fusion({{.*}}), kind=kCustom, calls=%fused_triton{{.*}}, backend_config={{.*}}"kind":"__triton"
// CHECK: ROOT %{{.*}} = (f32[256]{0}, f32[64,64]{1,0}) tuple(%[[LOOP]], %[[TRITON]])
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest,
       DoesNotFuseTritonFusionWithoutBlockLevelFusionConfig) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_loop {
  p0 = f32[256]{0} parameter(0)
  ROOT neg = f32[256]{0} negate(p0)
}

fused_triton {
  p1 = f32[64,64]{1,0} parameter(0)
  ROOT exp = f32[64,64]{1,0} exponential(p1)
}

ENTRY main {
  p0 = f32[256]{0} parameter(0)
  p1 = f32[64,64]{1,0} parameter(1)
  f0 = f32[256]{0} fusion(p0), kind=kLoop, calls=fused_loop
  f1 = f32[64,64]{1,0} fusion(p1), kind=kCustom, calls=fused_triton,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
  ROOT out = (f32[256]{0}, f32[64,64]{1,0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, FusesPhasedBlockClusterAndLoopFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

phase0_comp {
  p0 = f32[512]{0} parameter(0)
  ROOT neg = f32[512]{0} negate(p0)
}

phase1_comp {
  p0 = f32[512]{0} parameter(0)
  ROOT rev = f32[512]{0} reverse(p0), dimensions={0}
}

phased_cluster {
  p0 = f32[512]{0} parameter(0)
  f0 = f32[512]{0} fusion(p0), kind=kLoop, calls=phase0_comp
  f1 = f32[512]{0} fusion(f0), kind=kLoop, calls=phase1_comp
  ROOT t = (f32[512]{0}, f32[512]{0}) tuple(f0, f1)
}

fused_loop {
  p1 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p1)
}

ENTRY main {
  p0 = f32[512]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  c0 = (f32[512]{0}, f32[512]{0}) fusion(p0), kind=kCustom,
    calls=phased_cluster,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
  c0_1 = f32[512]{0} get-tuple-element(c0), index=1
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_loop
  ROOT out = (f32[512]{0}, f32[1024]{0}) tuple(c0_1, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: horizontal_fusion_computation
// CHECK-DAG: %[[PHASED:.*]] = (f32[512]{0}, f32[512]{0}) fusion({{.*}}), kind=kCustom, calls=%phased_cluster{{.*}}, backend_config={{.*}}"kind":"__phased_block_cluster"
// CHECK-DAG: %[[LOOP:.*]] = f32[1024]{0} fusion({{.*}}), kind=kLoop, calls=%fused_loop
// CHECK: ROOT %{{.*}} = (f32[512]{0}, f32[512]{0}, f32[1024]{0}) tuple(%[[PHASED]]#0, %[[PHASED]]#1, %[[LOOP]])
// CHECK: ENTRY %main
// CHECK: %[[HF:.*]] = (f32[512]{0}, f32[512]{0}, f32[1024]{0}) fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation, backend_config={{.*}}"kind":"__thread_block_specialized"
// CHECK: ROOT %{{.*}} = (f32[512]{0}, f32[1024]{0}) tuple(%[[HF]]#1, %[[HF]]#2)
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest,
       DoesNotFuseTritonFusionWithTmaOrWarpSpecialization) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_loop {
  p0 = f32[256]{0} parameter(0)
  ROOT neg = f32[256]{0} negate(p0)
}

fused_triton_tma {
  p1 = f32[64,64]{1,0} parameter(0)
  ROOT exp = f32[64,64]{1,0} exponential(p1)
}

fused_triton_ws {
  p2 = f32[64,64]{1,0} parameter(0)
  ROOT tanh = f32[64,64]{1,0} tanh(p2)
}

ENTRY main {
  p0 = f32[256]{0} parameter(0)
  p1 = f32[64,64]{1,0} parameter(1)
  p2 = f32[64,64]{1,0} parameter(2)
  f0 = f32[256]{0} fusion(p0), kind=kLoop, calls=fused_loop
  f1 = f32[64,64]{1,0} fusion(p1), kind=kCustom, calls=fused_triton_tma,
    backend_config={"fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16","64"]}],
        "num_warps":"2",
        "num_ctas":1,
        "num_stages":1,
        "is_tma_allowed":true
      }
    }}
  f2 = f32[64,64]{1,0} fusion(p2), kind=kCustom, calls=fused_triton_ws,
    backend_config={"fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16","64"]}],
        "num_warps":"2",
        "num_ctas":1,
        "num_stages":1,
        "is_warp_specialization_allowed":true
      }
    }}
  ROOT out = (f32[256]{0}, f32[64,64]{1,0}, f32[64,64]{1,0}) tuple(f0, f1, f2)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest,
       FusesNestedTupleMultiOutputFusionAndRewritesGteChains) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

argmax {
  lhs_val = f32[] parameter(0)
  lhs_idx = s32[] parameter(1)
  rhs_val = f32[] parameter(2)
  rhs_idx = s32[] parameter(3)
  cmp = pred[] compare(lhs_val, rhs_val), direction=GE
  max_val = f32[] select(cmp, lhs_val, rhs_val)
  max_idx = s32[] select(cmp, lhs_idx, rhs_idx)
  ROOT out = (f32[], s32[]) tuple(max_val, max_idx)
}

nested_mof {
  p0 = f32[16,64]{1,0} parameter(0)
  p1 = f32[16,64]{1,0} parameter(1)
  iota = s32[16,64]{1,0} iota(), iota_dimension=1
  neg_inf = f32[] constant(-inf)
  zero = s32[] constant(0)
  sum0 = f32[16,64]{1,0} add(p0, p1)
  red0 = (f32[16]{0}, s32[16]{0}) reduce(sum0, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  diff1 = f32[16,64]{1,0} subtract(p0, p1)
  red1 = (f32[16]{0}, s32[16]{0}) reduce(diff1, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  ROOT t = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      tuple(red0, red1)
}

fused_loop {
  p2 = f32[512]{0} parameter(0)
  ROOT exp = f32[512]{0} exponential(p2)
}

ENTRY main {
  p0 = f32[16,64]{1,0} parameter(0)
  p1 = f32[16,64]{1,0} parameter(1)
  p2 = f32[512]{0} parameter(2)
  f0 = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      fusion(p0, p1), kind=kInput, calls=nested_mof
  r0 = (f32[16]{0}, s32[16]{0}) get-tuple-element(f0), index=0
  idx0 = s32[16]{0} get-tuple-element(r0), index=1
  r1 = (f32[16]{0}, s32[16]{0}) get-tuple-element(f0), index=1
  idx1 = s32[16]{0} get-tuple-element(r1), index=1
  f1 = f32[512]{0} fusion(p2), kind=kLoop, calls=fused_loop
  ROOT out = (s32[16]{0}, s32[16]{0}, f32[512]{0}) tuple(idx0, idx1, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: horizontal_fusion_computation
// CHECK-DAG: %[[MOF:.*]] = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0})) fusion({{.*}}), kind=kInput, calls=%nested_mof
// CHECK-DAG: %[[LOOP:.*]] = f32[512]{0} fusion({{.*}}), kind=kLoop, calls=%fused_loop
// CHECK: ROOT %{{.*}} = (f32[16]{0}, s32[16]{0}, f32[16]{0}, s32[16]{0}, f32[512]{0}) tuple(%[[MOF]]#0#0, %[[MOF]]#0#1, %[[MOF]]#1#0, %[[MOF]]#1#1, %[[LOOP]])
// CHECK: ENTRY %main
// CHECK: %[[HF:.*]] = (f32[16]{0}, s32[16]{0}, f32[16]{0}, s32[16]{0}, f32[512]{0}) fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation, backend_config={{.*}}"kind":"__thread_block_specialized"
// CHECK: ROOT %{{.*}} = (s32[16]{0}, s32[16]{0}, f32[512]{0}) tuple(%[[HF]]#1, %[[HF]]#3, %[[HF]]#4)
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest, ReconstructsNestedTupleWhenUsedDirectly) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

argmax {
  lhs_val = f32[] parameter(0)
  lhs_idx = s32[] parameter(1)
  rhs_val = f32[] parameter(2)
  rhs_idx = s32[] parameter(3)
  cmp = pred[] compare(lhs_val, rhs_val), direction=GE
  max_val = f32[] select(cmp, lhs_val, rhs_val)
  max_idx = s32[] select(cmp, lhs_idx, rhs_idx)
  ROOT out = (f32[], s32[]) tuple(max_val, max_idx)
}

nested_mof {
  p0 = f32[16,64]{1,0} parameter(0)
  iota = s32[16,64]{1,0} iota(), iota_dimension=1
  neg_inf = f32[] constant(-inf)
  zero = s32[] constant(0)
  red0 = (f32[16]{0}, s32[16]{0}) reduce(p0, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  neg = f32[16,64]{1,0} negate(p0)
  red1 = (f32[16]{0}, s32[16]{0}) reduce(neg, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  ROOT t = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      tuple(red0, red1)
}

fused_loop {
  p1 = f32[512]{0} parameter(0)
  ROOT exp = f32[512]{0} exponential(p1)
}

ENTRY main {
  p0 = f32[16,64]{1,0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_loop
  f1 = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      fusion(p0), kind=kInput, calls=nested_mof
  r1 = (f32[16]{0}, s32[16]{0}) get-tuple-element(f1), index=1
  ROOT out = (f32[512]{0}, ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0})),
              (f32[16]{0}, s32[16]{0})) tuple(f0, f1, r1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: ENTRY %main
// CHECK: %[[HF:.*]] = (f32[512]{0}, f32[16]{0}, s32[16]{0}, f32[16]{0}, s32[16]{0}) fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation
// CHECK-DAG: %[[R1_TUPLE:.*]] = (f32[16]{0}, s32[16]{0}) tuple(%{{.*}}, %{{.*}})
// CHECK-DAG: %[[F1_TUPLE:.*]] = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0})) tuple(%{{.*}}, %{{.*}})
// CHECK: ROOT %{{.*}} = (f32[512]{0}, ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0})), (f32[16]{0}, s32[16]{0})) tuple(%{{.*}}, %[[F1_TUPLE]], %[[R1_TUPLE]])
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest, PreservesControlDependenciesOnNestedTupleGte) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule repro, is_scheduled=true

argmax {
  lhs_val = f32[] parameter(0)
  lhs_idx = s32[] parameter(1)
  rhs_val = f32[] parameter(2)
  rhs_idx = s32[] parameter(3)
  cmp = pred[] compare(lhs_val, rhs_val), direction=GE
  max_val = f32[] select(cmp, lhs_val, rhs_val)
  max_idx = s32[] select(cmp, lhs_idx, rhs_idx)
  ROOT out = (f32[], s32[]) tuple(max_val, max_idx)
}

nested_mof {
  p0 = f32[16,64]{1,0} parameter(0)
  iota = s32[16,64]{1,0} iota(), iota_dimension=1
  neg_inf = f32[] constant(-inf)
  zero = s32[] constant(0)
  red0 = (f32[16]{0}, s32[16]{0}) reduce(p0, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  neg = f32[16,64]{1,0} negate(p0)
  red1 = (f32[16]{0}, s32[16]{0}) reduce(neg, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  ROOT t = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      tuple(red0, red1)
}

fused_loop {
  p1 = f32[512]{0} parameter(0)
  ROOT exp = f32[512]{0} exponential(p1)
}

ENTRY main {
  p0 = f32[16,64]{1,0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      fusion(p0), kind=kInput, calls=nested_mof
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_loop
  r0 = (f32[16]{0}, s32[16]{0}) get-tuple-element(f0), index=0
  idx0 = s32[16]{0} get-tuple-element(r0), index=1
  cc = f32[512]{0} custom-call(p1), custom_call_target="bar",
      control-predecessors={r0}
  ROOT out = (s32[16]{0}, f32[512]{0}, f32[512]{0}) tuple(idx0, f1, cc)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* idx0 = root->operand(0);
  ASSERT_EQ(idx0->opcode(), HloOpcode::kGetTupleElement);
  const HloInstruction* outer_fusion = idx0->operand(0);
  ASSERT_EQ(outer_fusion->opcode(), HloOpcode::kFusion);
  const HloInstruction* cc = root->operand(2);
  EXPECT_THAT(cc->control_predecessors(), ::testing::ElementsAre(outer_fusion));
}

TEST_F(HorizontalFusionTest, PreservesControlPredecessorOnNestedTupleGte) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, is_scheduled=true

argmax {
  lhs_val = f32[] parameter(0)
  lhs_idx = s32[] parameter(1)
  rhs_val = f32[] parameter(2)
  rhs_idx = s32[] parameter(3)
  cmp = pred[] compare(lhs_val, rhs_val), direction=GE
  max_val = f32[] select(cmp, lhs_val, rhs_val)
  max_idx = s32[] select(cmp, lhs_idx, rhs_idx)
  ROOT out = (f32[], s32[]) tuple(max_val, max_idx)
}

nested_mof {
  p0 = f32[16,64]{1,0} parameter(0)
  iota = s32[16,64]{1,0} iota(), iota_dimension=1
  neg_inf = f32[] constant(-inf)
  zero = s32[] constant(0)
  red0 = (f32[16]{0}, s32[16]{0}) reduce(p0, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  neg = f32[16,64]{1,0} negate(p0)
  red1 = (f32[16]{0}, s32[16]{0}) reduce(neg, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  ROOT t = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      tuple(red0, red1)
}

fused_loop {
  p1 = f32[512]{0} parameter(0)
  ROOT exp = f32[512]{0} exponential(p1)
}

ENTRY main {
  p0 = f32[16,64]{1,0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      fusion(p0), kind=kInput, calls=nested_mof
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_loop
  cc = f32[512]{0} custom-call(f1), custom_call_target="bar"
  r0 = (f32[16]{0}, s32[16]{0}) get-tuple-element(f0), index=0,
      control-predecessors={cc}
  idx0 = s32[16]{0} get-tuple-element(r0), index=1
  ROOT out = (s32[16]{0}, f32[512]{0}) tuple(idx0, cc)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* idx0 = root->operand(0);
  const HloInstruction* cc = root->operand(1);
  ASSERT_EQ(idx0->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_THAT(idx0->control_predecessors(), ::testing::ElementsAre(cc));
}

TEST_F(HorizontalFusionTest,
       PreservesTwoSidedControlDependenciesOnRemovedNestedTupleGte) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, is_scheduled=true

argmax {
  lhs_val = f32[] parameter(0)
  lhs_idx = s32[] parameter(1)
  rhs_val = f32[] parameter(2)
  rhs_idx = s32[] parameter(3)
  cmp = pred[] compare(lhs_val, rhs_val), direction=GE
  max_val = f32[] select(cmp, lhs_val, rhs_val)
  max_idx = s32[] select(cmp, lhs_idx, rhs_idx)
  ROOT out = (f32[], s32[]) tuple(max_val, max_idx)
}

nested_mof {
  p0 = f32[16,64]{1,0} parameter(0)
  iota = s32[16,64]{1,0} iota(), iota_dimension=1
  neg_inf = f32[] constant(-inf)
  zero = s32[] constant(0)
  red0 = (f32[16]{0}, s32[16]{0}) reduce(p0, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  neg = f32[16,64]{1,0} negate(p0)
  red1 = (f32[16]{0}, s32[16]{0}) reduce(neg, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  ROOT t = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      tuple(red0, red1)
}

fused_loop {
  p1 = f32[512]{0} parameter(0)
  ROOT exp = f32[512]{0} exponential(p1)
}

ENTRY main {
  p0 = f32[16,64]{1,0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      fusion(p0), kind=kInput, calls=nested_mof
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_loop
  cc1 = f32[512]{0} custom-call(f1), custom_call_target="bar1"
  r0 = (f32[16]{0}, s32[16]{0}) get-tuple-element(f0), index=0,
      control-predecessors={cc1}
  idx0 = s32[16]{0} get-tuple-element(r0), index=1
  cc2 = f32[512]{0} custom-call(p1), custom_call_target="bar2",
      control-predecessors={r0}
  ROOT out = (s32[16]{0}, f32[512]{0}, f32[512]{0}) tuple(idx0, cc1, cc2)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* idx0 = root->operand(0);
  const HloInstruction* cc1 = root->operand(1);
  const HloInstruction* cc2 = root->operand(2);
  ASSERT_EQ(idx0->opcode(), HloOpcode::kGetTupleElement);
  const HloInstruction* outer_fusion = idx0->operand(0);
  ASSERT_EQ(outer_fusion->opcode(), HloOpcode::kFusion);
  EXPECT_THAT(idx0->control_predecessors(), ::testing::ElementsAre(cc1));
  EXPECT_THAT(cc2->control_predecessors(),
              ::testing::UnorderedElementsAre(outer_fusion, cc1));
}

TEST_F(HorizontalFusionTest,
       PreservesControlDependenciesOnReconstructedNestedTupleGte) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, is_scheduled=true

argmax {
  lhs_val = f32[] parameter(0)
  lhs_idx = s32[] parameter(1)
  rhs_val = f32[] parameter(2)
  rhs_idx = s32[] parameter(3)
  cmp = pred[] compare(lhs_val, rhs_val), direction=GE
  max_val = f32[] select(cmp, lhs_val, rhs_val)
  max_idx = s32[] select(cmp, lhs_idx, rhs_idx)
  ROOT out = (f32[], s32[]) tuple(max_val, max_idx)
}

nested_mof {
  p0 = f32[16,64]{1,0} parameter(0)
  iota = s32[16,64]{1,0} iota(), iota_dimension=1
  neg_inf = f32[] constant(-inf)
  zero = s32[] constant(0)
  red0 = (f32[16]{0}, s32[16]{0}) reduce(p0, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  neg = f32[16,64]{1,0} negate(p0)
  red1 = (f32[16]{0}, s32[16]{0}) reduce(neg, iota, neg_inf, zero),
      dimensions={1}, to_apply=argmax
  ROOT t = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      tuple(red0, red1)
}

fused_loop {
  p1 = f32[512]{0} parameter(0)
  ROOT exp = f32[512]{0} exponential(p1)
}

ENTRY main {
  p0 = f32[16,64]{1,0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = ((f32[16]{0}, s32[16]{0}), (f32[16]{0}, s32[16]{0}))
      fusion(p0), kind=kInput, calls=nested_mof
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_loop
  cc1 = f32[512]{0} custom-call(f1), custom_call_target="bar1"
  r0 = (f32[16]{0}, s32[16]{0}) get-tuple-element(f0), index=0,
      control-predecessors={cc1}
  idx0 = s32[16]{0} get-tuple-element(r0), index=1
  cc2 = f32[512]{0} custom-call(p1), custom_call_target="bar2",
      control-predecessors={r0}
  ROOT out = ((f32[16]{0}, s32[16]{0}), s32[16]{0}, f32[512]{0}, f32[512]{0})
      tuple(r0, idx0, cc1, cc2)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* new_tuple = root->operand(0);
  const HloInstruction* idx0 = root->operand(1);
  const HloInstruction* cc1 = root->operand(2);
  const HloInstruction* cc2 = root->operand(3);
  ASSERT_EQ(new_tuple->opcode(), HloOpcode::kTuple);
  ASSERT_EQ(idx0->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_THAT(new_tuple->control_predecessors(), ::testing::ElementsAre(cc1));
  EXPECT_THAT(idx0->control_predecessors(), ::testing::ElementsAre(cc1));
  EXPECT_THAT(cc2->control_predecessors(), ::testing::ElementsAre(new_tuple));
}

}  // namespace
}  // namespace xla::gpu
