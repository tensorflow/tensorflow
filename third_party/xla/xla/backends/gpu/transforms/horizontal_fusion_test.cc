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

#include "xla/backends/gpu/transforms/horizontal_fusion.h"

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
// CHECK: %[[HF:.*]] = (f32[1024]{0}, f32[512]{0}) fusion({{.*}}), kind=kLoop, calls=%horizontal_fusion_computation
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

}  // namespace
}  // namespace xla::gpu
