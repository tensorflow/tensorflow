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

#include "xla/backends/gpu/transforms/multi_phase_fusion.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;
using ::absl_testing::IsOkAndHolds;

bool IsPhasedBlockClusterFusion(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kFusion &&
         IsGpuFusionKind(*instr, kPhasedBlockClusterFusionKind);
}

class MultiPhaseFusionTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription device_info_ = TestGpuDeviceInfo::H100SXMDeviceInfo();
};

TEST_F(MultiPhaseFusionTest, FusesStraightLineChainOfSmallLoopFusions) {
  constexpr absl::string_view kHlo = R"hlo(
phase_0 { ROOT neg = f32[512] negate(f32[512] parameter(0)) }
phase_1 {
  bc = f32[2,512] broadcast(f32[512] parameter(0)), dimensions={1}
  rev = f32[1024] reverse(f32[1024] reshape(bc)), dimensions={0}
  ROOT add = f32[1024] add(rev, f32[1024] parameter(1))
}
phase_2 { ROOT exp = f32[1024] exponential(f32[1024] parameter(0)) }

ENTRY main {
  p0 = f32[512] parameter(0)
  p1 = f32[1024] parameter(1)
  f0 = f32[512] fusion(p0), kind=kLoop, calls=phase_0
  f1 = f32[1024] fusion(f0, p1), kind=kLoop, calls=phase_1
  ROOT f2 = f32[1024] fusion(f1), kind=kLoop, calls=phase_2
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  const HloInstruction* multi_phase_fusion = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::Fusion(&multi_phase_fusion, m::Parameter(0), m::Parameter(1))
              .WithPredicate(IsPhasedBlockClusterFusion)
              .WithOneUse(),
          2)));

  auto f0_nested = m::Fusion(m::Parameter(0))
                       .WithFusionKind(HloInstruction::FusionKind::kLoop);
  auto f1_nested = m::Fusion(f0_nested, m::Parameter(1))
                       .WithFusionKind(HloInstruction::FusionKind::kLoop);
  auto f2_nested =
      m::Fusion(f1_nested).WithFusionKind(HloInstruction::FusionKind::kLoop);
  EXPECT_THAT(multi_phase_fusion->fused_expression_root(),
              GmockMatch(m::Tuple(f0_nested, f1_nested, f2_nested)));
}

TEST_F(MultiPhaseFusionTest,
       PreservesIntermediateOutputsAndDeduplicatesOperands) {
  constexpr absl::string_view kHlo = R"hlo(
phase_0 { ROOT neg = f32[256] negate(f32[256] parameter(0)) }
phase_1 {
  ROOT add = f32[256] add(f32[256] parameter(0), f32[256] parameter(1))
}

ENTRY main {
  p0 = f32[256] parameter(0)
  f0 = f32[256] fusion(p0), kind=kLoop, calls=phase_0
  ROOT f1 = f32[256] fusion(f0, p0), kind=kLoop, calls=phase_1
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  const HloInstruction* multi_phase_fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::Fusion(&multi_phase_fusion, m::Parameter(0))
                      .WithPredicate(IsPhasedBlockClusterFusion)
                      .WithOneUse(),
                  1)));

  auto f0_nested = m::Fusion(m::Parameter(0))
                       .WithFusionKind(HloInstruction::FusionKind::kLoop);
  auto f1_nested = m::Fusion(f0_nested, m::Parameter(0))
                       .WithFusionKind(HloInstruction::FusionKind::kLoop);
  EXPECT_THAT(multi_phase_fusion->fused_expression_root(),
              GmockMatch(m::Tuple(f0_nested, f1_nested)));
}

TEST_F(MultiPhaseFusionTest, SplitsChainAtLargeFusion) {
  // `large` requires > 8 CTAs (`f32[1048576]`), so `f0 -> large` cannot fuse,
  // while the small downstream chain `f2 -> f3` does fuse.
  constexpr absl::string_view kHlo = R"hlo(
small_0 { ROOT neg = f32[512] negate(f32[512] parameter(0)) }
large_comp {
  bc = f32[2048,512] broadcast(f32[512] parameter(0)), dimensions={1}
  ROOT exp = f32[1048576] exponential(f32[1048576] reshape(bc))
}
small_2 {
  ROOT sl = f32[512] slice(f32[1048576] parameter(0)), slice={[0:512]}
}
small_3 { ROOT exp = f32[512] exponential(f32[512] parameter(0)) }

ENTRY main {
  p0 = f32[512] parameter(0)
  f0 = f32[512] fusion(p0), kind=kLoop, calls=small_0
  large = f32[1048576] fusion(f0), kind=kLoop, calls=large_comp
  f2 = f32[512] fusion(large), kind=kLoop, calls=small_2
  ROOT f3 = f32[512] fusion(f2), kind=kLoop, calls=small_3
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  auto f0 = m::Fusion(m::Parameter(0))
                .WithFusionKind(HloInstruction::FusionKind::kLoop);
  auto large = m::Fusion(f0).WithFusionKind(HloInstruction::FusionKind::kLoop);
  const HloInstruction* multi_phase_fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::Fusion(&multi_phase_fusion, large)
                      .WithPredicate(IsPhasedBlockClusterFusion)
                      .WithOneUse(),
                  1)));

  auto f2_nested = m::Fusion(m::Parameter(0))
                       .WithFusionKind(HloInstruction::FusionKind::kLoop);
  auto f3_nested =
      m::Fusion(f2_nested).WithFusionKind(HloInstruction::FusionKind::kLoop);
  EXPECT_THAT(multi_phase_fusion->fused_expression_root(),
              GmockMatch(m::Tuple(f2_nested, f3_nested)));
}

TEST_F(MultiPhaseFusionTest, DoesNotFuseWhenCycleWouldBeFormed) {
  // `f0 -> custom_call -> f1` where `f1` also consumes `f0`: fusing `{f0, f1}`
  // would form a cycle with `custom_call`.
  constexpr absl::string_view kHlo = R"hlo(
comp_0 { ROOT neg = f32[256] negate(f32[256] parameter(0)) }
comp_1 { ROOT add = f32[256] add(f32[256] parameter(0), f32[256] parameter(1)) }

ENTRY main {
  p0 = f32[256] parameter(0)
  f0 = f32[256] fusion(p0), kind=kLoop, calls=comp_0
  custom_call = f32[256] custom-call(f0), custom_call_target="dummy"
  ROOT f1 = f32[256] fusion(f0, custom_call), kind=kLoop, calls=comp_1
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(false));
}

TEST_F(MultiPhaseFusionTest, StopsChainAtSkipPhaseDependency) {
  // `f2` skip-reads `f0` in addition to `f1`, so `{f0, f1}` fuses while `f2`
  // remains outside.
  constexpr absl::string_view kHlo = R"hlo(
comp_0 { ROOT neg = f32[256] negate(f32[256] parameter(0)) }
comp_1 { ROOT exp = f32[256] exponential(f32[256] parameter(0)) }
comp_2 { ROOT add = f32[256] add(f32[256] parameter(0), f32[256] parameter(1)) }

ENTRY main {
  p0 = f32[256] parameter(0)
  f0 = f32[256] fusion(p0), kind=kLoop, calls=comp_0
  f1 = f32[256] fusion(f0), kind=kLoop, calls=comp_1
  ROOT f2 = f32[256] fusion(f1, f0), kind=kLoop, calls=comp_2
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  const HloInstruction* multi_phase_fusion_0 = nullptr;
  const HloInstruction* multi_phase_fusion_1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Fusion(m::GetTupleElement(
                        m::Fusion(&multi_phase_fusion_1, m::Parameter(0))
                            .WithPredicate(IsPhasedBlockClusterFusion),
                        1),
                    m::GetTupleElement(m::Fusion(&multi_phase_fusion_0), 0))
              .WithFusionKind(HloInstruction::FusionKind::kLoop)));
  EXPECT_EQ(multi_phase_fusion_0, multi_phase_fusion_1);
}

TEST_F(MultiPhaseFusionTest, FusesUnfusedOpsAndDeduplicatesInnerOperands) {
  constexpr absl::string_view kHlo = R"hlo(
phase_1 {
  bc = f32[2,512] broadcast(f32[512] parameter(0)), dimensions={1}
  rev = f32[1024] reverse(f32[1024] reshape(bc)), dimensions={0}
  ROOT add = f32[1024] add(rev, f32[1024] parameter(1))
}

ENTRY main {
  p0 = f32[512] parameter(0)
  p1 = f32[1024] parameter(1)
  neg = f32[512] negate(p0)
  f1 = f32[1024] fusion(neg, p1), kind=kLoop, calls=phase_1
  ROOT sq = f32[1024] multiply(f1, f1)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  const HloInstruction* multi_phase_fusion = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::Fusion(&multi_phase_fusion, m::Parameter(0), m::Parameter(1))
              .WithPredicate(IsPhasedBlockClusterFusion)
              .WithOneUse(),
          2)));

  const HloInstruction* sq_nested = nullptr;
  auto neg_nested = m::Fusion(m::Parameter(0))
                        .WithFusionKind(HloInstruction::FusionKind::kLoop);
  auto f1_nested = m::Fusion(neg_nested, m::Parameter(1))
                       .WithFusionKind(HloInstruction::FusionKind::kLoop);
  ASSERT_THAT(multi_phase_fusion->fused_expression_root(),
              GmockMatch(m::Tuple(
                  neg_nested, f1_nested,
                  m::Fusion(&sq_nested, f1_nested)
                      .WithFusionKind(HloInstruction::FusionKind::kLoop))));
  EXPECT_THAT(sq_nested->fused_expression_root(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(0))));
}

TEST_F(MultiPhaseFusionTest, DoesNotFuseInsideCalledReducerComputation) {
  constexpr absl::string_view kHlo = R"hlo(
reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  neg = f32[] negate(lhs)
  exp = f32[] exponential(neg)
  ROOT sum = f32[] add(exp, rhs)
}

ENTRY main {
  p0 = f32[32,128] parameter(0)
  init = f32[] constant(0.0)
  ROOT red = f32[32] reduce(p0, init), dimensions={1}, to_apply=reducer
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(false));
}

TEST_F(MultiPhaseFusionTest,
       VectorizesSmallF32LoopStagesOnHopperAndBlackwellOnly) {
  // Without small-input vectorization, `f32[4096]` requires 32 blocks,
  // exceeding the 8-block cluster limit on pre-Hopper (`A100SXMDeviceInfo`). On
  // Hopper (`H100SXMDeviceInfo`) and Blackwell (`B200SXMDeviceInfo`), 4x
  // unrolling shrinks `f32[4096]` to 8 blocks.
  constexpr absl::string_view kF32ChainHlo = R"hlo(
phase_0 { ROOT neg = f32[4096] negate(f32[4096] parameter(0)) }
phase_1 { ROOT exp = f32[4096] exponential(f32[4096] parameter(0)) }

ENTRY main {
  p0 = f32[4096] parameter(0)
  f0 = f32[4096] fusion(p0), kind=kLoop, calls=phase_0
  ROOT f1 = f32[4096] fusion(f0), kind=kLoop, calls=phase_1
}
)hlo";
  MultiPhaseFusion a100_pass(TestGpuDeviceInfo::A100SXMDeviceInfo());
  MultiPhaseFusion h100_pass(TestGpuDeviceInfo::H100SXMDeviceInfo());
  MultiPhaseFusion b200_pass(TestGpuDeviceInfo::B200SXMDeviceInfo());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> f32_a100_module,
                       ParseAndReturnVerifiedModule(kF32ChainHlo));
  EXPECT_THAT(RunHloPass(&a100_pass, f32_a100_module.get()),
              IsOkAndHolds(false));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> f32_h100_module,
                       ParseAndReturnVerifiedModule(kF32ChainHlo));
  EXPECT_THAT(RunHloPass(&h100_pass, f32_h100_module.get()),
              IsOkAndHolds(true));
  EXPECT_THAT(
      f32_h100_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::Fusion().WithPredicate(IsPhasedBlockClusterFusion).WithOneUse(),
          1)));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> f32_b200_module,
                       ParseAndReturnVerifiedModule(kF32ChainHlo));
  EXPECT_THAT(RunHloPass(&b200_pass, f32_b200_module.get()),
              IsOkAndHolds(true));
  EXPECT_THAT(
      f32_b200_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::Fusion().WithPredicate(IsPhasedBlockClusterFusion).WithOneUse(),
          1)));
}

TEST_F(MultiPhaseFusionTest, VectorizesSmallS8LoopStagesOnBlackwellOnly) {
  // On Blackwell (`B200SXMDeviceInfo`), narrow-type unrolling
  // (`unroll_factor = 16`) shrinks `s8[16384]` to 8 blocks, whereas Hopper
  // (`H100SXMDeviceInfo`, `unroll_factor = 4`) still requires 32 blocks.
  constexpr absl::string_view kS8ChainHlo = R"hlo(
phase_0 { ROOT neg = s8[16384] negate(s8[16384] parameter(0)) }
phase_1 { ROOT add = s8[16384] add(s8[16384] parameter(0), s8[16384] parameter(1)) }

ENTRY main {
  p0 = s8[16384] parameter(0)
  f0 = s8[16384] fusion(p0), kind=kLoop, calls=phase_0
  ROOT f1 = s8[16384] fusion(f0, p0), kind=kLoop, calls=phase_1
}
)hlo";
  MultiPhaseFusion h100_pass(TestGpuDeviceInfo::H100SXMDeviceInfo());
  MultiPhaseFusion b200_pass(TestGpuDeviceInfo::B200SXMDeviceInfo());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> s8_h100_module,
                       ParseAndReturnVerifiedModule(kS8ChainHlo));
  EXPECT_THAT(RunHloPass(&h100_pass, s8_h100_module.get()),
              IsOkAndHolds(false));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> s8_b200_module,
                       ParseAndReturnVerifiedModule(kS8ChainHlo));
  EXPECT_THAT(RunHloPass(&b200_pass, s8_b200_module.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      s8_b200_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::Fusion().WithPredicate(IsPhasedBlockClusterFusion).WithOneUse(),
          1)));
}

}  // namespace
}  // namespace xla::gpu
