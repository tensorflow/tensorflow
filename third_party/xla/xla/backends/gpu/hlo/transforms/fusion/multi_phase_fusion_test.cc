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

#include "xla/backends/gpu/hlo/transforms/fusion/multi_phase_fusion.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

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

TEST_F(MultiPhaseFusionTest, FusesChainWithSkipPhaseDependency) {
  // `f2` consumes `f1` and also skip-reads `f0`. All 3 stages fuse into a
  // single 3-phase `__phased_block_cluster` fusion, even when `f2` precedes
  // `f1` in `f0`'s user list.
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

  const HloInstruction* multi_phase_fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::Fusion(&multi_phase_fusion, m::Parameter(0))
                      .WithPredicate(IsPhasedBlockClusterFusion)
                      .WithOneUse(),
                  2)));

  const HloInstruction* f0_nested = nullptr;
  const HloInstruction* f1_nested = nullptr;
  EXPECT_THAT(multi_phase_fusion->fused_expression_root(),
              GmockMatch(m::Tuple(
                  m::Fusion(&f0_nested, m::Parameter(0))
                      .WithFusionKind(HloInstruction::FusionKind::kLoop),
                  m::Fusion(&f1_nested, m::Fusion())
                      .WithFusionKind(HloInstruction::FusionKind::kLoop),
                  m::Fusion(m::Fusion(), m::Fusion())
                      .WithFusionKind(HloInstruction::FusionKind::kLoop))));
  const HloInstruction* f2_nested =
      multi_phase_fusion->fused_expression_root()->operand(2);
  EXPECT_EQ(f2_nested->operand(0), f1_nested);
  EXPECT_EQ(f2_nested->operand(1), f0_nested);
}

TEST_F(MultiPhaseFusionTest, FusesTupleSkipPhaseDependency) {
  // `f0` produces a 2-tuple `(gte0, gte1)`; `f1` consumes `gte0`, and `f2`
  // consumes `f1` together with a skip-phase read of `gte1`.
  constexpr absl::string_view kHlo = R"hlo(
mof_0 {
  p0 = f32[256]{0} parameter(0)
  ROOT t = (f32[256]{0}, f32[256]{0}) tuple(f32[256]{0} negate(p0), f32[256]{0} cosine(p0))
}
phase_1 { ROOT exp = f32[256]{0} exponential(f32[256]{0} parameter(0)) }
phase_2 { ROOT add = f32[256]{0} add(f32[256]{0} parameter(0), f32[256]{0} parameter(1)) }

ENTRY main {
  p0 = f32[256]{0} parameter(0)
  f0 = (f32[256]{0}, f32[256]{0}) fusion(p0), kind=kLoop, calls=mof_0
  gte0 = f32[256]{0} get-tuple-element(f0), index=0
  gte1 = f32[256]{0} get-tuple-element(f0), index=1
  f1 = f32[256]{0} fusion(gte0), kind=kLoop, calls=phase_1
  ROOT f2 = f32[256]{0} fusion(f1, gte1), kind=kLoop, calls=phase_2
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
                      .WithPredicate(IsPhasedBlockClusterFusion),
                  3)));
  EXPECT_EQ(multi_phase_fusion->fused_expression_root()->operand_count(), 4);
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

TEST_F(MultiPhaseFusionTest, FusesInputTransposeReductionAndLoopChain) {
  constexpr absl::string_view kHlo = R"hlo(
add_f32 {
  ROOT sum = f32[] add(f32[] parameter(0), f32[] parameter(1))
}

transpose_phase {
  ROOT tr = f32[64,64]{1,0} transpose(f32[64,64]{1,0} parameter(0)), dimensions={1,0}
}

reduce_phase {
  p0 = f32[64,64]{1,0} parameter(0)
  init = f32[] constant(0.0)
  ROOT red = f32[64]{0} reduce(p0, init), dimensions={1}, to_apply=add_f32
}

epilogue_phase {
  ROOT exp = f32[64]{0} exponential(f32[64]{0} parameter(0))
}

ENTRY main {
  p0 = f32[64,64]{1,0} parameter(0)
  f0 = f32[64,64]{1,0} fusion(p0), kind=kInput, calls=transpose_phase
  f1 = f32[64]{0} fusion(f0), kind=kInput, calls=reduce_phase
  ROOT f2 = f32[64]{0} fusion(f1), kind=kLoop, calls=epilogue_phase
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
                  2)));

  auto f0_nested = m::Fusion(m::Parameter(0))
                       .WithFusionKind(HloInstruction::FusionKind::kInput);
  auto f1_nested =
      m::Fusion(f0_nested).WithFusionKind(HloInstruction::FusionKind::kInput);
  auto f2_nested =
      m::Fusion(f1_nested).WithFusionKind(HloInstruction::FusionKind::kLoop);
  EXPECT_THAT(multi_phase_fusion->fused_expression_root(),
              GmockMatch(m::Tuple(f0_nested, f1_nested, f2_nested)));
}

TEST_F(MultiPhaseFusionTest,
       FusesMultiOutputReductionStageAndRewiresTupleElements) {
  constexpr absl::string_view kHlo = R"hlo(
add_f32 { ROOT sum = f32[] add(f32[] parameter(0), f32[] parameter(1)) }
max_f32 { ROOT mx = f32[] maximum(f32[] parameter(0), f32[] parameter(1)) }

mof_reduce {
  p0 = f32[32,128]{1,0} parameter(0)
  z = f32[] constant(0.0)
  ninf = f32[] constant(-inf)
  r0 = f32[32]{0} reduce(p0, z), dimensions={1}, to_apply=add_f32
  r1 = f32[32]{0} reduce(p0, ninf), dimensions={1}, to_apply=max_f32
  ROOT t = (f32[32]{0}, f32[32]{0}) tuple(r0, r1)
}

combine_loop {
  ROOT diff = f32[32]{0} subtract(f32[32]{0} parameter(0), f32[32]{0} parameter(1))
}

ENTRY main {
  p0 = f32[32,128]{1,0} parameter(0)
  f0 = (f32[32]{0}, f32[32]{0}) fusion(p0), kind=kInput, calls=mof_reduce
  gte0 = f32[32]{0} get-tuple-element(f0), index=0
  gte1 = f32[32]{0} get-tuple-element(f0), index=1
  f1 = f32[32]{0} fusion(gte0, gte1), kind=kLoop, calls=combine_loop
  ROOT out = (f32[32]{0}, f32[32]{0}) tuple(gte0, f1)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  const HloInstruction* multi_phase_fusion = nullptr;
  const HloInstruction* multi_phase_fusion_dup = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::Fusion(&multi_phase_fusion, m::Parameter(0))
                                 .WithPredicate(IsPhasedBlockClusterFusion),
                             0),
          m::GetTupleElement(m::Fusion(&multi_phase_fusion_dup), 2))));
  EXPECT_EQ(multi_phase_fusion, multi_phase_fusion_dup);

  const HloInstruction* f0_nested = nullptr;
  ASSERT_THAT(multi_phase_fusion->fused_expression_root(),
              GmockMatch(m::Tuple(
                  m::GetTupleElement(
                      m::Fusion(&f0_nested, m::Parameter(0))
                          .WithFusionKind(HloInstruction::FusionKind::kInput),
                      0),
                  m::GetTupleElement(m::Fusion(), 1),
                  m::Fusion(m::GetTupleElement(m::Fusion(), 0),
                            m::GetTupleElement(m::Fusion(), 1))
                      .WithFusionKind(HloInstruction::FusionKind::kLoop))));
  EXPECT_NE(f0_nested, nullptr);
}

TEST_F(MultiPhaseFusionTest, WrapsUnfusedReduceAsInputAndLoopAsLoop) {
  constexpr absl::string_view kHlo = R"hlo(
add_f32 { ROOT sum = f32[] add(f32[] parameter(0), f32[] parameter(1)) }

ENTRY main {
  p0 = f32[32,128]{1,0} parameter(0)
  init = f32[] constant(0.0)
  red = f32[32]{0} reduce(p0, init), dimensions={1}, to_apply=add_f32
  ROOT exp = f32[32]{0} exponential(red)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  const HloInstruction* multi_phase_fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::Fusion(&multi_phase_fusion, m::Parameter(0), m::Constant())
                      .WithPredicate(IsPhasedBlockClusterFusion)
                      .WithOneUse(),
                  1)));

  auto red_nested = m::Fusion(m::Parameter(0), m::Parameter(1))
                        .WithFusionKind(HloInstruction::FusionKind::kInput);
  auto exp_nested =
      m::Fusion(red_nested).WithFusionKind(HloInstruction::FusionKind::kLoop);
  EXPECT_THAT(multi_phase_fusion->fused_expression_root(),
              GmockMatch(m::Tuple(red_nested, exp_nested)));
}

TEST_F(MultiPhaseFusionTest, SkipsReductionWithNonWarpMultipleThreads) {
  // `MultiRowReductionFusion` assigns `3 * 8 = 24` threads (< warp size 32) to
  // `reduce(f32[3,8])`, which cannot use partial-CTA `bar.sync` or guarded
  // full-warp shuffles inside a larger-CTA kernel.
  constexpr absl::string_view kHlo = R"hlo(
add_f32 { ROOT sum = f32[] add(f32[] parameter(0), f32[] parameter(1)) }

reduce_24_threads {
  p0 = f32[3,8]{1,0} parameter(0)
  init = f32[] constant(0.0)
  ROOT red = f32[3]{0} reduce(p0, init), dimensions={1}, to_apply=add_f32
}

epilogue {
  ROOT exp = f32[3]{0} exponential(f32[3]{0} parameter(0))
}

ENTRY main {
  p0 = f32[3,8]{1,0} parameter(0)
  f0 = f32[3]{0} fusion(p0), kind=kInput, calls=reduce_24_threads
  ROOT f1 = f32[3]{0} fusion(f0), kind=kLoop, calls=epilogue
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(false));
}

TEST_F(MultiPhaseFusionTest, ReconstructsWholeTupleWhenTupleStageIsEntryRoot) {
  constexpr absl::string_view kHlo = R"hlo(
pre_loop { ROOT exp = f32[256]{0} exponential(f32[256]{0} parameter(0)) }

mof_loop {
  p0 = f32[256]{0} parameter(0)
  ROOT t = (f32[256]{0}, f32[256]{0}) tuple(f32[256]{0} negate(p0), f32[256]{0} cosine(p0))
}

ENTRY main {
  p0 = f32[256]{0} parameter(0)
  f0 = f32[256]{0} fusion(p0), kind=kLoop, calls=pre_loop
  ROOT f1 = (f32[256]{0}, f32[256]{0}) fusion(f0), kind=kLoop, calls=mof_loop
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  const HloInstruction* multi_phase_fusion = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::Fusion(&multi_phase_fusion, m::Parameter(0))
                                 .WithPredicate(IsPhasedBlockClusterFusion),
                             1),
          m::GetTupleElement(m::Fusion(), 2))));
  EXPECT_NE(multi_phase_fusion, nullptr);
}

TEST_F(MultiPhaseFusionTest, FusesConcatenateStagesWithSubWarpThreads) {
  // `ConcatenateFusion` emits neither CTA barriers nor warp shuffles, so
  // `phase_1` and `phase_2` (`1 block x 8 threads` < warp size 32) can fuse
  // with `phase_0` (`1 block x 32 threads`).
  constexpr absl::string_view kHlo = R"hlo(
add_s32 { ROOT sum = s32[] add(s32[] parameter(0), s32[] parameter(1)) }

phase_0_comp {
  ROOT red = s32[8]{0} reduce(s32[8,32]{1,0} parameter(0), s32[] constant(0)),
    dimensions={1}, to_apply=add_s32
}

phase_1_comp {
  ROOT cat = s32[9]{0} concatenate(s32[1]{0} parameter(0), s32[8]{0} parameter(1)),
    dimensions={0}
}

phase_2_comp {
  sl = s32[8]{0} slice(s32[9]{0} parameter(1)), slice={[0:8]}
  sum = s32[8]{0} add(sl, s32[8]{0} parameter(2))
  ROOT cat = s32[9]{0} concatenate(s32[1]{0} parameter(0), sum), dimensions={0}
}

ENTRY main {
  p0 = s32[8,32]{1,0} parameter(0)
  z = s32[1]{0} constant({0})
  phase_0 = s32[8]{0} fusion(p0), kind=kInput, calls=phase_0_comp
  phase_1 = s32[9]{0} fusion(z, phase_0), kind=kInput, calls=phase_1_comp
  ROOT phase_2 = s32[9]{0} fusion(z, phase_1, phase_0), kind=kInput,
    calls=phase_2_comp
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  const HloInstruction* multi_phase_fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::Fusion(&multi_phase_fusion, m::Parameter(0), m::Constant())
                      .WithPredicate(IsPhasedBlockClusterFusion)
                      .WithOneUse(),
                  2)));
  auto p0_nested = m::Fusion(m::Parameter(0))
                       .WithFusionKind(HloInstruction::FusionKind::kInput);
  auto p1_nested = m::Fusion(m::Parameter(1), p0_nested)
                       .WithFusionKind(HloInstruction::FusionKind::kInput);
  auto p2_nested = m::Fusion(m::Parameter(1), p1_nested, p0_nested)
                       .WithFusionKind(HloInstruction::FusionKind::kInput);
  EXPECT_THAT(multi_phase_fusion->fused_expression_root(),
              GmockMatch(m::Tuple(p0_nested, p1_nested, p2_nested)));
}

TEST_F(MultiPhaseFusionTest, SplitsDiamondDagIntoDependencyOrderedChains) {
  // Tuple producer `f0` fans out to independent branches `f1` and `f2`, which
  // join at `f3` while `f1` is also returned externally. Rather than
  // serializing `f1` and `f2` into one fusion, the pass splits the diamond into
  // two dependency-ordered chains: `{f0, f1}` and `{f2, f3}`.
  constexpr absl::string_view kHlo = R"hlo(
mof_0 {
  p0 = f32[256]{0} parameter(0)
  ROOT t = (f32[256]{0}, f32[256]{0}) tuple(f32[256]{0} negate(p0), f32[256]{0} cosine(p0))
}
branch_1 { ROOT exp = f32[256]{0} exponential(f32[256]{0} parameter(0)) }
branch_2 { ROOT sn = f32[256]{0} sine(f32[256]{0} parameter(0)) }
join_3 { ROOT add = f32[256]{0} add(f32[256]{0} parameter(0), f32[256]{0} parameter(1)) }

ENTRY main {
  p0 = f32[256]{0} parameter(0)
  f0 = (f32[256]{0}, f32[256]{0}) fusion(p0), kind=kLoop, calls=mof_0
  gte0 = f32[256]{0} get-tuple-element(f0), index=0
  gte1 = f32[256]{0} get-tuple-element(f0), index=1
  f1 = f32[256]{0} fusion(gte0), kind=kLoop, calls=branch_1
  f2 = f32[256]{0} fusion(gte1), kind=kLoop, calls=branch_2
  f3 = f32[256]{0} fusion(f1, f2), kind=kLoop, calls=join_3
  ROOT out = (f32[256]{0}, f32[256]{0}) tuple(f1, f3)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  const HloInstruction* first_fusion = nullptr;
  const HloInstruction* second_fusion = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::Fusion(&first_fusion, m::Parameter(0))
                                 .WithPredicate(IsPhasedBlockClusterFusion),
                             2),
          m::GetTupleElement(
              m::Fusion(&second_fusion, m::GetTupleElement(m::Fusion(), 1),
                        m::GetTupleElement(m::Fusion(), 2))
                  .WithPredicate(IsPhasedBlockClusterFusion)
                  .WithOneUse(),
              1))));
  EXPECT_EQ(first_fusion->fused_expression_root()->operand_count(), 3);
  EXPECT_EQ(second_fusion->fused_expression_root()->operand_count(), 2);
  EXPECT_EQ(second_fusion->operand(0)->operand(0), first_fusion);
  EXPECT_EQ(second_fusion->operand(1)->operand(0), first_fusion);
}

TEST_F(MultiPhaseFusionTest, DoesNotSerializeIndependentProducersInJoinDag) {
  constexpr absl::string_view kJoinHlo = R"hlo(
root_0 { ROOT neg = f32[256]{0} negate(f32[256]{0} parameter(0)) }
root_1 { ROOT exp = f32[256]{0} exponential(f32[256]{0} parameter(0)) }
join_2 { ROOT add = f32[256]{0} add(f32[256]{0} parameter(0), f32[256]{0} parameter(1)) }

ENTRY main {
  p0 = f32[256]{0} parameter(0)
  p1 = f32[256]{0} parameter(1)
  f0 = f32[256]{0} fusion(p0), kind=kLoop, calls=root_0
  f1 = f32[256]{0} fusion(p1), kind=kLoop, calls=root_1
  ROOT f2 = f32[256]{0} fusion(f0, f1), kind=kLoop, calls=join_2
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kJoinHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));
  auto f1_unfused = m::Fusion(m::Parameter(1))
                        .WithFusionKind(HloInstruction::FusionKind::kLoop);
  const HloInstruction* multi_phase_fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::Fusion(&multi_phase_fusion, m::Parameter(0), f1_unfused)
                      .WithPredicate(IsPhasedBlockClusterFusion)
                      .WithOneUse(),
                  1)));
  EXPECT_EQ(multi_phase_fusion->fused_expression_root()->operand_count(), 2);
}

TEST_F(MultiPhaseFusionTest, DoesNotSerializeIndependentSiblingsInFanOut) {
  constexpr absl::string_view kFanOutHlo = R"hlo(
root_0 { ROOT neg = f32[256]{0} negate(f32[256]{0} parameter(0)) }
branch_1 { ROOT exp = f32[256]{0} exponential(f32[256]{0} parameter(0)) }
branch_2 { ROOT sn = f32[256]{0} sine(f32[256]{0} parameter(0)) }

ENTRY main {
  p0 = f32[256]{0} parameter(0)
  f0 = f32[256]{0} fusion(p0), kind=kLoop, calls=root_0
  f1 = f32[256]{0} fusion(f0), kind=kLoop, calls=branch_1
  f2 = f32[256]{0} fusion(f0), kind=kLoop, calls=branch_2
  ROOT out = (f32[256]{0}, f32[256]{0}) tuple(f1, f2)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kFanOutHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  const HloInstruction* multi_phase_fusion = nullptr;
  const HloInstruction* multi_phase_fusion_dup = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::Fusion(&multi_phase_fusion, m::Parameter(0))
                                 .WithPredicate(IsPhasedBlockClusterFusion),
                             1),
          // Matches the unfused sibling branch (`f2`, `kLoop`), which now reads
          // `f0`'s output from the newly formed multi-phase fusion.
          m::Fusion(m::GetTupleElement(m::Fusion(&multi_phase_fusion_dup), 0))
              .WithFusionKind(HloInstruction::FusionKind::kLoop))));
  EXPECT_EQ(multi_phase_fusion, multi_phase_fusion_dup);
  EXPECT_EQ(multi_phase_fusion->fused_expression_root()->operand_count(), 2);
}

TEST_F(MultiPhaseFusionTest, AvoidsProducerCycleInJoinDag) {
  // `f1 -> ext -> f0` creates an external path into `f0`, so `{f0, f2}` fuses
  // while `f1` stays outside to avoid forming a cycle.
  constexpr absl::string_view kCycleHlo = R"hlo(
root_1 { ROOT exp = f32[256]{0} exponential(f32[256]{0} parameter(0)) }
stage_0 { ROOT neg = f32[256]{0} negate(f32[256]{0} parameter(0)) }
join_2 { ROOT add = f32[256]{0} add(f32[256]{0} parameter(0), f32[256]{0} parameter(1)) }

ENTRY main {
  p0 = f32[256]{0} parameter(0)
  f1 = f32[256]{0} fusion(p0), kind=kLoop, calls=root_1
  ext = f32[256]{0} custom-call(f1), custom_call_target="dummy"
  f0 = f32[256]{0} fusion(ext), kind=kLoop, calls=stage_0
  ROOT f2 = f32[256]{0} fusion(f0, f1), kind=kLoop, calls=join_2
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kCycleHlo));
  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));
  auto f1_unfused = m::Fusion(m::Parameter(0))
                        .WithFusionKind(HloInstruction::FusionKind::kLoop);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::Fusion(m::CustomCall(f1_unfused), f1_unfused)
                      .WithPredicate(IsPhasedBlockClusterFusion)
                      .WithOneUse(),
                  1)));
}

// Adds a `kLoop` fusion computation that sums `num_params` `f32[256]`
// parameters (and optionally returns `(sum, -sum)` as a 2-tuple).
HloComputation* AddSumFusionComputation(VerifiedHloModule* module,
                                        absl::string_view name, int num_params,
                                        bool tuple_output = false) {
  const Shape shape = ShapeUtil::MakeShape(F32, {256});
  HloComputation::Builder builder(name);
  HloInstruction* acc = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_0"));
  for (int i = 1; i < num_params; ++i) {
    HloInstruction* p = builder.AddInstruction(
        HloInstruction::CreateParameter(i, shape, absl::StrCat("param_", i)));
    acc = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, acc, p));
  }
  if (tuple_output) {
    HloInstruction* neg = builder.AddInstruction(
        HloInstruction::CreateUnary(shape, HloOpcode::kNegate, acc));
    builder.AddInstruction(HloInstruction::CreateTuple({acc, neg}));
  }
  return module->AddEmbeddedComputation(builder.Build());
}

HloComputation* AddUnaryFusionComputation(VerifiedHloModule* module,
                                          absl::string_view name,
                                          HloOpcode opcode) {
  const Shape shape = ShapeUtil::MakeShape(F32, {256});
  HloComputation::Builder builder(name);
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_0"));
  builder.AddInstruction(HloInstruction::CreateUnary(shape, opcode, p0));
  return module->AddEmbeddedComputation(builder.Build());
}

TEST_F(MultiPhaseFusionTest, SplitsDagWhenExceedingMaxKernelArgs) {
  // `f0` reads `p0..p127` (128 unique outer operands, 2 tuple outputs) and `f1`
  // reads `gte0`, shared `p0`, and `p128..p252` (+125 new outer operands, +1
  // output), hitting `253 + 3 == 256` (`kMaxKernelArgs`). Appending `f2` (+1
  // output = 257) exceeds the cap, so `{f0, f1}` and `{f2, f3}` form two
  // separate fusions.
  const Shape shape = ShapeUtil::MakeShape(F32, {256});
  std::unique_ptr<VerifiedHloModule> module = CreateNewVerifiedModule();
  HloComputation* mof_comp = AddSumFusionComputation(
      module.get(), "mof_128", /*num_params=*/128, /*tuple_output=*/true);
  HloComputation* sum_comp =
      AddSumFusionComputation(module.get(), "sum_127", /*num_params=*/127);
  HloComputation* cos_comp =
      AddUnaryFusionComputation(module.get(), "cos_comp", HloOpcode::kCos);
  HloComputation* exp_comp =
      AddUnaryFusionComputation(module.get(), "exp_comp", HloOpcode::kExp);

  HloComputation::Builder entry_builder(TestName());
  std::vector<HloInstruction*> params;
  params.reserve(253);
  for (int i = 0; i < 253; ++i) {
    params.push_back(entry_builder.AddInstruction(
        HloInstruction::CreateParameter(i, shape, absl::StrCat("p", i))));
  }
  HloInstruction* f0 =
      entry_builder.AddInstruction(HloInstruction::CreateFusion(
          mof_comp->root_instruction()->shape(),
          HloInstruction::FusionKind::kLoop,
          absl::MakeSpan(params).subspan(0, 128), mof_comp));
  HloInstruction* gte0 = entry_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f0, 0));
  std::vector<HloInstruction*> f1_operands = {gte0, params[0]};
  f1_operands.insert(f1_operands.end(), params.begin() + 128, params.end());
  HloInstruction* f1 =
      entry_builder.AddInstruction(HloInstruction::CreateFusion(
          shape, HloInstruction::FusionKind::kLoop, f1_operands, sum_comp));
  HloInstruction* f2 =
      entry_builder.AddInstruction(HloInstruction::CreateFusion(
          shape, HloInstruction::FusionKind::kLoop, {f1}, cos_comp));
  entry_builder.AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, {f2}, exp_comp));
  module->AddEntryComputation(entry_builder.Build());

  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(true));

  const HloInstruction* first_fusion = nullptr;
  const HloInstruction* second_fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::Fusion(&second_fusion,
                            m::GetTupleElement(
                                m::Fusion(&first_fusion)
                                    .WithPredicate(IsPhasedBlockClusterFusion),
                                2))
                      .WithPredicate(IsPhasedBlockClusterFusion)
                      .WithOneUse(),
                  1)));
  EXPECT_EQ(first_fusion->operand_count(), 253);
  EXPECT_EQ(first_fusion->shape().tuple_shapes_size(), 3);
  EXPECT_THAT(second_fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Fusion(), m::Fusion())));
}

TEST_F(MultiPhaseFusionTest, SkipsSeedStageExceedingMaxKernelArgs) {
  const Shape shape = ShapeUtil::MakeShape(F32, {256});
  std::unique_ptr<VerifiedHloModule> module = CreateNewVerifiedModule();
  HloComputation* sum_comp =
      AddSumFusionComputation(module.get(), "sum_256", /*num_params=*/256);
  HloComputation* exp_comp =
      AddUnaryFusionComputation(module.get(), "exp_comp", HloOpcode::kExp);

  HloComputation::Builder entry_builder(TestName());
  std::vector<HloInstruction*> params;
  params.reserve(256);
  for (int i = 0; i < 256; ++i) {
    params.push_back(entry_builder.AddInstruction(
        HloInstruction::CreateParameter(i, shape, absl::StrCat("p", i))));
  }
  HloInstruction* f0 =
      entry_builder.AddInstruction(HloInstruction::CreateFusion(
          shape, HloInstruction::FusionKind::kLoop, params, sum_comp));
  entry_builder.AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, {f0}, exp_comp));
  module->AddEntryComputation(entry_builder.Build());

  MultiPhaseFusion pass(device_info_);
  EXPECT_THAT(RunHloPass(&pass, module.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla::gpu
