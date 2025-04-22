/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/nest_gemm_fusion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

using ::testing::ElementsAre;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

namespace xla {

namespace gpu {
namespace {

// Wraps a matcher for a fusion instruction's output tile sizes.
// Proto matchers would be nice, but b/229726259 is P2.
MATCHER_P(OutputTileSizesIs, matcher, "") {
  auto backend_config = arg.template backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    *result_listener << "failed to get backend config: "
                     << backend_config.status();
    return false;
  }
  FusionBackendConfig fusion_backend_config =
      backend_config->fusion_backend_config();
  if (!fusion_backend_config.has_block_level_fusion_config()) {
    *result_listener << "has no block level fusion config";
    return false;
  }
  if (fusion_backend_config.kind() != "__triton_nested_gemm_fusion") {
    *result_listener << "fusion kind is not __triton_nested_gemm_fusion";
    return false;
  }
  auto output_tile_sizes =
      fusion_backend_config.block_level_fusion_config().output_tiles(0).sizes();
  return ExplainMatchResult(matcher, output_tile_sizes, result_listener);
}

class NestGemmFusionTest : public HloHardwareIndependentTestBase {
 protected:
  const se::GpuComputeCapability compute_capability_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo(se::CudaComputeCapability::Ampere())
          .gpu_compute_capability()};
};

TEST_F(NestGemmFusionTest, BasicTest) {
  absl::string_view hlo = R"(
dot {
  lhs = f32[8192,512] parameter(0)
  rhs = f32[512,512] parameter(1)
  ROOT  dot = f32[8192,512] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = f32[8192,512] parameter(0)
  p1 = f32[512,512] parameter(1)
  ROOT fusion = f32[8192,512] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"64", "block_n":"256", "block_k":"32",
          "split_k":"1", "num_stages":"5", "num_warps":"4", "num_ctas":"3"
        }
      }
    }
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());

  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(match::Fusion(&fusion)));
  EXPECT_THAT(*fusion, OutputTileSizesIs(ElementsAre(64, 256)));

  BlockLevelFusionConfig block_level_fusion_config =
      fusion->backend_config<GpuBackendConfig>()
          ->fusion_backend_config()
          .block_level_fusion_config();
  EXPECT_THAT(block_level_fusion_config.output_tiles(0).sizes(),
              ElementsAre(64, 256));
  EXPECT_THAT(block_level_fusion_config.num_warps(), 4);
  EXPECT_THAT(block_level_fusion_config.num_ctas(), 3);
  EXPECT_THAT(block_level_fusion_config.num_stages(), 5);

  const HloInstruction* lhs = nullptr;
  const HloInstruction* rhs = nullptr;
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(match::Dot(match::Fusion(&lhs), match::Fusion(&rhs))));
  EXPECT_THAT(*lhs, OutputTileSizesIs(ElementsAre(64, 32)));
  EXPECT_THAT(*rhs, OutputTileSizesIs(ElementsAre(32, 256)));

  // The old GEMM config should have been deleted.
  EXPECT_FALSE(fusion->backend_config<GpuBackendConfig>()
                   ->fusion_backend_config()
                   .has_triton_gemm_config());
}

// Tests hoisting of bitcasts which would otherwise trigger unsatisfiable
// constraints during symbolic tile analysis.
TEST_F(NestGemmFusionTest, BitcastsAreHoistedOutOfGemmFusions) {
  absl::string_view hlo = R"(
dot {
  lhs = f32[21] parameter(0)
  bitcast = f32[3,7]{0,1} bitcast(lhs)
  rhs = f32[7,11] parameter(1)
  ROOT dot = f32[3,11] dot(bitcast, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = f32[21] parameter(0)
  p1 = f32[7,11] parameter(1)
  ROOT fusion = f32[3,11] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"32", "block_n":"64", "block_k":"16",
          "split_k":"1", "num_stages":"1", "num_warps":"1", "num_ctas":"1"
        }
      }
    }
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());

  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(match::Fusion(&fusion)));
  EXPECT_THAT(fusion->operand(0), GmockMatch(match::Bitcast()));
  EXPECT_THAT(*fusion, OutputTileSizesIs(ElementsAre(32, 64)));

  const HloInstruction* lhs = nullptr;
  const HloInstruction* rhs = nullptr;
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(match::Dot(match::Fusion(&lhs), match::Fusion(&rhs))));
  EXPECT_THAT(*lhs, OutputTileSizesIs(ElementsAre(32, 16)));
  EXPECT_THAT(*rhs, OutputTileSizesIs(ElementsAre(16, 64)));
}

TEST_F(NestGemmFusionTest, SupportsTwoBitcastsFromSameParameter) {
  absl::string_view hlo = R"(
dot {
  p0 = f32[32] parameter(0)
  lhs = f32[4,8] bitcast(p0)
  rhs = f32[8,4] bitcast(p0)
  ROOT dot = f32[4,4] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = f32[32] parameter(0)
  ROOT fusion = f32[4,4] fusion(p0),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"4", "block_n":"4", "block_k":"8",
          "split_k":"1", "num_stages":"1", "num_warps":"1", "num_ctas":"1"
        }
      }
    }
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());

  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(match::Fusion(&fusion)));
  EXPECT_THAT(*fusion, OutputTileSizesIs(ElementsAre(4, 4)));

  const HloInstruction* lhs = nullptr;
  const HloInstruction* rhs = nullptr;
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(match::Dot(match::Fusion(&lhs), match::Fusion(&rhs))));
  EXPECT_THAT(*lhs, OutputTileSizesIs(ElementsAre(4, 8)));
  EXPECT_THAT(*rhs, OutputTileSizesIs(ElementsAre(8, 4)));
}

TEST_F(NestGemmFusionTest, BitcastsCanBeHoistedPastOtherBitcasts) {
  absl::string_view hlo = R"(
dot {
  lhs = f32[3,7] parameter(0)
  bitcast0 = f32[21] bitcast(lhs)
  bitcast1 = f32[3,7] bitcast(bitcast0)
  rhs = f32[7,11] parameter(1)
  ROOT dot = f32[3,11] dot(bitcast1, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = f32[3, 7] parameter(0)
  p1 = f32[7,11] parameter(1)
  ROOT fusion = f32[3,11] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"32", "block_n":"64", "block_k":"16",
          "split_k":"1", "num_stages":"1", "num_warps":"1", "num_ctas":"1"
        }
      }
    }
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(NestGemmFusionTest, BitcastsCanBeHoistedPastElementwiseEpilogues) {
  absl::string_view hlo = R"(
dot {
  lhs = f32[3,7] parameter(0)
  rhs = f32[7,11] parameter(1)
  dot = f32[3,11] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast = f32[33] bitcast(dot)
  ROOT add = f32[33] add(bitcast, bitcast)
}

ENTRY entry {
  p0 = f32[3, 7] parameter(0)
  p1 = f32[7,11] parameter(1)
  ROOT fusion = f32[33] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"32", "block_n":"64", "block_k":"16",
          "split_k":"1", "num_stages":"1", "num_warps":"1", "num_ctas":"1"
        }
      }
    }
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

// We cannot hoist bitcasts past transposes, but we don't need to hoist
// because the bitcast is not rank-expanding and symbolic tile analysis
// works fine.
TEST_F(NestGemmFusionTest, BitcastsCannotBeHoistedPastTransposes) {
  absl::string_view hlo = R"(
dot {
  p0 = f32[72,36,2] parameter(0)
  transpose0 = f32[72,2,36] transpose(p0), dimensions={0,2,1}
  bitcast0 = f32[144,36] bitcast(transpose0)
  p1 = f32[36,3] parameter(1)
  dot = f32[144,3] dot(bitcast0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast1 = f32[144,3] bitcast(dot)
  ROOT transpose1 = f32[3,144] transpose(bitcast1), dimensions={1,0}
}

ENTRY entry {
  p0 = f32[72,36,2] parameter(0)
  p1 = f32[36,3] parameter(1)
  ROOT fusion = f32[3,144] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_gemm","triton_gemm_config":{
          "block_m":"128","block_n":"16","block_k":"32",
          "split_k":"1","num_stages":"4","num_warps":"4","num_ctas":"1"
        }
      }
    }
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(NestGemmFusionTest, TritonFusionEmitterDeviceLegacyTestSample1) {
  absl::string_view hlo = R"(
dot {
  p0 = f16[1,16,17,3] parameter(0)
  bitcast0 = f16[16,51] bitcast(f16[1,16,17,3] p0)
  p1 = s8[16,17,3] parameter(1)
  bitcast1 = s8[16,51] bitcast(s8[16,17,3] p1)
  convert = f16[16,51] convert(s8[16,51] bitcast1)
  bitcast2 = f16[51,16]{0,1} bitcast(f16[16,51] convert)
  dot = f16[16,16] dot(bitcast0, bitcast2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT bitcast3 = f16[1,16,16] bitcast(f16[16,16] dot)
}

ENTRY entry {
  p0 = f16[1,16,17,3] parameter(0)
  p1 = s8[16,17,3] parameter(1)
  ROOT fusion = f16[1,16,16] fusion(f16[1,16,17,3] p0, s8[16,17,3] p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_gemm","triton_gemm_config":{
          "block_m":"16","block_n":"16","block_k":"32",
          "split_k":"1","num_stages":"1","num_warps":"4","num_ctas":"1"
        }
      }
    }
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(NestGemmFusionTest, TritonFusionEmitterDeviceLegacyTestSample2) {
  absl::string_view hlo = R"(
dot {
  p0 = pred[3,122,96,12] parameter(0)
  transpose = pred[3,96,12,122] transpose(p0), dimensions={0,2,3,1}
  bitcast0 = pred[3456,122] bitcast(transpose)
  convert0 = f16[3456,122] convert(bitcast0)
  p1 = pred[1,5,122] parameter(1)
  bitcast1 = pred[5,122] bitcast(p1)
  convert1 = f16[5,122] convert(bitcast1)
  bitcast2 = f16[122,5]{0,1} bitcast(convert1)
  dot.1 = f16[3456,5] dot(convert0, bitcast2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT bitcast3 = f16[3,96,12,1,5] bitcast(dot.1)
}

ENTRY entry_computation {
  p0 = pred[3,122,96,12] parameter(0)
  p1 = pred[1,5,122] parameter(1)
  ROOT gemm_fusion_dot = f16[3,96,12,1,5] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_gemm","triton_gemm_config":{
          "block_m":"4","block_n":"16","block_k":"128",
          "split_k":"1","num_stages":"1","num_warps":"4","num_ctas":"1"
        }
      }
    }
})";
  // Note: block sizes were 16,16,32, but that now fails to satisfy constraints.
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(NestGemmFusionTest, TritonFusionEmitterDeviceLegacyTestSample3) {
  absl::string_view hlo = R"(
dot {
  p0 = f32[1,40] parameter(0)
  bitcast0 = f32[40] bitcast(p0)
  bitcast1 = f32[40,1] bitcast(bitcast0)
  p1 = f32[1,40,250000] parameter(1)
  bitcast2 = f32[40,250000] bitcast(p1)
  dot = f32[1,250000] dot(bitcast1, bitcast2), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  bitcast3 = f32[250000] bitcast(dot)
  ROOT bitcast4 = f32[1,250000] bitcast(bitcast3)
}

ENTRY entry_computation {
  p0 = f32[1,40] parameter(0)
  p1 = f32[1,40,250000] parameter(1)
  ROOT gemm_fusion_dot.2 = f32[1,250000] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_gemm","triton_gemm_config":{
          "block_m":"16","block_n":"16","block_k":"32",
          "split_k":"1","num_stages":"1","num_warps":"4","num_ctas":"1"
        }
      }
    }
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(NestGemmFusionTest, ConcatenationsAreHoistedWithinNestedGemmFusions) {
  absl::string_view hlo = R"(
HloModule t

triton_gemm {
  parameter_0 = f32[2,3,10]{2,1,0} parameter(0)
  parameter_1 = f32[2,10,128]{2,1,0} parameter(1)
  parameter_2 = f32[2,10,256]{2,1,0} parameter(2)
  concatenate = f32[2,10,384]{2,1,0} concatenate(parameter_1, parameter_2), dimensions={2}
  ROOT dot = f32[2,3,384]{2,1,0} dot(parameter_0, concatenate),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  parameter_0 = f32[2,3,10]{2,1,0} parameter(0)
  parameter_1 = f32[2,10,128]{2,1,0} parameter(1)
  parameter_2 = f32[2,10,256]{2,1,0} parameter(2)
  ROOT dot = f32[2,3,384]{2,1,0} fusion(parameter_0, parameter_1, parameter_2),
    kind=kCustom, calls=triton_gemm,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":64,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":2,
                         "num_ctas":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  HloComputation* fusion_computation = module->entry_computation()
                                           ->root_instruction()
                                           ->fused_instructions_computation();
  HloInstruction* dot_lhs;
  HloInstruction* dot_rhs;
  EXPECT_THAT(
      fusion_computation->root_instruction(),
      GmockMatch(match::Dot(match::Fusion(&dot_lhs), match::Fusion(&dot_rhs))));
  EXPECT_THAT(dot_rhs->fused_instructions_computation()->root_instruction(),
              GmockMatch(match::Concatenate(match::Fusion(), match::Fusion())));
}

TEST_F(NestGemmFusionTest, UnsupportedComputationsAreRejected) {
  // Fusions other than kTritonNestedGemmFusionKind are not supported so
  // we expect that the pass will fail as the resulting computation is not
  // supported.
  absl::string_view hlo = R"(
identity {
  ROOT result = f32[128,128]{1,0} parameter(0)
}

triton_dot {
  p0 = f32[128,128]{1,0} parameter(0)
  cp0 = f32[128,128]{1,0} fusion(p0), kind=kCustom, calls=identity
  p1 = f32[128,128]{1,0} parameter(1)
  ROOT result = f32[128,128]{1,0} dot(cp0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[128,128]{1,0} parameter(0)
  p1 = f32[128,128]{1,0} parameter(1)
  ROOT result = f32[128,128] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    "triton_gemm_config": {
      "block_m":32,"block_n":16,"block_k":128,
      "split_k":1,"num_stages":1,"num_warps":4, "num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  absl::StatusOr<bool> result =
      NestGemmFusion(compute_capability_).Run(module.get());
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInternal)) << result.status();
}

TEST_F(NestGemmFusionTest, BitcastsAreHoistedPastCompare) {
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = s32[11,24,128]{2,1,0} parameter(0)
  p1 = s32[11,24,128]{2,1,0} parameter(1)
  eq = pred[11,24,128]{2,1,0} compare(p0, p1), direction=EQ
  eq_reshape = pred[264,128]{1,0} bitcast(eq)
  eq_f32 = f32[264,128]{1,0} convert(eq_reshape)
  p2 = f32[128,8]{1,0} parameter(2)
  ROOT result = f32[264,8]{1,0} dot(eq_f32, p2),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s32[11,24, 128]{2,1,0} parameter(0)
  p1 = s32[11,24,128]{2,1,0} parameter(1)
  p2 = f32[128,8]{1,0} parameter(2)
  ROOT result = f32[264,8] fusion(p0, p1, p2), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {
      "block_m":32,"block_n":16,"block_k":128,
      "split_k":1,"num_stages":1,"num_warps":4, "num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(NestGemmFusionTest, BitcastsAreHoistedUpThroughBroadcasts) {
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[11,1,24,1] parameter(0)
  p0_broadcast = f32[11,1,24,1,128] broadcast(p0), dimensions={0,1,2,3}
  p0_reshape = f32[264,128] bitcast(p0_broadcast)

  p1 = f32[128,8]{1,0} parameter(1)
  ROOT result = f32[264,8]{1,0} dot(p0_reshape, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[11,1,24,1] parameter(0)
  p1 = f32[128,8] parameter(1)
  ROOT result = f32[264,8] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
// Broadcast fusion:
CHECK: {{.*}} {
CHECK-NEXT: [[broadcast_p0:[^ ]+]] = f32[264]{0} parameter(0)
CHECK-NEXT: ROOT {{.*}} = f32[264,128]{1,0} broadcast([[broadcast_p0]]), dimensions={0}
CHECK-NEXT: }
CHECK: ENTRY {{.*}} {
CHECK: [[entry_p0:[^ ]+]] = f32[11,1,24,1]{3,2,1,0} parameter(0)
CHECK: {{.*}} = f32[264]{0} bitcast([[entry_p0]])
)"),
      IsOkAndHolds(true));
}

TEST_F(NestGemmFusionTest, BitcastOfOperandAndBroadcastDimsIsNotHoisted) {
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[3,4] parameter(0)
  p0_broadcast = f32[3,4,5] broadcast(p0), dimensions={0,1}
  p0_cos = f32[3,4,5] cosine(p0_broadcast)
  // Bitcast mixes operand and broadcasted dimensions and cannot be hoisted.
  p0_reshape = f32[3,20] bitcast(p0_cos)

  p1 = f32[20,7]{1,0} parameter(1)
  ROOT result = f32[3,7]{1,0} dot(p0_reshape, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[3,4] parameter(0)
  p1 = f32[20,7] parameter(1)
  ROOT result = f32[3,7] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_TRUE(!NestGemmFusion(compute_capability_).Run(module.get()).ok());
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  // Cos should not be rewritten as we cannot hoist bitcast.
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: f32[3,4,5]{2,1,0} cosine
)"),
      IsOkAndHolds(true));
}

TEST_F(NestGemmFusionTest, BitcastsAreHoistedUpThroughBroadcastDiamonds) {
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[3,5] parameter(0)
  b0 = f32[3,5,77,1] broadcast(p0), dimensions={0,1}
  b1 = f32[3,5,1] broadcast(p0), dimensions={0,1}
  b2 = f32[3,5,77,1] broadcast(b1), dimensions={0,1,3}
  sum = add(b0, b2)
  sum_reshape = f32[15,77] bitcast(sum)
  p1 = f32[77,8]{1,0} parameter(1)
  ROOT result = f32[15,8] dot(sum_reshape, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[3,5] parameter(0)
  p1 = f32[77,8] parameter(1)
  ROOT result = f32[15,8] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: [[p0:[^ ]+]] = f32[15]{0} parameter(0)
CHECK-DAG: {{.*}} = f32[15,77]{1,0} broadcast([[p0]]), dimensions={0}
CHECK-DAG: [[br:[^ ]+]] = f32[15]{0} broadcast([[p0]]), dimensions={0}
CHECK-DAG: {{.*}} = f32[15,77]{1,0} broadcast([[br]]), dimensions={0}
)"),
      IsOkAndHolds(true));
}

TEST_F(NestGemmFusionTest, BitcastsAreHoistedOverBroadcasts) {
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[11,1,24,1] parameter(0)
  p0_broadcast = f32[11,1,24,1,128,1] broadcast(p0), dimensions={0,1,2,5}
  p0_reshape = f32[264,128] bitcast(p0_broadcast)

  p1 = f32[128,8]{1,0} parameter(1)
  ROOT result = f32[264,8]{1,0} dot(p0_reshape, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[11,1,24,1] parameter(0)
  p1 = f32[128,8] parameter(1)
  ROOT result = f32[264,8] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
// Broadcast fusion:
CHECK: {{.*}} {
CHECK-NEXT: [[broadcast_p0:[^ ]+]] = f32[264]{0} parameter(0)
CHECK-NEXT: ROOT {{.*}} = f32[264,128]{1,0} broadcast([[broadcast_p0]]), dimensions={0}
CHECK-NEXT: }
CHECK: ENTRY {{.*}} {
CHECK: [[entry_p0:[^ ]+]] = f32[11,1,24,1]{3,2,1,0} parameter(0)
CHECK: {{.*}} = f32[264]{0} bitcast([[entry_p0]])
)"),
      IsOkAndHolds(true));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
