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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
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

class NestGemmFusionTest : public HloHardwareIndependentTestBase,
                           public ::testing::WithParamInterface<HloOpcode> {
 protected:
  const se::GpuComputeCapability compute_capability_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo(se::CudaComputeCapability::Ampere())
          .gpu_compute_capability()};
};

TEST_P(NestGemmFusionTest, BasicTest) {
  HloOpcode opcode = GetParam();
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

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
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
TEST_P(NestGemmFusionTest, BitcastsAreHoistedOutOfGemmFusions) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  lhs = f32[21] parameter(0)
  bitcast = f32[3,7]{0,1} $0(lhs)
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

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
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

TEST_P(NestGemmFusionTest, SupportsTwoBitcastsFromSameParameter) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  p0 = f32[32] parameter(0)
  lhs = f32[4,8] $0(p0)
  rhs = f32[8,4] $0(p0)
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

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
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

TEST_P(NestGemmFusionTest, BitcastsCanBeHoistedPastOtherBitcasts) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  lhs = f32[3,7] parameter(0)
  bitcast0 = f32[21] $0(lhs)
  bitcast1 = f32[3,7] $0(bitcast0)
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_P(NestGemmFusionTest, BitcastsCanBeHoistedPastElementwiseEpilogues) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  lhs = f32[3,7] parameter(0)
  rhs = f32[7,11] parameter(1)
  dot = f32[3,11] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast = f32[33] $0(dot)
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_P(NestGemmFusionTest, BitcastsCanBeHoistedPastConvertEpilogues) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  lhs = f32[3,7] parameter(0)
  rhs = f32[7,11] parameter(1)
  dot = f32[3,11] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast = f32[33] $0(dot)
  ROOT convert = f16[33] convert(bitcast)
}

ENTRY entry {
  p0 = f32[3, 7] parameter(0)
  p1 = f32[7,11] parameter(1)
  ROOT fusion = f16[33] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"32", "block_n":"64", "block_k":"16",
          "split_k":"1", "num_stages":"1", "num_warps":"1", "num_ctas":"1"
        }
      }
    }
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: f16[3,11]{1,0} convert(
CHECK: f16[3,11]{1,0} fusion(
)"),
      IsOkAndHolds(true));

  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

// We cannot hoist bitcasts past transposes, but we don't need to hoist
// because the bitcast is not rank-expanding and symbolic tile analysis
// works fine.
TEST_P(NestGemmFusionTest, BitcastsCannotBeHoistedPastTransposes) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  p0 = f32[72,36,2] parameter(0)
  transpose0 = f32[72,2,36] transpose(p0), dimensions={0,2,1}
  bitcast0 = f32[144,36] $0(transpose0)
  p1 = f32[36,3] parameter(1)
  dot = f32[144,3] dot(bitcast0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast1 = f32[144,3] $0(dot)
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_P(NestGemmFusionTest, TritonFusionEmitterDeviceLegacyTestSample1) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  p0 = f16[1,16,17,3] parameter(0)
  bitcast0 = f16[16,51] $0(f16[1,16,17,3] p0)
  p1 = s8[16,17,3] parameter(1)
  bitcast1 = s8[16,51] $0(s8[16,17,3] p1)
  convert = f16[16,51] convert(s8[16,51] bitcast1)
  bitcast2 = f16[51,16]{0,1} $0(f16[16,51] convert)
  dot = f16[16,16] dot(bitcast0, bitcast2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT bitcast3 = f16[1,16,16] $0(f16[16,16] dot)
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_P(NestGemmFusionTest, TritonFusionEmitterDeviceLegacyTestSample2) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  p0 = pred[3,122,96,12] parameter(0)
  transpose = pred[3,96,12,122] transpose(p0), dimensions={0,2,3,1}
  bitcast0 = pred[3456,122] $0(transpose)
  convert0 = f16[3456,122] convert(bitcast0)
  p1 = pred[1,5,122] parameter(1)
  bitcast1 = pred[5,122] $0(p1)
  convert1 = f16[5,122] convert(bitcast1)
  bitcast2 = f16[122,5]{0,1} $0(convert1)
  dot.1 = f16[3456,5] dot(convert0, bitcast2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT bitcast3 = f16[3,96,12,1,5] $0(dot.1)
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_P(NestGemmFusionTest, TritonFusionEmitterDeviceLegacyTestSample3) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  p0 = f32[1,40] parameter(0)
  bitcast0 = f32[40] $0(p0)
  bitcast1 = f32[40,1] $0(bitcast0)
  p1 = f32[1,40,250000] parameter(1)
  bitcast2 = f32[40,250000] $0(p1)
  dot = f32[1,250000] dot(bitcast1, bitcast2), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  bitcast3 = f32[250000] $0(dot)
  ROOT bitcast4 = f32[1,250000] $0(bitcast3)
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_P(NestGemmFusionTest, ConcatenationsAreHoistedWithinNestedGemmFusions) {
  HloOpcode opcode = GetParam();
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
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

TEST_P(NestGemmFusionTest, UnsupportedComputationsAreRejected) {
  HloOpcode opcode = GetParam();
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  absl::StatusOr<bool> result =
      NestGemmFusion(compute_capability_).Run(module.get());
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInternal)) << result.status();
}

TEST_P(NestGemmFusionTest, BitcastsAreHoistedPastCompare) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = s32[11,24,128]{2,1,0} parameter(0)
  p1 = s32[11,24,128]{2,1,0} parameter(1)
  eq = pred[11,24,128]{2,1,0} compare(p0, p1), direction=EQ
  eq_reshape = pred[264,128]{1,0} $0(eq)
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_P(NestGemmFusionTest, BitcastsAreHoistedUpThroughBroadcasts) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[11,1,24,1] parameter(0)
  p0_broadcast = f32[11,1,24,1,128] broadcast(p0), dimensions={0,1,2,3}
  p0_reshape = f32[264,128] $0(p0_broadcast)

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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
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

TEST_P(NestGemmFusionTest, BitcastOfOperandAndBroadcastDimsIsNotHoistedUp) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[3,4] parameter(0)
  p1 = f32[64,7]{1,0} parameter(1)
  broadcast = f32[3,4,16] broadcast(p0), dimensions={0,1}
  // Bitcast mixes operand and broadcasted dimensions and cannot be hoisted.
  reshape = f32[3,64] $0(broadcast)
  ROOT dot = f32[3,7]{1,0} dot(reshape, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[3,4] parameter(0)
  p1 = f32[64,7] parameter(1)
  ROOT result = f32[3,7] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  // We can nest the fusion including the broadcast.
  EXPECT_TRUE(NestGemmFusion(compute_capability_).Run(module.get()).ok());
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  // Cos should not be rewritten as we cannot hoist bitcast.
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           absl::Substitute(R"(
CHECK:      f32[3,4,16]{2,1,0} broadcast
CHECK-NEXT: f32[3,64]{1,0} $0
)",
                                            HloOpcodeString(opcode))),
              IsOkAndHolds(true));
}

TEST_P(NestGemmFusionTest, BitcastOfOperandAndBroadcastDimsIsNotHoistedDown) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[6,7] parameter(0)
  p1 = f32[5,7]{1,0} parameter(1)
  dot = f32[6,5]{1,0} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  // Bitcast mixes operand and broadcasted dimensions and cannot be hoisted.
  reshape = f32[2,3,5] $0(dot)
  ROOT broadcast = f32[2,4,3,5] broadcast(reshape), dimensions={0,2,3}
}

ENTRY e {
  p0 = f32[6,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  ROOT result = f32[2,4,3,5] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  // We can nest the fusion including the broadcast.
  EXPECT_TRUE(NestGemmFusion(compute_capability_).Run(module.get()).ok());
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  // Cos should not be rewritten as we cannot hoist bitcast.
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           absl::Substitute(R"(
CHECK:      f32[2,3,5]{2,1,0} $0
CHECK-NEXT: f32[2,4,3,5]{3,2,1,0} broadcast
)",
                                            HloOpcodeString(opcode))),
              IsOkAndHolds(true));
}

TEST_P(NestGemmFusionTest, BitcastsAreHoistedUpThroughBroadcastDiamonds) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[3,5] parameter(0)
  b0 = f32[3,5,77,1] broadcast(p0), dimensions={0,1}
  b1 = f32[3,5,1] broadcast(p0), dimensions={0,1}
  b2 = f32[3,5,77,1] broadcast(b1), dimensions={0,1,3}
  sum = add(b0, b2)
  sum_reshape = f32[15,77] $0(sum)
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
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

TEST_P(NestGemmFusionTest, BitcastsAreHoistedOverBroadcasts) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[11,1,24,1] parameter(0)
  p0_broadcast = f32[11,1,24,1,128,1] broadcast(p0), dimensions={0,1,2,5}
  p0_reshape = f32[264,128] $0(p0_broadcast)

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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           R"(
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

TEST_P(NestGemmFusionTest, BitcastsLayoutIsPreserved) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

gemm_dot {
  p0 = pred[3,122,96,12] parameter(0)
  bitcast0 = pred[3,122,1152] $0(p0)
  transpose0 = pred[3,1152,122] transpose(bitcast0), dimensions={0,2,1}
  // bitcast1 = pred[3,96,12,122] $0(transpose0)
  bitcast2 = pred[3456,122] $0(transpose0)
  convert0 = f16[3456,122] convert(bitcast2)
  p1 = pred[1,5,122] parameter(1)
  bitcast3 = pred[5,122] $0(p1)
  convert1 = f16[5,122] convert(bitcast3)
  bitcast4 = f16[122,5]{0,1} $0(convert1)
  dot0 = f16[3456,5]{1,0} dot(convert0, bitcast4), lhs_contracting_dims={1},
    rhs_contracting_dims={0}
  ROOT bitcast5 = f16[3,96,12,1,5] $0(dot0)
}

ENTRY e {
  p0 = pred[3,122,96,12] parameter(0)
  p1 = pred[1,5,122] parameter(1)
  ROOT fusion = f16[3,96,12,1,5] fusion(p0, p1), kind=kCustom, calls=gemm_dot,
    backend_config={"fusion_backend_config":{kind:"__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":32,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           absl::Substitute(R"(
CHECK: {{.*}} {
CHECK: transpose
CHECK: [[bitcast_or_reshape:[^ ]+]] = pred[3456,122]{1,0} $0({{.*}})
CHECK: ROOT {{.*}} = f16[3456,122]{1,0} convert([[bitcast_or_reshape]])
CHECK-NEXT: }
CHECK: {{.*}} {
CHECK-NOT: $0
CHECK: ROOT {{.*}} = f16[122,5]{0,1} convert({{.*}})
CHECK-NEXT: }
CHECK: ENTRY {{.*}} {
CHECK: {{.*}} = pred[122,5]{0,1} bitcast({{.*}})
)",
                                            HloOpcodeString(opcode))),
              IsOkAndHolds(true));
}

TEST_P(NestGemmFusionTest, CheckDimensionsOfBroadcastAfterBitcastIsHoisted) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  p0 = bf16[1,8] parameter(0)
  broadcast0 = bf16[1,8,8] broadcast(p0), dimensions={0,2}
  lhs = bf16[1,2,4,8] $0(broadcast0)

  p1 = bf16[1,8] parameter(1)
  broadcast1 = bf16[1,8,8] broadcast(p1), dimensions={0,2}
  rhs = bf16[1,2,4,8] $0(broadcast1)

  ROOT dot = bf16[2,1,4,4] dot(lhs, rhs),
    lhs_contracting_dims={3}, lhs_batch_dims={1,0},
    rhs_contracting_dims={3}, rhs_batch_dims={1,0}
}

ENTRY entry {
  p0 = bf16[1,8] parameter(0)
  ROOT fusion = bf16[2,1,4,4] fusion(p0, p0), kind=kCustom, calls=dot,
    backend_config={"fusion_backend_config":{kind:"__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":32,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: bf16[1,2,4,8]{{.*}} broadcast({{.*}}), dimensions={0,3}
CHECK: bf16[1,2,4,8]{{.*}} broadcast({{.*}}), dimensions={0,3}
)"),
      IsOkAndHolds(true));
}

TEST_P(NestGemmFusionTest, BitcastsAreHoistedUpThroughTransposes) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[7,6] parameter(0)
  transpose = f32[6,7] transpose(p0), dimensions={1,0}
  bitcast = f32[2,3,7] $0(transpose)
  p1 = f32[2,5,7] parameter(1)
  ROOT result = f32[2,3,5] dot(bitcast, p1),
    lhs_contracting_dims={2}, lhs_batch_dims={0},
    rhs_contracting_dims={2}, rhs_batch_dims={0}
}

ENTRY e {
  p0 = f32[7,6] parameter(0)
  p1 = f32[2,5,7] parameter(1)
  ROOT result = f32[2,3,5] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK:      ROOT transpose
CHECK-SAME: f32[2,3,7]{2,1,0} transpose
CHECK-SAME: dimensions={1,2,0}
)"),
      IsOkAndHolds(true));
}

TEST_P(NestGemmFusionTest,
       RankReducingBitcastsAreNotHoistedUpThroughTransposes) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[2,3,7] parameter(0)
  transpose = f32[3,2,7] transpose(p0), dimensions={1,0,2}
  bitcast = f32[6,7] $0(transpose)
  p1 = f32[5,7] parameter(1)
  ROOT dot = f32[6,5] dot(bitcast, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f32[2,3,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  ROOT result = f32[6,5] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK:      transpose
CHECK-SAME: f32[3,2,7]{2,1,0} transpose
CHECK-SAME: dimensions={1,0,2}
)"),
      IsOkAndHolds(true));
}

TEST_P(NestGemmFusionTest,
       RankReducingBitcastsAreNotHoistedDownThroughTransposes) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[6,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  dot = f32[6,5] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bitcast = f32[2,3,5] $0(dot)
  ROOT transpose = f32[3,2,5] transpose(bitcast), dimensions={1,0,2}
}

ENTRY e {
  p0 = f32[6,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  ROOT result = f32[3,2,5] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           absl::Substitute(R"(
CHECK:      f32[2,3,5]{2,1,0} $0
CHECK-NEXT: f32[3,2,5]{2,1,0} transpose
)",
                                            HloOpcodeString(opcode))),
              IsOkAndHolds(true));
}

TEST_P(NestGemmFusionTest, HoistingBitcastDoesNotIntroduceArtificialDimension) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
gemm_dot {
  p0 = f16[3,122,1152] parameter(0)
  transpose = f16[3,1152,122] transpose(p0), dimensions={0,2,1}
  bitcast0 = f16[3,96,12,122] $0(transpose)
  bitcast1 = f16[3456,122] $0(bitcast0)
  p1 = f16[122,5] parameter(1)
  ROOT dot = f16[3456,5]{1,0} dot(bitcast1, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[3,122,1152] parameter(0)
  p1 = f16[122,5] parameter(1)
  ROOT fusion = f16[3456,5] fusion(p0, p1), kind=kCustom, calls=gemm_dot,
    backend_config={"fusion_backend_config":{kind:"__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":32,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}
}
          )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  // Checks that transpose is on rank 3 tensor from hoisting bitcast1, not rank
  // 4 tensor from hoisting bitcast0 first and then failing to hoist bitcast1.
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK:      transpose
CHECK-SAME: f16[3,1152,122]{2,1,0} transpose
CHECK-SAME: dimensions={0,2,1}
)"),
      IsOkAndHolds(true));
}

TEST_P(NestGemmFusionTest, BitcastsAreHoistedDownThroughTransposes) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[2,3,7] parameter(0)
  p1 = f32[2,5,7] parameter(1)
  dot = f32[2,3,5] dot(p0, p1),
    lhs_contracting_dims={2}, lhs_batch_dims={0},
    rhs_contracting_dims={2}, rhs_batch_dims={0}
  bitcast = f32[6,5] $0(dot)
  ROOT transpose = f32[5,6] transpose(bitcast), dimensions={1,0}
}

ENTRY e {
  p0 = f32[2,3,7] parameter(0)
  p1 = f32[2,5,7] parameter(1)
  ROOT result = f32[5,6] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK:      ROOT transpose
CHECK-SAME: f32[5,2,3]{2,1,0} transpose
CHECK-SAME: dimensions={2,0,1}
)"),
      IsOkAndHolds(true));
}

TEST_P(NestGemmFusionTest, BitcastsAreHoistedDownThroughBroadcasts) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[3,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  dot = f32[3,5] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bitcast = f32[15] $0(dot)
  ROOT broadcast = f32[2,15,6] broadcast(bitcast), dimensions={1}
}

ENTRY e {
  p0 = f32[3,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  ROOT result = f32[2,15,6] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK:      ROOT broadcast
CHECK-SAME: f32[2,3,5,6]{3,2,1,0} broadcast
CHECK-SAME: dimensions={1,2}
)"),
      IsOkAndHolds(true));
}

TEST_P(NestGemmFusionTest,
       BitcastsAreNotHoistedDownThroughBroadcastsWithNonDefaultLayout) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[6,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  dot = f32[6,5] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bitcast = f32[2,3,5]{2,1,0} $0(dot)
  ROOT broadcast = f32[2,3,5]{2,0,1} broadcast(bitcast), dimensions={0,1,2}
}

ENTRY e {
  p0 = f32[6,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  ROOT result = f32[2,3,5]{2,0,1} fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           absl::Substitute(R"(
CHECK:      f32[2,3,5]{2,1,0} $0(dot)
CHECK-NEXT: f32[2,3,5]{2,0,1} broadcast
)",
                                            HloOpcodeString(opcode))),
              IsOkAndHolds(true));
}

TEST_P(NestGemmFusionTest, BitcastRootsAreHoistedDown) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[3,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  dot = f32[3,5] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  ROOT bitcast = f32[15] $0(dot)
}

ENTRY e {
  p0 = f32[3,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  ROOT result = f32[15] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(hlo, HloOpcodeString(opcode))));
  EXPECT_THAT(NestGemmFusion(compute_capability_).Run(module.get()),
              IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: ROOT dot
)"),
      IsOkAndHolds(true));
}

INSTANTIATE_TEST_SUITE_P(NestGemmFusionTestSuite, NestGemmFusionTest,
                         ::testing::ValuesIn({HloOpcode::kReshape,
                                              HloOpcode::kBitcast}),
                         [](const ::testing::TestParamInfo<HloOpcode>& info) {
                           return std::string(HloOpcodeString(info.param));
                         });

}  // namespace
}  // namespace gpu
}  // namespace xla
