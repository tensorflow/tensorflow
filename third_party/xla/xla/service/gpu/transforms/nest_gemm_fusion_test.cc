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

#include <ostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

using ::testing::ElementsAre;
using ::tsl::testing::IsOkAndHolds;

namespace xla {

// Gtest hook to pretty-print an HloInstruction.
static void PrintTo(const HloInstruction& hlo, std::ostream* os) {
  *os << hlo.ToString();
}

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
  auto output_tile_sizes =
      fusion_backend_config.block_level_fusion_config().output_tiles(0).sizes();
  return ExplainMatchResult(matcher, output_tile_sizes, result_listener);
}

class NestGemmFusionTest : public HloTestBase {};

TEST_F(NestGemmFusionTest, BasicTest) {
  absl::string_view hlo = R"(
dot {
  lhs = bf16[8192,512] parameter(0)
  rhs = bf16[512,512] parameter(1)
  ROOT  dot = bf16[8192,512] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = bf16[8192,512] parameter(0)
  p1 = bf16[512,512] parameter(1)
  ROOT fusion = bf16[8192,512] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"64", "block_n":"256", "block_k":"32",
          "split_k":"1", "num_stages":"1", "num_warps":"1", "num_ctas":"1"
        }
      }
    }
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion().Run(module.get()), IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());

  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(match::Fusion(&fusion)));
  EXPECT_THAT(*fusion, OutputTileSizesIs(ElementsAre(64, 256)));

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
  EXPECT_THAT(NestGemmFusion().Run(module.get()), IsOkAndHolds(true));
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
  EXPECT_THAT(NestGemmFusion().Run(module.get()), IsOkAndHolds(true));
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
  EXPECT_THAT(NestGemmFusion().Run(module.get()), IsOkAndHolds(true));
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
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion().Run(module.get()), IsOkAndHolds(true));
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
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion().Run(module.get()), IsOkAndHolds(true));
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
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion().Run(module.get()), IsOkAndHolds(true));
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
}
)";
  // Note: block sizes were 16,16,32, but that now fails to satisfy constraints.
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion().Run(module.get()), IsOkAndHolds(true));
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
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_THAT(NestGemmFusion().Run(module.get()), IsOkAndHolds(true));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
