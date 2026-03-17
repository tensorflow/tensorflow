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

#include "xla/backends/gpu/transforms/hoist_fused_bitcasts.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

using ::absl_testing::IsOkAndHolds;

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

class HoistFusedBitcastsReshapeTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<HloOpcode> {
 protected:
  HoistFusedBitcastsReshapeTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }
  const se::DeviceDescription device_description_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo(
          se::GpuComputeCapability{se::CudaComputeCapability::Ampere()})};
  mlir::MLIRContext mlir_context_;

  std::unique_ptr<VerifiedHloModule> RunHoistFusedBitcasts(
      absl::string_view hlo, const bool expect_change = true) {
    std::unique_ptr<VerifiedHloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    EXPECT_THAT(HoistFusedBitcasts().Run(module.get()),
                IsOkAndHolds(expect_change));
    EXPECT_OK(verifier().Run(module.get()).status());
    return module;
  }
};

// Tests hoisting of bitcasts which would otherwise trigger unsatisfiable
// constraints during symbolic tile analysis.
TEST_P(HoistFusedBitcastsReshapeTest, BitcastsAreHoistedOutOfGemmFusions) {
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

  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));

  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: dot {
CHECK-NEXT: [[lhs:[^ ]+]] = f32[3,7]{0,1} parameter(0)
CHECK-NEXT: [[rhs:[^ ]+]] = f32[7,11]{1,0} parameter(1)
CHECK-NEXT: ROOT {{.*}} = f32[3,11]{1,0} dot([[lhs]], [[rhs]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}
CHECK-NEXT: }
CHECK: ENTRY
CHECK: bitcast
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest, BitcastsCanBeHoistedPastOtherBitcasts) {
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
  RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastsCanBeHoistedPastElementwiseEpilogues) {
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
  RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastsCanBeHoistedPastConvertEpilogues) {
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: f16[3,11]{1,0} convert(
CHECK: f16[3,11]{1,0} fusion(
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       ResumeBitcastSinkingAfterIncompatibleOps) {
  // Even though we cannot hoist the bitcast1 past the transpose we still can
  // remove bitcast2.
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  p0 = f32[2,32] parameter(0)
  p1 = f32[64,32] parameter(1)
  dot = f32[2,64] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bitcast1 = f32[2,32,2] $0(dot)
  transpose1 = f32[2,2,32] transpose(bitcast1), dimensions={0,2,1}
  ROOT bitcast2 = f32[1,2,1,2,32] $0(transpose1)
}

ENTRY entry {
  p0 = f32[2,32] parameter(0)
  p1 = f32[64,32] parameter(1)
  ROOT fusion = f32[1,2,1,2,32] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_gemm","triton_gemm_config":{
          "block_m":"128","block_n":"16","block_k":"32",
          "split_k":"1","num_stages":"4","num_warps":"4","num_ctas":"1"
        }
      }
    }
})";
  auto module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
  CHECK: ROOT {{.*}} = f32[2,2,32]{2,1,0} transpose
  CHECK: ENTRY
  CHECK: ROOT {{.*}} = f32[1,2,1,2,32]{4,3,2,1,0} bitcast
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest, BitcastsKeepElementSizeInBits) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
dot {
  lhs = s8[21]{0:E(4)} parameter(0)
  c1 = s8[21] convert(lhs)
  c2 = f32[21] convert(c1)
  b0 = f32[3,7] $0(c2)
  rhs = f32[7,11] parameter(1)
  dot = f32[3,11] dot(b0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  b1 = f32[33] $0(dot)
  ROOT c = s8[33]{0:E(4)} convert(b1)
}

ENTRY entry {
  p0 = s8[21]{0:E(4)} parameter(0)
  p1 = f32[7,11] parameter(1)
  ROOT fusion = s8[33]{0:E(4)} fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"32", "block_n":"64", "block_k":"16",
          "split_k":"1", "num_stages":"1", "num_warps":"1", "num_ctas":"1"
        }
      }
    }
})";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
  CHECK: ENTRY
  CHECK: {{.*}} = s8[3,7]{1,0:E(4)} bitcast({{.*}})
  CHECK: [[fusion:[^ ]+]] = s8[3,11]{1,0:E(4)} fusion({{.*}})
  CHECK: ROOT {{.*}} = s8[33]{0:E(4)} bitcast([[fusion]])
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       TritonFusionEmitterDeviceLegacyTestSample1) {
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
  RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       TritonFusionEmitterDeviceLegacyTestSample2) {
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
  RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       TritonFusionEmitterDeviceLegacyTestSample3) {
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
  RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
}

TEST_P(HoistFusedBitcastsReshapeTest, BitcastsAreHoistedPastCompare) {
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
  RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
}

TEST_P(HoistFusedBitcastsReshapeTest, BitcastsAreHoistedUpThroughBroadcasts) {
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
// Broadcast fusion:
CHECK: {{.*}} {
CHECK-NEXT: [[dot_p0:[^ ]+]] = f32[264]{0} parameter(0)
CHECK-NEXT: {{.*}} = f32[264,128]{1,0} broadcast([[dot_p0]]), dimensions={0}
CHECK: ENTRY {{.*}} {
CHECK: [[entry_p0:[^ ]+]] = f32[11,1,24,1]{3,2,1,0} parameter(0)
CHECK: {{.*}} = f32[264]{0} bitcast([[entry_p0]])
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastsAreHoistedUpThroughBroadcastsWithTrivialDimensions) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[11,24,1] parameter(0)
  p0_broadcast = f32[11,1,24,1,128] broadcast(p0), dimensions={0,2,3}
  p0_reshape = f32[264,128] $0(p0_broadcast)
  p1 = f32[128,8]{1,0} parameter(1)
  ROOT result = f32[264,8]{1,0} dot(p0_reshape, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[11,24,1] parameter(0)
  p1 = f32[128,8] parameter(1)
  ROOT result = f32[264,8] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}}
)";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
// Broadcast fusion:
CHECK: {{.*}} {
CHECK-NEXT: [[dot_p0:[^ ]+]] = f32[264]{0} parameter(0)
CHECK-NEXT: {{.*}} = f32[264,128]{1,0} broadcast([[dot_p0]]), dimensions={0}
CHECK: ENTRY {{.*}} {
CHECK: [[entry_p0:[^ ]+]] = f32[11,24,1]{{.*}} parameter(0)
CHECK: {{.*}} = f32[264]{0} bitcast([[entry_p0]])
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastOfOperandAndBroadcastDimsIsNotHoistedUp) {
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)),
                            /*expect_change=*/false);
  // Cos should not be rewritten as we cannot hoist bitcast.
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           absl::Substitute(R"(
CHECK:      f32[3,4,16]{2,1,0} broadcast
CHECK-NEXT: f32[3,64]{1,0} $0
)",
                                            HloOpcodeString(opcode))),
              IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastOfOperandAndBroadcastDimsIsNotHoistedDown) {
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)),
                            /*expect_change=*/false);
  // Cos should not be rewritten as we cannot hoist bitcast.
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           absl::Substitute(R"(
CHECK:      f32[2,3,5]{2,1,0} $0
CHECK-NEXT: f32[2,4,3,5]{3,2,1,0} broadcast
)",
                                            HloOpcodeString(opcode))),
              IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastsAreHoistedUpThroughBroadcastDiamonds) {
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: [[p0:[^ ]+]] = f32[15]{0} parameter(0)
CHECK-DAG: {{.*}} = f32[15,77]{1,0} broadcast([[p0]]), dimensions={0}
CHECK-DAG: [[br:[^ ]+]] = f32[15]{0} broadcast([[p0]]), dimensions={0}
CHECK-DAG: {{.*}} = f32[15,77]{1,0} broadcast([[br]]), dimensions={0}
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest, BitcastsAreHoistedOverBroadcasts) {
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           R"(
// Broadcast fusion:
CHECK: {{.*}} {
CHECK-NEXT: [[dot_p0:[^ ]+]] = f32[264]{0} parameter(0)
CHECK-NEXT: {{.*}} = f32[264,128]{1,0} broadcast([[dot_p0]]), dimensions={0}
CHECK: ENTRY {{.*}} {
CHECK: [[entry_p0:[^ ]+]] = f32[11,1,24,1]{3,2,1,0} parameter(0)
CHECK: {{.*}} = f32[264]{0} bitcast([[entry_p0]])
)"),

              IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest, BitcastsLayoutIsPreserved) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

gemm_dot {
  p0 = pred[3,122,96,12] parameter(0)
  bitcast0 = pred[3,122,1152] $0(p0)
  transpose0 = pred[3,1152,122] transpose(bitcast0), dimensions={0,2,1}
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           absl::Substitute(R"(
CHECK: {{.*}} {
CHECK: [[re:[^ ]+]] = pred[3456,122]{1,0} $0({{.*}})
CHECK: {{.*}} = f16[3456,122]{1,0} convert([[re]])
CHECK-NOT: $0
CHECK: {{.*}} = f16[122,5]{0,1} convert({{.*}})
CHECK-NEXT: }
CHECK: ENTRY {{.*}} {
CHECK: {{.*}} = pred[122,5]{0,1} bitcast({{.*}})
)",
                                            HloOpcodeString(opcode))),
              IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       CheckDimensionsOfBroadcastAfterBitcastIsHoisted) {
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

  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: bf16[1,2,4,8]{{.*}} broadcast({{.*}}), dimensions={3}
CHECK: bf16[1,2,4,8]{{.*}} broadcast({{.*}}), dimensions={3}
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest, BitcastsAreHoistedUpThroughTransposes) {
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: {{.*}} {
CHECK-NEXT: [[p0:[^ ]*]] = f32[7,2,3]{2,1,0} parameter(0)
CHECK-NEXT: {{.*}} = f32[2,3,7]{2,1,0} transpose([[p0]]), dimensions={1,2,0}
CHECK ENTRY
CHECK f32[7,2,3]{2,1,0} bitcast
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastsWithSize1DimensionsAreHoistedUpThroughTransposes) {
  const HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[7,6] parameter(0)
  transpose = f32[6,7] transpose(p0), dimensions={1,0}
  bitcast = f32[1,6,7] $0(transpose)
  p1 = f32[1,5,7] parameter(1)
  ROOT result = f32[1,6,5] dot(bitcast, p1),
    lhs_contracting_dims={2}, lhs_batch_dims={0},
    rhs_contracting_dims={2}, rhs_batch_dims={0}
}

ENTRY e {
  p0 = f32[7,6] parameter(0)
  p1 = f32[1,5,7] parameter(1)
  ROOT result = f32[1,6,5] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: {{.*}} {
CHECK-NEXT: [[p0:[^ ]+]] = f32[7,1,6]{2,1,0} parameter(0)
CHECK-NEXT: {{.*}} = f32[1,6,7]{2,1,0} transpose([[p0]]), dimensions={1,2,0}
CHECK-NOT: bitcast
CHECK: }
CHECK ENTRY {{.*}} {
CHECK: bitcast
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastsWithSize1DimensionsSeparatedAreHoistedUpThroughTransposes) {
  const HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  lhs = f32[16,24,320] parameter(0)
  rhs = f32[320,2] parameter(1)
  dot = f32[16,24,2] dot(lhs, rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}

  bitcast = f32[1, 384, 2] $0(dot)
  ROOT transpose = f32[1, 2, 384] transpose(bitcast), dimensions={0, 2, 1}
}

ENTRY e {
  p0 = f32[16,24,320] parameter(0)
  p1 = f32[320,2] parameter(1)
  ROOT result = f32[1,2,384] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: {{.*}} {
CHECK-NEXT: [[p0:[^ ]+]] = f32[16,24,320]{2,1,0} parameter(0)
CHECK-NEXT: [[p1:[^ ]+]] = f32[320,2]{1,0} parameter(1)
CHECK-NEXT: [[dot:[^ ]+]] = f32[16,24,2]{2,1,0} dot([[p0]], [[p1]])
CHECK-NEXT: ROOT {{.*}} = f32[2,16,24]{2,1,0} transpose([[dot]]), dimensions={2,0,1}
CHECK: }
CHECK: ENTRY {{.*}} {
CHECK: [[fusion:[^ ]+]] = f32[2,16,24]{2,1,0} fusion
CHECK: bitcast([[fusion]])
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastBetweenDotAndTransposeWithDegenerateDimension) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  lhs = f32[16,16] parameter(0)
  rhs = f32[16,16] parameter(1)
  dot = f32[16,16] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast = f32[1, 2, 128] $0(dot)
  ROOT transpose = f32[1, 128, 2] transpose(bitcast), dimensions={0, 2, 1}
}

ENTRY entry {
  p0 = f32[16,16] parameter(0)
  p1 = f32[16,16] parameter(1)
  ROOT fusion = f32[1, 128, 2] fusion(p0, p1),
    kind=kCustom, calls=triton_dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"16", "block_n":"16", "block_k":"16",
          "split_k":"1", "num_stages":"1", "num_warps":"1", "num_ctas":"1"
        }
      }
    }
}
)";

  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)),
                            /*expect_change=*/false);
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: bitcast
CHECK: transpose
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       RankReducingBitcastsAreNotHoistedUpThroughTransposes) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[2,7,3] parameter(0)
  transpose = f32[3,2,7] transpose(p0), dimensions={2,0,1}
  $0 = f32[6,7] $0(transpose)
  p1 = f32[5,7] parameter(1)
  ROOT dot = f32[6,5] dot($0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f32[2,7,3] parameter(0)
  p1 = f32[5,7] parameter(1)
  ROOT result = f32[6,5] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)),
                            /*expect_change=*/false);
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK:      transpose
CHECK-SAME: f32[3,2,7]{2,1,0} transpose
CHECK-SAME: dimensions={2,0,1}
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       RankReducingBitcastsAreNotHoistedDownThroughTransposes) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[6,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  dot = f32[6,5] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  $0 = f32[2,3,5] $0(dot)
  ROOT transpose = f32[2,5,3] transpose($0), dimensions={0,2,1}
}

ENTRY e {
  p0 = f32[6,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  ROOT result = f32[2,5,3] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)),
                            /*expect_change=*/false);
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           absl::Substitute(R"(
CHECK:      f32[2,3,5]{2,1,0} $0
CHECK-NEXT: f32[2,5,3]{2,1,0} transpose
)",
                                            HloOpcodeString(opcode))),
              IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       HoistingBitcastDoesNotIntroduceArtificialDimension) {
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
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

TEST_P(HoistFusedBitcastsReshapeTest, BitcastsAreHoistedDownThroughTransposes) {
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK:      ROOT transpose
CHECK-SAME: f32[5,2,3]{2,1,0} transpose
CHECK-SAME: dimensions={2,0,1}
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest, BitcastsAreHoistedDownThroughBroadcasts) {
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK:      ROOT broadcast
CHECK-SAME: f32[3,5,6,2]{2,1,0,3} broadcast
CHECK-SAME: dimensions={0,1}
)"),
      IsOkAndHolds(true));
}

// TODO(b/467306121): handle the case when we need to sink the reshape through
// broadcast.
TEST_P(HoistFusedBitcastsReshapeTest,
       DISABLED_BitcastsAreHoistedDownThroughBroadcastsWithTrivialDimensions) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[3,7] parameter(0)
  p1 = f32[6,7] parameter(1)
  dot = f32[3,6] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bitcast = f32[3,2,3] $0(dot)
  ROOT broadcast = f32[3,2,1,3,7] broadcast(bitcast), dimensions={0,1,3}
}

ENTRY e {
  p0 = f32[3,7] parameter(0)
  p1 = f32[6,7] parameter(1)
  ROOT result = f32[3,2,1,3,7] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK:      ROOT broadcast
CHECK-SAME: f32[3,5,6,2]{2,1,0,3} broadcast
CHECK-SAME: dimensions={0,1}
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)),
                            /*expect_change=*/false);
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           absl::Substitute(R"(
CHECK:      f32[2,3,5]{2,1,0} $0(dot)
CHECK-NEXT: f32[2,3,5]{2,0,1} broadcast
)",
                                            HloOpcodeString(opcode))),
              IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest, BitcastRootsAreHoistedDown) {
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
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: ROOT dot
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastAreHoistedDownThroughBinaryElementwiseOps) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
triton_dot {
  p0 = f32[3,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  p2 = f32[15] parameter(2)
  dot = f32[3,5] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  $0 = f32[15] $0(dot)
  ROOT add = f32[15] add($0, p2)
}

ENTRY e {
  p0 = f32[3,7] parameter(0)
  p1 = f32[5,7] parameter(1)
  p2 = f32[15] parameter(2)
  ROOT result = f32[15] fusion(p0, p1, p2), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}}
)";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: ROOT add = f32[3,5]{1,0} add
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastsWithNonDefaultLayoutAreHoistedOutThroughBroadcast) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[7,2]{0,1} parameter(0)
  broadcast.1 = f32[15,7,2]{1,0,2} broadcast(p0), dimensions={1,2}
  $0.1 = f32[2,7,15]{1,2,0} $0(broadcast.1)
  p1 = f32[2,15,15]{2,1,0} parameter(1)
  dot = f32[2,7,15]{2,1,0} dot($0.1, p1),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
  $0.2 = f32[15,14]{0,1} $0(dot)
  ROOT broadcast.2 = f32[15,11,14]{0,2,1} broadcast($0.2), dimensions={0,2}
}

ENTRY e {
  p0 = f32[7,2]{0,1} parameter(0)
  p1 = f32[2,15,15]{2,1,0} parameter(1)
  ROOT result = f32[15,11,14]{0,2,1} fusion(p0, p1),
    kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}}
)";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK-NOT: bitcast
CHECK-NOT: reshape
CHECK: f32[2,7,15]{1,2,0} broadcast({{.*}}), dimensions={0,1}
CHECK-NOT: bitcast
CHECK-NOT: reshape
CHECK: f32[2,7,15,11]{2,1,0,3} broadcast({{.*}}), dimensions={0,1,2}
CHECK: ENTRY
CHECK: f32[7,2]{0,1} parameter(0)
CHECK: f32[2,7]{1,0} bitcast(p0
CHECK: result = f32[2,7,15,11]{2,1,0,3} fusion
CHECK: ROOT {{.*}} = f32[15,11,14]{0,2,1} bitcast(result)
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest,
       BitcastsWithNonDefaultLayoutAreHoistedOutThroughTranspose) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[2,3,7]{0,2,1} parameter(0)
  $0.1 = f32[7,3,2]{2,0,1} $0(p0)
  transpose.1 = f32[3,2,7]{2,0,1} transpose($0.1), dimensions={1,2,0}
  p1 = f32[3,5,7]{2,1,0} parameter(1)
  dot = f32[3,2,5]{2,1,0} dot(transpose.1, p1),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
  $0.2 = f32[5,3,2]{0,2,1} $0(dot)
  ROOT transpose.2 = f32[2,3,5]{0,2,1} transpose($0.2), dimensions={2,1,0}
}

ENTRY e {
  p0 = f32[2,3,7]{0,2,1} parameter(0)
  p1 = f32[3,5,7]{2,1,0} parameter(1)
  ROOT result = f32[2,3,5]{0,2,1} fusion(p0, p1),
    kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}}
)";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK-NOT: bitcast
CHECK-NOT: reshape
CHECK: f32[3,2,7]{2,0,1} transpose({{.*}}), dimensions={1,2,0}
CHECK-NOT: bitcast
CHECK-NOT: reshape
CHECK: f32[3,5,2]{2,1,0} transpose({{.*}}), dimensions={0,2,1}
CHECK: ENTRY
CHECK: f32[2,3,7]{0,2,1} parameter(0)
CHECK: f32[7,3,2]{2,0,1} bitcast(p0
CHECK: result = f32[3,5,2]{2,1,0} fusion
CHECK: ROOT {{.*}} = f32[2,3,5]{0,2,1} bitcast(result)
)"),
      IsOkAndHolds(true));
}

TEST_P(HoistFusedBitcastsReshapeTest, MultipleBitcastsAreHoistedOut) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[3,3]{1,0} parameter(0)
  $0.1 = f32[3,3]{1,0} $0(p0)
  $0.2 = f32[3,3]{1,0} $0($0.1)
  p1 = f32[3,3]{1,0} parameter(1)
  dot = f32[3,3]{1,0} dot($0.2, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  $0.3 = f32[3,3]{1,0} $0(dot)
  ROOT $0.4 = f32[3,3]{0,1} $0($0.3)
}

ENTRY e {
  p0 = f32[3,3]{1,0} parameter(0)
  ROOT result = f32[3,3]{0,1} fusion(p0, p0),
    kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}}
)";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK-NOT: bitcast
CHECK-NOT: reshape
CHECK: ENTRY
)"),
      IsOkAndHolds(true));
}

// TODO(b/393299275): this test was not written correctly and now fails.
TEST_P(HoistFusedBitcastsReshapeTest,
       DISABLED_BitcastsAreNotHoistedOutThroughLayoutChangingTranspose) {
  HloOpcode opcode = GetParam();
  absl::string_view hlo = R"(
HloModule t

triton_dot {
  p0 = f32[7,2]{1,0} parameter(0)
  $0.1 = f32[2,7]{0,1} $0(p0)
  transpose.1 = f32[2,7]{1,0} transpose($0.1), dimensions={0,1}
  p1 = f32[5,7]{1,0} parameter(1)
  dot = f32[2,5]{1,0} dot(transpose.1, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  $0.2 = f32[5,2]{0,1} $0(dot)
  ROOT transpose.2 = f32[5,2]{1,0} transpose($0.2), dimensions={0,1}
}

ENTRY e {
  p0 = f32[7,2]{1,0} parameter(0)
  p1 = f32[5,7]{1,0} parameter(1)
  ROOT result = f32[5,2]{1,0} fusion(p0, p1),
    kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":16,"block_k":8,
    "split_k":1,"num_stages":1,"num_warps":4,"num_ctas":1}}}}
)";
  std::unique_ptr<VerifiedHloModule> module =
      RunHoistFusedBitcasts(absl::Substitute(hlo, HloOpcodeString(opcode)));
  EXPECT_THAT(RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()),
                           absl::Substitute(R"(
CHECK: $0.1 = f32[2,7]{0,1} $0
CHECK: $0.2 = f32[5,2]{0,1} $0
CHECK: ENTRY
CHECK-NOT: bitcast
CHECK-NOT: reshape
        )",
                                            HloOpcodeString(opcode))),
              IsOkAndHolds(true));
}

INSTANTIATE_TEST_SUITE_P(HoistFusedBitcastsReshapeTestSuite,
                         HoistFusedBitcastsReshapeTest,
                         ::testing::ValuesIn({HloOpcode::kReshape,
                                              HloOpcode::kBitcast}),
                         [](const ::testing::TestParamInfo<HloOpcode>& info) {
                           return std::string(HloOpcodeString(info.param));
                         });

struct CommonFactorsTestCase {
  std::vector<int64_t> from, to;
  absl::InlinedVector<std::pair<int64_t, int64_t>, 8> expected;
};

class CommonFactorsMergingTrivialRangesTest
    : public ::testing::TestWithParam<CommonFactorsTestCase> {};

TEST_P(CommonFactorsMergingTrivialRangesTest, Example) {
  const CommonFactorsTestCase& test_case = GetParam();
  EXPECT_EQ(test_case.expected, detail::CommonFactorsMergingTrivialRanges(
                                    test_case.from, test_case.to));
}

INSTANTIATE_TEST_SUITE_P(
    CommonFactorsMergingTrivialRangesTestSuite,
    CommonFactorsMergingTrivialRangesTest,
    ::testing::Values(
        CommonFactorsTestCase{{1}, {}, {{0, 0}, {1, 0}}},
        CommonFactorsTestCase{{}, {1}, {{0, 0}, {0, 1}}},
        CommonFactorsTestCase{{}, {}, {{0, 0}}},
        CommonFactorsTestCase{{1, 2, 0}, {2, 0, 3}, {{0, 0}, {3, 3}}},
        CommonFactorsTestCase{{2, 3, 0}, {1, 0, 1000}, {{0, 0}, {3, 3}}},
        CommonFactorsTestCase{{1, 1, 1}, {1, 1}, {{0, 0}, {1, 1}, {3, 2}}},
        CommonFactorsTestCase{{1, 1, 3}, {3, 1, 1}, {{0, 0}, {3, 3}}},
        CommonFactorsTestCase{{2, 6}, {4, 3}, {{0, 0}, {2, 2}}},
        CommonFactorsTestCase{{1, 2, 6}, {4, 1, 3, 1}, {{0, 0}, {3, 4}}},
        CommonFactorsTestCase{{2, 3, 4, 5}, {6, 20}, {{0, 0}, {2, 1}, {4, 2}}},
        CommonFactorsTestCase{
            {2, 3, 4, 5, 6}, {6, 20, 6}, {{0, 0}, {2, 1}, {4, 2}, {5, 3}}},
        CommonFactorsTestCase{{2, 2, 2, 2}, {4, 4}, {{0, 0}, {2, 1}, {4, 2}}},
        CommonFactorsTestCase{
            {2, 5, 1, 3}, {1, 10, 3, 1}, {{0, 0}, {2, 2}, {4, 4}}}),
    [](const ::testing::TestParamInfo<CommonFactorsTestCase>& info) {
      return absl::StrCat(absl::StrJoin(info.param.from, "_"), "_to_",
                          absl::StrJoin(info.param.to, "_"));
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
