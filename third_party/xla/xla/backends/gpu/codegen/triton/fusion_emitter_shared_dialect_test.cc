/* Copyright 2025 The OpenXLA Authors.

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

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/triton/xtile_test_base.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

// *****************************************************************************
// Tests for emitting a shared dialect between XLA:CPU and XLA:GPU.
//
// These tests are currently relying on the triton specific fusion emitter. The
// plan is to use these tests whilst migrating the triton emitter to a shared
// emitter. The idea is for these tests to be backend agnostic once the shared
// emitter becomes a reality.
// *****************************************************************************

class XTileDialectTest : public HloHardwareIndependentTestBase,
                         public XTileTestBase {};

TEST_F(XTileDialectTest, HloTransposeIsLoweredToStableHloTranspose) {
  constexpr absl::string_view kHloText = R"(
HloModule t

transpose_fusion {
  p0 = f32[137,115]{1,0} parameter(0)
  ROOT transpose = f32[115,137]{1,0} transpose(p0), dimensions={1, 0}
}

ENTRY e {
  p0 = f32[137,115]{1,0} parameter(0)
  ROOT custom-call = f32[115,137]{1,0} fusion(p0), kind=kCustom,
    calls=transpose_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 32}};

  TF_EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("transpose_fusion"),
      block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.transpose %[[ARG:.*]], dims = [1, 0] : (tensor<32x16xf32>) -> tensor<16x32xf32>
)"));
}

TEST_F(XTileDialectTest, HloBitcastIsLoweredToTensorBitcast) {
  constexpr absl::string_view kHloText = R"(
HloModule t, is_scheduled=true

bitcast_fusion {
  p0 = f32[150,160] parameter(0)
  ROOT bitcast_convert = s32[150,160] bitcast(p0)
}

ENTRY e {
  p0 = f32[150,160] parameter(0)
  ROOT custom-call = s32[150,160] fusion(p0), kind=kCustom,
    calls=bitcast_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 32}};

  TF_EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("bitcast_fusion"), block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = tensor.bitcast %[[ARG:.*]] : tensor<16x32xf32> to tensor<16x32xi32>
)"));
}

TEST_F(XTileDialectTest, HloIotaIsLoweredToStableHloIota) {
  constexpr absl::string_view kHloText = R"(
HloModule t, is_scheduled=true

iota_fusion {
  ROOT iota = s32[256] iota(), iota_dimension=0
}

ENTRY e {
  ROOT custom-call = s32[256] fusion(), kind=kCustom,
    calls=iota_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16}};

  TF_EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("iota_fusion"), block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.iota dim = 0 : tensor<16xi32>
)"));
}

TEST_F(XTileDialectTest, HloBroadcastInDimIsLoweredToStableHloBroadcastInDim) {
  constexpr absl::string_view kHloText = R"(
HloModule t

broadcast_in_dim_fusion {
  p0 = f32[150,160] parameter(0)
  ROOT broadcast = f32[150,160,31] broadcast(p0), dimensions={0,1}
}

ENTRY e {
  p0 = f32[150,160] parameter(0)
  ROOT custom-call = f32[150,160,31] fusion(p0), kind=kCustom,
    calls=broadcast_in_dim_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 32, 8}};

  TF_EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("broadcast_in_dim_fusion"),
      block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.broadcast_in_dim %[[ARG:.*]], dims = [0, 1] : (tensor<16x32xf32>) -> tensor<16x32x8xf32>
)"));
}

TEST_F(XTileDialectTest,
       HloZeroDimensionalBroadcastIsLoweredToStableHloBroadcastInDim) {
  constexpr absl::string_view kHloText = R"(
HloModule t

broadcast_in_dim_fusion {
  p0 = f32[] parameter(0)
  ROOT broadcast = f32[150,160,31] broadcast(p0), dimensions={}
}

ENTRY e {
  p0 = f32[] parameter(0)
  ROOT custom-call = f32[150,160,31] fusion(p0), kind=kCustom,
    calls=broadcast_in_dim_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 32, 8}};

  TF_EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("broadcast_in_dim_fusion"),
      block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.broadcast_in_dim %[[ARG:.*]], dims = [] : (tensor<f32>) -> tensor<16x32x8xf32>
)"));
}

TEST_F(XTileDialectTest, HloReduceIsLoweredToStableHloReduce) {
  constexpr absl::string_view kHloText = R"(
HloModule t

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

reduce_fusion {
  p0 = f32[150,160] parameter(0)
  const = f32[] constant(0.0)
  ROOT broadcast = f32[160] reduce(p0, const), dimensions={0}, to_apply=add
}

ENTRY e {
  p0 = f32[150,160] parameter(0)
  ROOT custom-call = f32[160] fusion(p0), kind=kCustom,
    calls=reduce_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16}};

  TF_EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("reduce_fusion"), block_level_parameters,
      R"(
CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
CHECK: %[[MASKED_INPUT:.*]] = xtile.mask {{.*}}
CHECK: %[[RES:.*]] = stablehlo.reduce(%[[MASKED_INPUT]] init: %[[INIT]]) applies stablehlo.add across dimensions = [0] : (tensor<256x16xf32>, tensor<f32>) -> tensor<16xf32>
)"));
}

TEST_F(XTileDialectTest, HloReshapeIsLoweredToStableHloReshape) {
  constexpr absl::string_view kHloText = R"(
HloModule t, is_scheduled=true

reshape_fusion {
  p0 = s32[150] parameter(0)
  ROOT reshape = s32[15, 10] reshape(p0)
}

ENTRY e {
  p0 = s32[150] parameter(0)
  ROOT custom-call = s32[15, 10] fusion(p0), kind=kCustom,
    calls=reshape_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{1, 16}};

  TF_EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("reshape_fusion"), block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.reshape %[[ARG:.*]] : (tensor<16xi32>) -> tensor<1x16xi32>
)"));
}

TEST_F(XTileDialectTest, HloDotIsLoweredToStableHloDot) {
  constexpr absl::string_view kHloText = R"(
HloModule t

flhs {
  ROOT flhs.p0 = f32[150,160] parameter(0)
}

frhs {
  ROOT frhs.p0 = f32[160,31] parameter(0)
}

dot_fusion {
  fdot.p0 = f32[150,160] parameter(0)
  fdot.p1 = f32[160,31] parameter(1)
  fdot.lhs = f32[150,160] fusion(fdot.p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "8"]}]
      }
    }
  }
  fdot.rhs = f32[160,31]{1,0} fusion(fdot.p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "8"]}]
      }
    }
  }

  ROOT dot = f32[150,31] dot(fdot.lhs, fdot.rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[150, 160] parameter(0)
  p1 = f32[160, 31] parameter(1)
  ROOT custom-call = f32[150,31] fusion(p0, p1), kind=kCustom,
    calls=dot_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{32, 8}};

  TF_EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("dot_fusion"), block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.dot_general %[[ARG0:.*]], %[[ARG1:.*]], contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x8xf32>) -> tensor<32x8xf32>
CHECK: %[[ADD_RES:.*]] = arith.addf %[[ARG2:.*]], %[[RES]] : tensor<32x8xf32>
)"));
}

TEST_F(XTileDialectTest, HloScaledDotIsLoweredToXTileDotScaled) {
  constexpr absl::string_view kHloText = R"(
HloModule m
flhs (p0: f8e5m2[128,128]) -> f8e5m2[128,128] {
  ROOT p0 = f8e5m2[128,128]{1,0} parameter(0)
}
frhs (p0: f8e5m2[128,256]) -> f8e5m2[128,256] {
  ROOT p0 = f8e5m2[128,256]{1,0} parameter(0)
}
flhs_scale (p0: f8e8m0fnu[128,4]) -> f8e8m0fnu[128,4] {
  ROOT p0 = f8e8m0fnu[128,4]{1,0} parameter(0)
}
frhs_scale (p0: f8e8m0fnu[4,256]) -> f8e8m0fnu[4,256] {
  ROOT p0 = f8e8m0fnu[4,256]{1,0} parameter(0)
}

triton_dot {
  lhs = f8e5m2[128,128] parameter(0)
  lhs1 = f8e5m2[128,128]{1,0} fusion(lhs),
    kind=kCustom,
    calls=flhs,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","128"]}],
          "num_warps":"4",
          "num_stages":"1",
          "num_ctas":"1",
        }
      }
    }
  rhs = f8e5m2[128,256] parameter(1)
  rhs1 = f8e5m2[128,256]{1,0} fusion(rhs),
    kind=kCustom,
    calls=frhs,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","256"]}],
          "num_warps":"4",
          "num_stages":"1",
          "num_ctas":"1",
        }
      }
    }
  lhs_scale = f8e8m0fnu[128,4] parameter(2)
  lhs_scale1 = f8e8m0fnu[128,4]{1,0} fusion(lhs_scale),
    kind=kCustom,
    calls=flhs_scale,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","128"]}],
          "num_warps":"4",
          "num_stages":"1",
          "num_ctas":"1",
        }
      }
    }
  rhs_scale = f8e8m0fnu[4,256] parameter(3)
  rhs_scale1 = f8e8m0fnu[4,256]{1,0} fusion(rhs_scale),
    kind=kCustom,
    calls=frhs_scale,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128", "256"]}],
          "num_warps":"4",
          "num_stages":"1",
          "num_ctas":"1",
        }
      }
    }
  ROOT _ = bf16[128,256]{1,0} scaled-dot(lhs1, rhs1, lhs_scale1, rhs_scale1),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0}
}

ENTRY e {
  lhs = f8e5m2[128,128]{1,0} parameter(0)
  rhs = f8e5m2[128,256]{1,0} parameter(1)
  lhs_scale = f8e8m0fnu[128,4]{1,0} parameter(2)
  rhs_scale = f8e8m0fnu[4,256]{1,0} parameter(3)
  ROOT _ = bf16[128,256]{1,0} fusion(lhs, rhs, lhs_scale, rhs_scale),
    kind=kCustom,
    calls=triton_dot,
    backend_config={
      "fusion_backend_config": {
        kind: "__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128", "256"]}],
          "num_warps":"4",
          "num_stages":"1",
          "num_ctas":"1"
        }
      }
    }
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  auto& debug_options = module->mutable_config().mutable_debug_options();
  debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(true);

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{128, 256}};

  TF_EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("triton_dot"), block_level_parameters,
      R"(
      CHECK: %[[DOT:.*]] = xtile.dot_scaled %[[LHS:.*]] scale %[[LHS_SCALE:.*]], %[[RHS:.*]] scale %[[RHS_SCALE:.*]] {fastMath = true} : tensor<128x128xf8E5M2>, tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
      CHECK: %[[RES:.*]] = arith.addf %{{.*}}, %[[DOT]] : tensor<128x256xf32>
      )"));
}

TEST_F(XTileDialectTest, HloAllReduceIsLoweredToStableHloAllReduce) {
  constexpr absl::string_view kHloText =
      R"(
      HloModule wrapped_module_all-reduce-start

      %apply_op {
        %x = f32[] parameter(0)
        %y = f32[] parameter(1)
        ROOT %apply_op = f32[] add(%x, %y)
      }

      %wrapped_all-reduce-start {
        %param = f32[65536]{0} parameter(0)
        ROOT %all-reduce-start = f32[65536]{0} all-reduce-start(%param), replica_groups={{0,1}}, to_apply=%apply_op
      }

      ENTRY %entry {
        %param = f32[65536]{0} parameter(0)
        ROOT %fusion = f32[65536]{0} fusion(%param), kind=kLoop, calls=%wrapped_all-reduce-start, backend_config={"fusion_backend_config":{"kind":"__triton_collective","block_level_fusion_config":{"num_warps":"16","output_tiles":[{"sizes":["4096"]}],"num_ctas":1,"num_stages":1,"is_tma_allowed":false,"is_warp_specialization_allowed":false}}}
      }
    )";

  // The HLO is not valid so we parse and return unverified. This is the same
  // HLO that gets generated in the collective_emitter_tests.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnUnverifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{4096}};

  TF_EXPECT_OK(CreateXTileIrAndFileCheck(
      *hlo_module->GetComputationWithName("wrapped_all-reduce-start"),
      block_level_parameters,
      R"(
CHECK: stablehlo.all_reduce
CHECK: stablehlo.add
)"));
}

TEST_F(XTileDialectTest, HloUnsignedIntIsLoweredToStableHloUnsignedInt) {
  constexpr absl::string_view kHloText = R"(
HloModule t, is_scheduled=true

add_fusion {
  p0 = u32[150] parameter(0)
  ROOT add = u32[150] add(p0, p0)
}

ENTRY e {
  p0 = u32[150] parameter(0)
  ROOT custom-call = u32[150] fusion(p0), kind=kCustom,
    calls=add_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16}};

  TF_EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("add_fusion"), block_level_parameters,
      R"(
CHECK: stablehlo.add{{.*}}: tensor<16xui32>
)"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
