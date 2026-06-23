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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/triton/xtile_test_base.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/xla.pb.h"

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

class XTileDialectTestParameterized
    : public XTileDialectTest,
      public ::testing::WithParamInterface<bool> {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_enable_tiling_propagation(
        GetParam());
    return debug_options;
  }
};

INSTANTIATE_TEST_SUITE_P(XTileDialectTestParameterized,
                         XTileDialectTestParameterized, testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "ExperimentalEmitter"
                                             : "LegacyEmitter";
                         });

TEST_P(XTileDialectTestParameterized,
       HloTransposeIsLoweredToStableHloTranspose) {
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 32}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("transpose_fusion"),
      block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.transpose %[[ARG:.*]], dims = [1, 0] : (tensor<32x16xf32>) -> tensor<16x32xf32>
)"));
}

TEST_P(XTileDialectTestParameterized, HloBitcastIsLoweredToTensorBitcast) {
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 32}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("bitcast_fusion"), block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = tensor.bitcast %[[ARG:.*]] : tensor<16x32xf32> to tensor<16x32xi32>
)"));
}

TEST_P(XTileDialectTestParameterized, HloIotaIsLoweredToStableHloIota) {
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("iota_fusion"), block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.iota dim = 0 : tensor<16xi32>
)"));
}

TEST_P(XTileDialectTestParameterized,
       HloBroadcastInDimIsLoweredToStableHloBroadcastInDim) {
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 32, 8}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("broadcast_in_dim_fusion"),
      block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.broadcast_in_dim %[[ARG:.*]], dims = [0, 1] : (tensor<16x32xf32>) -> tensor<16x32x8xf32>
)"));
}

TEST_P(XTileDialectTestParameterized,
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 32, 8}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("broadcast_in_dim_fusion"),
      block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.broadcast_in_dim %[[ARG:.*]], dims = [] : (tensor<f32>) -> tensor<16x32x8xf32>
)"));
}

TEST_P(XTileDialectTestParameterized, HloReduceIsLoweredToStableHloReduce) {
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("reduce_fusion"), block_level_parameters,
      R"(
CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
CHECK: %[[MASKED_INPUT:.*]] = xtile.mask {{.*}}
CHECK: %[[RES:.*]] = stablehlo.reduce(%[[MASKED_INPUT]] init: %[[INIT]]) applies stablehlo.add across dimensions = [0] : (tensor<256x16xf32>, tensor<f32>) -> tensor<16xf32>
)"));
}

TEST_P(XTileDialectTestParameterized, HloScanIsLoweredToXTileScan) {
  if (!GetParam()) {
    GTEST_SKIP() << "Skipping test for legacy emitter.";
  }

  constexpr absl::string_view kHloText = R"(
HloModule t

scan_computation {
  p_carry = f32[] parameter(0)
  p_input = f32[] parameter(1)
  add = f32[] add(p_carry, p_input)
  ROOT tuple = (f32[], f32[]) tuple(add, add)
}

scan_fusion {
  p0 = f32[1024] parameter(0)
  p1 = f32[] parameter(1)
  scan = (f32[1024], f32[]) scan(p0, p1), dimensions={0}, num_carries=1, is_associative=true, to_apply=scan_computation
  ROOT gte = f32[1024] get-tuple-element(scan), index=0
}

ENTRY e {
  p0 = f32[1024] parameter(0)
  p1 = f32[] parameter(1)
  ROOT custom-call = f32[1024] fusion(p0, p1), kind=kCustom,
    calls=scan_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{1024}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("scan_fusion"), block_level_parameters,
      R"(
// CHECK-LABEL: @xtile_dialect_fn
// CHECK:         %[[EXTRACT0:.*]] = xtile.extract %arg0[%c0] [1024] [1] : memref<1024xf32> -> tensor<1024xf32>
// CHECK:         %[[EXTRACT1:.*]] = xtile.extract %arg1[] [] [] : memref<f32> -> tensor<f32>
// CHECK:         %[[OUTPUT:.*]], %{{.*}} = xtile.scan(%[[EXTRACT0]]) inits(%[[EXTRACT1]])
// CHECK-SAME:        dimension = 0 {scan_dim_size = 1024 : i64}
// CHECK-SAME:        : (tensor<1024xf32>), (tensor<f32>) -> (tensor<1024xf32>), (tensor<1024xf32>) {
// CHECK:         ^bb0(%[[INPUT:.*]]: tensor<f32>, %[[CARRY:.*]]: tensor<f32>):
// CHECK:           %[[ADD:.*]] = stablehlo.add %[[INPUT]], %[[CARRY]] : tensor<f32>
// CHECK:           stablehlo.return %[[ADD]], %[[ADD]] : tensor<f32>, tensor<f32>
// CHECK:         xtile.insert %[[OUTPUT]] into %arg2
)"));
}

TEST_P(XTileDialectTestParameterized, HloReshapeIsLoweredToStableHloReshape) {
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{1, 16}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("reshape_fusion"), block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.reshape %[[ARG:.*]] : (tensor<16xi32>) -> tensor<1x16xi32>
)"));
}

TEST_P(XTileDialectTestParameterized, HloDotIsLoweredToStableHloDot) {
  constexpr absl::string_view kHloText = R"(
HloModule t

dot_fusion {
  fdot.p0 = f32[150,160] parameter(0)
  fdot.p1 = f32[160,31] parameter(1)
  ROOT dot = f32[150,31] dot(fdot.p0, fdot.p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    backend_config={sizes:[8]}
}

ENTRY e {
  p0 = f32[150, 160] parameter(0)
  p1 = f32[160, 31] parameter(1)
  ROOT custom-call = f32[150,31] fusion(p0, p1), kind=kCustom,
    calls=dot_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{32, 8}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("dot_fusion"), block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.dot_general %[[ARG0:.*]], %[[ARG1:.*]], contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x8xf32>) -> tensor<32x8xf32>
CHECK: %[[ADD_RES:.*]] = arith.addf %[[ARG2:.*]], %[[RES]] : tensor<32x8xf32>
)"));
}

TEST_P(XTileDialectTestParameterized, HloScaledDotIsLoweredToXTileDotScaled) {
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_dot {
  lhs = f8e5m2[128,128] parameter(0)
  rhs = f8e5m2[128,256] parameter(1)
  lhs_scale = f8e8m0fnu[128,4] parameter(2)
  rhs_scale = f8e8m0fnu[4,256] parameter(3)
  ROOT _ = bf16[128,256]{1,0} scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    backend_config={sizes:[128]}
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  auto& debug_options = module->mutable_config().mutable_debug_options();
  debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(true);

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{128, 256}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("triton_dot"), block_level_parameters,
      R"(
      CHECK: %[[DOT:.*]] = xtile.dot_scaled %[[LHS:.*]] scale %[[LHS_SCALE:.*]], %[[RHS:.*]] scale %[[RHS_SCALE:.*]] {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, fastMath = true} : tensor<128x128xf8E5M2>, tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
      CHECK: %[[RES:.*]] = arith.addf %{{.*}}, %[[DOT]] : tensor<128x256xf32>
      )"));
}

TEST_P(XTileDialectTestParameterized,
       HloAllReduceIsLoweredToStableHloAllReduce) {
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnUnverifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{4096}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *hlo_module->GetComputationWithName("wrapped_all-reduce-start"),
      block_level_parameters,
      R"(
CHECK: stablehlo.all_reduce
CHECK: stablehlo.add
)"));
}

TEST_P(XTileDialectTestParameterized,
       HloUnsignedIntIsLoweredToStableHloUnsignedInt) {
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16}};

  EXPECT_OK(CreateXTileIrAndFileCheck(
      *module->GetComputationWithName("add_fusion"), block_level_parameters,
      R"(
CHECK: stablehlo.add{{.*}}: tensor<16xui32>
)"));
}

TEST_F(XTileDialectTest, HloAllGatherDotLowering) {
  constexpr absl::string_view kHloText = R"(
    HloModule nested_all_gather_dot

    %ag_dot {
      %param0 = f32[128,128]{1,0} parameter(0)
      %param1 = f32[128,128]{1,0} parameter(1)
      %all-gather1 = f32[256,128]{1,0} all-gather(%param0),
        replica_groups={{0,1}}, dimensions={0}
      %all-gather2 = f32[512,128]{1,0} all-gather(%all-gather1),
        replica_groups={{0,1}}, dimensions={0}
      ROOT %dot = f32[512,128]{1,0} dot(%all-gather2, %param1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0},
        backend_config={sizes:[128]}
    }

    ENTRY %entry {
      %param0 = f32[128,128]{1,0} parameter(0)
      %param1 = f32[128,128]{1,0} parameter(1)
      ROOT %fusion = f32[512,128]{1,0} fusion(%param0, %param1),
        kind=kLoop, calls=%ag_dot,
        backend_config={
          "fusion_backend_config": {
            "kind": "__triton_collective",
            "block_level_fusion_config": {
              "num_warps": "4",
              "output_tiles": [{"sizes": ["128", "128"]}],
              "num_ctas": 1,
              "num_stages": 1,
              "is_tma_allowed": false,
              "is_warp_specialization_allowed": false
            }
          }
        }
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHloText));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_enable_tiling_propagation(true);

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{128, 128}};

  EXPECT_OK(CreateXTileIrAndFileCheck(*module->GetComputationWithName("ag_dot"),
                                      block_level_parameters, R"(
    CHECK: xtile.entry_func @xtile_dialect_fn(%arg0: memref<2xi64>
    CHECK: %[[SELECT1:.*]] = xtile.select_buffer %arg0[%{{.*}}]
    CHECK-SAME: : memref<2xi64> -> memref<2xi64>
    CHECK: %[[SELECT2:.*]] = xtile.select_buffer %[[SELECT1]][%{{.*}}]
    CHECK-SAME: : memref<2xi64> -> memref<128x128xf32>
    CHECK: %[[LHS_TILE:.*]] = xtile.extract %[[SELECT2]]
    CHECK: %[[RHS_TILE:.*]] = xtile.extract %arg1
    CHECK: stablehlo.dot_general %[[LHS_TILE]], %[[RHS_TILE]]
    )"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
