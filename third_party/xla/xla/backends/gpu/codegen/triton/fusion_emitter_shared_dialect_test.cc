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
#include "xla/backends/gpu/codegen/triton/test_utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/tests/hlo_test_base_with_symbolic_expr_context.h"
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

using XTileDialectTest = HloTestBaseWithSymbolicExprContext;

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
      this, *module->GetComputationWithName("transpose_fusion"),
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
      this, *module->GetComputationWithName("bitcast_fusion"),
      block_level_parameters,
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
      this, *module->GetComputationWithName("iota_fusion"),
      block_level_parameters,
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
      this, *module->GetComputationWithName("broadcast_in_dim_fusion"),
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
      this, *module->GetComputationWithName("broadcast_in_dim_fusion"),
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
      this, *module->GetComputationWithName("reduce_fusion"),
      block_level_parameters,
      R"(
CHECK: %[[REDUCE_INPUT:.*]] = arith.select {{.*}}
CHECK: %[[INIT_VALUE_TO_TENSOR:.*]] = xtile.to_tensor %{{.*}} : f32
CHECK: %[[RES:.*]] = stablehlo.reduce(%[[REDUCE_INPUT]] init: %[[INIT_VALUE_TO_TENSOR]]) across dimensions = [0] : (tensor<256x16xf32>, tensor<f32>) -> tensor<16xf32>
CHECK: reducer(%[[ARG_0:.*]]: tensor<f32>, %[[ARG_1:.*]]: tensor<f32>)  {
CHECK:   %[[SUM:.*]] = arith.addf %[[ARG_0]], %[[ARG_1]] : tensor<f32>
CHECK:   stablehlo.return %[[SUM]] : tensor<f32>
CHECK: }
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
      this, *module->GetComputationWithName("reshape_fusion"),
      block_level_parameters,
      R"(
CHECK: %[[RES:.*]] = stablehlo.reshape %[[ARG:.*]] : (tensor<16xi32>) -> tensor<1x16xi32>
)"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
