/* Copyright 2023 The OpenXLA Authors.

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

#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "Eigen/Core"
#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/codegen/triton/test_utils.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

constexpr ErrorSpec kExactMatch{/*aabs=*/0, /*arel=*/0};

class TritonEmitterTest : public GpuCodegenTest {
 public:
  const stream_executor::GpuComputeCapability& GpuComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability();
  }
};

// TODO(bchetioui): turn this into a general binary elementwise test.
TEST_F(TritonEmitterTest, MinimumIsEmittedCorrectly) {
  constexpr absl::string_view kHloText = R"(
computation {
  p0 = f32[8,4] parameter(0)
  p1 = f32[8,4] parameter(1)
  ROOT minimum = f32[8,4] minimum(p0, p1)
}

ENTRY entry_computation {
  p0 = f32[8,4] parameter(0)
  p1 = f32[8,4] parameter(1)
  ROOT fusion = f32[8,4] fusion(p0, p1), kind=kCustom,
    calls=computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1", "4"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, ReductionOnMinormostAxisIsEmittedCorrectly) {
  constexpr absl::string_view kHloText = R"(
HloModule m

region {
  param_0.1 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(param_0.1, param_1)
}

fused_computation {
  param_0.2 = f32[8,4] parameter(0)
  constant = f32[] constant(0)
  ROOT reduce = f32[8] reduce(param_0.2, constant), dimensions={1},
    to_apply=region
}

ENTRY entry_computation {
  param_0.3 = f32[8,4] parameter(0)
  ROOT fusion = f32[8] fusion(param_0.3), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["4"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "fused_computation", R"(
CHECK:  "tt.reduce"(%[[LOAD:.*]]) <{axis = 1 : i32}>
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest,
       ReductionOnMinormostAxisWithExtraOutputIsEmittedCorrectly) {
  constexpr absl::string_view kHloText = R"(
HloModule m

region {
  param_0.1 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0.1, param_1)
}

fused_computation {
  param_0.2 = f32[128,512] parameter(0)
  abs = f32[128,512] abs(param_0.2)
  constant = f32[] constant(-inf)
  reduce = f32[128] reduce(abs, constant), dimensions={1}, to_apply=region
  ROOT tuple = (f32[128], f32[128,512]) tuple(reduce, abs)
}

ENTRY entry_computation {
  param_0.3 = f32[128,512] parameter(0)
  ROOT fusion = (f32[128], f32[128,512]) fusion(param_0.3), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64"]},{"sizes":["64","512"]}],
          "num_warps":"2",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "fused_computation", R"(
CHECK-COUNT-1:  triton_xla.extract
CHECK:  %[[ABS:.*]] = math.absf
CHECK: %[[REDUCE:.*]] = "tt.reduce"(%[[ABS:.*]]) <{axis = 1 : i32}>
CHECK:  triton_xla.insert %[[REDUCE]] {{.*}} : tensor<64xf32> into tensor<128xf32>
CHECK:  triton_xla.insert %[[ABS]] {{.*}} : tensor<64x512xf32> into tensor<128x512xf32>
)"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, ReductionToScalarWithExtraOutputIsEmittedCorrectly) {
  constexpr absl::string_view kHloText = R"(
HloModule m

region {
  param_0.1 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0.1, param_1)
}

fused_computation {
  param_0.2 = f32[512] parameter(0)
  abs = f32[512] abs(param_0.2)
  constant = f32[] constant(-inf)
  reduce = f32[] reduce(abs, constant), dimensions={0}, to_apply=region
  ROOT tuple = (f32[], f32[512]) tuple(reduce, abs)
}

ENTRY entry_computation {
  param_0.3 = f32[512] parameter(0)
  ROOT fusion = (f32[], f32[512]) fusion(param_0.3), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":[]},{"sizes":["512"]}],
          "num_warps":"2",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "fused_computation", R"(
CHECK-COUNT-1:  triton_xla.extract
CHECK:  %[[ABS:.*]] = math.absf
CHECK: %[[REDUCE:.*]] = "tt.reduce"(%[[ABS:.*]]) <{axis = 0 : i32}>
CHECK:  tensor.insert %[[REDUCE]] {{.*}} : tensor<f32>
CHECK:  triton_xla.insert %[[ABS]] {{.*}} : tensor<512xf32> into tensor<512xf32>
)"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest,
       SliceWithTilingThatNeedsPaddingHasBoundaryCheckForBothRoots) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0.1 = f32[64] parameter(0)
  abs = f32[64] abs(param_0.1)
  slice = f32[63] slice(abs), slice={[0:63]}
  negate = f32[63] negate(slice)
  ROOT tuple = (f32[63], f32[63]) tuple(negate, slice)
}

ENTRY entry_computation {
  param_0.2 = f32[64] parameter(0)
  ROOT fusion = (f32[63], f32[63]) fusion(param_0.2), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32"]},{"sizes":["32"]}],
          "num_warps":"2",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, SliceWithExtraOutputThatCanReuseTileDueToPadding) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0.1 = f32[64] parameter(0)
  abs = f32[64] abs(param_0.1)
  slice = f32[63] slice(abs), slice={[0:63]}
  negate = f32[63] negate(slice)
  ROOT tuple = (f32[63], f32[64]) tuple(negate, abs)
}

ENTRY entry_computation {
  param_0.2 = f32[64] parameter(0)
  ROOT fusion = (f32[63], f32[64]) fusion(param_0.2), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32"]},{"sizes":["32"]}],
          "num_warps":"2",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, BitcastReduceWithStride1Tiling) {
  constexpr absl::string_view kHloText = R"(
HloModule m

region {
  param_0.1 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT add = f32[] add(param_0.1, param_1)
}

fused_computation {
  param_0.2 = f32[64] parameter(0)
  abs = f32[64] abs(param_0.2)
  bitcast = f32[4,4,4] bitcast(abs)
  constant = f32[] constant(0)
  reduce = f32[4,4] reduce(bitcast, constant), dimensions={1}, to_apply=region
  ROOT tuple = (f32[4,4], f32[64]) tuple(reduce, abs)
}

ENTRY entry_computation {
  param_0.3 = f32[64] parameter(0)
  ROOT fusion = (f32[4,4], f32[64]) fusion(param_0.3), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1", "4"]},{"sizes":["16"]}],
          "num_warps":"2",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "fused_computation", R"(
CHECK-COUNT-1:  triton_xla.extract
CHECK: tt.reduce
CHECK-COUNT-2:  triton_xla.insert
)"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, BitcastReduceWithStride4Tiling) {
  constexpr absl::string_view kHloText = R"(
HloModule m

region {
  param_0.1 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT add = f32[] add(param_0.1, param_1)
}

fused_computation {
  param_0.2 = f32[64] parameter(0)
  abs = f32[64] abs(param_0.2)
  bitcast = f32[4,4,4] bitcast(abs)
  constant = f32[] constant(0)
  reduce = f32[4,4] reduce(bitcast, constant), dimensions={1}, to_apply=region
  ROOT tuple = (f32[4,4], f32[64]) tuple(reduce, abs)
}

ENTRY entry_computation {
  param_0.3 = f32[64] parameter(0)
  ROOT fusion = (f32[4,4], f32[64]) fusion(param_0.3), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1", "1"]},{"sizes":["4"]}],
          "num_warps":"2",
          "num_ctas":"1",
          "num_stages":"1"}}}

})";
  auto status =
      CreateTritonIrAndFileCheck(this, kHloText, "fused_computation", "");
  EXPECT_THAT(
      status,
      tsl::testing::StatusIs(
          tsl::error::UNIMPLEMENTED,
          ::testing::HasSubstr("Unsupported case of multi-output fusion")));
}

TEST_F(TritonEmitterTest, ReductionOnIntermediateAxisIsEmittedCorrectly) {
  constexpr absl::string_view kHloText = R"(
HloModule t
maximum {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(Arg_0, Arg_1)
}

triton_reduction_computation {
  parameter_0 = f32[5,5,5,5,3] parameter(0)
  constant_0 = f32[] constant(0)
  ROOT reduction = f32[5,5,5,3] reduce(parameter_0, constant_0), dimensions={2}, to_apply=maximum
}

ENTRY main {
  param_0 = f32[5,5,5,5,3] parameter(0)
  ROOT triton_reduction = f32[5,5,5,3] fusion(param_0), kind=kCustom,
    calls=triton_reduction_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4", "2", "5", "1"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          "triton_reduction_computation", R"(
CHECK:  tt.make_range
CHECK-COUNT-4:  tt.expand_dims
CHECK:  "tt.reduce"(%[[SELECT:.*]]) <{axis = 2 : i32}>
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, TestReductionWithTileSizeLargerThanSourceTensor) {
  constexpr absl::string_view kHloText = R"(
HloModule t
maximum {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(Arg_0, Arg_1)
}

triton_reduction_computation {
  parameter_0 = f32[5,3] parameter(0)
  constant_0 = f32[] constant(0)
  ROOT reduce = f32[3] reduce(parameter_0, constant_0), dimensions={0}, to_apply=maximum
}

ENTRY main {
  param_0 = f32[5,3] parameter(0)
  ROOT triton_reduction = f32[3] fusion(param_0), kind=kCustom,
    calls=triton_reduction_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["3"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          "triton_reduction_computation", R"(
; Make sure input reduction tile is padded with a neutral value.
CHECK:  %[[LOAD:.*]] = triton_xla.extract
CHECK:  %[[RANGE:.*]] = tt.make_range
CHECK:  %[[EXPAND:.*]] = tt.expand_dims %[[RANGE]]
CHECK:  %[[BROADCAST:.*]] = tt.broadcast %[[EXPAND]]
CHECK:  %[[CMPI:.*]] = arith.cmpi slt, %[[BROADCAST]]
CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[LOAD]]
CHECK:  "tt.reduce"(%[[SELECT]]) <{axis = 0 : i32}>
CHECK:  ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32):
CHECK:    %[[MAXIMUM:.*]] = arith.maximumf %[[ARG2]], %[[ARG3]] : f32
CHECK:    tt.reduce.return %[[MAXIMUM]] : f32
CHECK:  })
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonEmitterTest, TestGenericEmitterWithSoftMaxSingleParameter) {
  constexpr absl::string_view kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0),
    kind=kCustom, calls=triton_softmax_computation, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1", "128"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          "triton_softmax_computation", R"(
CHECK:        func.func @triton_fn(%[[P0:.*]]: {{.*}}, %[[P1:.*]]: {{.*}})
CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
CHECK-DAG:        %[[C1:.*]] = arith.constant 1 : index
CHECK-DAG:        %[[PID:.*]] = tt.get_program_id x : i32
CHECK-DAG:        %[[PID_I64:.*]] = arith.extsi %[[PID]] : i32 to i64
CHECK-DAG:        %[[PID_INDEX:.*]] = arith.index_castui %[[PID_I64]] : i64 to index
CHECK-DAG:        triton_xla.tile %[[P0]][%[[PID_INDEX]], %[[C0]]][%[[C1]], %[[C1]]] {layout = array<i64:1, 0>} : !triton_xla.tiled_tensor<1x128|125x127xf32>
CHECK-NEXT:       triton_xla.extract
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG2:[^:]*]]: f32, %[[ARG3:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG2]], %[[ARG3]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<1x128xf32>) -> tensor<1xf32>
CHECK:            arith.mulf
CHECK-SAME:       tensor<1x128xf32>
CHECK:            triton_xla.tile %[[P1]][%[[PID_INDEX]], %[[C0]]][%[[C1]], %[[C1]]] {layout = array<i64:1, 0>} : !triton_xla.tiled_tensor<1x128|125x127xf32>
CHECK-NEXT:       triton_xla.insert
CHECK:            return
CHECK:        }
)"));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonEmitterTest, TestGenericEmitterWithMultipleParameters) {
  constexpr absl::string_view kHloText = R"(
HloModule t

add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[127]{0} parameter(1)
  broadcast_0 = f32[125,127]{1,0} broadcast(param_1), dimensions={1}
  multiply_0 = f32[125,127]{1,0} multiply(param_0, broadcast_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[127]{0} parameter(1)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0, param_1),
    kind=kCustom, calls=triton_softmax_computation,
    backend_config={"fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1", "128"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          "triton_softmax_computation", R"(
CHECK:         func.func @triton_fn(
CHECK-SAME:                      %[[P0:[A-Za-z0-9_]*]]: tensor<125x127xf32>
CHECK-SAME:                      %[[P1:[A-Za-z0-9_]*]]: tensor<127xf32>
CHECK-SAME:                      %[[P2:[A-Za-z0-9_]*]]: tensor<125x127xf32>
CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
CHECK-DAG:        %[[C1:.*]] = arith.constant 1 : index
CHECK-DAG:        %[[PID:.*]] = tt.get_program_id x : i32
CHECK-DAG:        %[[PID_I64:.*]] = arith.extsi %[[PID]] : i32 to i64
CHECK-DAG:        %[[PID_INDEX:.*]] = arith.index_castui %[[PID_I64]] : i64 to index
CHECK-DAG:        %[[TILE_0:.*]] = triton_xla.tile %[[P0]][%[[PID_INDEX]], %[[C0]]][%[[C1]], %[[C1]]] {layout = array<i64:1, 0>} : !triton_xla.tiled_tensor<1x128|125x127xf32>
CHECK-DAG:        triton_xla.extract %[[TILE_0]][%[[C0]], %[[C0]]] : tensor<125x127xf32> to tensor<1x128xf32>
CHECK-DAG:        %[[TILE_1:.*]] = triton_xla.tile %[[P1]][%[[C0]]][%[[C1]]] {layout = array<i64:0>} : !triton_xla.tiled_tensor<128|127xf32>
CHECK-DAG:        triton_xla.extract %[[TILE_1]][%[[C0]]] : tensor<127xf32> to tensor<128xf32>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG3:[^:]*]]: f32, %[[ARG4:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG3]], %[[ARG4]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<1x128xf32>) -> tensor<1xf32>
CHECK:            arith.mulf

CHECK-DAG:        %[[TILE_2:.*]] = triton_xla.tile %[[P2]][%[[PID_INDEX]], %[[C0]]][%[[C1]], %[[C1]]] {layout = array<i64:1, 0>} : !triton_xla.tiled_tensor<1x128|125x127xf32>
CHECK-DAG:        triton_xla.insert {{.*}} into %[[TILE_2]][%[[C0]], %[[C0]]] : tensor<1x128xf32> into tensor<125x127xf32>
)"));
}

TEST_F(TritonEmitterTest, TestGenericEmitterWithMultipleTiledDimensions) {
  constexpr absl::string_view kHloText = R"(
HloModule t

max {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT max = f32[] maximum(Arg_0, Arg_1)
}

triton_softmax_computation {
  param_0 = f32[10,125,127]{2,1,0} parameter(0)
  param_1 = f32[127]{0} parameter(1)
  param_2 = f32[10,125]{1,0} parameter(2)
  broadcast_0 = f32[10,125,127]{2,1,0} broadcast(param_1), dimensions={2}
  multiply_0 = f32[10,125,127]{2,1,0} multiply(param_0, broadcast_0)
  broadcast_1 = f32[10,125,127]{2,1,0} broadcast(param_2), dimensions={0,1}
  multiply_1 = f32[10,125,127]{2,1,0} multiply(multiply_0, broadcast_1)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[10,125]{1,0} reduce(multiply_1, constant_0), dimensions={2}, to_apply=max
  broadcast_4 = f32[10,125,127]{2,1,0} broadcast(reduce_0), dimensions={0,1}
  ROOT multiply = f32[10,125,127]{2,1,0} multiply(multiply_1, broadcast_4)
}

ENTRY main {
  param_0 = f32[10,125,127]{2,1,0} parameter(0)
  param_1 = f32[127]{0} parameter(1)
  param_2 = f32[10,125]{1,0} parameter(2)
  ROOT triton_softmax = f32[10,125,127]{2,1,0} fusion(param_0, param_1, param_2),
    kind=kCustom, calls=triton_softmax_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes": ["1", "1", "127"]}],
          "num_warps": "1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";

  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = #xla.indexing_map<"(d0) -> (d0 floordiv 125), domain: d0 in [0, 1249]">
CHECK:        #[[MAP1:.*]] = #xla.indexing_map<"(d0) -> (d0 mod 125), domain: d0 in [0, 1249]">
CHECK:        func.func @triton_fn(%[[P0:.*]]: {{.*}}, %[[P1:.*]]: {{.*}}, %[[P2:.*]]: {{.*}}, %[[P3:.*]]: {{.*}})
CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
CHECK-DAG:        %[[C1:.*]] = arith.constant 1 : index
CHECK-DAG:        %[[PID:.*]] = tt.get_program_id x : i32
CHECK-DAG:        %[[PID_I64:.*]] = arith.extsi %[[PID]] : i32 to i64
CHECK-DAG:        %[[PID_INDEX:.*]] = arith.index_castui %[[PID_I64]] : i64 to index
CHECK-DAG:        %[[ROW_INDEX:.*]] = xla.apply_indexing #[[MAP]](%[[PID_INDEX]]
CHECK-DAG:        %[[COL_INDEX:.*]] = xla.apply_indexing #[[MAP1]](%[[PID_INDEX]]

CHECK-DAG:        triton_xla.tile %[[P0]][%[[ROW_INDEX]], %[[COL_INDEX]], %[[C0]]][%[[C1]], %[[C1]], %[[C1]]] {layout = array<i64:2, 1, 0>} : !triton_xla.tiled_tensor<1x1x128|10x125x127xf32>
CHECK-NEXT:       triton_xla.extract {{.*}} : tensor<10x125x127xf32> to tensor<1x1x128xf32>
CHECK-DAG:        triton_xla.tile %[[P1]][%[[C0]]][%[[C1]]] {layout = array<i64:0>} : !triton_xla.tiled_tensor<128|127xf32>
CHECK-NEXT:       triton_xla.extract {{.*}} : tensor<127xf32> to tensor<128xf32>

CHECK-DAG:        triton_xla.tile %[[P2]][%[[ROW_INDEX]], %[[COL_INDEX]]][%[[C1]], %[[C1]]] {layout = array<i64:1, 0>} : !triton_xla.tiled_tensor<1x1|10x125xf32>
CHECK-NEXT:       triton_xla.extract {{.*}} : tensor<10x125xf32> to tensor<1x1xf32>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG4:[^:]*]]: f32, %[[ARG5:[^:]*]]: f32):
CHECK-NEXT:           %[[MAX:.*]] = arith.maximumf %[[ARG4]], %[[ARG5]] : f32
CHECK-NEXT:           tt.reduce.return %[[MAX]] : f32
CHECK-NEXT:       }) : (tensor<1x1x128xf32>) -> tensor<1x1xf32>

CHECK-DAG:        triton_xla.tile %[[P3]][%[[ROW_INDEX]], %[[COL_INDEX]], %[[C0]]][%[[C1]], %[[C1]], %[[C1]]] {layout = array<i64:2, 1, 0>} : !triton_xla.tiled_tensor<1x1x128|10x125x127xf32>
CHECK-NEXT:       triton_xla.insert {{.*}} : tensor<1x1x128xf32> into tensor<10x125x127xf32>
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(
    TritonEmitterTest,
    DiamondWithAdditionalDiamondParameterBroadcastedAlongReductionDimProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  constexpr absl::string_view kHloText = R"(
HloModule h1

max_computation {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT _ = f32[] maximum(x, y)
}

triton_softmax_computation {
  parameter_1 = f32[32]{0} parameter(1)
  broadcast_1 = f32[32,16]{1,0} broadcast(parameter_1), dimensions={0}
  parameter_0 = f32[32,16]{1,0} parameter(0)
  add_0 = f32[32,16]{1,0} add(broadcast_1, parameter_0)
  c = f32[] constant(0)
  reduce_0 = f32[32]{0} reduce(parameter_0, c), dimensions={1}, to_apply=max_computation
  broadcast_0 = f32[32,16]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT _ = f32[32,16]{1,0} add(add_0, broadcast_0)
}

ENTRY main {
  parameter_1 = f32[32]{0} parameter(1)
  parameter_0 = f32[32,16]{1,0} parameter(0)
  ROOT _ = f32[32,16]{1,0} fusion(parameter_0, parameter_1), kind=kCustom,
    calls=triton_softmax_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","16"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, NestedReducerFusionGetsCodegenedCorrectly) {
  if (!SupportsBF16(GpuComputeCapability())) {
    GTEST_SKIP() << "BF16 not supported.";
  }

  constexpr absl::string_view kHloText = R"(
HloModule softmax

fused_convert {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  convert0 = bf16[] convert(p0)
  convert1 = bf16[] convert(p1)
  add = bf16[] add(convert0, convert1)
  ROOT output = f32[] convert(add)
}

add_computation {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT fusion = f32[] fusion(p0, p1), kind=kLoop, calls=fused_convert
}

triton_softmax_computation {
  p0 = pred[10,128]{1,0} parameter(0)
  p0_f32 = f32[10,128]{1,0} convert(p0)
  zero = f32[] constant(0)
  reduce = f32[10]{0} reduce(p0_f32, zero), dimensions={1}, to_apply=add_computation
  broadcast = f32[10,128]{1,0} broadcast(reduce), dimensions={0}
  ROOT add = f32[10,128]{1,0} add(p0_f32, broadcast)
}

ENTRY main {
  p0 = pred[10,128]{1,0} parameter(0)
  ROOT softmax = f32[10,128] fusion(p0), kind=kCustom,
    calls=triton_softmax_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","128"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0,
                                                           /*arel=*/0}));
}

TEST_F(
    TritonEmitterTest,
    DiamondWithAdditionalDiamondParameterBroadcastedAlongBatchDimProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  constexpr absl::string_view kHloText = R"(
HloModule h1

max_computation {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT _ = f32[] maximum(x, y)
}

triton_softmax_computation {
  parameter_1 = f32[32]{0} parameter(1)
  broadcast_1 = f32[16,32]{1,0} broadcast(parameter_1), dimensions={1}
  parameter_0 = f32[16,32]{1,0} parameter(0)
  add_0 = f32[16,32]{1,0} add(broadcast_1, parameter_0)
  c = f32[] constant(0)
  reduce_0 = f32[16]{0} reduce(parameter_0, c), dimensions={1}, to_apply=max_computation
  broadcast_0 = f32[16,32]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT _ = f32[16,32]{1,0} add(add_0, broadcast_0)
}

ENTRY main {
  parameter_0 = f32[16,32]{1,0} parameter(0)
  parameter_1 = f32[32]{0} parameter(1)
  ROOT _ = f32[16,32]{1,0} fusion(parameter_0,parameter_1), kind=kCustom,
    calls=triton_softmax_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","32"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(
    TritonEmitterTest,
    DiamondWithAdditionalSplatDiamondScalarParameterProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  constexpr absl::string_view kHloText = R"(
HloModule h1

max_computation {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT _ = f32[] maximum(x,y)
}

triton_softmax_computation {
  parameter_1 = f32[] parameter(1)
  broadcast_1 = f32[64,32,16]{2,1,0} broadcast(parameter_1), dimensions={}
  parameter_0 = f32[64,32,16]{2,1,0} parameter(0)
  add_0 = f32[64,32,16]{2,1,0} add(broadcast_1, parameter_0)
  c = f32[] constant(0)
  reduce_0 = f32[64,32]{1,0} reduce(parameter_0, c), dimensions={2}, to_apply=max_computation
  broadcast_0 = f32[64,32,16]{2,1,0} broadcast(reduce_0), dimensions={0,1}
  ROOT _ = f32[64,32,16]{2,1,0} add(add_0, broadcast_0)
}

ENTRY main {
  parameter_1 = f32[64,32,16]{2,1,0} parameter(1)
  parameter_0 = f32[] parameter(0)
  ROOT _ = f32[64,32,16]{2,1,0} fusion(parameter_1, parameter_0), kind=kCustom,
    calls=triton_softmax_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","1","16"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  TF_ASSERT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          "triton_softmax_computation", R"(
// CHECK:         #xla.indexing_map<"(d0) -> (d0 floordiv 32), domain: d0 in [0, 2047]">
// CHECK:         #xla.indexing_map<"(d0) -> (d0 mod 32), domain: d0 in [0, 2047]">
// CHECK-LABEL:   func.func @triton_fn(
// CHECK-SAME:                       %[[P0:[A-Za-z0-9_]*]]: tensor<64x32x16xf32>
// CHECK-SAME:                       %[[P1:[A-Za-z0-9_]*]]: tensor<f32>
// CHECK-SAME:                       %[[P2:[A-Za-z0-9_]*]]: tensor<64x32x16xf32>
// CHECK-DAG:       tensor.extract {{.*}} : tensor<f32>
// CHECK-DAG:       triton_xla.extract {{.*}} : tensor<64x32x16xf32> to tensor<1x1x16xf32>
// CHECK:           triton_xla.insert {{.*}} : tensor<1x1x16xf32> into tensor<64x32x16xf32>
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(
    TritonEmitterTest,
    DiamondWithAdditionalBroadcastOf1DParameterAlongNonReductionDimensionsProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  constexpr absl::string_view kHloText = R"(
HloModule h1

max_computation {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT _ = f32[] maximum(x,y)
}

triton_softmax_computation {
  parameter_1 = f32[16]{0} parameter(1)
  broadcast_1 = f32[64,32,16]{2,1,0} broadcast(f32[16]{0} parameter_1), dimensions={2}
  parameter_0 = f32[64,32,16]{2,1,0} parameter(0)
  add_0 = f32[64,32,16]{2,1,0} add(f32[64,32,16]{2,1,0} broadcast_1, f32[64,32,16]{2,1,0} parameter_0)
  c = f32[] constant(0)
  reduce_0 = f32[64,32]{1,0} reduce(f32[64,32,16]{2,1,0} parameter_0, f32[] c), dimensions={2}, to_apply=max_computation
  broadcast_0 = f32[64,32,16]{2,1,0} broadcast(f32[64,32]{1,0} reduce_0), dimensions={0,1}
  ROOT _ = f32[64,32,16]{2,1,0} add(f32[64,32,16]{2,1,0} add_0, f32[64,32,16]{2,1,0} broadcast_0)
}

ENTRY main {
  parameter_1 = f32[64,32,16]{2,1,0} parameter(1)
  parameter_0 = f32[16]{0} parameter(0)
  ROOT _ = f32[64,32,16]{2,1,0} fusion(f32[64,32,16]{2,1,0} parameter_1, f32[16]{0} parameter_0), kind=kCustom,
    calls=triton_softmax_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","1","16"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
}
)";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonEmitterTest, EmitterFailsIfComputeCapabilityIsBelowAmpere) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  ROOT add = f32[10,10] add(p0, p1)
}

ENTRY entry {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  ROOT r = f32[10,10] fusion(p0, p1),
    kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","1"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloText));
  const HloFusionInstruction* triton_fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(
      TritonWrapper("test_fn", triton_fusion,
                    se::CudaComputeCapability{se::CudaComputeCapability::kVolta,
                                              /*minor=*/0},
                    dev_info, BlockLevelParameters(), &llvm_module,
                    mlir_context),
      tsl::testing::StatusIs(
          absl::StatusCode::kFailedPrecondition,
          ::testing::HasSubstr("Triton support is only enabled for Ampere GPUs "
                               "(compute capability 8.0) and up, but got")));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonEmitterTest,
       EmitterFailsIfFusionBackendConfigDoesNotSatisfyConstraints) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

max_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0, param_1)
}

fused_computation {
  param_0 = f32[8192,50304] parameter(0)
  constant = f32[] constant(-inf)
  reduce = f32[8192] reduce(param_0, constant), dimensions={1}, to_apply=max_computation
  broadcast = f32[8192,50304] broadcast(reduce), dimensions={0}
  ROOT subtract = f32[8192,50304] subtract(param_0, broadcast)
}

ENTRY entry_computation {
  param_0 = f32[8192,50304] parameter(0)
  ROOT fusion = f32[8192,50304] fusion(param_0),
    kind=kCustom, calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1024","1"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})"));
  const HloFusionInstruction* triton_fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto compute_capability = se::CudaComputeCapability{
      se::CudaComputeCapability::kHopper, /*minor=*/0};
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(compute_capability);
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{1024, 1}};
  block_level_parameters.num_warps = 1;

  // Because of reduce, we need to load full rows from param_0 and the load tile
  // will be 1024 * 65536 = 67108864 elements, that is larger than the limit of
  // 1048576.
  EXPECT_THAT(
      TritonWrapper("test_fn", triton_fusion, compute_capability, dev_info,
                    block_level_parameters, &llvm_module, mlir_context),
      tsl::testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr(
              "Tile parameters 1024, 1 do not satisfy constraints.")));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should b
// moved to deviceless test file.
TEST_F(TritonEmitterTest, TestGenericEmitterReductionFusion) {
  constexpr absl::string_view kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_reduction_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  parameter_1 = f32[125]{0} parameter(1)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  ROOT multiply = f32[125]{0} multiply(parameter_1, reduce_0)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[125]{0} parameter(1)
  ROOT triton_reduction = f32[125]{0} fusion(param_0, param_1),
    kind=kCustom, calls=triton_reduction_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";

  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          "triton_reduction_computation", R"(
CHECK:        func.func @triton_fn(%[[P0:[A-Za-z0-9_]*]]: tensor<125x127xf32>
CHECK-SAME:                      %[[P1:[A-Za-z0-9_]*]]: tensor<125xf32>
CHECK-SAME:                      %[[P2:[A-Za-z0-9_]*]]: tensor<125xf32>
CHECK-DAG:        triton_xla.extract {{.*}} : tensor<125xf32> to tensor<1xf32>
CHECK-DAG:        triton_xla.extract {{.*}} : tensor<125x127xf32> to tensor<1x128xf32>
CHECK:            tt.reduce
CHECK:              (tensor<1x128xf32>) -> tensor<1xf32>
CHECK:            arith.mulf {{.*}} tensor<1xf32>
CHECK:            triton_xla.insert {{.*}} : tensor<1xf32> into tensor<125xf32>
)"));
}

TEST_F(TritonEmitterTest,
       TestGenericEmitterWithReductonAndMultidimensionalTile) {
  constexpr absl::string_view kHloText = R"(
HloModule t
max {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT max = f32[] maximum(Arg_0, Arg_1)
}

triton_reduction_computation {
  parameter_0 = f32[4,12,125,127]{3,2,1,0} parameter(0)
  constant_0 = f32[] constant(-inf)
  ROOT reduce = f32[4,12,125]{2,1,0} reduce(parameter_0, constant_0), dimensions={3}, to_apply=max
}

ENTRY main {
  param_0 = f32[4,12,125,127]{3,2,1,0} parameter(0)
  ROOT triton_reduce = f32[4,12,125]{2,1,0} fusion(param_0),
    kind=kCustom, calls=triton_reduction_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2","5","16"]}],
        "num_warps":"4",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, TestSoftMaxWithTileElementsNotAllContiguous) {
  constexpr absl::string_view kHloText = R"(
HloModule m

region {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(param_0, param_1)
}

triton_softmax_computation {
  constant.1 = f32[] constant(0)
  broadcast.2 = f32[4,4,8] broadcast(constant.1), dimensions={}
  param_0.1 = f32[4,4,8] parameter(0)
  constant = f32[] constant(0)
  reduce = f32[4,4] reduce(param_0.1, constant), dimensions={2}, to_apply=region
  broadcast = f32[4,4,8] broadcast(reduce), dimensions={0,1}
  multiply = f32[4,4,8] multiply(broadcast.2, broadcast)
  ROOT add.2 = f32[4,4,8] add(multiply, broadcast)
}

ENTRY entry_computation {
  param_0.2 = f32[4,4,8] parameter(0)
  ROOT fusion = f32[4,4,8] fusion(param_0.2), kind=kCustom,
    calls=triton_softmax_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2","2","8"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}

})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                           /*arel=*/1e-6}));
}

TEST_F(TritonEmitterTest, TestSliceWithTileThatNeedsMasking) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  p = f32[128,32] parameter(0)
  ROOT slice = f32[12,5] slice(p), slice={[116:128], [20:25]}
}

ENTRY entry_computation {
  p = f32[128,32] parameter(0)
  ROOT fusion = f32[12,5] fusion(p), kind=kCustom, calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8","4"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{0, 0}));
}

TEST_F(TritonEmitterTest, TestSliceWithTileElementsNotAllContiguous) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0.1 = f32[16,16,32] parameter(0)
  slice = f32[4,4,8] slice(param_0.1), slice={[2:10:2], [2:6], [3:11]}
  slice.1 = f32[4,4,8] slice(param_0.1), slice={[4:8], [8:16:2], [13:21]}
  ROOT add.3 = f32[4,4,8] add(slice, slice.1)
}

ENTRY entry_computation {
  param_0.2 = f32[16,16,32] parameter(0)
  ROOT fusion = f32[4,4,8] fusion(param_0.2), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2","2","8"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                           /*arel=*/1e-6}));
}

TEST_F(TritonEmitterTest, TestSliceWithTileElementsNotAllContiguousUnaligned) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  p = f32[7,7,75] parameter(0)
  ROOT slice = f32[3,2,14] slice(p), slice={[1:6:2], [2:6:3], [35:75:3]}
}

ENTRY entry_computation {
  p = f32[7,7,75] parameter(0)
  ROOT fusion = f32[3,2,14] fusion(p),
    kind=kCustom, calls=fused_computation, backend_config={
      "fusion_backend_config": {
        "kind":"__triton",
        "block_level_fusion_config": {
          "output_tiles":[{"sizes":["2","2","8"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{0, 0}));
}

TEST_F(TritonEmitterTest, ReshapeIntoBroadcastIsLoweredCorrectly) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  param_0 = f32[128,256]{1,0} parameter(0)
  reshape = f32[64,2,256]{2,1,0} reshape(param_0)
  ROOT broadcast = f32[64,2,256,2]{3,2,1,0} broadcast(reshape), dimensions={0,1,2}
}

ENTRY main {
  param_0 = f32[128,256]{1,0} parameter(0)
  ROOT triton_fusion = f32[64,2,256,2]{3,2,1,0} fusion(param_0), kind=kCustom,
    calls=triton_computation, backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["2","2","2","2"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK: tt.reshape
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, BitcastIntoBroadcastIsLoweredCorrectly) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  param_0 = f32[128,256]{1,0} parameter(0)
  bitcast = f32[64,2,256]{2,1,0} bitcast(param_0)
  ROOT broadcast = f32[64,2,256,2]{3,2,1,0} broadcast(bitcast), dimensions={0,1,2}
}

ENTRY main {
  param_0 = f32[128,256]{1,0} parameter(0)
  ROOT triton_fusion = f32[64,2,256,2]{3,2,1,0} fusion(param_0), kind=kCustom,
    calls=triton_computation, backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["4","2","8","2"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK: tt.reshape
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, BitcastNormalizedLayoutsIsLoweredCorrectly) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  p = s8[5,42] parameter(0)
  ROOT bitcast = s8[5,6,7] bitcast(p)
}

ENTRY entry_computation {
  p = s8[5,42] parameter(0)
  ROOT fusion = s8[5,6,7] fusion(p), kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["2","4","1"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:     triton_xla.extract
CHECK-NOT: tt.trans
CHECK:     tt.reshape
CHECK-NOT: tt.trans
CHECK:     triton_xla.insert
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, BitcastNonNormalizedInputLayoutIsLoweredCorrectly) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  p = s8[42,5]{0,1} parameter(0)
  ROOT bitcast = s8[5,6,7] bitcast(p)
}

ENTRY entry_computation {
  p = s8[42,5]{0,1} parameter(0)
  ROOT fusion = s8[5,6,7] fusion(p), kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2","4","1"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:     triton_xla.extract
CHECK:     tt.trans
CHECK:     tt.reshape
CHECK-NOT: tt.trans
CHECK:     triton_xla.insert
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, BitcastNonNormalizedOutputLayoutIsLoweredCorrectly) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  p = s8[5,42] parameter(0)
  ROOT bitcast = s8[5,6,7]{1,2,0} bitcast(p)
}

ENTRY entry_computation {
  p = s8[5,42] parameter(0)
  ROOT fusion = s8[5,6,7]{1,2,0} fusion(p), kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2","4","1"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:     triton_xla.extract
CHECK-NOT: tt.trans
CHECK:     tt.reshape
CHECK:     tt.trans
CHECK:     triton_xla.insert
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest,
       BitcastNonNormalizedInputOutputLayoutIsLoweredCorrectly) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  p = s8[42,5]{0,1} parameter(0)
  ROOT bitcast = s8[5,6,7]{1,2,0} bitcast(p)
}

ENTRY entry_computation {
  p = s8[42,5]{0,1} parameter(0)
  ROOT fusion = s8[5,6,7]{1,2,0} fusion(p), kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2","4","1"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:     triton_xla.extract
CHECK:     tt.trans
CHECK:     tt.reshape
CHECK:     tt.trans
CHECK:     triton_xla.insert
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, BitcastTransposeOnlyIsLoweredCorrectly) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  p = s8[42,5]{0,1} parameter(0)
  ROOT bitcast = s8[5,42] bitcast(p)
}

ENTRY entry_computation {
  p = s8[42,5]{0,1} parameter(0)
  ROOT fusion = s8[5,42] fusion(p), kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4","1"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:     triton_xla.extract
CHECK:     tt.trans
CHECK-NOT: tt.reshape
CHECK-NOT: tt.trans
CHECK:     triton_xla.insert
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest,
       BitcastInBetweenReductionAndSlicedBroadcastIsLoweredCorrectly) {
  // Regression test for b/392099316
  constexpr absl::string_view kHloText = R"(
triton_computation {
  p0 = bf16[2048,4,256]{2,1,0} parameter(0)
  c0 = bf16[] constant(0)
  reduce = bf16[2048,4]{1,0} reduce(p0, c0), dimensions={2}, to_apply={
    a = bf16[] parameter(0)
    b = bf16[] parameter(1)
    ROOT maximum = bf16[] maximum(a, b)
  }
  add_unnecessary_dim = bf16[1,2048,4]{2,1,0} bitcast(reduce)
  upcast = f32[1,2048,4]{2,1,0} convert(add_unnecessary_dim)
  some_high_precision_op = f32[1,2048,4]{2,1,0} sqrt(upcast)
  downcast = bf16[1,2048,4]{2,1,0} convert(some_high_precision_op)
  remove_dim = bf16[2048,4]{1,0} bitcast(downcast)
  broadcast = bf16[2048,4,256]{2,1,0} broadcast(remove_dim), dimensions={0,1}
  ROOT slice = bf16[2048,4,128]{2,1,0} slice(broadcast),
    slice={[0:2048], [0:4], [0:128]}
}

ENTRY main {
  %p0 = bf16[2048,4,256]{2,1,0} parameter(0)
  ROOT fusion = bf16[2048,4,128]{2,1,0} fusion(p0), kind=kCustom,
  calls=triton_computation, backend_config={
    "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8","4","128"]}],
        "num_warps":"8",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:     triton_xla.extract
CHECK:     tt.reduce
CHECK:     tt.broadcast
CHECK:     triton_xla.insert
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// TODO(b/353484968): move this test to a deviceless file.
TEST_F(TritonEmitterTest, GenericEmitterLowersBroadcastFrom0dOperandCorrectly) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  param_0 = f32[] parameter(0)
  ROOT broadcast = f32[127,125]{1,0} broadcast(param_0), dimensions={}
}

ENTRY main {
  param_0 = f32[] parameter(0)
  ROOT triton_fusion = f32[127,125]{1,0} fusion(param_0), kind=kCustom,
    calls=triton_computation, backend_config={
      "fusion_backend_config":{
        "kind":"__triton","block_level_fusion_config":{
          "output_tiles":[{"sizes":["8","4"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:       tt.splat {{.*}} f32 -> tensor<8x4xf32>
)"));
}

TEST_F(TritonEmitterTest, PredOutputIsStoredCorrectly) {
  // The 'pred' element type in XLA is unpacked and uses i8 for storage.  This
  // is the only sub-byte type to have this behavior.
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_computation {
  param_0 = f32[15] parameter(0)
  param_1 = f32[15] parameter(1)
  ROOT compare = pred[15] compare(param_0, param_1), direction=GE
}

ENTRY main {
  param_0 = f32[15] parameter(0)
  param_1 = f32[15] parameter(1)
  ROOT triton_fusion = pred[15] fusion(param_0, param_1), kind=kCustom,
    calls=triton_computation, backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["4"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:      %[[CASTED_OUT:.*]] = arith.extui
CHECK-SAME:   tensor<4xi1> to tensor<4xi8>
CHECK:      triton_xla.insert %[[CASTED_OUT]]
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, PredInputIsLoadedCorrectly) {
  // The 'pred' element type in XLA is unpacked and uses i8 for storage.  This
  // is the only sub-byte type to have this behavior.
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_computation {
  param_0 = pred[15] parameter(0)
  param_1 = f32[15] parameter(1)
  param_2 = f32[15] parameter(2)
  // To highlight the issue, we need to construct something with type i1 inside
  // the kernel and combine it with a parameter.
  compare = pred[15] compare(param_1, param_2), direction=GE
  and = pred[15] and(compare, param_0)
  ROOT select = f32[15] select(and, param_1, param_2)
}

ENTRY main {
  param_0 = pred[15] parameter(0)
  param_1 = f32[15] parameter(1)
  param_2 = f32[15] parameter(2)
  ROOT triton_fusion = f32[15] fusion(param_0, param_1, param_2),
    kind=kCustom, calls=triton_computation, backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["4"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:      %[[I8_PARAM:.*]] = triton_xla.extract {{.*}} : tensor<15xi8> to tensor<4xi8>
CHECK:      arith.trunci %[[I8_PARAM]] : tensor<4xi8> to tensor<4xi1>
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, Transpose3D) {
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_computation {
  param_0 = f32[15,7,3] parameter(0)
  ROOT transpose = f32[3,15,7]{2,1,0} transpose(param_0), dimensions={2,0,1}
}

ENTRY main {
  param_0 = f32[15,7,3] parameter(0)
  ROOT triton_fusion = f32[3,15,7] fusion(param_0),
    kind=kCustom, calls=triton_computation, backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","8","4"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:      %[[TILE:.*]] = triton_xla.extract {{.*}} : tensor<15x7x3xf32> to tensor<8x4x1xf32>
CHECK:      tt.trans %[[TILE]] {order = array<i32: 2, 0, 1>} : tensor<8x4x1xf32> -> tensor<1x8x4xf32>
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, Transpose3DWithExtraOutput) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0.1 = f32[15,7,3] parameter(0)
  abs = f32[15,7,3] abs(param_0.1)
  transpose = f32[3,15,7] transpose(abs), dimensions={2,0,1}
  ROOT tuple = (f32[3,15,7], f32[15,7,3]) tuple(transpose, abs)
}

ENTRY entry_computation {
  param_0.2 = f32[15,7,3] parameter(0)
  ROOT fusion = (f32[3,15,7], f32[15,7,3]) fusion(param_0.2), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","8","4"]},{"sizes":["4","8","1"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "fused_computation", R"(
CHECK:         %[[TILE:.*]] = triton_xla.extract {{.*}} : tensor<15x7x3xf32> to tensor<8x4x1xf32>
CHECK-NOT:     triton_xla.extract
CHECK:         %[[ABS:.*]] = math.absf %[[TILE]]
CHECK:         tt.trans %[[ABS]] {order = array<i32: 2, 0, 1>} : tensor<8x4x1xf32> -> tensor<1x8x4xf32>
CHECK-COUNT-2: triton_xla.insert
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// TODO(b/353484968): Delete this test once we have constraints to only
// propagate tile sizes that are a power of 2.
TEST_F(TritonEmitterTest, Transpose3D_TileFullDimThatIsNotPowerOf2) {
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_computation {
  param_0 = f32[3,8,20] parameter(0)
  ROOT transpose = f32[8,3,20] transpose(param_0), dimensions={1,0,2}
}

ENTRY main {
  param_0 = f32[3,8,20] parameter(0)
  ROOT triton_fusion = f32[8,3,20] fusion(param_0),
    kind=kCustom, calls=triton_computation, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","1", "20"]}],
        "num_warps":"4",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, StridedIota4DIsCodegeneratedCorrectly) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  iota = f32[3,4,1000,5] iota(), iota_dimension=2
  ROOT slice = f32[3,4,182,5] slice(iota), slice={[0:3], [0:4], [91:1000:5], [0:5]}
}

ENTRY main {
  ROOT triton_fusion = f32[3,4,182,5] fusion(),
    kind=kCustom, calls=triton_computation, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","2","64","8"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";

  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:      %[[RANGE:.*]] = tt.make_range {{.*}} : tensor<64xi32>
CHECK:      arith.muli{{.*}} %[[RANGE]]
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

class IotaEmitterParametrizedTest
    : public TritonEmitterTest,
      public ::testing::WithParamInterface<PrimitiveType> {};

TEST_P(IotaEmitterParametrizedTest, Iota4DIsCodegeneratedCorrectly) {
  auto data_type = GetParam();
  const std::string kHloText =
      absl::Substitute(R"(
triton_computation {
  ROOT iota = $0[3,4,1000,5] iota(), iota_dimension=2
}

ENTRY main {
  ROOT triton_fusion = $0[3,4,1000,5] fusion(),
    kind=kCustom, calls=triton_computation, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","2","64","8"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})",
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:      %[[RANGE:.*]] = tt.make_range {{.*}} : tensor<64xi32>
CHECK:      arith.addi{{.*}} %[[RANGE]]
            // Omit the data type below, since it depends on a test parameter
            // and is not abbreviated the same as in HLO.
CHECK:      tt.broadcast {{.*}} -> tensor<1x2x64x8x
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

std::string TypeTestParamToString(
    const ::testing::TestParamInfo<PrimitiveType>& data) {
  return primitive_util::LowercasePrimitiveTypeName(data.param);
}

INSTANTIATE_TEST_SUITE_P(IotaEmitterParametrizedTestSuite,
                         IotaEmitterParametrizedTest,
                         ::testing::ValuesIn({S8, S16, S32, S64, BF16, F16, F32,
                                              F64}),
                         TypeTestParamToString);

TEST_F(TritonEmitterTest, ReducePrecisionIsLoweredCorrectly) {
  const std::string kHloText = R"(
triton_computation {
  p = f32[5,7] parameter(0)
  ROOT rp = f32[5,7] reduce-precision(p), exponent_bits=2, mantissa_bits=2
}

ENTRY entry_computation {
  p = f32[5,7] parameter(0)
  ROOT fusion = f32[5,7] fusion(p), kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles": [{"sizes":["4","4"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:     triton_xla.extract
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, Chaining0DElementwiseScalarsIsSupported) {
  const std::string kHloText = R"(
triton_computation {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  exp0 = f32[] exponential(p0)
  exp1 = f32[] exponential(p1)
  neg0 = f32[] negate(exp0)
  neg1 = f32[] negate(exp1)
  add = f32[] add(neg0, neg1)
  mul = f32[] multiply(add, add)
  div = f32[] divide(mul, p0)
  conv = bf16[] convert(div)
  const = bf16[] constant(0.5)
  ROOT sub = bf16[] subtract(conv, const)
}

ENTRY entry_computation {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT fusion = bf16[] fusion(p0, p1), kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
        "output_tiles": [{"sizes":[]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:     tensor.extract {{.*}} tensor<f32>
CHECK:     tt.extern_elementwise {{.*}} (f32) -> f32
CHECK:     arith.subf {{.*}} f32
CHECK:     tensor.extract {{.*}} tensor<f32>
CHECK:     tt.extern_elementwise {{.*}} (f32) -> f32
CHECK:     arith.subf {{.*}} f32
CHECK:     arith.addf {{.*}} f32
CHECK:     arith.mulf {{.*}} f32
CHECK:     arith.divf {{.*}} f32
CHECK:     arith.truncf {{.*}} f32 to bf16
CHECK:     arith.subf {{.*}} bf16
CHECK:     tensor.insert {{.*}} tensor<bf16>
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/6e-1, /*arel=*/6e-1}));
}

TEST_F(TritonEmitterTest, Multiple0DBroadcastsAreSupported) {
  const std::string kHloText = R"(
add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

triton_computation {
  p = f32[] parameter(0)
  exp = f32[] exponential(p)
  b1 = f32[10] broadcast(exp), dimensions={}
  b2 = f32[10,10] broadcast(exp), dimensions={}
  b3 = f32[10,10] broadcast(b1), dimensions={0}
  add = f32[10,10] add(b2,b3)
  c = f32[] constant(0)
  reduce1 = f32[10] reduce(add, c), dimensions={0}, to_apply=add
  ROOT reduce2 = f32[] reduce(reduce1, c), dimensions={0}, to_apply=add
}

ENTRY entry_computation {
  p = f32[] parameter(0)
  ROOT fusion = f32[] fusion(p), kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles": [{"sizes":[]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:     tensor.extract
CHECK:     tt.splat
CHECK:     arith.addf
CHECK:     tt.reduce
CHECK:     tensor.insert {{.*}} tensor<f32>
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/6e-1, /*arel=*/6e-1}));
}

TEST_F(TritonEmitterTest, ReshapeTo0DIsSupported) {
  const std::string kHloText = R"(
triton_computation {
  p0 = f32[1,1,1,1] parameter(0)
  p1 = f32[1] parameter(1)
  reshape1 = f32[] reshape(p0)
  reshape2 = f32[] reshape(p1)
  ROOT add = f32[] add(reshape1, reshape2)
}

ENTRY entry_computation {
  p0 = f32[1,1,1,1] parameter(0)
  p1 = f32[1] parameter(1)
  ROOT fusion = f32[] fusion(p0, p1), kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":[]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:     tt.reshape
CHECK:     tt.reduce{{.*}}axis = 0
CHECK-NOT: tt.reshape
CHECK:     tt.reduce{{.*}}axis = 0
CHECK:     tensor.insert {{.*}} tensor<f32>
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{0, 0}));
}

// Reproducer from b/380277401.
TEST_F(TritonEmitterTest, IntraWarpReduceOfReduceIsCorrect) {
  const std::string kHloText = R"(
add {
  x = s32[] parameter(0)
  y = s32[] parameter(1)
  ROOT add = s32[] add(x, y)
}

triton_computation {
  p = s32[4,8] parameter(0)
  bitcast = s32[4,2,4] bitcast(p)

  zero = s32[] constant(0)
  reduce_1 = s32[4,2] reduce(bitcast, zero), dimensions={2}, to_apply=add
  ROOT reduce_2 = s32[2] reduce(reduce_1, zero), dimensions={0}, to_apply=add
}

ENTRY entry_computation {
  i = s32[32] iota(), iota_dimension=0
  p = s32[4,8] bitcast(i)

  ROOT r = s32[2] fusion(p), kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["2"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "triton_computation", R"(
CHECK:     triton_xla.extract
CHECK:     tt.reshape
CHECK:     tt.reduce
CHECK:     tt.reduce
CHECK:     triton_xla.insert
)"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// Reproducer from b/384110192.
TEST_F(TritonEmitterTest,
       FusionWithOutputContainingMoreThanInt32MaxElementsExecutesCorrectly) {
  // The point here is to check the output of the Triton fusion. The `slice` op
  // at the end is inserted to allow the comparison of output to run in a
  // reasonable amount of time, and has been proven to still correctly capture
  // the indexing overflow behaviour of the Triton fusion that we're checking
  // for.
  constexpr absl::string_view kTritonHloText = R"(
computation {
  p0 = s8[256]{0} parameter(0)
  ROOT broadcast = s8[16777217,256]{1,0} broadcast(p0), dimensions={1}
}

ENTRY entry_computation {
  p0 = s8[256]{0} parameter(0)
  fusion = s8[16777217,256]{1,0} fusion(p0), kind=kCustom,
    calls=computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["2","256"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
  ROOT slice = s8[1000,256]{1,0} slice(fusion), slice={[16776217:16777217], [0:256]}
})";

  constexpr absl::string_view kEmittersHloText = R"(
computation {
  p0 = s8[256]{0} parameter(0)
  ROOT broadcast = s8[16777217,256]{1,0} broadcast(p0), dimensions={1}
}

ENTRY entry_computation {
  p0 = s8[256]{0} parameter(0)
  fusion = s8[16777217,256]{1,0} fusion(p0), kind=kCustom,
    calls=computation
  ROOT slice = s8[1000,256]{1,0} slice(fusion), slice={[16776217:16777217], [0:256]}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> triton_module,
                          ParseAndReturnVerifiedModule(kTritonHloText));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> emitters_module,
                          ParseAndReturnVerifiedModule(kEmittersHloText));

  const Shape& triton_fusion_shape = triton_module->entry_computation()
                                         ->root_instruction()
                                         ->operand(0)
                                         ->shape();

  ASSERT_GT(Product(triton_fusion_shape.dimensions()), 1l << 32);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(triton_module),
                                      std::move(emitters_module), kExactMatch,
                                      /*run_hlo_passes=*/false));
}

TEST_F(TritonEmitterTest, ConvertF16ToF8E5M2Exhaustive) {
  // TODO(b/396595945): enable post-Ampere once Triton respects RTNE semantics
  // on H100.
  if (auto cc =
          std::get_if<se::CudaComputeCapability>(&GpuComputeCapability())) {
    if (cc->IsAtLeastHopper()) {
      GTEST_SKIP() << "Skipping tests above Ampere, Triton's conversion isn't "
                      "always correct";
    }
  }

  constexpr absl::string_view kHloTextTemplate = R"(
computation {
  p0 = f16[65536]{0} parameter(0)
  ROOT convert = f8e5m2[65536]{0} convert(p0)
}

ENTRY entry_computation {
  p0 = f16[65536]{0} constant({$0})
  ROOT fusion = f8e5m2[65536]{0} fusion(p0), kind=kCustom,
    calls=computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";

  std::vector<Eigen::half> all_f16_values;
  for (int i = 0; i < 65536; i++) {
    all_f16_values.push_back(
        Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(i)));
  }

  std::string hlo_text =
      absl::Substitute(kHloTextTemplate, absl::StrJoin(all_f16_values, ", "));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), kExactMatch));
}

TEST_F(TritonEmitterTest, FP8ToFP8EndToEnd) {
  if (auto cc =
          std::get_if<se::CudaComputeCapability>(&GpuComputeCapability())) {
    if (!cc->IsAtLeastHopper()) {
      GTEST_SKIP() << "Doesn't pass on pre-Hopper GPUs.";
    }
  }

  const std::string hlo_text = R"(
HloModule t

triton_dot {
  parameter_0 = f8e5m2[32,32]{1,0} parameter(0)
  parameter_1 = f8e4m3fn[32,32]{1,0} parameter(1)
  convert = f8e4m3fn[32,32]{1,0} convert(parameter_0)
  ROOT dot = f32[32,32]{1,0} dot(convert, parameter_1),
                lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  parameter_0 = f8e5m2[32,32]{1,0} parameter(0)
  parameter_1 = f8e4m3fn[32,32]{1,0} parameter(1)
  ROOT gemm_fusion_dot = f32[32,32]{1,0} fusion(parameter_0, parameter_1),
       kind=kCustom, calls=triton_dot,
       backend_config={
       "fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":
         {"block_m":"32","block_n":"32","block_k":"32","split_k":"1",
          "num_stages":"1","num_warps":"4","num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text,
                                       ErrorSpec{/*aabs=*/1.0, /*arel=*/1e-3}));
}

TEST_F(TritonEmitterTest, SingleTileDotWithNestedFusionsIsEmittedCorrectly) {
  // Simplest case when everything fits into one tile that is useful for
  // debugging. This also tests support for empty nested fusions.
  const std::string kHloText = R"(
flhs {
  ROOT flhs.p0 = f32[16,16] parameter(0)
}

frhs {
  frhs.p0 = f32[16,16] parameter(0)
  ROOT frhs.root = f32[16,16] abs(frhs.p0)
}

fdot {
  fdot.p0 = f32[16,16] parameter(0)
  fdot.p1 = f32[16,16] parameter(1)
  fdot.lhs = f32[16,16] fusion(fdot.p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "16"]}]
      }
    }
  }
  fdot.rhs = f32[16,16]{1,0} fusion(fdot.p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "16"]}]
      }
    }
  }
  ROOT fdot.root = f32[16,16]{1,0} dot(fdot.lhs, fdot.rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_f32_f32_f32
}

ENTRY entry {
  entry.p0 = f32[16,16] parameter(0)
  entry.p1 = f32[16,16] parameter(1)
  ROOT fusion = f32[16,16] fusion(entry.p0, entry.p1),
    kind=kCustom, calls=fdot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","16"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  // We expect that for loop instruction will be optimized away.
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText, "fdot", R"(
CHECK: tt.dot {{.*}} -> tensor<16x16xf32>
)"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

TEST_F(TritonEmitterTest, DotWithNestedFusionsIsEmittedCorrectly) {
  const std::string kHloText = R"(
flhs {
  flhs.p0 = f32[32,256] parameter(0)
  ROOT lhs.root = f32[32,256] negate(flhs.p0)
}

frhs {
  frhs.p0 = f32[256,512] parameter(0)
  ROOT frhs.root = f32[256,512] abs(frhs.p0)
}

fdot {
  fdot.p0 = f32[32,256] parameter(0)
  fdot.p1 = f32[256,512] parameter(1)
  fdot.lhs = f32[32,256] fusion(fdot.p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "32"]}]
      }
    }
  }
  fdot.rhs = f32[256,512]{1,0} fusion(fdot.p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "64"]}]
      }
    }
  }
  ROOT fdot.root = f32[32,512]{1,0} dot(fdot.lhs, fdot.rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_f32_f32_f32
}

ENTRY entry {
  entry.p0 = f32[32,256] parameter(0)
  entry.p1 = f32[256,512] parameter(1)
  ROOT fusion = f32[32,512] fusion(entry.p0, entry.p1),
    kind=kCustom, calls=fdot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16", "64"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText, "fdot", R"(
CHECK:      func.func @triton_fn(%[[ARG0:[A-Za-z0-9_]*]]: tensor<32x256xf32>
CHECK-SAME:                    %[[ARG1:[A-Za-z0-9_]*]]: tensor<256x512xf32>
CHECK-SAME:                    %[[ARG2:[A-Za-z0-9_]*]]: tensor<32x512xf32>
CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : i64
CHECK-DAG:  %[[C8:.*]] = arith.constant 8 : i64
CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : i64
CHECK:      {{.*}} = scf.for {{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
CHECK-SAME: iter_args({{.*}}) -> (tensor<16x64xf32>)  : i64 {
CHECK-DAG:  triton_xla.tile %[[ARG0]]
CHECK-DAG:  triton_xla.tile %[[ARG1]]
CHECK-DAG:  arith.subf {{.*}} : tensor<16x32xf32>
CHECK-DAG:  math.absf {{.*}} : tensor<32x64xf32>
CHECK-DAG:  tt.dot {{.*}} tensor<16x32xf32> * tensor<32x64xf32> -> tensor<16x64xf32>
CHECK:      scf.yield {{.*}} : tensor<16x64xf32>
CHECK-COUNT-1: triton_xla.insert
)"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

TEST_F(TritonEmitterTest, MaskedDotIsEmittedCorrectly) {
  const std::string kHloText = R"(
flhs {
  flhs.p0 = f32[32,299] parameter(0)
  ROOT lhs.root = f32[32,299] cosine(flhs.p0)
}

frhs {
  frhs.p0 = f32[299,512] parameter(0)
  ROOT frhs.root = f32[299,512] cosine(frhs.p0)
}

fdot {
  fdot.p0 = f32[32,299] parameter(0)
  fdot.p1 = f32[299,512] parameter(1)
  fdot.lhs = f32[32,299] fusion(fdot.p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "32"]}]
      }
    }
  }
  fdot.rhs = f32[299,512]{1,0} fusion(fdot.p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "64"]}]
      }
    }
  }
  ROOT fdot.root = f32[32,512]{1,0} dot(fdot.lhs, fdot.rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_f32_f32_f32
}

ENTRY entry {
  entry.p0 = f32[32,299] parameter(0)
  entry.p1 = f32[299,512] parameter(1)
  ROOT fusion = f32[32,512] fusion(entry.p0, entry.p1),
    kind=kCustom, calls=fdot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16", "64"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

TEST_F(TritonEmitterTest, DotWithMajorLhsContractingDimIsEmittedCorrectly) {
  const std::string kHloText = R"(
lhs {
  ROOT p0 = f32[299,32] parameter(0)
}

rhs {
  ROOT p0 = f32[299,512] parameter(0)
}

fdot {
  p0 = f32[299,32] parameter(0)
  p1 = f32[299,512] parameter(1)
  lhs = f32[299,32] fusion(p0), kind=kCustom, calls=lhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "16"]}]
      }
    }
  }
  rhs = f32[299,512]{1,0} fusion(p1), kind=kCustom, calls=rhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "64"]}]
      }
    }
  }
  ROOT dot = f32[32,512]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={0}, rhs_contracting_dims={0},
    algorithm=dot_f32_f32_f32
}

ENTRY entry {
  // Take in boolean inputs for the test, in order to allow exact accumulation.
  p0 = pred[299,32] parameter(0)
  p1 = pred[299,512] parameter(1)
  p0_f32 = f32[299,32] convert(p0)
  p1_f32 = f32[299,512] convert(p1)
  ROOT fusion = f32[32,512] fusion(p0_f32, p1_f32),
    kind=kCustom, calls=fdot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16", "64"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, DotWithMinorRhsContractingDimIsEmittedCorrectly) {
  const std::string kHloText = R"(
lhs {
  ROOT p0 = f32[32,299] parameter(0)
}

rhs {
  ROOT p0 = f32[512,299] parameter(0)
}

fdot {
  p0 = f32[32,299] parameter(0)
  p1 = f32[512,299] parameter(1)
  lhs = f32[32,299] fusion(p0), kind=kCustom, calls=lhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "32"]}]
      }
    }
  }
  rhs = f32[512,299]{1,0} fusion(p1), kind=kCustom, calls=rhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64", "32"]}]
      }
    }
  }
  ROOT dot = f32[32,512]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={1},
    algorithm=dot_f32_f32_f32
}

ENTRY entry {
  // Take in boolean inputs for the test, in order to allow exact accumulation.
  p0 = pred[32,299] parameter(0)
  p1 = pred[512,299] parameter(1)
  p0_f32 = f32[32,299] convert(p0)
  p1_f32 = f32[512,299] convert(p1)
  ROOT fusion = f32[32,512] fusion(p0_f32, p1_f32),
    kind=kCustom, calls=fdot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16", "64"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest,
       DotWithAdditionalDimensionsWithUnitTileSizesIsEmittedCorrectly) {
  const std::string kHloText = R"(
lhs {
  ROOT p0 = f32[2,3,32,125] parameter(0)
}

rhs {
  ROOT p0 = f32[2,125,3,256] parameter(0)
}

fdot {
  p0 = f32[2,3,32,125] parameter(0)
  p1 = f32[2,125,3,256] parameter(1)
  lhs = f32[2,3,32,125] fusion(p0), kind=kCustom, calls=lhs,
    backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1", "1", "16", "32"]}]
      }
    }
  }
  rhs = f32[2,125,3,256] fusion(p1), kind=kCustom, calls=rhs,
    backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1", "32", "1", "64"]}]
      }
    }
  }
  ROOT dot = f32[2,3,32,256] dot(lhs, rhs),
    lhs_batch_dims={0,1}, rhs_batch_dims={0,2},
    lhs_contracting_dims={3}, rhs_contracting_dims={1},
    algorithm=dot_f32_f32_f32
}

ENTRY entry {
  // Take in boolean inputs for the test, in order to allow exact accumulation.
  p0 = pred[2,3,32,125] parameter(0)
  p1 = pred[2,125,3,256] parameter(1)
  p0_f32 = f32[2,3,32,125] convert(p0)
  p1_f32 = f32[2,125,3,256] convert(p1)
  ROOT fusion = f32[2,3,32,256] fusion(p0_f32, p1_f32),
    kind=kCustom, calls=fdot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1", "1", "16", "64"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, ConcatenateOfNestsIsEmittedCorrectly) {
  const std::string kHloText = R"(
nest0 {
  p0 = s32[128] parameter(0)
  ROOT abs = s32[128] abs(p0)
}

nest1 {
  p0 = s32[128] parameter(0)
  ROOT negate = s32[128] negate(p0)
}

nest2 {
  ROOT p0 = s32[128] parameter(0)
}

concatenate_fusion {
  p0 = s32[128] parameter(0)
  p1 = s32[128] parameter(1)
  p2 = s32[128] parameter(2)

  fusion0 = s32[128] fusion(p0), kind=kCustom, calls=nest0, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
  fusion1 = s32[128] fusion(p1), kind=kCustom, calls=nest1, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
  fusion2 = s32[128] fusion(p2), kind=kCustom, calls=nest2, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}

  ROOT concatenate = s32[384] concatenate(fusion0, fusion1, fusion2), dimensions={0}
}

ENTRY main {
  p0 = s32[128] parameter(0)
  p1 = s32[128] parameter(1)
  p2 = s32[128] parameter(2)
  ROOT fusion = s32[384] fusion(p0, p1, p2), kind=kCustom,
    calls=concatenate_fusion, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";

  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(this, kHloText, "concatenate_fusion", R"(
    // Check that we generate three branches. This is a bit of an implementation
    // detail, so it doesn't seem worth enforcing a lot here.
    CHECK-COUNT-2: scf.if
  )"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, NestedFusionOfNestedFusionsExecutesCorrectly) {
  const std::string kHloText = R"(
lhs {
  p0 = f32[32,299] parameter(0)
  ROOT cos = f32[32,299] cosine(p0)
}

nest0 {
  p0 = f32[299,128] parameter(0)
  ROOT abs = f32[299,128] abs(p0)
}

nest1 {
  p0 = f32[299,128] parameter(0)
  ROOT negate = f32[299,128] negate(p0)
}

nest2 {
  ROOT p0 = f32[299,128] parameter(0)
}

nest3 {
  p0 = f32[299,128] parameter(0)
  ROOT cos = f32[299,128] cosine(p0)
}

rhs {
  p0 = f32[299,128] parameter(0)
  p1 = f32[299,128] parameter(1)
  p2 = f32[299,128] parameter(2)
  p3 = f32[299,128] parameter(3)

  fusion0 = f32[299,128] fusion(p0), kind=kCustom, calls=nest0, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "64"]}]
      }
    }
  }
  fusion1 = f32[299,128] fusion(p1), kind=kCustom, calls=nest1, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "64"]}]
      }
    }
  }
  fusion2 = f32[299,128] fusion(p2), kind=kCustom, calls=nest2, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "64"]}]
      }
    }
  }
  fusion3 = f32[299,128] fusion(p3), kind=kCustom, calls=nest3, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "64"]}]
      }
    }
  }

  concatenate = f32[299,512] concatenate(fusion0, fusion1, fusion2, fusion3), dimensions={1}
  ROOT cos = f32[299,512] cosine(concatenate)
}

dot {
  p0 = f32[32,299] parameter(0)
  p1 = f32[299,128] parameter(1)
  p2 = f32[299,128] parameter(2)
  p3 = f32[299,128] parameter(3)
  p4 = f32[299,128] parameter(4)
  lhs = f32[32,299] fusion(p0), kind=kCustom, calls=lhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "32"]}]
      }
    }
  }
  rhs = f32[299,512]{1,0} fusion(p1, p2, p3, p4), kind=kCustom, calls=rhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "64"]}]
      }
    }
  }
  ROOT dot = f32[32,512]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_f32_f32_f32
}

ENTRY entry {
  p0 = f32[32,299] parameter(0)
  p1 = f32[299,128] parameter(1)
  p2 = f32[299,128] parameter(2)
  p3 = f32[299,128] parameter(3)
  p4 = f32[299,128] parameter(4)
  ROOT fusion = f32[32,512] fusion(p0, p1, p2, p3, p4),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16", "64"]}], "num_warps":"1",
          "num_ctas":"1", "num_stages":"1"
        }
      }
    }
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

TEST_F(TritonEmitterTest, DotFromBroadcastIsEmittedCorrectly) {
  // TODO(b/393299275): add a deviceless test to run the whole pipeline as
  // other passes might change the module but we are starting from a fixed
  // state.
  const std::string kHloText = R"(
HloModule module

flhs (parameter_0: f32[264]) -> f32[264,128] {
  parameter_0 = f32[264]{0} parameter(0)
  ROOT flhs.1 = f32[264,128]{1,0} broadcast(parameter_0), dimensions={0}
}

frhs (parameter_0.1: f32[128,8]) -> f32[128,8] {
  ROOT parameter_0.1 = f32[128,8]{1,0} parameter(0)
}

triton_dot (p0: f32[264], p1: f32[128,8]) -> f32[264,8] {
  p0 = f32[264]{0} parameter(0)
  lhs = f32[264,128]{1,0} fusion(p0), kind=kCustom, calls=flhs, backend_config={"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion","block_level_fusion_config":{"num_warps":"1","output_tiles":[{"sizes":["32","16"]}]}}}
  p1 = f32[128,8]{1,0} parameter(1)
  rhs = f32[128,8]{1,0} fusion(p1), kind=kCustom, calls=frhs, backend_config={"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion","block_level_fusion_config":{"num_warps":"1","output_tiles":[{"sizes":["16","16"]}]}}}
  ROOT result = f32[264,8]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_f32_f32_f32
}

ENTRY e (p0.1: f32[11,1,24,1], p1.1: f32[128,8]) -> f32[264,8] {
  p0.1 = f32[11,1,24,1]{3,2,1,0} parameter(0)
  bitcast = f32[264]{0} bitcast(p0.1)
  p1.1 = f32[128,8]{1,0} parameter(1)
  ROOT result.1 = f32[264,8]{1,0} fusion(bitcast, p1.1), kind=kCustom,
    calls=triton_dot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","16"]}],
          "num_warps":"1",
          "num_stages":"1",
          "num_ctas":"1"}}}
}
)";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

// The template is parametrized by the type of the lhs/rhs, the type of the
// dot output, and the algorithm.
constexpr absl::string_view kHloForDotAlgorithmTestTemplate = R"(
lhs {
  ROOT p0 = $0[512,512] parameter(0)
}

rhs {
  ROOT p0 = $0[512,512] parameter(0)
}

dot {
  p0 = $0[512,512] parameter(0)
  p1 = $0[512,512] parameter(1)
  lhs = $0[512,512] fusion(p0), kind=kCustom, calls=lhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "32"]}]
    }}}
  rhs = $0[512,512]{1,0} fusion(p1), kind=kCustom, calls=rhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "64"]}]
    }}}
  ROOT dot = $1[512,512] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=$2
}

ENTRY entry {
  p0 = $0[512,512] parameter(0)
  p1 = $0[512,512] parameter(1)
  ROOT fusion = $1[512,512] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_warps":"1", "num_ctas":"1", "num_stages":"1"
    }}}
})";

std::string GetDotAlgorithmHlo(PrimitiveType in_ty, PrimitiveType out_ty,
                               PrecisionConfig::Algorithm algorithm) {
  constexpr absl::string_view kAlgorithmPrefix = "ALG_";
  std::string in_ty_str = primitive_util::LowercasePrimitiveTypeName(in_ty);
  std::string out_ty_str = primitive_util::LowercasePrimitiveTypeName(out_ty);
  std::string algorithm_str = PrecisionConfig::Algorithm_Name(algorithm).substr(
      kAlgorithmPrefix.size());
  return absl::Substitute(kHloForDotAlgorithmTestTemplate, in_ty_str,
                          out_ty_str, algorithm_str);
}

// TODO(b/407744579): narrow down the error specs for the various dot
// algorithms.
//
// The non-default values are either taken from the pre-existing
// `dot_algorithms_test` as of 2025-04-01, or approximated. It's not clear
// whether even the pre-existing values were derived to adhere precisely to the
// numerical expectations of the corresponding algorithms. We should narrow this
// down in the future.
ErrorSpec ErrorSpecForDotAlgorithm(PrecisionConfig::Algorithm algorithm) {
  // A default error spec, not particularly tuned to any algorithm.
  ErrorSpec default_error_spec{/*aabs=*/1e-4, /*arel=*/1e-6};
  switch (algorithm) {
    case PrecisionConfig::ALG_UNSET:
      // Give a loose tolerance to ALG_UNSET, as the expected behaviour is
      // not deducible from the algorithm name alone.
      return ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-3};
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-3};
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      return default_error_spec;
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/2e-6, /*arel=*/2e-6};
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
      return default_error_spec;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      // Taken from `dot_algorithms_test`.
      return ErrorSpec{/*aabs=*/0, /*arel=*/6e-5};
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
      // Taken from `dot_algorithms_test`.
      return ErrorSpec{/*aabs=*/0, /*arel=*/7e-6};
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/2e-6, /*arel=*/2e-6};
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3};
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/2e-6, /*arel=*/3e-6};
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/2e-6, /*arel=*/2e-6};
    case PrecisionConfig::ALG_DOT_BF16_BF16_BF16:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
      return kExactMatch;
    // Keep in order to make the switch exhaustive.
    case PrecisionConfig_Algorithm_PrecisionConfig_Algorithm_INT_MIN_SENTINEL_DO_NOT_USE_:  // NOLINT(whitespace/line_length)
    case PrecisionConfig_Algorithm_PrecisionConfig_Algorithm_INT_MAX_SENTINEL_DO_NOT_USE_:  // NOLINT(whitespace/line_length)
      LOG(FATAL) << "Unsupported algorithm: " << algorithm;
  }
}

class TritonEmitterTestWithAlgorithmParam
    : public TritonEmitterTest,
      public ::testing::WithParamInterface<PrecisionConfig::Algorithm> {};

// Regroups tests for dot algorithms that have no ambiguous type parameters as
// per `algorithm_util::GetAllowedOperandsTypeForAlgorithm` and
// `algorithm_util::GetDotAccumulatorType`, and do not decompose each tiled step
// into multiple `dot` operations. We call these algorithms "basic" algorithms
// here.
using BasicDotAlgorithmEmitterTest = TritonEmitterTestWithAlgorithmParam;

constexpr std::array kBasicAlgorithms = {
    PrecisionConfig::ALG_DOT_F16_F16_F16,
    PrecisionConfig::ALG_DOT_F32_F32_F32,
    PrecisionConfig::ALG_DOT_F64_F64_F64,
    PrecisionConfig::ALG_DOT_F16_F16_F32,
    PrecisionConfig::ALG_DOT_BF16_BF16_F32,
    PrecisionConfig::ALG_DOT_TF32_TF32_F32,
};

TEST_P(BasicDotAlgorithmEmitterTest, BasicAlgorithmIsEmittedCorrectly) {
  auto algorithm = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<PrimitiveType> allowed_types,
      algorithm_util::GetAllowedOperandsTypeForAlgorithm(algorithm));
  ASSERT_EQ(allowed_types.size(), 1);
  PrimitiveType in_ty = allowed_types.front();
  TF_ASSERT_OK_AND_ASSIGN(PrimitiveType out_ty,
                          algorithm_util::GetDotAccumulatorType(algorithm));
  const std::string kHloText = GetDotAlgorithmHlo(in_ty, out_ty, algorithm);

  TF_EXPECT_OK(CreateTritonIrAndFileCheck(
      this, kHloText, "dot",
      absl::Substitute(R"(
  CHECK:  tt.dot{{.*}} : tensor<16x32x$0> * tensor<32x64x$0> -> tensor<16x64x$1>
  )",
                       primitive_util::LowercasePrimitiveTypeName(in_ty),
                       primitive_util::LowercasePrimitiveTypeName(out_ty))));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpecForDotAlgorithm(algorithm)));
}

std::string DotAlgorithmTestToString(
    const ::testing::TestParamInfo<PrecisionConfig::Algorithm>& data) {
  return PrecisionConfig::Algorithm_Name(data.param);
}

INSTANTIATE_TEST_SUITE_P(BasicDotAlgorithmEmitterTestSuite,
                         BasicDotAlgorithmEmitterTest,
                         ::testing::ValuesIn(kBasicAlgorithms),
                         DotAlgorithmTestToString);

// Regroups tests for dot algorithms that issue several dot instructions.
using MultiDotAlgorithmEmitterTest = TritonEmitterTestWithAlgorithmParam;

constexpr std::array kMultiDotAlgorithms = {
    PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3,
    PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6,
    PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3,
    PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9,
};

TEST_P(MultiDotAlgorithmEmitterTest, MultiDotAlgorithmIsEmittedCorrectly) {
  auto algorithm = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(PrimitiveType out_ty,
                          algorithm_util::GetDotAccumulatorType(algorithm));
  PrimitiveType in_ty =
      algorithm == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3 ? F32 : BF16;
  // Dummy value to ensure that the dot count is explicitly set.
  int dot_count_for_algorithm = 0x1337;
  std::string input_precision_string = "";
  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
      dot_count_for_algorithm = 3;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
      dot_count_for_algorithm = 6;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      dot_count_for_algorithm = 9;
      break;
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      // Triton implements TF32x3 as a specific precision mode.
      input_precision_string = "tf32x3";
      dot_count_for_algorithm = 1;
      break;
    default:
      // Unreachable.
      ASSERT_TRUE(false);
  }

  const std::string kHloText = GetDotAlgorithmHlo(in_ty, out_ty, algorithm);

  TF_EXPECT_OK(CreateTritonIrAndFileCheck(
      this, kHloText, "dot",
      absl::Substitute(R"(
  CHECK-COUNT-$2:  tt.dot{{.*}}$3{{.*}} : tensor<16x32x$0> * tensor<32x64x$0> -> tensor<16x64x$1>
  )",
                       primitive_util::LowercasePrimitiveTypeName(in_ty),
                       primitive_util::LowercasePrimitiveTypeName(out_ty),
                       dot_count_for_algorithm, input_precision_string)));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpecForDotAlgorithm(algorithm)));
}

INSTANTIATE_TEST_SUITE_P(MultiDotAlgorithmEmitterTestSuite,
                         MultiDotAlgorithmEmitterTest,
                         ::testing::ValuesIn(kMultiDotAlgorithms),
                         DotAlgorithmTestToString);

// Regroups tests that use TF32 precision by definition.
using TF32DotAlgorithmEmitterTest = TritonEmitterTestWithAlgorithmParam;

constexpr std::array kTF32DotAlgorithms = {
    PrecisionConfig::ALG_DOT_TF32_TF32_F32,
    PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3};

TEST_P(TF32DotAlgorithmEmitterTest, TF32AlgorithmsUseTF32InputPrecision) {
  auto algorithm = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<PrimitiveType> allowed_types,
      algorithm_util::GetAllowedOperandsTypeForAlgorithm(algorithm));
  ASSERT_EQ(allowed_types.size(), 1);
  PrimitiveType in_ty = allowed_types.front();
  TF_ASSERT_OK_AND_ASSIGN(PrimitiveType out_ty,
                          algorithm_util::GetDotAccumulatorType(algorithm));
  const std::string kHloText = GetDotAlgorithmHlo(in_ty, out_ty, algorithm);

  std::string input_precision_string =
      algorithm == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3 ? "tf32x3"
                                                             : "tf32";

  TF_EXPECT_OK(CreateTritonIrAndFileCheck(
      this, kHloText, "dot",
      absl::Substitute(R"(
  CHECK:  tt.dot{{.*}} inputPrecision = $2 : tensor<16x32x$0> * tensor<32x64x$0> -> tensor<16x64x$1>
  )",
                       primitive_util::LowercasePrimitiveTypeName(in_ty),
                       primitive_util::LowercasePrimitiveTypeName(out_ty),
                       input_precision_string)));
  // No need to `RunAndCompare` here, these algorithms are already covered by
  // other tests.
}

INSTANTIATE_TEST_SUITE_P(TF32DotAlgorithmEmitterTestSuite,
                         TF32DotAlgorithmEmitterTest,
                         ::testing::ValuesIn(kTF32DotAlgorithms),
                         DotAlgorithmTestToString);

class DotUnsetAlgorithmEmitterTest
    : public TritonEmitterTest,
      public ::testing::WithParamInterface<PrimitiveType> {};

TEST_P(DotUnsetAlgorithmEmitterTest, UnsetAlgorithmIsEmittedCorrectly) {
  // This currently assumes that the dot output type is the same as the input
  // type. This is not enforced by the verifier/HLO spec, but is currently true
  // for our emitters, and is enforced by `support_test.cc`. This test may
  // require upgrading if we ever consider emitting code for truly mixed type
  // `dot`s.
  PrimitiveType ty = GetParam();
  if (!internal::IsResultTypeSupportedByAlgUnsetDot(ty,
                                                    GpuComputeCapability())) {
    GTEST_SKIP() << primitive_util::LowercasePrimitiveTypeName(ty)
                 << " is not supported on this platform.";
  }

  ErrorSpec error_spec = ErrorSpecForDotAlgorithm(PrecisionConfig::ALG_UNSET);
  // For 8-bit floating point types, we need to allow large errors.
  if (primitive_util::IsFloatingPointType(ty) &&
      primitive_util::BitWidth(ty) == 8) {
    error_spec = ErrorSpec{/*aabs=*/1e0, /*arel=*/1e-1};
  }

  const std::string kHloText =
      GetDotAlgorithmHlo(ty, ty, PrecisionConfig::ALG_UNSET);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, error_spec));
}

std::vector<PrimitiveType> AllXlaDataTypesSupportedByAlgUnsetDotLowering() {
  // We don't have a pointer to stream executor available here so we can't
  // detect the particular device we're running on with a canonical API call.
  // Instead, we just return a superset of the supported types (i.e. those that
  // are supported on the latest device), and filter out the unsupported types
  // in the test body.
  std::vector<PrimitiveType> supported_types;
  absl::c_copy_if(AllXlaDataTypes(), std::back_inserter(supported_types),
                  [](PrimitiveType type) {
                    return internal::IsResultTypeSupportedByAlgUnsetDot(
                        type, se::CudaComputeCapability::Blackwell());
                  });
  return supported_types;
}

INSTANTIATE_TEST_SUITE_P(
    DotUnsetAlgorithmEmitterTestSuite, DotUnsetAlgorithmEmitterTest,
    ::testing::ValuesIn(AllXlaDataTypesSupportedByAlgUnsetDotLowering()),
    TypeTestParamToString);

}  // namespace
}  // namespace gpu
}  // namespace xla
