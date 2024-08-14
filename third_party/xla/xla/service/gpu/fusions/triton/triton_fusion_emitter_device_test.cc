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

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "xla/autotuning.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusions/triton/triton_fusion_emitter.h"
#include "xla/service/gpu/fusions/triton/triton_test_utils.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class TritonEmitterTest : public GpuCodegenTest {
 public:
  const stream_executor::GpuComputeCapability& GpuComputeComp() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability();
  }
};

TEST_F(TritonEmitterTest, ReductionOnMinormostAxisIsEmittedCorrectly) {
  const std::string kHloText = R"(
HloModule t
maximum {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(Arg_0, Arg_1)
}

triton_reduction_computation {
  parameter_0 = f32[8,4] parameter(0)
  constant_0 = f32[] constant(0)
  ROOT reduce = f32[8] reduce(parameter_0, constant_0), dimensions={1}, to_apply=maximum
}

ENTRY main {
  param_0 = f32[8,4] parameter(0)
  ROOT triton_reduction = f32[8] fusion(param_0), kind=kCustom, calls=triton_reduction_computation, backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tile_sizes":["4"],"num_warps":"1"}}}
})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({4}),
                                          "triton_reduction_computation", R"(
CHECK:  "tt.reduce"(%[[LOAD:.*]]) <{axis = 1 : i32}>
)"));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(TritonEmitterTest, ReductionOnMajormostAxisIsEmittedCorrectly) {
  const std::string kHloText = R"(
HloModule t
maximum {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(Arg_0, Arg_1)
}

triton_reduction_computation {
  parameter_0 = f32[8,4] parameter(0)
  constant_0 = f32[] constant(0)
  ROOT reduce = f32[4] reduce(parameter_0, constant_0), dimensions={0}, to_apply=maximum
}

ENTRY main {
  param_0 = f32[8,4] parameter(0)
  ROOT triton_reduction = f32[4] fusion(param_0), kind=kCustom, calls=triton_reduction_computation, backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tile_sizes":["4"],"num_warps":"1"}}}
})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({4}),
                                          "triton_reduction_computation", R"(
CHECK:  "tt.reduce"(%[[LOAD:.*]]) <{axis = 0 : i32}>
)"));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(TritonEmitterTest, ReductionOnIntermediateAxisIsEmittedCorrectly) {
  const std::string kHloText = R"(
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
  ROOT triton_reduction = f32[5,5,5,3] fusion(param_0), kind=kCustom, calls=triton_reduction_computation, backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tile_sizes":["4", "2", "5", "1"],"num_warps":"1"}}}
})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({4, 2, 5, 1}),
                                          "triton_reduction_computation", R"(
CHECK:  tt.make_range
CHECK-COUNT-4:  tt.expand_dims
CHECK:  "tt.reduce"(%[[SELECT:.*]]) <{axis = 2 : i32}>
)"));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(TritonEmitterTest, TestReductionWithTileSizeLargerThanSourceTensor) {
  const std::string kHloText = R"(
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
  ROOT triton_reduction = f32[3] fusion(param_0), kind=kCustom, calls=triton_reduction_computation, backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tile_sizes":["3"],"num_warps":"1"}}}
})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({3}),
                                          "triton_reduction_computation", R"(
; Make sure input reduction tile is padded with a neutral value.
CHECK:  %[[LOAD:.*]] = tt.load
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

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonEmitterTest, TestSoftmaxEmitterWithSingleParameter) {
  const std::string kHloText = R"(
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
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton"}}
})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({1, 127}),
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = #xla_gpu.indexing_map<(d0) -> (d0 * 127)
CHECK:        tt.func @triton_fn(%[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK:            %[[PID:.*]] = tt.get_program_id x : i32
CHECK:            arith.index_castui %[[PID]] : i32 to index
CHECK:            tt.addptr %[[P0]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<1x128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<1x128xf32>>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG2:[^:]*]]: f32, %[[ARG3:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG2]], %[[ARG3]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<1x128xf32>) -> tensor<1xf32>
CHECK:            arith.mulf
CHECK-SAME:       tensor<1x128xf32>
CHECK:            tt.addptr %[[P1]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<1x128xf32>>
CHECK-NEXT:       tt.store
CHECK-SAME:       {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<1x128xf32>>
CHECK:            tt.return
CHECK:        }
)"));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonEmitterTest, TestGenericEmitterWithMultipleParameters) {
  const std::string kHloText = R"(
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
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0, param_1), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton"}}
}
)";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({1, 127}),
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = #xla_gpu.indexing_map<(d0) -> (d0 * 127)
CHECK-LABEL:  tt.func @triton_fn(
CHECK-SAME:                      %[[P0:[A-Za-z0-9_]*]]: !tt.ptr<f32>
CHECK-SAME:                      %[[P1:[A-Za-z0-9_]*]]: !tt.ptr<f32>
CHECK-SAME:                      %[[P2:[A-Za-z0-9_]*]]: !tt.ptr<f32>
CHECK-DAG:        %[[PID:.*]] = tt.get_program_id x : i32
CHECK-DAG:        %[[PID_INDEX:.*]] = arith.index_castui %[[PID]] : i32 to index
CHECK-DAG:        %[[C127_i64:.*]] = arith.constant 127 : i64
CHECK-DAG:        %[[ZERO_OFFSET:.*]] = arith.constant 0 : i64
CHECK-DAG:        %[[ROW_OFFSET_INDEX:.*]] = xla_gpu.apply_indexing #[[MAP]](%[[PID_INDEX]]
CHECK-DAG:        %[[ROW_OFFSET:.*]] = arith.index_castui %[[ROW_OFFSET_INDEX]] : index to i64
CHECK-DAG:        %[[ARG0:.*]] = tt.addptr %[[P0]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
CHECK-DAG:        tt.make_tensor_ptr {{.*}} : <tensor<1x128xf32>>
CHECK-DAG:        tt.load {{.*}} : !tt.ptr<tensor<1x128xf32>>
CHECK-DAG:        %[[ARG1:.*]] = tt.addptr %[[P1]], %[[ZERO_OFFSET]] : !tt.ptr<f32>, i64
CHECK-DAG:        tt.make_tensor_ptr {{.*}} : <tensor<128xf32>>
CHECK-DAG:        tt.load {{.*}} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG3:[^:]*]]: f32, %[[ARG4:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG3]], %[[ARG4]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<1x128xf32>) -> tensor<1xf32>
CHECK-DAG:        tt.addptr %[[P2]]
CHECK-DAG:        tt.make_tensor_ptr {{.*}} : <tensor<1x128xf32>>
CHECK-DAG:        tt.store {{.*}} : !tt.ptr<tensor<1x128xf32>>
)"));
}

TEST_F(TritonEmitterTest, TestGenericEmitterWithMultipleTiledDimensions) {
  const std::string kHloText = R"(
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
    backend_config={"fusion_backend_config":
      {"kind":"__triton",
       "block_level_fusion_config": {"output_tile_sizes": ["1", "1", "127"],
                                     "num_warps": "1"}}}
})";

  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({1, 1, 127}),
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = #xla_gpu.indexing_map<(d0) -> (d0 * 127)
CHECK:        tt.func @triton_fn(%[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P2:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P3:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK-DAG:        %[[PID:.*]] = tt.get_program_id x : i32
CHECK-DAG:        %[[PID_INDEX:.*]] = arith.index_castui %[[PID]] : i32 to index
CHECK-DAG:        %[[PID_i64:.*]] = arith.index_castui %[[PID_INDEX]] : index to i64
CHECK-DAG:        %[[C127_i64:.*]] = arith.constant 127 : i64
CHECK-DAG:        %[[ZERO_OFFSET:.*]] = arith.constant 0 : i64
CHECK-DAG:        %[[ROW_OFFSET_INDEX:.*]] = xla_gpu.apply_indexing #[[MAP]](%[[PID_INDEX]]
CHECK-DAG:        %[[ROW_OFFSET:.*]] = arith.index_castui %[[ROW_OFFSET_INDEX]] : index to i64
CHECK-DAG:        tt.addptr %[[P0]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
CHECK-DAG:        tt.addptr %[[P1]], %[[ZERO_OFFSET]] : !tt.ptr<f32>, i64
CHECK-DAG:        tt.addptr %[[P2]], %[[PID_i64]] : !tt.ptr<f32>, i64
CHECK-DAG:        tt.load {{.*}} : !tt.ptr<tensor<1x1x128xf32>>
CHECK-DAG:        tt.load {{.*}} : !tt.ptr<tensor<128xf32>>
CHECK-DAG:        tt.load {{.*}} : !tt.ptr<tensor<1x1xf32>>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG4:[^:]*]]: f32, %[[ARG5:[^:]*]]: f32):
CHECK-NEXT:           %[[MAX:.*]] = arith.maximumf %[[ARG4]], %[[ARG5]] : f32
CHECK-NEXT:           tt.reduce.return %[[MAX]] : f32
CHECK-NEXT:       }) : (tensor<1x1x128xf32>) -> tensor<1x1xf32>
CHECK:            tt.store {{.*}} : !tt.ptr<tensor<1x1x128xf32>>
)"));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(
    TritonEmitterTest,
    DiamondWithAdditionalDiamondParameterBroadcastedAlongReductionDimProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  const std::string kHloText = R"(
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
  ROOT _ = f32[32,16]{1,0} fusion(parameter_0, parameter_1), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tile_sizes":["1","16"],"num_warps":"1"}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(TritonEmitterTest, NestedReducerFusionGetsCodegenedCorrectly) {
  if (!SupportsBF16(GpuComputeComp())) {
    GTEST_SKIP() << "BF16 not supported.";
  }

  const std::string kHloText = R"(
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
  ROOT softmax = f32[10,128] fusion(p0), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tile_sizes":["1","128"],"num_warps":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0,
                                                           /*arel=*/0}));
}

TEST_F(
    TritonEmitterTest,
    DiamondWithAdditionalDiamondParameterBroadcastedAlongBatchDimProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  const std::string kHloText = R"(
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
  ROOT _ = f32[16,32]{1,0} fusion(parameter_0,parameter_1), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tile_sizes":["1","32"],"num_warps":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(
    TritonEmitterTest,
    DiamondWithAdditionalSplatDiamondScalarParameterProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  const std::string kHloText = R"(
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
    backend_config={"fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{"output_tile_sizes":["1","1","16"],
                                   "num_warps":"1"}}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  TF_ASSERT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({1, 1, 16}),
                                          "triton_softmax_computation", R"(
// CHECK:         #[[MAP:.*]] = #xla_gpu.indexing_map<(d0) -> (d0 * 16)
// CHECK-LABEL:   tt.func @triton_fn(
// CHECK-SAME:                       %[[P0:[A-Za-z0-9_]*]]: !tt.ptr<f32>
// CHECK-SAME:                       %[[P1:[A-Za-z0-9_]*]]: !tt.ptr<f32>
// CHECK-SAME:                       %[[P2:[A-Za-z0-9_]*]]: !tt.ptr<f32>
// CHECK-DAG:       tt.load {{.*}} : !tt.ptr<f32>
// CHECK-DAG:       tt.load {{.*}} : !tt.ptr<tensor<1x1x16xf32>>
// CHECK:           tt.store {{.*}} : !tt.ptr<tensor<1x1x16xf32>>
)"));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(
    TritonEmitterTest,
    DiamondWithAdditionalBroadcastOf1DParameterAlongNonReductionDimensionsProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  const std::string kHloText = R"(
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
  ROOT _ = f32[64,32,16]{2,1,0} fusion(f32[64,32,16]{2,1,0} parameter_1, f32[16]{0} parameter_0), kind=kCustom, calls=%triton_softmax_computation, backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tile_sizes":["1","1","16"],"num_warps":"1"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonEmitterTest, EmitterFailsIfComputeCapabilityIsBelowAmpere) {
  const std::string kHloText = R"(
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
    backend_config={"fusion_backend_config": {
      "kind":"__triton",
      "block_level_fusion_config": {"output_tile_sizes": ["1","1"],
                                    "num_warps": "1"}}}
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
                    se::CudaComputeCapability{se::CudaComputeCapability::VOLTA,
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
    backend_config={"fusion_backend_config": {
      "kind":"__triton",
      "block_level_fusion_config": {"output_tile_sizes": ["1024","1"],
                                    "num_warps": "1"}}}
})"));
  const HloFusionInstruction* triton_fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto compute_capability =
      se::CudaComputeCapability{se::CudaComputeCapability::HOPPER, /*minor=*/0};
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(compute_capability);
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {1024, 1};
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
  const std::string kHloText = R"(
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
  ROOT triton_reduction = f32[125]{0} fusion(param_0, param_1), kind=kCustom, calls=triton_reduction_computation, backend_config={"fusion_backend_config": {"kind":"__triton"}}
})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({1}),
                                          "triton_reduction_computation", R"(
CHECK:        tt.func @triton_fn(%[[P0:[A-Za-z0-9_]*]]: !tt.ptr<f32>
CHECK-SAME:                      %[[P1:[A-Za-z0-9_]*]]: !tt.ptr<f32>
CHECK-SAME:                      %[[P2:[A-Za-z0-9_]*]]: !tt.ptr<f32>
CHECK-DAG:        tt.load {{.*}} : !tt.ptr<tensor<1xf32>>
CHECK-DAG:        tt.load {{.*}} : !tt.ptr<tensor<1x128xf32>>
CHECK:            tt.reduce
CHECK:              (tensor<1x128xf32>) -> tensor<1xf32>
CHECK:            arith.mulf {{.*}} tensor<1xf32>
CHECK:            tt.store {{.*}} : !tt.ptr<tensor<1xf32>>
)"));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonEmitterTest, TestGenericEmitterWithSoftMaxSingleParameter) {
  const std::string kHloText = R"(
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
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton"}}
})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({1, 127}),
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = #xla_gpu.indexing_map<(d0) -> (d0 * 127)
CHECK:        tt.func @triton_fn(%[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK:            %[[PID:.*]] = tt.get_program_id x : i32
CHECK:            arith.index_castui %[[PID]] : i32 to index
CHECK:            tt.addptr %[[P0]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<1x128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<1x128xf32>>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG2:[^:]*]]: f32, %[[ARG3:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG2]], %[[ARG3]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<1x128xf32>) -> tensor<1xf32>
CHECK:            arith.mulf
CHECK-SAME:       tensor<1x128xf32>
CHECK:            tt.addptr %[[P1]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<1x128xf32>>
CHECK-NEXT:       tt.store
CHECK-SAME:       {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<1x128xf32>>
CHECK:            tt.return
CHECK:        }
)"));
}

TEST_F(TritonEmitterTest,
       TestGenericEmitterWithReductonAndMultidimensionalTile) {
  const std::string kHloText = R"(
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
    backend_config={"fusion_backend_config":
      {"kind":"__triton",
      "block_level_fusion_config":{"output_tile_sizes":["2","5","16"],"num_warps":"4"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(TritonEmitterTest, TestSoftMaxWithTileElementsNotAllContiguous) {
  const std::string kHloText = R"(
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
  ROOT fusion = f32[4,4,8] fusion(param_0.2), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton","block_level_fusion_config":{"output_tile_sizes":["2","2","8"],"num_warps":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                           /*arel=*/1e-6}));
}

TEST_F(TritonEmitterTest, TestSliceWithTileElementsNotAllContiguous) {
  const std::string kHloText = R"(
HloModule m

region {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT add.2 = f32[] add(param_0, param_1)
}

fused_computation {
  param_0.1 = f32[16,16,32] parameter(0)
  slice = f32[4,4,8] slice(param_0.1), slice={[2:10:2], [2:6], [3:11]}
  slice.1 = f32[4,4,8] slice(param_0.1), slice={[4:8], [8:16:2], [13:21]}
  ROOT add.3 = f32[4,4,8] add(slice, slice.1)
}

ENTRY entry_computation {
  param_0.2 = f32[16,16,32] parameter(0)
  ROOT fusion = f32[4,4,8] fusion(param_0.2), kind=kCustom, calls=fused_computation, backend_config={"fusion_backend_config": {"kind":"__triton","block_level_fusion_config":{"output_tile_sizes":["2","2","8"],"num_warps":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                           /*arel=*/1e-6}));
}

TEST_F(TritonEmitterTest, ReshapeIntoBroadcastIsLoweredCorrectly) {
  const std::string kHloText = R"(
triton_computation {
  param_0 = f32[128,256]{1,0} parameter(0)
  reshape = f32[64,2,256]{2,1,0} reshape(param_0)
  ROOT broadcast = f32[64,2,256,2]{3,2,1,0} broadcast(reshape), dimensions={0,1,2}
}

ENTRY main {
  param_0 = f32[128,256]{1,0} parameter(0)
  ROOT triton_fusion = f32[64,2,256,2]{3,2,1,0} fusion(param_0), kind=kCustom,
    calls=triton_computation,
    backend_config={"fusion_backend_config":
      {"kind":"__triton",
      "block_level_fusion_config":{"output_tile_sizes":["2","2","2","2"],
                                   "num_warps":"1"}}}
})";
  // TODO(b/353490600): parse output tile sizes from backend config.
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({4, 2, 8, 2}),
                                          "triton_computation", R"(
CHECK: tt.reshape
)"));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(TritonEmitterTest, BitcastIntoBroadcastIsLoweredCorrectly) {
  const std::string kHloText = R"(
triton_computation {
  param_0 = f32[128,256]{1,0} parameter(0)
  bitcast = f32[64,2,256]{2,1,0} bitcast(param_0)
  ROOT broadcast = f32[64,2,256,2]{3,2,1,0} broadcast(bitcast), dimensions={0,1,2}
}

ENTRY main {
  param_0 = f32[128,256]{1,0} parameter(0)
  ROOT triton_fusion = f32[64,2,256,2]{3,2,1,0} fusion(param_0), kind=kCustom,
    calls=triton_computation,
    backend_config={"fusion_backend_config":
      {"kind":"__triton",
      "block_level_fusion_config":{"output_tile_sizes":["4","2","8","2"],
                                   "num_warps":"1"}}}
})";
  // TODO(b/353490600): parse output tile sizes from backend config.
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(this, kHloText,
                                          FromOutputTileSizes({2, 2, 2, 2}),
                                          "triton_computation", R"(
CHECK: tt.reshape
)"));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

// TODO(b/353484968): move this test to a deviceless file.
TEST_F(TritonEmitterTest, GenericEmitterLowersBroadcastFrom0dOperandCorrectly) {
  const std::string kHloText = R"(
triton_computation {
  // TODO(b/348565795): make this a 0D scalar directly once this is known to be
  // supported.
  param_0 = f32[1] parameter(0)
  reshape = f32[] reshape(param_0)
  ROOT broadcast = f32[127,125]{1,0} broadcast(reshape), dimensions={}
}

ENTRY main {
  param_0 = f32[1] parameter(0)
  ROOT triton_fusion = f32[127,125]{1,0} fusion(param_0), kind=kCustom,
    calls=triton_computation,
    backend_config={"fusion_backend_config":
      {"kind":"__triton",
      "block_level_fusion_config":{"output_tile_sizes":["8","4"],
                                   "num_warps":"1"}}}
})";
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(
      this, kHloText, FromOutputTileSizes({8, 4}), "triton_computation", R"(
CHECK:       tt.broadcast
CHECK-SAME:    tensor<1x1xf32> -> tensor<8x4xf32>
)"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
