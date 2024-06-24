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
#include "xla/service/gpu/fusions/in_place_dynamic_update_slice_mlir.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/service/gpu/fusions/mlir_emitter_test_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using MlirInPlaceDynamicUpdateSliceFusionTest =
    MlirEmitterTestBase<MlirInPlaceDynamicUpdateSliceFusion>;

TEST_F(MlirInPlaceDynamicUpdateSliceFusionTest, ThreadIndexing) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    fused_computation {
      in = f32[20,30] parameter(0)
      updates = f32[5,6] parameter(1)
      i0 = s32[] parameter(2)
      i1 = s32[] parameter(3)
      ROOT updated = f32[20,30] dynamic-update-slice(in, updates, i0, i1)
    }
    ENTRY entry {
      in = f32[20,30] parameter(0)
      updates = f32[5,6] parameter(1)
      i0 = s32[] constant(2)
      i1 = s32[] constant(3)
      ROOT fusion = f32[20,30] fusion(in, updates, i0, i1), kind=kLoop, calls=fused_computation
    }
  )"));
  thread_id_printer_.SetSymbolName(0, "chunk_id");
  thread_id_printer_.SetSymbolName(1, "unroll_id");

  auto* root = module->entry_computation()->root_instruction();

  auto analysis = AnalyzeFusion(*root, device_info_);
  MlirInPlaceDynamicUpdateSliceFusion fusion(analysis);

  auto thread_id_update_indexing = fusion.ComputeThreadIdToInputIndexing(
      /*root_index=*/0, /*hero_operand_index=*/1, &mlir_context_);
  EXPECT_THAT(thread_id_update_indexing->ToString(thread_id_printer_),
              MatchIndexingString(R"(
    (th_x, th_y, th_z, bl_x, bl_y, bl_z)[chunk_id, unroll_id] -> (
    th_x floordiv 6, th_x mod 6)
    domain:
    th_x in [0, 29]
    th_y in [0, 0]
    th_z in [0, 0]
    bl_x in [0, 0]
    bl_y in [0, 0]
    bl_z in [0, 0]
    chunk_id in [0, 0]
    unroll_id in [0, 0]
  )"));
  auto thread_id_dst_indexing = fusion.ComputeThreadIdToInputIndexing(
      /*root_index=*/0, /*hero_operand_index=*/0, &mlir_context_);
  EXPECT_THAT(thread_id_dst_indexing, ::testing::Eq(std::nullopt));
}

TEST_F(MlirInPlaceDynamicUpdateSliceFusionTest, SimpleDUS) {
  auto kHloString = R"(
    HloModule module

    fused_computation {
      in = f32[20,30] parameter(0)
      updates = f32[5,6] parameter(1)
      i0 = s32[] parameter(2)
      i1 = s32[] parameter(3)
      ROOT updated = f32[20,30] dynamic-update-slice(in, updates, i0, i1)
    }
    ENTRY entry {
      in = f32[20,30] parameter(0)
      updates = f32[5,6] parameter(1)
      i0 = s32[] constant(2)
      i1 = s32[] constant(3)
      ROOT fusion = f32[20,30] fusion(in, updates, i0, i1), kind=kLoop, calls=fused_computation
    }
  )";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK-DAG: #[[MAP_1:.*]] = affine_map<(d0) -> (d0 floordiv 6)>
    // CHECK-DAG: #[[MAP_2:.*]] = affine_map<(d0) -> (d0 mod 6)>
    // CHECK:     func.func @fused_computation
    // CHECK-SAME:  %arg0: tensor<20x30xf32>
    // CHECK-SAME:  %arg1: tensor<5x6xf32>
    // CHECK-SAME:  %arg2: tensor<i32>
    // CHECK-SAME:  %arg3: tensor<i32>
    // CHECK-SAME:  %arg4: tensor<20x30xf32>
    // CHECK-DAG:   %[[C_24:.*]] = arith.constant 24
    // CHECK-DAG:   %[[C_15:.*]] = arith.constant 15
    // CHECK-DAG:   %[[C_0:.*]] = arith.constant 0
    // CHECK:       %[[THREAD_ID:.*]] = gpu.thread_id  x
    // CHECK:       %[[INPUT_INDEX_0:.*]] = xla_gpu.apply_indexing #[[MAP_1]](%[[THREAD_ID]] in [0, 29])
    // CHECK:       %[[INPUT_INDEX_1:.*]] = xla_gpu.apply_indexing #[[MAP_2]](%[[THREAD_ID]] in [0, 29])
    // CHECK:       %[[I0:.*]] = xla_gpu.pure_call @fused_computation_i0
    // CHECK:       %[[I1:.*]] = xla_gpu.pure_call @fused_computation_i1
    // CHECK:       %[[IDX0:.*]] = arith.index_cast %[[I0]]
    // CHECK:       %[[MIN0:.*]] = arith.minsi %[[IDX0]], %[[C_15]]
    // CHECK:       %[[MAX0:.*]] = arith.maxsi %[[MIN0]], %[[C_0]]
    // CHECK:       %[[ADD0:.*]] = arith.addi %[[INPUT_INDEX_0]], %[[MAX0]]
    // CHECK:       %[[IDX1:.*]] = arith.index_cast %[[I1]]
    // CHECK:       %[[MIN1:.*]] = arith.minsi %[[IDX1]], %[[C_24]]
    // CHECK:       %[[MAX1:.*]] = arith.maxsi %[[MIN1]], %[[C_0]]
    // CHECK:       %[[ADD1:.*]] = arith.addi %[[INPUT_INDEX_1]], %[[MAX1]]
    // CHECK:       %[[UPDATE:.*]] = xla_gpu.pure_call @fused_computation_updates
    // CHECK:       %[[INSERT:.*]] = tensor.insert %[[UPDATE:.*]] into %arg4[%[[ADD0]], %[[ADD1]]]
    // CHECK:       return %[[INSERT]]
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirInPlaceDynamicUpdateSliceFusionTest, OutOfBoundDUS) {
  auto kHloString = R"(
    HloModule module

    fused_computation {
      in = f32[7,8] parameter(0)
      updates = f32[2,3] parameter(1)
      i0 = s32[] parameter(2)
      i1 = s32[] parameter(3)
      ROOT updated = f32[7,8] dynamic-update-slice(in, updates, i0, i1)
    }
    ENTRY entry {
      in = f32[7,8] parameter(0)
      updates = f32[2,3] parameter(1)
      i0 = s32[] constant(-20)
      i1 = s32[] constant(30)
      ROOT fusion = f32[7,8] fusion(in, updates, i0, i1), kind=kLoop, calls=fused_computation
    }
  )";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK-DAG: #[[MAP_1:.*]] = affine_map<(d0) -> (d0 floordiv 3)>
    // CHECK-DAG: #[[MAP_2:.*]] = affine_map<(d0) -> (d0 mod 3)>
    // CHECK:     func.func @fused_computation
    // CHECK-SAME:  %arg0: tensor<7x8xf32>
    // CHECK-SAME:  %arg1: tensor<2x3xf32>
    // CHECK-SAME:  %arg2: tensor<i32>
    // CHECK-SAME:  %arg3: tensor<i32>
    // CHECK-SAME:  %arg4: tensor<7x8xf32>
    // CHECK-DAG:   %[[C_5:.*]] = arith.constant 5
    // CHECK-DAG:   %[[C_0:.*]] = arith.constant 0
    // CHECK:       %[[THREAD_ID:.*]] = gpu.thread_id  x
    // CHECK:       %[[INPUT_INDEX_0:.*]] = xla_gpu.apply_indexing #[[MAP_1]](%[[THREAD_ID]] in [0, 5])
    // CHECK:       %[[INPUT_INDEX_1:.*]] = xla_gpu.apply_indexing #[[MAP_2]](%[[THREAD_ID]] in [0, 5])
    // CHECK:       %[[I0:.*]] = xla_gpu.pure_call @fused_computation_i0
    // CHECK:       %[[I1:.*]] = xla_gpu.pure_call @fused_computation_i1
    // CHECK:       %[[IDX0:.*]] = arith.index_cast %[[I0]]
    // CHECK:       %[[MIN0:.*]] = arith.minsi %[[IDX0]], %[[C_5]]
    // CHECK:       %[[MAX0:.*]] = arith.maxsi %[[MIN0]], %[[C_0]]
    // CHECK:       %[[ADD0:.*]] = arith.addi %[[INPUT_INDEX_0]], %[[MAX0]]
    // CHECK:       %[[IDX1:.*]] = arith.index_cast %[[I1]]
    // CHECK:       %[[MIN1:.*]] = arith.minsi %[[IDX1]], %[[C_5]]
    // CHECK:       %[[MAX1:.*]] = arith.maxsi %[[MIN1]], %[[C_0]]
    // CHECK:       %[[ADD1:.*]] = arith.addi %[[INPUT_INDEX_1]], %[[MAX1]]
    // CHECK:       %[[UPDATE:.*]] = xla_gpu.pure_call @fused_computation_updates
    // CHECK:       %[[INSERT:.*]] = tensor.insert %[[UPDATE:.*]] into %arg4[%[[ADD0]], %[[ADD1]]]
    // CHECK:       return %[[INSERT]]
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirInPlaceDynamicUpdateSliceFusionTest, BitcastDus) {
  auto kHloString = R"(
    HloModule module

    fused_computation {
      in = f32[20,30] parameter(0)
      updates = f32[5,6] parameter(1)
      i0 = s32[] parameter(2)
      i1 = s32[] parameter(3)
      updated = f32[20,30] dynamic-update-slice(in, updates, i0, i1)
      ROOT bitcast = f32[600] bitcast(updated)
    }
    ENTRY entry {
      in = f32[20,30] parameter(0)
      updates = f32[5,6] parameter(1)
      i0 = s32[] constant(2)
      i1 = s32[] constant(3)
      ROOT fusion = f32[600] fusion(in, updates, i0, i1), kind=kLoop, calls=fused_computation
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirInPlaceDynamicUpdateSliceFusionTest, MOFDus) {
  auto kHloString = R"(
    HloModule module

    fused_computation {
      p0 = f32[10,11,12] parameter(0)
      p1 = f32[1,11,12] parameter(1)
      p2 = f32[8,11,12] parameter(2)
      p3 = f32[1,11,12] parameter(3)
      p4 = s32[] parameter(4)
      c0 = s32[] constant(0)
      cmp = pred[] compare(p4, c0), direction=EQ
      broadcast = pred[1,11,12] broadcast(cmp), dimensions={}
      select = f32[1,11,12] select(broadcast, p1, p3)
      dus0 = f32[10,11,12] dynamic-update-slice(p0, select, c0, c0, c0)
      dus1 = f32[8,11,12] dynamic-update-slice(p2, select, c0, c0, c0)
      ROOT tuple = (f32[10,11,12], f32[8,11,12]) tuple(dus0, dus1)
    }
    ENTRY entry {
      p0 = f32[10,11,12] parameter(0)
      p1 = f32[1,11,12] parameter(1)
      p2 = f32[8,11,12] parameter(2)
      p3 = f32[1,11,12] parameter(3)
      p4 = s32[] parameter(4)
      ROOT fusion_root_multiple = (f32[10,11,12], f32[8,11,12])
        fusion(p0, p1, p2, p3, p4), kind=kLoop, calls=fused_computation
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirInPlaceDynamicUpdateSliceFusionTest, OperandSubgraphWithTwoRoots) {
  auto kHloString = R"(
    HloModule in_place_dus

    dus_fusion {
      param_0.8 = f32[512,512]{1,0} parameter(0)
      param_1.10 = f32[128,128]{1,0} parameter(1)
      param_3.32 = s32[] parameter(3)
      two = s32[] constant(2)
      param_3_mod_2 = s32[] remainder(param_3.32, two)
      one = s32[] constant(1)
      param_3_plus_one = s32[] add(param_3_mod_2, one)
      param_2.32 = s32[] parameter(2)
      param_2_plus_one = s32[] add(param_2.32, one)
      ROOT dynamic-update-slice.5.1 = f32[512,512]{1,0} dynamic-update-slice(
        param_0.8, param_1.10, param_2_plus_one, param_3_plus_one)
    }
    ENTRY entry {
      p0 = f32[512,512]{1,0} parameter(0)
      p1 = f32[128,128]{1,0} parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT dus = f32[512,512]{1,0} fusion(p0, p1, p2, p3), kind=kLoop, calls=dus_fusion
    }
  )";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK:     func.func @fused_computation(
    // CHECK-SAME:  %[[ARG0:[^:]+]]: tensor<512x512xf32>
    // CHECK-SAME:  , %[[ARG1:[^:]+]]: tensor<128x128xf32>
    // CHECK-SAME:  , %[[ARG2:[^:]+]]: tensor<i32>
    // CHECK-SAME:  , %[[ARG3:[^:]+]]: tensor<i32>
    // CHECK-SAME:  , %[[ARG4:[^:]+]]: tensor<512x512xf32>
    // CHECK-DAG:   %[[C_384:.*]] = arith.constant 384
    // CHECK-DAG:   %[[C_0:.*]] = arith.constant 0
    // CHECK:       %[[THREAD_ID:.*]] = gpu.thread_id  x
    // CHECK:       %[[BLOCK_ID:.*]] = gpu.block_id  x
    // CHECK:       %[[I0:.*]] = xla_gpu.pure_call @dus_fusion_param_2_plus_one
    // CHECK:       %[[I1:.*]] = xla_gpu.pure_call @dus_fusion_param_3_plus_one
    // CHECK:       %[[IDX0:.*]] = arith.index_cast %[[I0]]
    // CHECK:       %[[MIN0:.*]] = arith.minsi %[[IDX0]], %[[C_384]]
    // CHECK:       %[[MAX0:.*]] = arith.maxsi %[[MIN0]], %[[C_0]]
    // CHECK:       %[[ADD0:.*]] = arith.addi %[[BLOCK_ID]], %[[MAX0]]
    // CHECK:       %[[IDX1:.*]] = arith.index_cast %[[I1]]
    // CHECK:       %[[MIN1:.*]] = arith.minsi %[[IDX1]], %[[C_384]]
    // CHECK:       %[[MAX1:.*]] = arith.maxsi %[[MIN1]], %[[C_0]]
    // CHECK:       %[[ADD1:.*]] = arith.addi %[[THREAD_ID]], %[[MAX1]]
    // CHECK:       %[[UPDATE:.*]] = xla_gpu.pure_call @dus_fusion_param_1_10
    // CHECK:       %[[INSERT:.*]] = tensor.insert %[[UPDATE:.*]] into %[[ARG4]][%[[ADD0]], %[[ADD1]]]
    // CHECK:       return %[[INSERT]]
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-6}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
