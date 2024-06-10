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
#include "xla/service/gpu/fusions/scatter_mlir.h"

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

using MlirScatterFusionTest = MlirEmitterTestBase<MlirScatterFusion>;

TEST_F(MlirScatterFusionTest, ThreadIdIndexing) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    computation {
      %p0 = f32[] parameter(0)
      %p1 = f32[] parameter(1)
      %p2 = f32[] parameter(2)
      %p3 = f32[] parameter(3)
      ROOT %tuple = (f32[], f32[]) tuple(f32[] %p2, f32[] %p3)
    }
    scatter {
      %operand0 = f32[300,200] parameter(0)
      %operand1 = f32[300,200] parameter(1)
      %indices = s32[42,1] parameter(2)
      %update.1 = f32[42,10,20] parameter(3)
      %update.2 = f32[42,10,20]parameter(4)

      ROOT %scatter = (f32[300,200], f32[300,200]) scatter(
          f32[300,200] %operand0,
          f32[300,200] %operand1,
          s32[42,1] %indices,
          f32[42,10,20] %update.1,
          f32[42,10,20] %update.2
        ),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=computation
    }
    ENTRY entry {
      %operand0 = f32[300,200] parameter(0)
      %operand1 = f32[300,200] parameter(1)
      %indices = s32[42,1] parameter(2)
      %update.1 = f32[42,10,20] parameter(3)
      %update.2 = f32[42,10,20]parameter(4)
      ROOT %fusion = (f32[300,200], f32[300,200]) fusion(
        %operand0, %operand1, %indices, %update.1, %update.2),
        kind=kLoop, calls=scatter
    }
  )"));
  thread_id_printer_.SetSymbolName(0, "chunk_id");
  thread_id_printer_.SetSymbolName(1, "unroll_id");
  thread_id_printer_.SetSymbolName(2, "index_id");

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  MlirScatterFusion fusion(analysis);

  constexpr auto kUpdatesIndexing = R"(
    (th_x, th_y, th_z, bl_x, bl_y, bl_z)[chunk_id, unroll_id] -> (
    ((bl_x * 16 + th_x floordiv 8) floordiv 25) mod 42,
    ((bl_x * 32 + th_x floordiv 4) floordiv 5) mod 10,
    (bl_x * 128 + th_x) mod 20)
    domain:
    th_x in [0, 127]
    th_y in [0, 0]
    th_z in [0, 0]
    bl_x in [0, 65]
    bl_y in [0, 0]
    bl_z in [0, 0]
    chunk_id in [0, 0]
    unroll_id in [0, 0]
    th_x + bl_x * 128 in [0, 8399]
  )";
  EXPECT_THAT(
      fusion
          .ComputeThreadIdToInputIndexing(
              /*root_index=*/0, /*hero_operand_index=*/3, &mlir_context_)
          ->ToString(thread_id_printer_),
      MatchIndexingString(kUpdatesIndexing));
  EXPECT_THAT(
      fusion
          .ComputeThreadIdToInputIndexing(
              /*root_index=*/0, /*hero_operand_index=*/4, &mlir_context_)
          ->ToString(thread_id_printer_),
      MatchIndexingString(kUpdatesIndexing));
  EXPECT_THAT(
      fusion
          .ComputeThreadIdToInputIndexing(
              /*root_index=*/1, /*hero_operand_index=*/3, &mlir_context_)
          ->ToString(thread_id_printer_),
      MatchIndexingString(kUpdatesIndexing));
  EXPECT_THAT(
      fusion
          .ComputeThreadIdToInputIndexing(
              /*root_index=*/1, /*hero_operand_index=*/4, &mlir_context_)
          ->ToString(thread_id_printer_),
      MatchIndexingString(kUpdatesIndexing));

  constexpr auto kIndicesIndexing = R"(
    (th_x, th_y, th_z, bl_x, bl_y, bl_z)[chunk_id, unroll_id, index_id] -> (
    ((bl_x * 16 + th_x floordiv 8) floordiv 25) mod 42, 0)
    domain:
    th_x in [0, 127]
    th_y in [0, 0]
    th_z in [0, 0]
    bl_x in [0, 65]
    bl_y in [0, 0]
    bl_z in [0, 0]
    chunk_id in [0, 0]
    unroll_id in [0, 0]
    index_id in [0, 0]
    th_x + bl_x * 128 in [0, 8399]
  )";
  EXPECT_THAT(
      fusion
          .ComputeThreadIdToInputIndexing(
              /*root_index=*/0, /*hero_operand_index=*/2, &mlir_context_)
          ->ToString(thread_id_printer_),
      MatchIndexingString(kIndicesIndexing));
  EXPECT_THAT(
      fusion
          .ComputeThreadIdToInputIndexing(
              /*root_index=*/1, /*hero_operand_index=*/2, &mlir_context_)
          ->ToString(thread_id_printer_),
      MatchIndexingString(kIndicesIndexing));
}

TEST_F(MlirScatterFusionTest, Scatter_UniqueIndices) {
  auto kHloString = R"(
    HloModule module

    add {
      %p0 = f32[] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %sum = f32[] add(%p0, %p1)
    }
    scatter {
      %operand = f32[10,5]  parameter(0)
      %indices = s32[8,1] parameter(1)
      %update = f32[8,1,2] parameter(2)

      ROOT %scatter = f32[10,5] scatter(
          f32[10,5] %operand,
          s32[8,1] %indices,
          f32[8,1,2] %update
        ),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        unique_indices=true,
        to_apply=add
    }
    ENTRY entry {
      %c1 = f32[] constant(1)
      %c1_tensor = f32[10,5]  broadcast(%c1), dimensions={}
      %indices = s32[8,1] constant({{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}})
      %update = f32[8, 1, 2] parameter(0)
      ROOT %fusion = f32[10, 5] fusion(
        %c1_tensor, %indices, %update), kind=kLoop, calls=scatter
    }
  )";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0 floordiv 2)>
    // CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (d0 mod 2)>

    // CHECK-LABEL: func.func @fused_computation(
    // CHECK-SAME:    %[[OPERAND:[a-zA-Z0-9]*]]: tensor<10x5xf32>
    // CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]*]]: tensor<8x1xi32>
    // CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]*]]: tensor<8x1x2xf32>
    // CHECK-SAME:    %[[OUT:[a-zA-Z0-9]*]]: tensor<10x5xf32>

    // CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG:       %[[C9:.*]] = arith.constant 9 : index

    // CHECK:      %[[TH_X:.*]] = gpu.thread_id  x
    // CHECK:      %[[SLICE_ID:.*]] = xla_gpu.apply_indexing #[[$MAP0]](%[[TH_X]]
    // CHECK:      %[[SLICE_X:.*]] = xla_gpu.apply_indexing #[[$MAP1]](%[[TH_X]]

    // CHECK:      %[[UPD_ELEM:.*]] = xla_gpu.pure_call @scatter_update(
    // CHECK-SAME:  %[[OPERAND]], %[[INDICES]], %[[UPDATES]],
    // CHECK-SAME:  %[[SLICE_ID]], %[[C0]], %[[SLICE_X]])

    // CHECK:      xla_gpu.pure_call @scatter_indices(%[[OPERAND]], %[[INDICES]]
    // CHECK-SAME:  %[[UPDATES]], %[[SLICE_ID]], %[[C0]])

    // CHECK:      %[[IN_BOUNDS:.*]] = arith.cmpi ule
    // CHECK:      scf.if %[[IN_BOUNDS]] -> (tensor<10x5xf32>) {
    // CHECK:        %[[CURRENT:.*]] = xla_gpu.pure_call @scatter_operand(
    // CHECK-SAME:  %[[OPERAND]], %[[INDICES]], %[[UPDATES]],
    // CHECK-SAME:  %[[SLICE_X]])
    // CHECK:        %[[COMBINED:.*]] = arith.addf %[[CURRENT]], %[[UPD_ELEM]]
    // CHECK:        %[[UPDATED:.*]] = tensor.insert %[[COMBINED]]
    // CHECK-SAME:     into %[[OUT]][%{{.*}}, %[[SLICE_X]]] : tensor<10x5xf32>
    // CHECK:        scf.yield %[[UPDATED]] : tensor<10x5xf32>
    // CHECK:      } else {
    // CHECK:        scf.yield %[[OUT]] : tensor<10x5xf32>
    // CHECK:      }
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirScatterFusionTest, Scatter_Unsigned) {
  auto kHloString = R"(
    HloModule module

    add {
      %p0 = f32[] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %sum = f32[] add(%p0, %p1)
    }
    scatter {
      %operand = f32[10,5]  parameter(0)
      %indices = u32[24,1] parameter(1)
      %update = f32[24,2,3] parameter(2)

      ROOT %scatter = f32[10,5] scatter(%operand, %indices, %update),
          update_window_dims={1,2},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add
    }
    ENTRY entry {
      %c1 = f32[] constant(1)
      %c1_tensor = f32[10,5]  broadcast(%c1), dimensions={}
      %indices = u32[24,1] parameter(0)
      %update = f32[24, 2, 3] parameter(1)
      ROOT %fusion = f32[10, 5] fusion(%c1_tensor, %indices, %update),
          kind=kLoop, calls=scatter
    }
  )";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: func.func @fused_computation(
    // CHECK: %[[PARAM:.*]] = xla_gpu.pure_call @scatter_indices
    // CHECK: %[[INDEX:.*]] = arith.index_castui %[[PARAM]]
    // CHECK: arith.cmpi ule, %[[INDEX]]
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirScatterFusionTest, Scatter_Add) {
  auto kHloString = R"(
    HloModule module

    add {
      %p0 = f32[] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %sum = f32[] add(%p0, %p1)
    }
    scatter {
      %operand = f32[10,5]  parameter(0)
      %indices = s32[24,1] parameter(1)
      %update = f32[24,2,3] parameter(2)

      ROOT %scatter = f32[10,5] scatter(
          f32[10,5] %operand,
          s32[24,1] %indices,
          f32[24,2,3] %update
        ),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        unique_indices=false,
        to_apply=add
    }
    ENTRY entry {
      %c1 = f32[] constant(1)
      %c1_tensor = f32[10,5]  broadcast(%c1), dimensions={}
      %indices = s32[24,1] parameter(0)
      %update = f32[24, 2, 3] parameter(1)
      ROOT %fusion = f32[10, 5] fusion(
        %c1_tensor, %indices, %update), kind=kLoop, calls=scatter
    }
  )";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK-LABEL: func.func @fused_computation(
    // CHECK-SAME:    %[[OPERAND:[a-zA-Z0-9]*]]: tensor<10x5xf32>
    // CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]*]]: tensor<24x1xi32>
    // CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]*]]: tensor<24x2x3xf32>
    // CHECK-SAME:    %[[OUT:[a-zA-Z0-9]*]]: tensor<10x5xf32>

    // CHECK: %[[UPD_ELEM:.*]] = xla_gpu.pure_call @scatter_update
    // CHECK: %[[IN_BOUNDS:.*]] = arith.cmpi ule
    // CHECK: scf.if %[[IN_BOUNDS]] -> (tensor<10x5xf32>) {
    // CHECK:   %[[RMW:.*]] = xla_gpu.atomic_rmw %[[OUT]]
    // CHECK:   ^bb0(%[[CUR_VALUE:.*]]: f32):
    // CHECK:     %[[SUM:.*]] = arith.addf %[[CUR_VALUE]], %[[UPD_ELEM]]
    // CHECK:     xla_gpu.yield %[[SUM]] : f32
    // CHECK:   }
    // CHECK:   scf.yield %[[RMW]] : tensor<10x5xf32>
    // CHECK: } else {
    // CHECK:   scf.yield %[[OUT]] : tensor<10x5xf32>
    // CHECK: }
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirScatterFusionTest, Scatter_Overwrite) {
  auto kHloString = R"(
    HloModule module

    overwrite {
      %p0 = f32[] parameter(0)
      ROOT %p1 = f32[] parameter(1)
    }
    scatter {
      %operand = f32[10,5]  parameter(0)
      %indices = s32[3,1] parameter(1)
      %update = f32[3,2,3] parameter(2)

      ROOT %scatter = f32[10,5] scatter(
          f32[10,5] %operand,
          s32[3,1] %indices,
          f32[3,2,3] %update
        ),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        unique_indices=false,
        to_apply=overwrite
    }
    ENTRY entry {
      %c1 = f32[] constant(1)
      %c1_tensor = f32[10,5]  broadcast(%c1), dimensions={}
      %indices = s32[3,1] constant({ {0}, {3}, {6}})
      %update = f32[3, 2, 3] parameter(0)
      ROOT %fusion = f32[10, 5] fusion(
        %c1_tensor, %indices, %update), kind=kLoop, calls=scatter
    }
  )";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK-LABEL: func.func @fused_computation(
    // CHECK-SAME:    %[[OPERAND:[a-zA-Z0-9]*]]: tensor<10x5xf32>
    // CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]*]]: tensor<3x1xi32>
    // CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]*]]: tensor<3x2x3xf32>
    // CHECK-SAME:    %[[OUT:[a-zA-Z0-9]*]]: tensor<10x5xf32>

    // CHECK: %[[UPD_ELEM:.*]] = xla_gpu.pure_call @scatter_update
    // CHECK: %[[IN_BOUNDS:.*]] = arith.cmpi ule
    // CHECK: scf.if %[[IN_BOUNDS]] -> (tensor<10x5xf32>) {
    // CHECK:   %[[RMW:.*]] = xla_gpu.atomic_rmw %[[OUT]]
    // CHECK:   ^bb0(%[[CUR_VALUE:.*]]: f32):
    // CHECK:     xla_gpu.yield %[[UPD_ELEM]] : f32
    // CHECK:   }
    // CHECK:   scf.yield %[[RMW]] : tensor<10x5xf32>
    // CHECK: } else {
    // CHECK:   scf.yield %[[OUT]] : tensor<10x5xf32>
    // CHECK: }
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
