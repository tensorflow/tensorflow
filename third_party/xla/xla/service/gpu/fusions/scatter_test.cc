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
#include "xla/service/gpu/fusions/scatter.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/service/gpu/fusions/fusions.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class ScatterFusionTest : public HloTestBase {
 public:
  void SetUp() override {
    HloTestBase::SetUp();
    printer_ =
        AffineMapPrinter({"th_x", "th_y", "th_z", "bl_x", "bl_y", "bl_z"},
                         {"chunk_id", "unroll_id", "index_id"});
  }

 protected:
  AffineMapPrinter printer_;
  mlir::MLIRContext mlir_context_;
};

TEST_F(ScatterFusionTest, ScatterFusion) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    add (lhs: f32[], rhs: f32[]) -> f32[] {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    fused_computation {
      %input = f32[2,9] parameter(0)
      %indices = s32[3] parameter(1)
      %updates = f32[3,9] parameter(2)
      ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }

    ENTRY entry {
      %input = f32[2,9] parameter(0)
      %indices = s32[3] parameter(1)
      %updates = f32[3,9] parameter(2)
      ROOT %fusion = f32[2,9] fusion(%input, %indices, %updates), kind=kLoop, calls=fused_computation
    })")
                    .value();

  stream_executor::DeviceDescription device_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis_fused = AnalyzeFusion(*root, device_info);

  auto emitter =
      GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis_fused});
  auto scatter_fusion = dynamic_cast<ScatterFusion*>(emitter.get());
  ASSERT_NE(scatter_fusion, nullptr);
  EXPECT_EQ(scatter_fusion->launch_dimensions().launch_bound(),
            3 * 9 /* updates size */);
}

TEST_F(ScatterFusionTest, ThreadIdIndexing) {
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
  stream_executor::DeviceDescription device_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis_fused = AnalyzeFusion(*root, device_info);

  auto emitter =
      GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis_fused});
  auto fusion = dynamic_cast<ScatterFusion*>(emitter.get());
  ASSERT_NE(fusion, nullptr);

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
          ->ComputeThreadIdToInputIndexing(
              /*root_index=*/0, /*hero_operand_index=*/3, &mlir_context_)
          ->ToString(printer_),
      MatchIndexingString(kUpdatesIndexing));
  EXPECT_THAT(
      fusion
          ->ComputeThreadIdToInputIndexing(
              /*root_index=*/0, /*hero_operand_index=*/4, &mlir_context_)
          ->ToString(printer_),
      MatchIndexingString(kUpdatesIndexing));
  EXPECT_THAT(
      fusion
          ->ComputeThreadIdToInputIndexing(
              /*root_index=*/1, /*hero_operand_index=*/3, &mlir_context_)
          ->ToString(printer_),
      MatchIndexingString(kUpdatesIndexing));
  EXPECT_THAT(
      fusion
          ->ComputeThreadIdToInputIndexing(
              /*root_index=*/1, /*hero_operand_index=*/4, &mlir_context_)
          ->ToString(printer_),
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
          ->ComputeThreadIdToInputIndexing(
              /*root_index=*/0, /*hero_operand_index=*/2, &mlir_context_)
          ->ToString(printer_),
      MatchIndexingString(kIndicesIndexing));
  EXPECT_THAT(
      fusion
          ->ComputeThreadIdToInputIndexing(
              /*root_index=*/1, /*hero_operand_index=*/2, &mlir_context_)
          ->ToString(printer_),
      MatchIndexingString(kIndicesIndexing));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
