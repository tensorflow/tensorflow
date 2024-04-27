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
#include "xla/service/gpu/fusions/in_place_dynamic_update_slice.h"

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

class InPlaceDynamicUpdateSliceFusionTest : public HloTestBase {
 public:
  void SetUp() override {
    HloTestBase::SetUp();
    printer_ =
        AffineMapPrinter({"th_x", "th_y", "th_z", "bl_x", "bl_y", "bl_z"},
                         {"chunk_id", "unroll_id"});
  }

 protected:
  AffineMapPrinter printer_;
  mlir::MLIRContext mlir_context_;
};

TEST_F(InPlaceDynamicUpdateSliceFusionTest, ThreadIndexing) {
  auto module = ParseAndReturnVerifiedModule(R"(
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
  )")
                    .value();

  stream_executor::DeviceDescription device_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis_fused = AnalyzeFusion(*root, device_info);

  TF_ASSERT_OK_AND_ASSIGN(
      auto emitter,
      GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis_fused}));
  auto fusion = dynamic_cast<InPlaceDynamicUpdateSliceFusion*>(emitter.get());
  ASSERT_NE(fusion, nullptr);

  auto thread_id_update_indexing = fusion->ComputeThreadIdToInputIndexing(
      /*root_index=*/0, /*hero_operand_index=*/1, &mlir_context_);
  EXPECT_THAT(thread_id_update_indexing->ToString(printer_),
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
  auto thread_id_dst_indexing = fusion->ComputeThreadIdToInputIndexing(
      /*root_index=*/0, /*hero_operand_index=*/0, &mlir_context_);
  EXPECT_THAT(thread_id_dst_indexing, ::testing::Eq(std::nullopt));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
