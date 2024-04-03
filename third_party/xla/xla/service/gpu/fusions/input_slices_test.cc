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
#include "xla/service/gpu/fusions/input_slices.h"

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

class InputSlicesTest : public HloTestBase {
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

TEST_F(InputSlicesTest, ThreadIndexing) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    fused_computation {
      %input = f32[2,3,5,7]{2,1,0,3} parameter(0)
      slice0 = f32[1,2,3,5]{2,1,0,3} slice(input), slice={[0:1],[1:3],[0:3],[2:7]}
      slice1 = f32[1,2,3,5]{2,1,0,3} slice(input), slice={[0:1],[0:2],[0:3],[2:7]}
      ROOT tuple = (f32[1,2,3,5]{2,1,0,3}, f32[1,2,3,5]{2,1,0,3}) tuple(slice0, slice1)
    }

    ENTRY entry {
      %input = f32[2,3,5,7]{2,1,0,3} parameter(0)
      ROOT %fusion = (f32[1,2,3,5]{2,1,0,3}, f32[1,2,3,5]{2,1,0,3}) fusion(%input), kind=kLoop, calls=fused_computation
    })")
                    .value();

  stream_executor::DeviceDescription device_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis_fused = AnalyzeFusion(*root, device_info);

  TF_ASSERT_OK_AND_ASSIGN(
      auto emitter,
      GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis_fused}));
  auto fusion = dynamic_cast<InputSlicesFusion*>(emitter.get());
  ASSERT_NE(fusion, nullptr);

  auto thread_id_to_output_indexing =
      fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context_);
  EXPECT_THAT(thread_id_to_output_indexing->ToString(printer_),
              MatchIndexingString(R"(
    (th_x, th_y, th_z, bl_x, bl_y, bl_z)[chunk_id, unroll_id] -> (0,
      ((th_x + bl_x * 128) floordiv 3) mod 2,
       (th_x + bl_x * 128) mod 3,
       ((bl_x * 64 + th_x floordiv 2) floordiv 3) mod 5)
    domain:
    th_x in [0, 127]
    th_y in [0, 0]
    th_z in [0, 0]
    bl_x in [0, 1]
    bl_y in [0, 0]
    bl_z in [0, 0]
    chunk_id in [0, 0]
    unroll_id in [0, 0]
    th_x + bl_x * 128 in [0, 29]
  )"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
