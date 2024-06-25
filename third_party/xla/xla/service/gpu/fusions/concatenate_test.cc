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
#include "xla/service/gpu/fusions/concatenate.h"

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

class ConcatenateTest : public HloTestBase {
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

TEST_F(ConcatenateTest, ThreadIndexing) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    fused_computation {
      param0 = f32[200] parameter(0)
      param1 = f32[400] parameter(1)
      param2 = f32[300] parameter(2)
      ROOT concat = f32[900] concatenate(param0, param1, param2), dimensions={0}
    }
    ENTRY main {
      param0 = f32[200] parameter(0)
      param1 = f32[400] parameter(1)
      param2 = f32[300] parameter(2)
      ROOT fusion = f32[900] fusion(param0, param1, param2),
        calls=fused_computation, kind=kLoop
    }
  )")
                    .value();

  stream_executor::DeviceDescription device_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis_fused = AnalyzeFusion(*root, device_info);

  auto emitter =
      GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis_fused});
  auto fusion = dynamic_cast<ConcatenateFusion*>(emitter.get());
  ASSERT_NE(fusion, nullptr);

  constexpr auto kIndexing = R"(
    (th_x, th_y, th_z, bl_x, bl_y, bl_z)[chunk_id, unroll_id] -> (
    (bl_x * 128 + th_x) mod 400)
    domain:
    th_x in [0, 128)
    th_y in [0, 1)
    th_z in [0, 1)
    bl_x in [0, 4)
    bl_y in [0, 1)
    bl_z in [0, 1)
    chunk_id in [0, 1)
    unroll_id in [0, 1)
    th_x + bl_x * 128 in [0, 400)
  )";
  EXPECT_THAT(
      fusion
          ->ComputeThreadIdToInputIndexing(
              /*root_index=*/0, /*hero_operand_index=*/0, &mlir_context_)
          ->ToString(printer_),
      MatchIndexingString(kIndexing));
  EXPECT_THAT(
      fusion
          ->ComputeThreadIdToInputIndexing(
              /*root_index=*/0, /*hero_operand_index=*/1, &mlir_context_)
          ->ToString(printer_),
      MatchIndexingString(kIndexing));
  EXPECT_THAT(
      fusion
          ->ComputeThreadIdToInputIndexing(
              /*root_index=*/0, /*hero_operand_index=*/2, &mlir_context_)
          ->ToString(printer_),
      MatchIndexingString(kIndexing));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
