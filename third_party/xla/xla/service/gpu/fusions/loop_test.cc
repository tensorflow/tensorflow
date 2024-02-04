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
#include "xla/service/gpu/fusions/loop.h"

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/service/gpu/fusions/fusions.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class LoopTest : public HloTestBase {
 public:
  void SetUp() override {
    HloTestBase::SetUp();

    printer_ = AffineMapPrinter(
        {"th_x", "th_y", "th_z", "bl_x", "bl_y", "bl_z"}, {"unroll_factor"});
  }

 protected:
  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  AffineMapPrinter printer_;
  mlir::MLIRContext mlir_context_;
};

absl::StatusOr<std::unique_ptr<LoopFusion>> GetLoopFusion(
    const HloFusionAnalysis& analysis) {
  TF_ASSIGN_OR_RETURN(
      auto emitter, GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis}));
  auto fusion = dynamic_cast<LoopFusion*>(emitter.get());
  TF_RET_CHECK(fusion != nullptr);

  emitter.release();
  return std::unique_ptr<LoopFusion>{fusion};
}

TEST_F(LoopTest, ThreadIndexingUnrolled) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    neg {
      %input = f32[100,200,300] parameter(0)
      ROOT neg = f32[100,200,300] negate(%input)
    }

    ENTRY entry {
      %input = f32[100,200,300] parameter(0)
      ROOT %fusion = f32[100,200,300] fusion(%input), kind=kLoop, calls=neg
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  TF_ASSERT_OK_AND_ASSIGN(auto loop_fusion, GetLoopFusion(analysis));
  auto thread_id_to_output_indexing =
      loop_fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context_);

  EXPECT_THAT(thread_id_to_output_indexing->ToString(printer_),
              MatchIndexingString(R"(
              (th_x, th_y, th_z, bl_x, bl_y, bl_z)[unroll_factor] -> (
                (th_x * 4 + bl_x * 512 + unroll_factor) floordiv 60000,
                ((th_x * 4 + bl_x * 512 + unroll_factor) floordiv 300) mod 200,
                (th_x * 4 + bl_x * 512 + unroll_factor) mod 300)
              domain:
              th_x in [0, 127]
              th_y in [0, 0]
              th_z in [0, 0]
              bl_x in [0, 1007]
              bl_y in [0, 0]
              bl_z in [0, 0]
              unroll_factor in [0, 3]
             )"));
}

TEST_F(LoopTest, ThreadIndexingNotUnrolled) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    neg {
      %input = f32[20] parameter(0)
      ROOT neg = f32[20] negate(%input)
    }

    ENTRY entry {
      %input = f32[20] parameter(0)
      ROOT %fusion = f32[20] fusion(%input), kind=kLoop, calls=neg
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  TF_ASSERT_OK_AND_ASSIGN(auto loop_fusion, GetLoopFusion(analysis));
  auto thread_id_to_output_indexing =
      loop_fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context_);
  EXPECT_THAT(thread_id_to_output_indexing->ToString(printer_),
              MatchIndexingString(R"(
                (th_x, th_y, th_z, bl_x, bl_y, bl_z) -> (th_x)
                domain:
                th_x in [0, 19]
                th_y in [0, 0]
                th_z in [0, 0]
                bl_x in [0, 0]
                bl_y in [0, 0]
                bl_z in [0, 0]
  )"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
