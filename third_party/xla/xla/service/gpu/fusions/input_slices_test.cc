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

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

class InputSlicesTest : public HloTestBase {
 public:
  void SetUp() override {
    HloTestBase::SetUp();
    printer_.SetDimensionName(0, "th_x");
    printer_.SetDimensionName(1, "th_y");
    printer_.SetDimensionName(2, "th_z");
    printer_.SetDimensionName(3, "bl_x");
    printer_.SetDimensionName(4, "bl_y");
    printer_.SetDimensionName(5, "bl_z");
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
  ASSERT_NE(analysis_fused, std::nullopt);

  TF_ASSERT_OK_AND_ASSIGN(
      auto emitter,
      GetFusionEmitter(PreBufferAssignmentFusionInfo{*analysis_fused}));
  auto fusion = dynamic_cast<InputSlicesFusion*>(emitter.get());
  ASSERT_NE(fusion, nullptr);

  auto thread_id_to_output_indexing =
      fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context_);
  EXPECT_THAT(printer_.ToString(thread_id_to_output_indexing->affine_map),
              HasSubstr("(th_x, th_y, th_z, bl_x, bl_y, bl_z) -> "
                        "(0, "
                        "((th_x + bl_x * 128) floordiv 3) mod 2, "
                        "(th_x + bl_x * 128) mod 3, "
                        "((th_x + bl_x * 128) floordiv 6) mod 5)"));
  EXPECT_THAT(thread_id_to_output_indexing->domain,
              MatchDomain(ElementsAre(MatchRange(0, 127), MatchRange(0, 0),
                                      MatchRange(0, 0), MatchRange(0, 1),
                                      MatchRange(0, 0), MatchRange(0, 0)),
                          IsEmpty()));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
