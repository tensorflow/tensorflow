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
#include "xla/service/gpu/fusions/fusions.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class ScatterFusionTest : public HloTestBase {};

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
  ASSERT_NE(analysis_fused, std::nullopt);

  TF_ASSERT_OK_AND_ASSIGN(
      auto emitter,
      GetFusionEmitter(PreBufferAssignmentFusionInfo{*analysis_fused}));
  auto scatter_fusion = dynamic_cast<ScatterFusion*>(emitter.get());
  ASSERT_NE(scatter_fusion, nullptr);
  EXPECT_EQ(scatter_fusion->launch_dimensions().launch_bound(),
            3 * 9 /* updates size */);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
