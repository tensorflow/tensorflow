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

#include "xla/service/gpu/model/sol_gpu_cost_model_stats_collection.h"

#include <cstdint>
#include <functional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::mlir::MLIRContext;
using ::testing::Gt;
using ::testing::Property;

using ShapeSizeFn = std::function<int64_t(const Shape&)>;

class SolGpuCostModelStatsCollectionTest
    : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(se::CudaComputeCapability(9, 0));
  int pointer_size_ = 8;
  MLIRContext mlir_context_;
};

TEST_F(SolGpuCostModelStatsCollectionTest,
       RecordsRuntimeInformationForCollectives) {
  constexpr absl::string_view kHloText = R"(
  HloModule m

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT _ = f32[] add(x, y)
  }

  ENTRY ar {
    p0 = f32[8192,4096] parameter(0)

    ar-start = f32[8192,4096] all-reduce-start(p0), to_apply=add,
      replica_groups={{0,1,2,3,4,5,6,7}, {8,9,10,11,12,13,14,15}}
    ROOT ar-done = f32[8192,4096] all-reduce-done(ar-start)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          SolGpuCostModelStatsCollection(
                              device_info_, HloCostAnalysis::DefaultShapeSize,
                              pointer_size_, &mlir_context_)
                              .Run(module.get()));

  VLOG(1) << module->ToString();

  EXPECT_FALSE(changed);
  EXPECT_THAT(module->entry_computation()
                  ->root_instruction()
                  ->operand(0)
                  ->backend_config<GpuBackendConfig>()
                  ->reification_cost(),
              ElementsAre(Property(&ReificationCost::exec_time_us, Gt(0))));
}
TEST_F(SolGpuCostModelStatsCollectionTest,
       RecordsRuntimeInfoForAsyncStartReduceScatter) {
  constexpr absl::string_view kHloText = R"(
    HloModule async_rs_test
    %add.f32 (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }
    %async_rs {
      %p0 = f32[4096,128256] parameter(0)
      ROOT %rs = f32[512,128256] reduce-scatter(%p0), channel_id=1,
        replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, to_apply=%add.f32
    }
    ENTRY main {
      %param = f32[4096,128256] parameter(0)
      %rs_start = ((f32[4096,128256]), f32[512,128256], u32[])
        async-start(%param), calls=%async_rs
      ROOT %rs_done = f32[512,128256] async-done(%rs_start)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          SolGpuCostModelStatsCollection(
                              device_info_, HloCostAnalysis::DefaultShapeSize,
                              pointer_size_, &mlir_context_)
                              .Run(module.get()));
  VLOG(1) << module->ToString();
  EXPECT_FALSE(changed);
  HloInstruction* rs_start = FindInstruction(module.get(), "rs_start");
  ASSERT_NE(rs_start, nullptr);
  HloComputation* async_comp = rs_start->async_wrapped_computation();
  ASSERT_NE(async_comp, nullptr);
  HloInstruction* rs_instr = async_comp->root_instruction();

  EXPECT_THAT(rs_instr->backend_config<GpuBackendConfig>()->reification_cost(),
              ElementsAre(Property(&ReificationCost::exec_time_us, Gt(0))));
}
}  // namespace
}  // namespace xla::gpu
