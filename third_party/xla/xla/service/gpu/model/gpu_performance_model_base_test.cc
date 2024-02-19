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

#include "xla/service/gpu/model/gpu_performance_model_base.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class GpuPerformanceModelBaseTest : public HloTestBase {
 public:
  GpuHloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }

  GpuHloCostAnalysis::Options options_{ShapeSizeBytesFunction(),
                                       /*per_second_rates=*/{},
                                       /*count_multiple_input_accesses=*/true};
  // The reference times in the test cases below are measured
  // on A6000 by profiling the execution of the HLOs.
  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};
  GpuHloCostAnalysis analysis_{options_, &device_info_};

  GpuPerformanceModelBaseTest() : HloTestBase() {}
};

TEST_F(GpuPerformanceModelBaseTest, SharedOperandBytesAccessed_InPlaceDUS) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY entry_computation {
  param_0 = f32[8,16] parameter(0)
  param_1 = f32[4,4] parameter(1)
  c_0 = s32[] constant(0)
  log = f32[4,4] log(param_1)
  ROOT dynamic-update-slice = f32[8,16] dynamic-update-slice(param_0, log, c_0, c_0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  ASSERT_IS_OK(computation->Accept(&analysis_));

  auto dus_consumer = computation->root_instruction();
  auto log_producer = dus_consumer->mutable_operand(1);

  auto get_shared_operand_bytes_accessed = [&](const HloInstruction* operand) {
    return GpuPerformanceModelBase::GetSharedOperandBytesAccessed(
        &analysis_, log_producer, dus_consumer, operand);
  };

  EXPECT_EQ(get_shared_operand_bytes_accessed(dus_consumer->operand(0)), 0);
  EXPECT_EQ(get_shared_operand_bytes_accessed(log_producer->operand(0)), 64);
}

TEST_F(GpuPerformanceModelBaseTest, SharedOperandBytesAccessed_DUS) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY entry_computation {
  param_0 = f32[8,16] parameter(0)
  param_1 = f32[4,4] parameter(1)
  c_0 = s32[] constant(0)
  log = f32[8,16] log(param_0)
  ROOT dynamic-update-slice = f32[8,16] dynamic-update-slice(log, param_1, c_0, c_0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  ASSERT_IS_OK(computation->Accept(&analysis_));

  auto dus_consumer = computation->root_instruction();
  auto log_producer = dus_consumer->mutable_operand(0);

  auto get_shared_operand_bytes_accessed = [&](const HloInstruction* operand) {
    return GpuPerformanceModelBase::GetSharedOperandBytesAccessed(
        &analysis_, log_producer, dus_consumer, operand);
  };

  EXPECT_EQ(get_shared_operand_bytes_accessed(dus_consumer->operand(1)), 64);
  EXPECT_EQ(get_shared_operand_bytes_accessed(log_producer->operand(0)), 448);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
