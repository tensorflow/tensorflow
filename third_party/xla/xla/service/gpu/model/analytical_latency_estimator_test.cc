/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/model/analytical_latency_estimator.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {

namespace gpu {

namespace {

int64_t GetInstructionIndexInSchedule(
    absl::Span<HloInstruction* const> schedule, absl::string_view hlo_name) {
  return std::find_if(schedule.begin(), schedule.end(),
                      [hlo_name](HloInstruction* instruction) {
                        return instruction->name() == hlo_name;
                      }) -
         schedule.begin();
}

SchedulerConfig GetDefaultSchedulerConfig() {
  SchedulerConfig scheduler_config;
  return scheduler_config;
}

absl::StatusOr<bool> RunScheduler(
    HloModule* module, const SchedulerConfig& sched_config,
    std::unique_ptr<LatencyEstimator> latency_estimator =
        std::make_unique<ApproximateLatencyEstimator>()) {
  HloCostAnalysis::ShapeSizeFunction shape_size_bytes =
      [&shape_size_bytes](const Shape& shape) -> int64_t {
    int64_t shape_size = 0;
    if (shape.IsTuple()) {
      for (auto& sub_shape : shape.tuple_shapes()) {
        shape_size += shape_size_bytes(sub_shape);
      }
      return shape_size;
    }
    return ShapeUtil::ByteSizeOfElements(shape);
  };
  auto async_tracker = std::make_unique<AsyncTracker>(sched_config);
  auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
      shape_size_bytes, async_tracker.get(), latency_estimator.get(),
      sched_config);
  TF_ASSIGN_OR_RETURN(
      bool value, LatencyHidingScheduler(
                      std::move(latency_estimator), std::move(async_tracker),
                      std::move(scheduler_core), shape_size_bytes)
                      .Run(module));

  return value;
}

class AnalyticalLatencyHidingSchedulerTest : public GpuCodegenTest {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> ParseHloText(
      absl::string_view hlo_string) {
    return ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest());
  }
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }
};

TEST_F(AnalyticalLatencyHidingSchedulerTest, TestAnalyticalLatencyEstimator) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::PASCAL_)) {
    GTEST_SKIP() << "This test is for Pascal+ GPUs.";
  }
  const se::DeviceDescription dev_info =
      backend().default_stream_executor()->GetDeviceDescription();

  // The test below has 2 allreduces, ar2 should be have the larger latency
  // so we expect ar1 to be run first and ar2 to be overlapped with conv0.
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

region_20.995 {
  Arg_1.997 = f32[] parameter(1)
  Arg_0.996 = f32[] parameter(0)
  ROOT add.589 = f32[] add(Arg_0.996, Arg_1.997)
}

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[1024,2048,2048]{2,1,0} parameter(2)
  p3 = f32[2048,2048,2048]{2,1,0} parameter(3)
  all-reduce-start.1 = f32[1024,2048,2048]{2,1,0} all-reduce-start(p2), channel_id=8, replica_groups={{0}}, to_apply=region_20.995, backend_config="{\"is_sync\":false}"
  all-reduce-start.2 = f32[2048,2048,2048]{2,1,0} all-reduce-start(p3), channel_id=10, replica_groups={{0}}, to_apply=region_20.995, backend_config="{\"is_sync\":false}"

  all-reduce-done.1 = f32[1024,2048,2048]{2,1,0} all-reduce-done(all-reduce-start.1)
  all-reduce-done.2 = f32[2048,2048,2048]{2,1,0} all-reduce-done(all-reduce-start.2)
  conv0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb

  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1024,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}) tuple(conv0, all-reduce-done.1, all-reduce-done.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  hlo_module->mutable_config().set_num_partitions(8);

  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());

  auto scheduler_config = GetDefaultSchedulerConfig();
  auto latency_estimator = std::make_unique<AnalyticalLatencyEstimator>(
      scheduler_config, std::make_unique<ApproximateLatencyEstimator>(),
      dev_info, ShapeSizeBytesFunction(), hlo_module->entry_computation());
  EXPECT_TRUE(RunScheduler(hlo_module.get(), scheduler_config,
                           std::move(latency_estimator))
                  .ok());
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_schedule =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  int64_t ar2_index = GetInstructionIndexInSchedule(new_instruction_schedule,
                                                    "all-reduce-start.2");
  int64_t ar1_done_index = GetInstructionIndexInSchedule(
      new_instruction_schedule, "all-reduce-done.1");
  int64_t conv0_index =
      GetInstructionIndexInSchedule(new_instruction_schedule, "conv0");

  EXPECT_LT(ar1_done_index, ar2_index);
  EXPECT_LT(ar2_index, conv0_index);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
