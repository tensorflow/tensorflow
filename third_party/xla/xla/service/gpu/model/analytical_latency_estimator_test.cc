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

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {
namespace {

int64_t GetInstructionIndexInSchedule(
    absl::Span<HloInstruction* const> schedule, absl::string_view hlo_name) {
  return absl::c_find_if(schedule,
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
    const GpuAliasInfo* alias_info,
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
  std::shared_ptr<const SchedulingContext> scheduling_context =
      std::make_shared<const SchedulingContext>(
          module, std::move(latency_estimator), std::move(async_tracker),
          alias_info, shape_size_bytes);
  auto scheduler_core =
      std::make_unique<DefaultSchedulerCore>(scheduling_context, sched_config);
  ASSIGN_OR_RETURN(bool value, LatencyHidingScheduler(scheduling_context,
                                                      std::move(scheduler_core))
                                   .Run(module));

  return value;
}

class AnalyticalLatencyHidingSchedulerTest : public HloPjRtGpuTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> ParseHloText(
      absl::string_view hlo_string) {
    return ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest());
  }
  se::GpuComputeCapability GetGpuComputeCapability() {
    return device_description().gpu_compute_capability();
  }
  std::unique_ptr<GpuAliasInfo> GetAliasInfo() {
    return absl::down_cast<GpuCompiler*>(compiler())
        ->GetAliasInfo(device_description());
  }
};

TEST_F(AnalyticalLatencyHidingSchedulerTest, TestAnalyticalLatencyEstimator) {
  auto gpu_compute_capability = GetGpuComputeCapability();
  if (auto* c = gpu_compute_capability.cuda_compute_capability()) {
    if (!c->IsAtLeast(se::CudaComputeCapability::kPascal)) {
      GTEST_SKIP() << "This test is for Pascal+ GPUs.";
    }
    if (c->major == 12 && c->minor == 1) {
      // Skip this test for Spark. Because of the AllReduce, the test uses
      // gpu_collective_performance_model, which only makes sense in a
      // datacenter network setting.
      GTEST_SKIP() << "This test is for datacenter GPUs.";
    }
  } else if (auto* r = gpu_compute_capability.rocm_compute_capability()) {
    if (!r->gfx9_mi100_or_later()) {
      GTEST_SKIP() << "This test is for datacenter GPUs.";
    }
  }

  const se::DeviceDescription dev_info = device_description();

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
  all-reduce-start.1 = f32[1024,2048,2048]{2,1,0} all-reduce-start(p2), channel_id=8, replica_groups={{0}}, to_apply=region_20.995, backend_config={"collective_backend_config": {"is_sync": false}}
  all-reduce-start.2 = f32[2048,2048,2048]{2,1,0} all-reduce-start(p3), channel_id=10, replica_groups={{0}}, to_apply=region_20.995, backend_config={"collective_backend_config": {"is_sync": false}}

  all-reduce-done.1 = f32[1024,2048,2048]{2,1,0} all-reduce-done(all-reduce-start.1)
  all-reduce-done.2 = f32[2048,2048,2048]{2,1,0} all-reduce-done(all-reduce-start.2)
  conv0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb

  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1024,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}) tuple(conv0, all-reduce-done.1, all-reduce-done.2)
}
)";

  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  hlo_module->mutable_config().set_num_partitions(8);

  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());

  auto scheduler_config = GetDefaultSchedulerConfig();
  auto latency_estimator = std::make_unique<AnalyticalLatencyEstimator>(
      scheduler_config, std::make_unique<ApproximateLatencyEstimator>(),
      dev_info, HloCostAnalysis::DefaultShapeSize,
      hlo_module->entry_computation());
  auto alias_info = GetAliasInfo();
  EXPECT_TRUE(RunScheduler(hlo_module.get(), scheduler_config, alias_info.get(),
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

TEST_F(AnalyticalLatencyHidingSchedulerTest,
       TestAnalyticalLatencyEstimatorAllGather) {
  auto gpu_compute_capability = GetGpuComputeCapability();
  if (auto* c = gpu_compute_capability.cuda_compute_capability()) {
    if (!c->IsAtLeast(se::CudaComputeCapability::kPascal)) {
      GTEST_SKIP() << "This test is for Pascal+ GPUs.";
    }
    if (c->major == 12 && c->minor == 1) {
      GTEST_SKIP() << "This test is for datacenter GPUs.";
    }
  } else if (auto* r = gpu_compute_capability.rocm_compute_capability()) {
    if (!r->gfx9_mi100_or_later()) {
      GTEST_SKIP() << "This test is for datacenter GPUs.";
    }
  }

  const se::DeviceDescription dev_info = device_description();

  // Mirrors TestAnalyticalLatencyEstimator but for AllGather: 2 all-gathers
  // where ag2 has the larger latency, so we expect ag1 to be run first and
  // ag2 to be overlapped with conv0.
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true, num_partitions=8

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[128,2048,2048]{2,1,0} parameter(2)
  p3 = f32[256,2048,2048]{2,1,0} parameter(3)
  all-gather-start.1 = (f32[128,2048,2048]{2,1,0}, f32[1024,2048,2048]{2,1,0}) all-gather-start(p2), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=8, use_global_device_ids=true, backend_config={"collective_backend_config": {"is_sync": false}}
  all-gather-start.2 = (f32[256,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}) all-gather-start(p3), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=10, use_global_device_ids=true, backend_config={"collective_backend_config": {"is_sync": false}}

  all-gather-done.1 = f32[1024,2048,2048]{2,1,0} all-gather-done(all-gather-start.1)
  all-gather-done.2 = f32[2048,2048,2048]{2,1,0} all-gather-done(all-gather-start.2)
  conv0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb

  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1024,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}) tuple(conv0, all-gather-done.1, all-gather-done.2)
}
)";

  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  hlo_module->mutable_config().set_num_partitions(8);
  hlo_module->mutable_config().set_use_spmd_partitioning(true);

  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());

  auto scheduler_config = GetDefaultSchedulerConfig();
  auto latency_estimator = std::make_unique<AnalyticalLatencyEstimator>(
      scheduler_config, std::make_unique<ApproximateLatencyEstimator>(),
      dev_info, HloCostAnalysis::DefaultShapeSize,
      hlo_module->entry_computation());
  auto alias_info = GetAliasInfo();
  EXPECT_TRUE(RunScheduler(hlo_module.get(), scheduler_config, alias_info.get(),
                           std::move(latency_estimator))
                  .ok());
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_schedule =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  int64_t ag2_index = GetInstructionIndexInSchedule(new_instruction_schedule,
                                                    "all-gather-start.2");
  int64_t ag1_done_index = GetInstructionIndexInSchedule(
      new_instruction_schedule, "all-gather-done.1");
  int64_t conv0_index =
      GetInstructionIndexInSchedule(new_instruction_schedule, "conv0");

  EXPECT_LT(ag1_done_index, ag2_index);
  EXPECT_LT(ag2_index, conv0_index);
}

TEST_F(AnalyticalLatencyHidingSchedulerTest,
       TestAnalyticalLatencyEstimatorReduceScatter) {
  auto gpu_compute_capability = GetGpuComputeCapability();
  if (auto* c = gpu_compute_capability.cuda_compute_capability()) {
    if (!c->IsAtLeast(se::CudaComputeCapability::kPascal)) {
      GTEST_SKIP() << "This test is for Pascal+ GPUs.";
    }
    if (c->major == 12 && c->minor == 1) {
      GTEST_SKIP() << "This test is for datacenter GPUs.";
    }
  } else if (auto* r = gpu_compute_capability.rocm_compute_capability()) {
    if (!r->gfx9_mi100_or_later()) {
      GTEST_SKIP() << "This test is for datacenter GPUs.";
    }
  }

  const se::DeviceDescription dev_info = device_description();

  // Mirrors TestAnalyticalLatencyEstimator but for ReduceScatter: 2
  // reduce-scatters where rs2 has the larger latency, so we expect rs1 to
  // be run first and rs2 to be overlapped with conv0.
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true, num_partitions=8

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
  reduce-scatter-start.1 = ((f32[1024,2048,2048]{2,1,0}), f32[128,2048,2048]{2,1,0}) reduce-scatter-start(p2), channel_id=8, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, to_apply=region_20.995, use_global_device_ids=true, backend_config={"collective_backend_config": {"is_sync": false}}
  reduce-scatter-start.2 = ((f32[2048,2048,2048]{2,1,0}), f32[256,2048,2048]{2,1,0}) reduce-scatter-start(p3), channel_id=10, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, to_apply=region_20.995, use_global_device_ids=true, backend_config={"collective_backend_config": {"is_sync": false}}

  reduce-scatter-done.1 = f32[128,2048,2048]{2,1,0} reduce-scatter-done(reduce-scatter-start.1)
  reduce-scatter-done.2 = f32[256,2048,2048]{2,1,0} reduce-scatter-done(reduce-scatter-start.2)
  conv0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb

  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[256,2048,2048]{2,1,0}) tuple(conv0, reduce-scatter-done.1, reduce-scatter-done.2)
}
)";

  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  hlo_module->mutable_config().set_num_partitions(8);
  hlo_module->mutable_config().set_use_spmd_partitioning(true);

  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());

  auto scheduler_config = GetDefaultSchedulerConfig();
  auto latency_estimator = std::make_unique<AnalyticalLatencyEstimator>(
      scheduler_config, std::make_unique<ApproximateLatencyEstimator>(),
      dev_info, HloCostAnalysis::DefaultShapeSize,
      hlo_module->entry_computation());
  auto alias_info = GetAliasInfo();
  EXPECT_TRUE(RunScheduler(hlo_module.get(), scheduler_config, alias_info.get(),
                           std::move(latency_estimator))
                  .ok());
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_schedule =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  int64_t rs2_index = GetInstructionIndexInSchedule(new_instruction_schedule,
                                                    "reduce-scatter-start.2");
  int64_t rs1_done_index = GetInstructionIndexInSchedule(
      new_instruction_schedule, "reduce-scatter-done.1");
  int64_t conv0_index =
      GetInstructionIndexInSchedule(new_instruction_schedule, "conv0");

  EXPECT_LT(rs1_done_index, rs2_index);
  EXPECT_LT(rs2_index, conv0_index);
}

}  // namespace
}  // namespace xla::gpu
