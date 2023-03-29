/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/profile_guided_latency_estimator.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/latency_hiding_scheduler.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/protobuf.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {

namespace {

constexpr int kMaxConcurrentAsyncCollectivePermutes = 5;

int GetIndex(absl::Span<HloInstruction* const> instruction_sequence,
             absl::string_view hlo_name) {
  return absl::c_find_if(instruction_sequence,
                         [hlo_name](HloInstruction* instruction) {
                           return instruction->name() == hlo_name;
                         }) -
         instruction_sequence.begin();
}

SchedulerConfig GetDefaultSchedConfig() {
  SchedulerConfig sched_cfg;
  sched_cfg.collective_permute_overlap_limit =
      kMaxConcurrentAsyncCollectivePermutes;
  sched_cfg.send_recv_overlap_limit = INT32_MAX;
  return sched_cfg;
}

StatusOr<bool> RunScheduler(
    HloModule* module, SchedulerConfig sched_config = GetDefaultSchedConfig(),
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

}  // namespace

class LatencyHidingSchedulerTest : public HloTestBase {
 public:
  StatusOr<std::unique_ptr<HloModule>> ParseHloText(
      absl::string_view hlo_string) {
    TF_ASSIGN_OR_RETURN(
        auto hlo_module,
        ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest()));
    return StatusOr<std::unique_ptr<HloModule>>(std::move(hlo_module));
  }
};

TEST_F(LatencyHidingSchedulerTest, TestProfileGuidedLatencyEstimator) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[1024,2048,2048]{2,1,0} parameter(2)
  p3 = f32[2048,2048,2048]{2,1,0} parameter(3)
  cp1s = (f32[1024,2048,2048]{2,1,0}, f32[1024,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0},{0,3},{3,2}}
  cp2s = (f32[2048,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p3), source_target_pairs={{1,0},{0,3},{3,2}}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllToAll" op_name="c0"}
  cp1d = f32[1024,2048,2048]{2,1,0} collective-permute-done(cp1s)
  cp2d = f32[2048,2048,2048]{2,1,0} collective-permute-done(cp2s)
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1024,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}) tuple(c0, cp1d, cp2d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::string profiled_instructions_text_proto = R"pb(
    instructions { name: "cp1s" timestamp_us: 0.0 duration_us: 1.0 }
    instructions { name: "cp1d" timestamp_us: 1.0 duration_us: 40.0 }
    instructions { name: "cp2s" timestamp_us: 41.0 duration_us: 1.0 }
    instructions { name: "cp2d" timestamp_us: 42.0 duration_us: 80.0 }
    instructions { name: "c0" timestamp_us: 122.0 duration_us: 10.0 }
  )pb";
  ProfiledInstructionsProto profiled_instructions_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      profiled_instructions_text_proto, &profiled_instructions_proto));

  auto sched_config = GetDefaultSchedConfig();
  sched_config.collective_permute_overlap_limit = 2;
  sched_config.all_gather_overlap_limit = 2;
  auto latency_estimator = std::make_unique<ProfileGuidedLatencyEstimator>(
      sched_config, std::make_unique<ApproximateLatencyEstimator>(),
      profiled_instructions_proto);
  EXPECT_TRUE(RunScheduler(hlo_module.get(), GetDefaultSchedConfig(),
                           std::move(latency_estimator))
                  .ok());
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // cp2s should come first since the duration between cp2s->cp2d is double that
  // of cp1s->cp1d
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "cp1s"));
}

}  // namespace xla
