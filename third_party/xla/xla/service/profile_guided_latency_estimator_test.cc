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

#include "xla/service/profile_guided_latency_estimator.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {

namespace {

using ::tsl::testing::StatusIs;

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
  return sched_cfg;
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
  std::shared_ptr<const SchedulingContext> scheduling_context =
      std::make_shared<const SchedulingContext>(
          module, std::move(latency_estimator), std::move(async_tracker),
          shape_size_bytes);
  auto scheduler_core =
      std::make_unique<DefaultSchedulerCore>(scheduling_context, sched_config);
  TF_ASSIGN_OR_RETURN(
      bool value,
      LatencyHidingScheduler(scheduling_context, std::move(scheduler_core))
          .Run(module));

  return value;
}

}  // namespace

class LatencyHidingSchedulerTest : public HloHardwareIndependentTestBase,
                                   public ::testing::WithParamInterface<bool> {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> ParseHloText(
      absl::string_view hlo_string) {
    return ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest());
  }
};

TEST_P(LatencyHidingSchedulerTest, TestProfileGuidedLatencyEstimator) {
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
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  cp1d = f32[1024,2048,2048]{2,1,0} collective-permute-done(cp1s)
  cp2d = f32[2048,2048,2048]{2,1,0} collective-permute-done(cp2s)
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1024,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}) tuple(c0, cp1d, cp2d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());

  // Test parameter decided whether async latencies are read from latencies or
  // costs.
  std::string profiled_instructions_text_proto;
  if (GetParam()) {
    profiled_instructions_text_proto = R"pb(
      costs { name: "c0" cost_us: 10.0 }
      latencies { source: "cp1s" target: "cp1d" latency_us: 40.0 }
      latencies { source: "cp2s" target: "cp2d" latency_us: 80.0 }
    )pb";
  } else {
    profiled_instructions_text_proto = R"pb(
      costs { name: "c0" cost_us: 10.0 }
      costs { name: "cp1s" cost_us: 40.0 }
      costs { name: "cp2s" cost_us: 80.0 }
    )pb";
  }
  tensorflow::profiler::ProfiledInstructionsProto profiled_instructions_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      profiled_instructions_text_proto, &profiled_instructions_proto));

  auto sched_config = GetDefaultSchedConfig();
  sched_config.collective_permute_overlap_limit = 2;
  auto latency_estimator = std::make_unique<ProfileGuidedLatencyEstimator>(
      sched_config, std::make_unique<ApproximateLatencyEstimator>(),
      profiled_instructions_proto);
  EXPECT_TRUE(
      RunScheduler(hlo_module.get(), sched_config, std::move(latency_estimator))
          .ok());
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // cp2s should come first since the latency between cp2s->cp2d is double
  // that of cp1s->cp1d
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "cp1s"));
}

INSTANTIATE_TEST_SUITE_P(LatencyHidingSchedulerTest, LatencyHidingSchedulerTest,
                         ::testing::Bool());

using ProfileGuidedLatencyEstimatorTest = HloHardwareIndependentTestBase;

TEST_F(ProfileGuidedLatencyEstimatorTest,
       TestProfileGuidedLatencyEstimatorWithAsyncInstruction) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

add.1 {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  reduce-scatter-start = ((f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}), (f32[4,64,256]{2,1,0}, f32[4,64,256]{2,1,0})) reduce-scatter-start(p0, p1), channel_id=1, replica_groups={}, dimensions={0}, to_apply=add.1
  reduce-scatter-done = (f32[4,64,256]{2,1,0}, f32[4,64,256]{2,1,0}) reduce-scatter-done(reduce-scatter-start)
  ROOT gte = f32[4,64,256]{2,1,0} get-tuple-element(reduce-scatter-done), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::string profiled_instructions_text_proto = R"pb(
    costs { name: "reduce-scatter" cost_us: 120.0 }
  )pb";
  ;
  tensorflow::profiler::ProfiledInstructionsProto profiled_instructions_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      profiled_instructions_text_proto, &profiled_instructions_proto));

  auto sched_config = GetDefaultSchedConfig();
  auto latency_estimator = std::make_unique<ProfileGuidedLatencyEstimator>(
      sched_config, std::make_unique<ApproximateLatencyEstimator>(),
      profiled_instructions_proto);
  HloInstruction* rs_start =
      FindInstruction(hlo_module.get(), "reduce-scatter-start");
  HloInstruction* rs_done =
      FindInstruction(hlo_module.get(), "reduce-scatter-done");
  HloGraphNode rs_start_node = HloGraphNode(rs_start, 0);
  HloGraphNode rs_done_node = HloGraphNode(rs_done, 1);

  double latency =
      latency_estimator->GetLatencyBetween(rs_start_node, rs_done_node);
  EXPECT_EQ(latency, 120.0);
}

TEST_F(ProfileGuidedLatencyEstimatorTest,
       TestProfileGuidedLatencyEstimatorWithP2pInstruction) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true
ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  after-all.1 = token[] after-all()
  send.7.0 = (f32[16,64,256]{2,1,0}, u32[], token[]) send(p0, after-all.1), channel_id=1, frontend_attributes={_xla_send_recv_source_target_pairs="{{0,1}}"}
  send-done.7.0 = token[] send-done(send.7.0), channel_id=1
  recv.7.0 = (f32[16,64,256]{2,1,0}, u32[], token[]) recv(after-all.1), channel_id=1, frontend_attributes={_xla_send_recv_source_target_pairs="{{0,1}}"}
  recv-done.7.0 = (f32[16,64,256]{2,1,0}, token[]) recv-done(recv.7.0), channel_id=1
  ROOT recv-data = f32[16,64,256]{2,1,0} get-tuple-element(recv-done.7.0), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::string profiled_instructions_text_proto = R"pb(
    costs { name: "send.7.0" cost_us: 110.0 }
    costs { name: "recv.7.0" cost_us: 100.0 }
  )pb";
  ;
  tensorflow::profiler::ProfiledInstructionsProto profiled_instructions_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      profiled_instructions_text_proto, &profiled_instructions_proto));

  auto sched_config = GetDefaultSchedConfig();
  sched_config.schedule_send_recvs = true;
  auto latency_estimator = std::make_unique<ProfileGuidedLatencyEstimator>(
      sched_config, std::make_unique<ApproximateLatencyEstimator>(),
      profiled_instructions_proto);
  HloInstruction* send_start = FindInstruction(hlo_module.get(), "send.7.0");
  HloInstruction* send_done =
      FindInstruction(hlo_module.get(), "send-done.7.0");

  HloInstruction* recv_start = FindInstruction(hlo_module.get(), "recv.7.0");
  HloInstruction* recv_done =
      FindInstruction(hlo_module.get(), "recv-done.7.0");

  HloGraphNode send_start_node = HloGraphNode(send_start, 0);
  HloGraphNode send_done_node = HloGraphNode(send_done, 1);

  HloGraphNode recv_start_node = HloGraphNode(recv_start, 2);
  HloGraphNode recv_done_node = HloGraphNode(recv_done, 3);

  double send_latency =
      latency_estimator->GetLatencyBetween(send_start_node, send_done_node);
  double recv_latency =
      latency_estimator->GetLatencyBetween(recv_start_node, recv_done_node);

  EXPECT_EQ(send_latency, 110.0);
  EXPECT_EQ(recv_latency, 100.0);
}

TEST_F(ProfileGuidedLatencyEstimatorTest,
       ProfileGuidedLatencyEstimatorCheckAccuracyFailsIfMissingAggregator) {
  std::string kFdoProfile = "";
  absl::string_view kHloModule = R"(
    HloModule module

    ENTRY main {
      p0 = f32[1] parameter(0)
      ROOT add0 = f32[1] add(p0,p0)
    }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  tensorflow::profiler::ProfiledInstructionsProto fdo_profile;
  ASSERT_TRUE(
      tsl::protobuf::TextFormat::ParseFromString(kFdoProfile, &fdo_profile));

  auto sched_config = GetDefaultSchedConfig();
  auto latency_estimator = std::make_unique<ProfileGuidedLatencyEstimator>(
      sched_config, std::make_unique<ApproximateLatencyEstimator>(),
      fdo_profile);
  EXPECT_THAT(latency_estimator->CheckAccuracy(*hlo_module),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace xla
