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

#include "xla/service/memory_space_assignment/simulator.h"

#include <cstdint>
#include <memory>
#include <queue>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using memory_space_assignment::CostAnalysis;
using memory_space_assignment::CostAnalysisOptions;
using memory_space_assignment::RuntimeSimulator;

constexpr int64_t kPointerSize = 8;

int64_t ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

class MemorySpaceAssignmentSimulatorTest : public HloTestBase {
 protected:
  absl::Status Initialize(const HloModule* module) {
    HloCostAnalysis::Options tpu_device_options;
    tpu_device_options.shape_size = ShapeSize;
    // Assume 1 FLOP per second for testing.
    tpu_device_options.set_flops_per_second(1);
    hlo_cost_analysis_ = std::make_unique<HloCostAnalysis>(tpu_device_options);
    TF_RETURN_IF_ERROR(
        module->entry_computation()->Accept(hlo_cost_analysis_.get()));
    hlo_cost_analysis_costs_ =
        std::make_unique<memory_space_assignment::HloCostAnalysisCosts>(
            *hlo_cost_analysis_);
    CostAnalysisOptions _options;
    TF_ASSIGN_OR_RETURN(
        cost_analysis_,
        CostAnalysis::Create(*hlo_cost_analysis_costs_, _options, *module));
    runtime_simulator_ =
        std::make_unique<xla::memory_space_assignment::RuntimeSimulator>(
            cost_analysis_.get());
    return absl::OkStatus();
  }
  std::unique_ptr<HloCostAnalysis> hlo_cost_analysis_;
  std::unique_ptr<memory_space_assignment::HloCostAnalysisCosts>
      hlo_cost_analysis_costs_;
  std::unique_ptr<CostAnalysis> cost_analysis_;
  std::unique_ptr<RuntimeSimulator> runtime_simulator_;
};

TEST_F(MemorySpaceAssignmentSimulatorTest, SingleLayerNestedLoop) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true

      %body {
        %constant.1 = s32[] constant(1)
        %param = (s32[]) parameter(0)
        %count = s32[] get-tuple-element(%param), index=0
        %increment = s32[] add(s32[] %count, s32[] %constant.1)
        ROOT %loop_result = (s32[]) tuple(%increment)
      }

      %condition {
        %param = (s32[]) parameter(0)
        %constant.42 = s32[] constant(42)
        %condition_input = s32[] get-tuple-element(%param), index=0
        ROOT %greater = pred[] compare(s32[] %constant.42, s32[] %condition_input), direction=GT
      }

      ENTRY Entry {
        %dummy_input = s32[] parameter(0)
        %constant.0 = s32[] constant(0)
        ROOT %while = (s32[]) while(tuple(%constant.0)), condition=%condition, body=%body
      }

    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(Initialize(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_live_range,
                          HloLiveRange::Run(module->schedule(), *alias_analysis,
                                            module->entry_computation()));

  // Since the HLO does not contain memory access, pass an empty allocation
  // sequence for test.
  memory_space_assignment::AllocationSequence allocations;
  // The while loop has 42 iterations, and each iteration has 2 FLOP (for
  // %increment and %greater). Thus, the total FLOPs are 84 FLOPs.
  float expected_elapsed_time = 84;
  EXPECT_EQ(runtime_simulator_->SimulateElapsedTimeWithoutAsyncCopies(
                *hlo_live_range, allocations),
            expected_elapsed_time);
}

TEST_F(MemorySpaceAssignmentSimulatorTest,
       AsyncCopyTransferForSharedBandwidth) {
  int64_t buffer_size_1 = 100;
  int64_t buffer_size_2 = 200;
  auto parameter_1 = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {buffer_size_1}), "parameter_1");
  auto copy_start_1 = HloInstruction::CreateCopyStart(
      ShapeUtil::MakeShape(F32, {buffer_size_1}), parameter_1.get());
  auto parameter_2 = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {buffer_size_2}), "parameter_2");
  auto copy_start_2 = HloInstruction::CreateCopyStart(
      ShapeUtil::MakeShape(F32, {buffer_size_2}), parameter_2.get());
  std::queue<const HloInstruction*> memory_access_queue_to_share_bandwidth;
  memory_access_queue_to_share_bandwidth.push(copy_start_1.get());
  memory_access_queue_to_share_bandwidth.push(copy_start_2.get());
  absl::flat_hash_map<const HloInstruction*, float> remaining_size_of_buffers =
      {{copy_start_1.get(), buffer_size_1 * 4},
       {copy_start_2.get(), buffer_size_2 * 4}};
  float default_memory_bytes_per_second = 1;
  float bytes_to_transfer_step1 = 400;
  // The bandwidth is shared, so only uses half of the bandwidth.
  float expected_elapsed_time_step1 =
      bytes_to_transfer_step1 / (0.5 * default_memory_bytes_per_second);
  EXPECT_EQ(RuntimeSimulator::SimulateAsyncCopyTransfer(
                bytes_to_transfer_step1, memory_access_queue_to_share_bandwidth,
                remaining_size_of_buffers, default_memory_bytes_per_second),
            expected_elapsed_time_step1);

  float bytes_to_transfer_step2 = 900;
  // After the first step, there are 1200-400=800 bytes left in the shared
  // queue. Thus, for the next 800 bytes transfer, the bandwidth is shared.
  // Then, we can use all bandwidth for the rest of bytes.
  float shared_bandwidth_bytes =
      ((buffer_size_1 + buffer_size_2) * 4 - bytes_to_transfer_step1);
  float expected_elapsed_time_step2 =
      shared_bandwidth_bytes / (0.5 * default_memory_bytes_per_second) +
      (bytes_to_transfer_step2 - shared_bandwidth_bytes) /
          default_memory_bytes_per_second;
  EXPECT_EQ(RuntimeSimulator::SimulateAsyncCopyTransfer(
                bytes_to_transfer_step2, memory_access_queue_to_share_bandwidth,
                remaining_size_of_buffers, default_memory_bytes_per_second),
            expected_elapsed_time_step2);

  // Now, both copy instructions in the shared queue
  // finishes. The bandwidth is not shared anymore.
  float bytes_to_transfer_step3 = 150;
  float expected_elapsed_time_step3 =
      bytes_to_transfer_step3 / (default_memory_bytes_per_second);
  EXPECT_EQ(RuntimeSimulator::SimulateAsyncCopyTransfer(
                bytes_to_transfer_step3, memory_access_queue_to_share_bandwidth,
                remaining_size_of_buffers, default_memory_bytes_per_second),
            expected_elapsed_time_step3);
}
TEST_F(MemorySpaceAssignmentSimulatorTest, DrainMemoryAccessQueueInTimeWindow) {
  int64_t buffer_size_1 = 100;
  int64_t buffer_size_2 = 200;
  auto parameter_1 = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {buffer_size_1}), "parameter_1");
  auto copy_start_1 = HloInstruction::CreateCopyStart(
      ShapeUtil::MakeShape(F32, {buffer_size_1}), parameter_1.get());
  auto parameter_2 = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {buffer_size_2}), "parameter_2");
  auto copy_start_2 = HloInstruction::CreateCopyStart(
      ShapeUtil::MakeShape(F32, {buffer_size_2}), parameter_2.get());

  std::queue<const HloInstruction*> read_queue, write_queue;
  read_queue.push(copy_start_1.get());
  write_queue.push(copy_start_2.get());

  absl::flat_hash_map<const HloInstruction*, float> remaining_size_of_buffers =
      {{copy_start_1.get(), buffer_size_1 * 4},
       {copy_start_2.get(), buffer_size_2 * 4}};

  float default_memory_bytes_per_second = 1;

  float time_window_1 = 100;

  RuntimeSimulator::ProcessAsyncCopyInTimeWindow(
      time_window_1, read_queue, write_queue, remaining_size_of_buffers,
      default_memory_bytes_per_second);

  // During the first time window, both queues are not empty, so the bandwidth
  // is shared. Each of the request at the front of the queue process 100 sec *
  // 0.5 bytes/sec = 50 bytes.
  float expected_remained_bytes_1 =
      buffer_size_1 * 4 - time_window_1 * 0.5 * default_memory_bytes_per_second;
  float expected_remained_bytes_2 =
      buffer_size_2 * 4 - time_window_1 * 0.5 * default_memory_bytes_per_second;
  EXPECT_EQ(remaining_size_of_buffers.at(copy_start_1.get()),
            expected_remained_bytes_1);
  EXPECT_EQ(remaining_size_of_buffers.at(copy_start_2.get()),
            expected_remained_bytes_2);

  float time_window_2 = 700;
  RuntimeSimulator::ProcessAsyncCopyInTimeWindow(
      time_window_2, read_queue, write_queue, remaining_size_of_buffers,
      default_memory_bytes_per_second);
  // Like the first time window, the queues share the bandwidth in the second
  // time window. After the second time window, the front queue for the read
  // queue is drained and remove from the queue, since all 400 bytes are
  // processed.
  EXPECT_TRUE(read_queue.empty());
  expected_remained_bytes_2 -=
      time_window_2 * 0.5 * default_memory_bytes_per_second;
  EXPECT_EQ(remaining_size_of_buffers.at(copy_start_2.get()),
            expected_remained_bytes_2);
  float time_window_3 = 100;
  RuntimeSimulator::ProcessAsyncCopyInTimeWindow(
      time_window_3, read_queue, write_queue, remaining_size_of_buffers,
      default_memory_bytes_per_second);
  // Since the read queue is empty, the write queue can use the full bandwidth.
  expected_remained_bytes_2 -= time_window_3 * default_memory_bytes_per_second;
  EXPECT_EQ(remaining_size_of_buffers.at(copy_start_2.get()),
            expected_remained_bytes_2);
}
}  // namespace
}  // namespace xla
