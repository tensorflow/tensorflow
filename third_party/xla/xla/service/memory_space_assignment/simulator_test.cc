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
#include <list>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/shape.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using memory_space_assignment::CostAnalysis;
using memory_space_assignment::CostAnalysisOptions;
using memory_space_assignment::RuntimeSimulator;

using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

constexpr int64_t kAlternateMemorySpace = 1;

class MemorySpaceAssignmentSimulatorTest : public HloTestBase {
 protected:
  absl::Status Initialize(absl::string_view hlo_string) {
    TF_ASSIGN_OR_RETURN(module_, ParseAndReturnVerifiedModule(hlo_string));
    for (HloInstruction* inst : module_->entry_computation()->instructions()) {
      instruction_map_[inst->name()] = inst;
      // Construct an allocation for the instruction if it is in the alternate
      // memory.
      if (inst->shape().has_layout() &&
          inst->shape().layout().memory_space() == kAlternateMemorySpace) {
        std::unique_ptr<xla::memory_space_assignment::Allocation> allocation =
            std::make_unique<memory_space_assignment::PinnedAllocation>(
                HloPosition{inst, {}},
                memory_space_assignment::MemorySpace::kAlternate,
                HeapSimulator::Chunk::FromOffsetSize(-1, -1),
                /*start_time=*/0,
                /*end_time=*/1, /*is_scoped_allocation=*/false);
        for (HloInstruction* user : inst->users()) {
          allocation->AddUse(HloUse{user, 0});
        }
        allocations_.push_back(std::move(allocation));
      }
    }
    HloCostAnalysis::Options tpu_device_options;
    // Assume 1 FLOP per second for testing.
    tpu_device_options.set_flops_per_second(1);
    // Assume 1 byte per second for testing.
    tpu_device_options.set_bytes_per_second(1);
    hlo_cost_analysis_ = std::make_unique<HloCostAnalysis>(tpu_device_options);
    TF_RETURN_IF_ERROR(
        module_->entry_computation()->Accept(hlo_cost_analysis_.get()));
    hlo_cost_analysis_costs_ =
        std::make_unique<memory_space_assignment::HloCostAnalysisCosts>(
            *hlo_cost_analysis_);
    CostAnalysisOptions cost_analysis_options;
    // Assume 2 byte per second for testing.
    cost_analysis_options.alternate_mem_bandwidth_bytes_per_second = 2;
    cost_analysis_options.default_mem_bandwidth_bytes_per_second = 1.0;

    TF_ASSIGN_OR_RETURN(cost_analysis_,
                        CostAnalysis::Create(*hlo_cost_analysis_costs_,
                                             cost_analysis_options, *module_));

    TF_ASSIGN_OR_RETURN(alias_analysis_, HloAliasAnalysis::Run(module_.get()));
    TF_ASSIGN_OR_RETURN(hlo_live_range_,
                        HloLiveRange::Run(module_->schedule(), *alias_analysis_,
                                          module_->entry_computation()));
    runtime_simulator_ = std::make_unique<RuntimeSimulator>(
        cost_analysis_.get(), kAlternateMemorySpace);
    return absl::OkStatus();
  }
  absl::flat_hash_map<absl::string_view, const HloInstruction*>
      instruction_map_;
  std::unique_ptr<HloCostAnalysis> hlo_cost_analysis_;
  std::unique_ptr<memory_space_assignment::HloCostAnalysisCosts>
      hlo_cost_analysis_costs_;
  std::unique_ptr<CostAnalysis> cost_analysis_;
  std::unique_ptr<HloAliasAnalysis> alias_analysis_;
  std::unique_ptr<HloLiveRange> hlo_live_range_;
  memory_space_assignment::AllocationSequence allocations_;
  std::unique_ptr<RuntimeSimulator> runtime_simulator_;
  std::unique_ptr<HloModule> module_;
};

TEST_F(MemorySpaceAssignmentSimulatorTest, SingleLayerLoop) {
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
  TF_ASSERT_OK(Initialize(hlo_string));
  // The total elapsed time is the summation of the elapsed time of each
  // instruction. Here are the overhead of each instruction (secs):
  // %increment: 12 * 42
  // tuple(%constant.0): 8 * 1
  // %greater: 9 * 42
  // %loop_result: 8 * 42
  EXPECT_EQ(runtime_simulator_->SimulateElapsedTimeWithoutAsyncCopyLikes(
                *hlo_live_range_, allocations_),
            1226);
  EXPECT_EQ(
      runtime_simulator_->SimulateElapsedTime(module_.get(), allocations_),
      1226);
}

TEST_F(MemorySpaceAssignmentSimulatorTest, NestedLayerLoop) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
      %inner.body {
        %constant.1 = s32[] constant(1)
        %param = (s32[]) parameter(0)
        %count = s32[] get-tuple-element(%param), index=0
        %increment = s32[] add(s32[] %count, s32[] %constant.1)
        ROOT %loop_result = (s32[]) tuple(%increment)
      }
      %inner.condition {
        %param = (s32[]) parameter(0)
        %constant.42 = s32[] constant(42)
        %condition_input = s32[] get-tuple-element(%param), index=0
        ROOT %greater = pred[] compare(s32[] %constant.42, s32[] %condition_input), direction=GT
      }
      %outer.body {
        %constant.0 = s32[] constant(0)
        %constant.1 = s32[] constant(1)
        %param = (s32[]) parameter(0)
        %inner_while = (s32[]) while(tuple(%constant.0)), condition=%inner.condition, body=%inner.body
        %count = s32[] get-tuple-element(%param), index=0
        %increment = s32[] add(s32[] %count, s32[] %constant.1)
        ROOT %loop_result = (s32[]) tuple(%increment)
      }
      %outer.condition {
        %param = (s32[]) parameter(0)
        %constant.27 = s32[] constant(27)
        %condition_input = s32[] get-tuple-element(%param), index=0
        ROOT %greater = pred[] compare(s32[] %constant.27, s32[] %condition_input), direction=GT
      }
      ENTRY Entry {
        %constant.0 = s32[] constant(0)
        ROOT %while_outer = (s32[]) while(tuple(%constant.0)), condition=%outer.condition, body=%outer.body
      }
    )";
  TF_ASSERT_OK(Initialize(hlo_string));
  // The inner loop is derived from the SingleLayerLoop test, whose overhead is
  // 1226 seconds.

  // For the outer loop, the overhead of each instruction is:
  // %increment: 12 * 27
  // tuple(%constant.0): 8 * 1
  // %greater: 9 * 27
  // %loop_result: 8 * 27
  // Thus, the total overhead of the while_outer is 1226 * 27 + 12 * 27 + 8 * 1
  // + 9 * 27 + 8 * 27 = 33893

  EXPECT_EQ(runtime_simulator_->SimulateElapsedTimeWithoutAsyncCopyLikes(
                *hlo_live_range_, allocations_),
            33893);
  EXPECT_EQ(
      runtime_simulator_->SimulateElapsedTime(module_.get(), allocations_),
      33893);
}

TEST_F(MemorySpaceAssignmentSimulatorTest, SingleAsyncCopyOverhead) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
      ENTRY Entry {
        param_0 = f32[1,1,1024,2048] parameter(0)
        copy-start.1 = (f32[1,1,1024,2048]{0,1,2,3:S(1)}, f32[1,1,1024,2048], u32[]) copy-start(param_0)
        ROOT copy-done.1 = f32[1,1,1024,2048]{0,1,2,3:S(1)} copy-done(copy-start.1)
      }

    )";
  TF_ASSERT_OK(Initialize(hlo_string));

  // Since the HLO does not contain memory access, pass an empty allocation
  // sequence for test.
  memory_space_assignment::AllocationSequence allocations;
  // The SimulateElapsedTimeWithoutAsyncCopyLikes should not include the
  // overhead of async copies.
  EXPECT_EQ(runtime_simulator_->SimulateElapsedTimeWithoutAsyncCopyLikes(
                *hlo_live_range_, allocations_),
            0);
  // The expected elapsed time is 1024 * 2048 * 4 / 1 = 8388608.
  EXPECT_EQ(
      runtime_simulator_->SimulateElapsedTime(module_.get(), allocations_),
      8388608);
}

TEST_F(MemorySpaceAssignmentSimulatorTest, AsyncCopyWithComputationOverhead) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
      ENTRY Entry {
        param_0 = f32[8] parameter(0)
        param_1 = f32[2] parameter(1)
        copy-start.1 = (f32[8]{0:S(1)}, f32[8], u32[]) copy-start(param_0)
        neg_compute = f32[2] negate(param_1)
        ROOT copy-done.1 = f32[8]{0:S(1)} copy-done(copy-start.1)
      }

    )";
  TF_ASSERT_OK(Initialize(hlo_string));
  // The neg_compute read/write 16 bytes in total, thus, it requires 16 seconds
  // for default memory access. Since it only requires 2 FLOPs computation which
  // requires 2 seconds, it is a  memory-bound instruction which does not have
  // idle time to process async copies.
  // Workflow:
  // neg_compute: | 16 sec (memory-bound)  |
  // copy-done.1: |                        | read 32 bytes  |
  // time:        |     16 sec             |      32 sec    |
  EXPECT_EQ(
      runtime_simulator_->SimulateElapsedTime(module_.get(), allocations_), 48);
}

TEST_F(MemorySpaceAssignmentSimulatorTest, SingleAsyncSliceCopyOverhead) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
        ENTRY Entry {
        param_0 = f32[3072,2048] parameter(0)
        slice-start = ((f32[3072,2048]), f32[768,2048]{1,0:S(1)}, s32[]) slice-start(f32[3072,2048] param_0), slice={[1536:2304], [0:2048]}
        ROOT slice-done = f32[768,2048]{1,0:T(8,128)S(1)} slice-done(((f32[3072,2048]), f32[768,2048]{1,0:S(1)}, s32[]) slice-start)
      }
      )";
  TF_ASSERT_OK(Initialize(hlo_string));

  memory_space_assignment::AllocationSequence allocations;
  // The expected elapsed time is 768 * 2048 * 4 / 1 = 6291456.
  float expected_elapsed_time = 6291456;

  EXPECT_EQ(
      runtime_simulator_->SimulateElapsedTime(module_.get(), allocations_),
      expected_elapsed_time);
}

TEST_F(MemorySpaceAssignmentSimulatorTest,
       AsyncCopyAndAsyncSliceAndComputeOverhead) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
        ENTRY Entry {
        param_0 = f32[2048] parameter(0)
        param_1 = f32[64] parameter(1)
        param_2 = f32[128] parameter(2)
        slice-start = ((f32[2048]), f32[64]{0:S(1)}, s32[]) slice-start(f32[2048] param_0), slice={[0:64]}
        copy-start = (f32[64]{0:S(1)}, f32[64], u32[]) copy-start(f32[64] param_1)
        slice-done = f32[64]{0:S(1)} slice-done(((f32[2048]), f32[64]{0:S(1)}, s32[]) slice-start)
        copy-done = f32[64]{0:S(1)} copy-done(copy-start)
        copy-start-overlap = (f32[128]{0:S(1)}, f32[128], u32[]) copy-start(f32[128] param_2)
        add = f32[64]{0:S(1)} add(slice-done, copy-done)
        ROOT copy-done-overlap = f32[128]{0:S(1)} copy-done(copy-start-overlap)
      }
      )";
  TF_ASSERT_OK(Initialize(hlo_string));

  // The overhead of each instruction is:
  // slice-done: 64 * 4 / 1 = 256 sec (default memory access)
  // copy-done: 64 * 4 /1 = 256 sec (default memory access)
  // add: 3 * 64 * 4 / 2 = 384 sec (alternate memory access)
  // Since add does not access default memory, we can use process 384 bytes in
  // copy-start-overlap.
  // copy-done-overlap: (128 * 4 - 384) / 1 = 128 sec (default memory access)
  EXPECT_EQ(
      runtime_simulator_->SimulateElapsedTime(module_.get(), allocations_),
      1024);
}

class SimulateAsyncCopyLikeDoneTest
    : public MemorySpaceAssignmentSimulatorTest {
 protected:
  absl::Status Initialize(absl::string_view hlo_string) {
    TF_RETURN_IF_ERROR(
        MemorySpaceAssignmentSimulatorTest::Initialize(hlo_string));
    if (instruction_map_.contains("copy-start.1")) {
      outstanding_read_default_queue_.push_back(
          memory_space_assignment::OutstandingAsyncCopyLike{
              instruction_map_["copy-start.1"], 512});
    }
    if (instruction_map_.contains("copy-start.2")) {
      outstanding_write_default_queue_.push_back(
          memory_space_assignment::OutstandingAsyncCopyLike{
              instruction_map_["copy-start.2"], 128});
    }
    runtime_simulator_ = std::make_unique<RuntimeSimulator>(
        cost_analysis_.get(), kAlternateMemorySpace,
        outstanding_read_default_queue_, outstanding_write_default_queue_);
    return absl::OkStatus();
  }
  std::list<memory_space_assignment::OutstandingAsyncCopyLike>
      outstanding_read_default_queue_;
  std::list<memory_space_assignment::OutstandingAsyncCopyLike>
      outstanding_write_default_queue_;
};

TEST_F(SimulateAsyncCopyLikeDoneTest, AsyncCopyAlreadyCompleted) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
      ENTRY Entry {
        param_0 = f32[128] parameter(0)
        copy-start.1 = (f32[128]{0:S(1)}, f32[128], u32[]) copy-start(param_0)
        ROOT copy-done.1 = f32[128]{0:S(1)} copy-done(copy-start.1)
      }
    )";

  TF_ASSERT_OK(Initialize(hlo_string));

  const HloInstruction* copy_done_inst = instruction_map_["copy-done.1"];
  // Process the copy-start.1
  runtime_simulator_->SimulateAsyncCopyLikeDone(copy_done_inst);

  // There should be no request in the read/write queues.
  EXPECT_THAT(runtime_simulator_->GetOutstandingReadDefaultQueue(), IsEmpty());
  EXPECT_THAT(runtime_simulator_->GetOutstandingWriteDefaultQueue(), IsEmpty());
  // The function should return 0 for requests that are already completed.
  float elapsed_time_for_completed_copy =
      runtime_simulator_->SimulateAsyncCopyLikeDone(copy_done_inst);
  EXPECT_EQ(elapsed_time_for_completed_copy, 0);
  // There should be no request in the read/write queues.
  EXPECT_THAT(runtime_simulator_->GetOutstandingReadDefaultQueue(), IsEmpty());
  EXPECT_THAT(runtime_simulator_->GetOutstandingWriteDefaultQueue(), IsEmpty());
}

TEST_F(SimulateAsyncCopyLikeDoneTest, AsyncCopyFullBandwidth) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
      ENTRY Entry {
        param_0 = f32[128] parameter(0)
        copy-start.1 = (f32[128]{0:S(1)}, f32[128], u32[]) copy-start(param_0)
        ROOT copy-done.1 = f32[128]{0:S(1)} copy-done(copy-start.1)
      }
    )";

  TF_ASSERT_OK(Initialize(hlo_string));
  const HloInstruction* copy_done_inst = instruction_map_["copy-done.1"];

  // The elapsed time for copy-done.1 is 128 * 4 / 1 = 512.
  float copy_done_elapsed_time =
      runtime_simulator_->SimulateAsyncCopyLikeDone(copy_done_inst);
  EXPECT_EQ(copy_done_elapsed_time, 512);

  // There should be no request in the read/write queues.
  EXPECT_THAT(runtime_simulator_->GetOutstandingReadDefaultQueue(), IsEmpty());
  EXPECT_THAT(runtime_simulator_->GetOutstandingWriteDefaultQueue(), IsEmpty());
}

TEST_F(SimulateAsyncCopyLikeDoneTest, AsyncCopySharedBandwidth) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
      ENTRY Entry {
        param_0 = f32[128] parameter(0)
        param_1 = f32[32]{0:S(1)} parameter(1)
        copy-start.1 = (f32[128]{0:S(1)}, f32[128], u32[]) copy-start(param_0)
        copy-start.2 = (f32[32], f32[32]{0:S(1)}, u32[]) copy-start(param_1)
        copy-done.2 = f32[32] copy-done(copy-start.2)
        ROOT copy-done.1 = f32[128]{0:S(1)} copy-done(copy-start.1)
      }
    )";

  TF_ASSERT_OK(Initialize(hlo_string));

  const HloInstruction* copy_start_1_inst = instruction_map_["copy-start.1"];
  const HloInstruction* copy_done_2_inst = instruction_map_["copy-done.2"];

  // The copy-start.2 needs to share bandwidth with copy-start.1. Thus, it can
  // only use half bandwidth to access default memory. Thus, the elapsed time is
  // 32 * 4 / 0.5 = 256
  float copy_done_2_elapsed_time =
      runtime_simulator_->SimulateAsyncCopyLikeDone(copy_done_2_inst);
  EXPECT_EQ(copy_done_2_elapsed_time, 256);

  // The only write request (copy-start.2) should be completed.
  EXPECT_THAT(runtime_simulator_->GetOutstandingWriteDefaultQueue(), IsEmpty());

  // The read request has (128-32)*4 bytes left to process.
  EXPECT_THAT(
      runtime_simulator_->GetOutstandingReadDefaultQueue(),
      ElementsAreArray({memory_space_assignment::OutstandingAsyncCopyLike{
          copy_start_1_inst, 384}}));
}

TEST_F(SimulateAsyncCopyLikeDoneTest, AsyncCopyTransferPartialProcess) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
      ENTRY Entry {
        param_0 = f32[128] parameter(0)
        param_1 = f32[32]{0:S(1)} parameter(1)
        copy-start.1 = (f32[128]{0:S(1)}, f32[128], u32[]) copy-start(param_0)
        copy-start.2 = (f32[32], f32[32]{0:S(1)}, u32[]) copy-start(param_1)
        copy-done.2 = f32[32] copy-done(copy-start.2)
        ROOT copy-done.1 = f32[128]{0:S(1)} copy-done(copy-start.1)
      }
    )";

  TF_ASSERT_OK(Initialize(hlo_string));

  const HloInstruction* copy_start_1_inst = instruction_map_["copy-start.1"];
  const HloInstruction* copy_done_1_inst = instruction_map_["copy-done.1"];
  const HloInstruction* copy_done_2_inst = instruction_map_["copy-done.2"];

  // Execute copy-done.2.
  float copy_done_2_elapsed_time =
      runtime_simulator_->SimulateAsyncCopyLikeDone(copy_done_2_inst);
  // For copy-done.2, it requires to transfer 32*4 bytes
  // default-write request. At the same time, there is a 128*4 bytes
  // default-read request in the queue for copy-start.1. So the
  // elapsed time for copy-done.2 is 32*4 / (0.5*1) = 256.
  EXPECT_EQ(copy_done_2_elapsed_time, 256);
  // In parallel with copy-done.2, copy-start.1 is also being processed.
  // So the remaining bytes should be 128*4 - 32*4 = 384.
  EXPECT_THAT(
      runtime_simulator_->GetOutstandingReadDefaultQueue(),
      ElementsAreArray({memory_space_assignment::OutstandingAsyncCopyLike{
          copy_start_1_inst, 384}}));
  EXPECT_THAT(runtime_simulator_->GetOutstandingWriteDefaultQueue(), IsEmpty());

  // Execute copy-done.1.
  float copy_done_1_elapsed_time =
      runtime_simulator_->SimulateAsyncCopyLikeDone(copy_done_1_inst);
  // The copy-done.1 is the only request in the read-queue, and there is no
  // request in the write-queue. Thus, it can use the full bandwidth. The
  // elapsed time is 384 / 1 = 384.
  EXPECT_EQ(copy_done_1_elapsed_time, 384);
  // No request should be in the queue.
  EXPECT_THAT(runtime_simulator_->GetOutstandingReadDefaultQueue(), IsEmpty());
  EXPECT_THAT(runtime_simulator_->GetOutstandingWriteDefaultQueue(), IsEmpty());
}

TEST_F(SimulateAsyncCopyLikeDoneTest,
       SimulateComputeInstructionWithSingleAsyncCopy) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
      ENTRY Entry {
        param_0 = f32[128] parameter(0)
        param_1 = f32[32] parameter(1)
        copy-start.1 = (f32[128]{0:S(1)}, f32[128], u32[]) copy-start(param_0)
        neg = f32[32] negate(param_1)
        ROOT copy-done.1 = f32[128]{0:S(1)} copy-done(copy-start.1)
      }
    )";

  TF_ASSERT_OK(Initialize(hlo_string));
  const HloInstruction* copy_start_1_inst = instruction_map_["copy-start.1"];
  const HloInstruction* neg_inst = instruction_map_["neg"];

  float compute_elapsed_time =
      runtime_simulator_
          ->SimulateComputeInstruction(neg_inst,
                                       /*operands_in_alternate_memory=*/{},
                                       /*outputs_in_alternate_memory=*/{})
          .elapsed_time;

  // The compute operand requires 32 FLOPs and 32 * 4 * 2 bytes access, which
  // requires 32 and 256 secs respectively. Thus, it is default memory access
  // dominated, which does not have idle time to process the async copy.
  EXPECT_EQ(compute_elapsed_time, 256);
  EXPECT_THAT(
      runtime_simulator_->GetOutstandingReadDefaultQueue(),
      ElementsAreArray({memory_space_assignment::OutstandingAsyncCopyLike{
          copy_start_1_inst, 512}}));

  EXPECT_THAT(runtime_simulator_->GetOutstandingWriteDefaultQueue(), IsEmpty());
}

TEST_F(SimulateAsyncCopyLikeDoneTest,
       SimulateComputeInstructionWithSharedBandwidth) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
      ENTRY Entry {
        param_0 = f32[128] parameter(0)
        param_1 = f32[32]{0:S(1)} parameter(1)
        copy-start.1 = (f32[128]{0:S(1)}, f32[128], u32[]) copy-start(param_0)
        copy-start.2 = (f32[32], f32[32]{0:S(1)}, u32[]) copy-start(param_1)
        neg = f32[32] negate(param_1)
        copy-done.2 = f32[32] copy-done(copy-start.2)
        ROOT copy-done.1 = f32[128]{0:S(1)} copy-done(copy-start.1)
      }
    )";

  TF_ASSERT_OK(Initialize(hlo_string));

  const HloInstruction* copy_start_1_inst = instruction_map_["copy-start.1"];
  const HloInstruction* copy_start_2_inst = instruction_map_["copy-start.2"];

  // The instruction reads 32 * 4 bytes from alternate memory, which takes 64
  // secs. In this 64 secs, it does not access default memory. Thus, we can
  // process the async copies in this time. Both queues are not empty, so the
  // bandwidth is shared. Each of the request at the front of the queue process
  // 64 sec * 0.5 bytes/sec = 32 bytes.
  float compute_elapsed_time =
      runtime_simulator_
          ->SimulateComputeInstruction(
              instruction_map_["neg"],
              /*operands_in_alternate_memory=*/{{0, {}}},
              /*outputs_in_alternate_memory=*/{})
          .elapsed_time;
  // 64 secs for alternate memory access + 128 secs for default memory access
  EXPECT_EQ(compute_elapsed_time, 192);

  EXPECT_THAT(
      runtime_simulator_->GetOutstandingReadDefaultQueue(),
      ElementsAreArray({memory_space_assignment::OutstandingAsyncCopyLike{
          copy_start_1_inst, 480}}));

  EXPECT_THAT(
      runtime_simulator_->GetOutstandingWriteDefaultQueue(),
      ElementsAreArray({memory_space_assignment::OutstandingAsyncCopyLike{
          copy_start_2_inst, 96}}));
}

TEST_F(SimulateAsyncCopyLikeDoneTest,
       SimulateComputeInstructionWithFullBandwidth) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
      ENTRY Entry {
        param_0 = f32[128] parameter(0)
        param_1 = f32[32]{0:S(1)} parameter(1)
        copy-start.1 = (f32[128]{0:S(1)}, f32[128], u32[]) copy-start(param_0)
        neg = f32[32] negate(param_1)
        ROOT copy-done.1 = f32[128]{0:S(1)} copy-done(copy-start.1)
      }
    )";

  TF_ASSERT_OK(Initialize(hlo_string));

  const HloInstruction* copy_start_1_inst = instruction_map_["copy-start.1"];

  // Same as the 'SimulateComputeInstructionWithSharedBandwidth' test, there are
  // 64 secs idle time to process async copies. Since only the read queue is not
  // empty, we can use the full bandwidth and process 64 sec * 1 bytes/sec = 64
  // bytes.
  float compute_elapsed_time =
      runtime_simulator_
          ->SimulateComputeInstruction(
              instruction_map_["neg"],
              /*operands_in_alternate_memory=*/{{0, {}}},
              /*outputs_in_alternate_memory=*/{})
          .elapsed_time;
  // 64 secs for alternate memory access + 128 secs for default memory access
  EXPECT_EQ(compute_elapsed_time, 192);

  EXPECT_THAT(
      runtime_simulator_->GetOutstandingReadDefaultQueue(),
      ElementsAreArray({memory_space_assignment::OutstandingAsyncCopyLike{
          copy_start_1_inst, 448}}));
  EXPECT_THAT(runtime_simulator_->GetOutstandingWriteDefaultQueue(), IsEmpty());
}

TEST_F(SimulateAsyncCopyLikeDoneTest,
       SimulateComputeInstructionWithEmptyQueues) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true
      ENTRY Entry {
        param_0 = f32[128] parameter(0)
        ROOT neg = f32[128] negate(param_0)
      }
    )";

  TF_ASSERT_OK(Initialize(hlo_string));

  float compute_elapsed_time =
      runtime_simulator_
          ->SimulateComputeInstruction(instruction_map_["neg"],
                                       /*operands_in_alternate_memory=*/{},
                                       /*outputs_in_alternate_memory=*/{})
          .elapsed_time;
  // Execution time: 128 * 4 * 2 / 1 for default access
  EXPECT_EQ(compute_elapsed_time, 1024);
  // The queues should remain empty.
  EXPECT_THAT(runtime_simulator_->GetOutstandingReadDefaultQueue(), IsEmpty());
  EXPECT_THAT(runtime_simulator_->GetOutstandingWriteDefaultQueue(), IsEmpty());
}

}  // namespace
}  // namespace xla
