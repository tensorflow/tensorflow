/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"

#include <memory>
#include <stack>
#include <unordered_set>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/util/ptr_util.h"

namespace xla {
namespace gpu {
namespace {
void InitAndStartTimer(std::stack<std::unique_ptr<se::Timer>>* timers,
                       se::Stream* stream) {
  timers->push(absl::make_unique<se::Timer>(stream->parent()));
  stream->InitTimer(timers->top().get()).ThenStartTimer(timers->top().get());
}

uint64 GetCyclesTaken(std::stack<std::unique_ptr<se::Timer>>* timers,
                      const std::vector<StreamPool::Ptr>& sub_streams,
                      se::Stream* stream, double clock_rate_ghz) {
  CHECK_GT(timers->size(), 0);
  stream->ThenWaitFor(&sub_streams);
  stream->ThenStopTimer(timers->top().get());
  stream->BlockHostUntilDone().IgnoreError();
  double nanoseconds = timers->top()->Nanoseconds();
  timers->pop();
  return static_cast<uint64>(nanoseconds * clock_rate_ghz);
}
}  // namespace

HloExecutionProfiler::HloExecutionProfiler(
    bool do_profile, HloExecutionProfile* profile, se::Stream* stream,
    const std::vector<StreamPool::Ptr>& sub_streams,
    const HloComputation* computation)
    : do_profile_(do_profile),
      profile_(profile),
      stream_(stream),
      sub_streams_(sub_streams),
      computation_(computation) {
  if (do_profile_) {
    clock_rate_ghz_ = stream->parent()->GetDeviceDescription().clock_rate_ghz();
    InitAndStartTimer(&timers_, stream);
  }
}

void HloExecutionProfiler::FinishExecution() {
  CHECK(!finished_execution_) << "Call FinishExecution only once!";
  finished_execution_ = true;
  if (do_profile_) {
    profile_->set_total_cycles_executed(
        *computation_,
        GetCyclesTaken(&timers_, sub_streams_, stream_, clock_rate_ghz_));
  }
}

void HloExecutionProfiler::StartHloComputation() {
  if (do_profile_) {
    InitAndStartTimer(&timers_, stream_);
  }
}

void HloExecutionProfiler::FinishHloComputation(
    const HloComputation* computation) {
  if (do_profile_) {
    profile_->set_total_cycles_executed(
        *computation,
        GetCyclesTaken(&timers_, sub_streams_, stream_, clock_rate_ghz_));
  }
}

void HloExecutionProfiler::StartHloInstruction() {
  if (do_profile_) {
    InitAndStartTimer(&timers_, stream_);
  }
}

void HloExecutionProfiler::FinishHloInstruction(
    const HloInstruction* hlo_instruction) {
  if (do_profile_) {
    hlo_instructions_.erase(hlo_instruction);
    profile_->SetCyclesTakenBy(
        hlo_instruction,
        GetCyclesTaken(&timers_, sub_streams_, stream_, clock_rate_ghz_));
  }
}

std::unique_ptr<ScopedInstructionProfiler>
HloExecutionProfiler::MakeScopedInstructionProfiler(
    const HloInstruction* hlo_instruction) {
  if (do_profile_ && hlo_instruction != nullptr) {
    // Make sure that we are not already measuring the time for the same
    // 'hlo_instruction'.
    CHECK(hlo_instructions_.insert(hlo_instruction).second)
        << hlo_instruction->name();
  }
  return absl::make_unique<ScopedInstructionProfiler>(this, hlo_instruction);
}

}  // namespace gpu
}  // namespace xla
