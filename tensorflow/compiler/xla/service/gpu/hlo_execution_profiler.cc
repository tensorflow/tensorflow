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
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

HloExecutionProfiler::HloExecutionProfiler(
    bool do_profile, HloExecutionProfile* profile, se::Stream* stream,
    const std::vector<Pool<se::Stream>::SmartPtr>& sub_streams,
    const HloComputation* computation)
    : do_profile_(do_profile),
      profile_(profile),
      stream_(stream),
      sub_streams_(sub_streams),
      computation_(computation) {
  if (do_profile_) {
    clock_rate_ghz_ = stream->parent()->GetDeviceDescription().clock_rate_ghz();
    execution_timer_.reset(new se::Timer(stream->parent()));
    per_op_timer_.reset(new se::Timer(stream->parent()));
    stream->InitTimer(execution_timer_.get())
        .ThenStartTimer(execution_timer_.get());
    stream->InitTimer(per_op_timer_.get());
  }
}

void HloExecutionProfiler::FinishExecution() {
  CHECK(!finished_execution_) << "Call FinishExecution only once!";
  finished_execution_ = true;
  if (do_profile_) {
    stream_->ThenWaitFor(&sub_streams_);
    stream_->ThenStopTimer(execution_timer_.get());
    stream_->BlockHostUntilDone().IgnoreError();
    profile_->set_total_cycles_executed(
        *computation_,
        static_cast<uint64>(execution_timer_->Nanoseconds() * clock_rate_ghz_));
  }
}

void HloExecutionProfiler::StartOperation() {
  if (do_profile_) {
    stream_->ThenStartTimer(per_op_timer_.get());
  }
}

void HloExecutionProfiler::FinishOperation(
    const HloInstruction* hlo_instruction) {
  if (do_profile_) {
    stream_->ThenWaitFor(&sub_streams_);
    stream_->ThenStopTimer(per_op_timer_.get());
    stream_->BlockHostUntilDone().IgnoreError();
    profile_->SetCyclesTakenBy(
        hlo_instruction,
        static_cast<uint64>(per_op_timer_->Nanoseconds() * clock_rate_ghz_));
  }
}

}  // namespace gpu
}  // namespace xla
