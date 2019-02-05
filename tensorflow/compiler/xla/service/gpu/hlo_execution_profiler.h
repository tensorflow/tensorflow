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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_EXECUTION_PROFILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_EXECUTION_PROFILER_H_

#include <memory>
#include <stack>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

class ScopedInstructionProfiler;

// A helper class for profiling HLO in the course of GPU program execution.
// All of the profiling is guarded internally, to avoid the caller needing to
// have lots of conditionals sprinkled around.
class HloExecutionProfiler {
 public:
  // If profiling is enabled, start an execution timer running.
  explicit HloExecutionProfiler(bool do_profile, HloExecutionProfile* profile,
                                se::Stream* stream,
                                const std::vector<StreamPool::Ptr>& sub_streams,
                                const HloComputation* computation);

  // If profiling is enabled, sets the total cycle count on the profile from the
  // execution timer.
  void FinishExecution();

  // If profiling is enabled, starts a timer for a (sub)computation.
  void StartHloComputation();

  // If profiling is enabled stops the timer for a (sub)computation and records
  // the time that the computation took to execute in the profile.
  void FinishHloComputation(const HloComputation* computation);

  // If profiling is enabled, starts a per-operation timer.
  void StartHloInstruction();

  // If profiling is enabled, stops the per-operation timer and records the time
  // that the hlo_instruction took to execute in the profile.
  void FinishHloInstruction(const HloInstruction* hlo_instruction);

  // Returns a ScopedInstructionProfiler and triggers a call to
  // StartHloInstruction(). Once the returned ScopedInstructionProfiler goes
  // out of scope, it triggers a call to FinishHloInstruction().
  std::unique_ptr<ScopedInstructionProfiler> MakeScopedInstructionProfiler(
      const HloInstruction* hlo_instruction);

 private:
  const bool do_profile_;
  double clock_rate_ghz_;
  HloExecutionProfile* profile_;
  se::Stream* stream_;
  const std::vector<StreamPool::Ptr>& sub_streams_;
  const HloComputation* computation_;
  std::stack<std::unique_ptr<se::Timer>> timers_;
  // Contains the HLO instructions for which we are currently measuring the
  // time.
  std::unordered_set<const HloInstruction*> hlo_instructions_;
  bool finished_execution_ = false;
};

// This class can be used within the ExecuteOnStream() implementations of
// Thunks. It ensures that we always have a pair of matching
// StartHloInstruction() and FinishHloInstruction() calls to the profiler.
class ScopedInstructionProfiler {
 public:
  ScopedInstructionProfiler(HloExecutionProfiler* profiler,
                            const HloInstruction* hlo_instruction)
      : profiler_(profiler), hlo_instruction_(hlo_instruction) {
    if (hlo_instruction != nullptr) {
      profiler->StartHloInstruction();
    }
  }
  ~ScopedInstructionProfiler() {
    if (hlo_instruction_ != nullptr) {
      profiler_->FinishHloInstruction(hlo_instruction_);
    }
  }

 private:
  HloExecutionProfiler* profiler_;
  const HloInstruction* hlo_instruction_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_EXECUTION_PROFILER_H_
