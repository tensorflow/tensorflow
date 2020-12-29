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

  // If profiling is enabled stops the timer for a (sub)computation with the
  // given profile index and records the time that the computation took to
  // execute in the profile.
  void FinishHloComputation(absl::optional<size_t> profile_index);

  // If profiling is enabled, starts a per-operation timer.
  void StartHloInstruction();

  // If profiling is enabled, stops the per-operation timer and records the time
  // that at `profile_index`. Profile indices can be looked up from
  // HloProfileIndexMap.
  void FinishHloInstruction(size_t profile_index);

  // Returns a ScopedInstructionProfiler and triggers a call to
  // StartHloInstruction(). Once the returned ScopedInstructionProfiler goes
  // out of scope, it triggers a call to FinishHloInstruction().
  //
  // If profile_index < 0, it results in a no-op.
  std::unique_ptr<ScopedInstructionProfiler> MakeScopedInstructionProfiler(
      absl::optional<int64> profile_index);

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
  std::unordered_set<size_t> indices_;
  bool finished_execution_ = false;
};

// This class can be used within the ExecuteOnStream() implementations of
// Thunks. It ensures that we always have a pair of matching
// StartHloInstruction() and FinishHloInstruction() calls to the profiler.
class ScopedInstructionProfiler {
 public:
  ScopedInstructionProfiler(HloExecutionProfiler* profiler,
                            absl::optional<int64> index)
      : profiler_(profiler), index_(index) {
    if (index_.has_value()) {
      profiler->StartHloInstruction();
    }
  }
  ~ScopedInstructionProfiler() {
    if (index_.has_value()) {
      profiler_->FinishHloInstruction(*index_);
    }
  }

 private:
  HloExecutionProfiler* profiler_;
  absl::optional<int64> index_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_EXECUTION_PROFILER_H_
