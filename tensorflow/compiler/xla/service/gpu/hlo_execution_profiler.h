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
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pool.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// A helper class for profiling HLO in the course of GPU program execution.
// All of the profiling is guarded internally, to avoid the caller needing to
// have lots of conditionals sprinkled around.
class HloExecutionProfiler {
 public:
  // If profiling is enabled, start an execution timer running.
  explicit HloExecutionProfiler(
      bool do_profile, HloExecutionProfile* profile, se::Stream* stream,
      const std::vector<Pool<se::Stream>::SmartPtr>& sub_streams,
      const HloComputation* computation);

  // If profiling is enabled, sets the total cycle count on the profile from the
  // execution timer.
  void FinishExecution();

  // If profiling is enabled, starts the per-operation timer.
  void StartOperation();

  // If profiling is enabled, stops the per-operation timer and records the time
  // that the hlo_instruction took to execute in the profile.
  void FinishOperation(const HloInstruction* hlo_instruction);

 private:
  const bool do_profile_;
  double clock_rate_ghz_;
  HloExecutionProfile* profile_;
  se::Stream* stream_;
  const std::vector<Pool<se::Stream>::SmartPtr>& sub_streams_;
  const HloComputation* computation_;
  std::unique_ptr<se::Timer> execution_timer_;
  std::unique_ptr<se::Timer> per_op_timer_;
  bool finished_execution_ = false;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_EXECUTION_PROFILER_H_
