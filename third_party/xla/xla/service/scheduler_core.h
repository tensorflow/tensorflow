/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SCHEDULER_CORE_H_
#define XLA_SERVICE_SCHEDULER_CORE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/xla.pb.h"

namespace xla {

class HloScheduleGraph;

// Base class for the core scheduling algorithm.
class SchedulerCore {
 public:
  // Hook function to modify scheduling graph before scheduler runs.
  using GraphProcessingHook = std::function<absl::Status(HloScheduleGraph*)>;

  // Abstract base class for target defined state.
  struct TargetDefinedState {
    virtual ~TargetDefinedState() = default;
    virtual void Reset() {}
  };

  // Abstract base class for scheduling state.
  struct SchedulingState {
    virtual ~SchedulingState() = default;
    virtual void Reset() {
      if (target_defined_state) {
        target_defined_state->Reset();
      }
    }
    GraphProcessingHook graph_processing_hook;
    std::unique_ptr<TargetDefinedState> target_defined_state;
  };

  virtual absl::Status InitializeScheduler(const HloModule* module) = 0;

  virtual absl::Status CaptureScheduleProto() = 0;

  virtual absl::StatusOr<ScheduleProto> GetCapturedScheduleProto() = 0;

  virtual absl::StatusOr<std::shared_ptr<SchedulerCore::SchedulingState>>
  MakeSchedulingState(const HloComputation* computation) {
    return absl::UnimplementedError("Not implemented.");
  }
  virtual absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation) {
    return absl::UnimplementedError("Not implemented.");
  }
  virtual absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation,
      std::shared_ptr<SchedulingState> sched_state) {
    return absl::UnimplementedError("Not implemented.");
  }
  // Note: This function is not thread-safe. It must not be invoked during
  // concurrent BRKGA or simulated evaluation.
  virtual std::shared_ptr<SchedulingState> GetSchedulingState() {
    return nullptr;
  }

  virtual ~SchedulerCore() = default;
  virtual int64_t GetMemoryPeak() = 0;
  virtual void SetMemoryLimit(uint64_t new_limit) = 0;
  virtual uint64_t GetMemoryLimit() = 0;
  virtual int64_t GetRerunTimes() = 0;

  // Set a graph processing hook that will run before scheduling a computation.
  // Heuristics can use this to set scheduling preferences to the scheduling
  // graph nodes.
  virtual absl::Status SetGraphProcessingHook(const GraphProcessingHook& hook) {
    return absl::OkStatus();
  }
  virtual void SetEvaluatingConcurrently(bool val) {
    is_evaluating_concurrently_ = val;
  }
  virtual bool IsEvaluatingConcurrently() const {
    return is_evaluating_concurrently_;
  }

 protected:
  bool is_evaluating_concurrently_ = false;
};

}  // namespace xla

#endif  // XLA_SERVICE_SCHEDULER_CORE_H_
