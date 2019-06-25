/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/ipu_scheduler.h"

#include <map>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<HloSchedule> IpuScheduleModule(
    HloModule* module, const LogicalBuffer::SizeFunction& size_function,
    const IpuSchedulerAlgorithm& algorithm) {
  HloSchedule schedule(module);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TuplePointsToAnalysis> points_to_analysis,
                      TuplePointsToAnalysis::Run(module));
  std::unique_ptr<HloAliasAnalysis> alias_analysis =
      HloAliasAnalysis::NewEmptyAnalysis(module);
  absl::flat_hash_map<const HloComputation*, int64> memory_by_computation;
  for (auto* computation : module->MakeComputationPostOrder()) {
    if (!computation->IsFusionComputation()) {
      TF_ASSIGN_OR_RETURN(HloInstructionSequence computation_sequence,
                          algorithm(computation, *points_to_analysis,
                                    size_function, memory_by_computation));
      TF_ASSIGN_OR_RETURN(
          auto bytes,
          HeapSimulator::MinimumMemoryForComputation(
              *computation, computation_sequence, *alias_analysis,
              size_function, &memory_by_computation));

      memory_by_computation[computation] = bytes;
      schedule.set_sequence(computation, std::move(computation_sequence));
    }
  }
  VLOG(1) << "Module schedule:\n" << schedule;

  TF_RETURN_IF_ERROR(schedule.Verify());

  return std::move(schedule);
}
}  // namespace

IpuSchedulerAlgorithm MemorySchedulerAlgorithmToIPU(
    MemorySchedulerAlgorithm algorithm) {
  return [algorithm](HloComputation* computation,
                     const TuplePointsToAnalysis& points_to_analysis,
                     const LogicalBuffer::SizeFunction& size_function,
                     const absl::flat_hash_map<const HloComputation*, int64>&
                        memory_by_computation) {
    std::unique_ptr<HloAliasAnalysis> alias_analysis =
        HloAliasAnalysis::NewEmptyAnalysis(computation->parent());
    return algorithm(computation, points_to_analysis, *alias_analysis,
                     size_function, memory_by_computation);
  };
}

MemorySchedulerAlgorithm IpuToMemorySchedulerAlgorithm(
    IpuSchedulerAlgorithm algorithm) {
  return [algorithm](HloComputation* computation,
                     const TuplePointsToAnalysis& points_to_analysis,
                     const HloAliasAnalysis& alias_analysis,
                     const LogicalBuffer::SizeFunction& size_function,
                     const absl::flat_hash_map<const HloComputation*, int64>&
                     memory_by_computation) {
    return algorithm(computation, points_to_analysis, size_function,
                     memory_by_computation);
  };
}

StatusOr<IpuSchedulerAlgorithm> BestIpuSchedule(
    IpuSchedulerAlgorithm algorithm_a, IpuSchedulerAlgorithm algorithm_b) {
  if (!algorithm_a && !algorithm_b) {
    return xla::FailedPrecondition(
        "Cannot construct BestIpuSchedule when both inputs are invalid");
  }

  // Handle cases with invalid algorithms
  if (!algorithm_a) {
    return algorithm_b;
  }

  if (!algorithm_b) {
    return algorithm_a;
  }

  return IpuSchedulerAlgorithm{
      [algorithm_a, algorithm_b](
          HloComputation* computation,
          const TuplePointsToAnalysis& tuple_points_to_analysis,
          const LogicalBuffer::SizeFunction& size_function,
          const absl::flat_hash_map<const HloComputation*, int64>&
              memory_by_computation) -> StatusOr<HloInstructionSequence> {
        std::unique_ptr<HloAliasAnalysis> alias_analysis =
            HloAliasAnalysis::NewEmptyAnalysis(computation->parent());

        auto schedule_a_status =
            algorithm_a(computation, tuple_points_to_analysis, size_function,
                        memory_by_computation);

        auto schedule_b_status =
            algorithm_b(computation, tuple_points_to_analysis, size_function,
                        memory_by_computation);

        // If no valid schedule could be produced return the first failure
        if (!schedule_a_status.ok() && !schedule_b_status.ok()) {
          schedule_b_status.IgnoreError();
          return schedule_a_status;
        }

        // If schedule A is invalid, return B
        if (!schedule_a_status.ok()) {
          schedule_a_status.IgnoreError();
          return schedule_b_status;
        }

        // If schedule B is invalid, return A
        if (!schedule_b_status.ok()) {
          schedule_b_status.IgnoreError();
          return schedule_a_status;
        }

        // If both schedules succeeded, we must evaluate which is better
        // TODO(T9494): Replace the heap simulator
        auto schedule_a = schedule_a_status.ValueOrDie();
        auto schedule_b = schedule_b_status.ValueOrDie();

        TF_ASSIGN_OR_RETURN(
            const int64 schedule_a_memory,
            HeapSimulator::MinimumMemoryForComputation(
                *computation, schedule_a, *alias_analysis, size_function,
                &memory_by_computation));

        TF_ASSIGN_OR_RETURN(
            const int64 schedule_b_memory,
            HeapSimulator::MinimumMemoryForComputation(
                *computation, schedule_b, *alias_analysis, size_function,
                &memory_by_computation));

        // If schedule A is better than B, return A
        if (schedule_a_memory < schedule_b_memory) {
          return schedule_a;
        }

        // Otherwise return schedule A
        return schedule_b;
      }};
}

StatusOr<IpuSchedulerAlgorithm> BestIpuSchedule(
    const std::vector<IpuSchedulerAlgorithm>& algorithms) {
  if (algorithms.empty()) {
    return xla::FailedPrecondition(
        "Cannot construct BestIpuSchedule when the input is empty");
  }

  auto algo_predicate = [](IpuSchedulerAlgorithm algo) -> bool {
    return static_cast<bool>(algo);
  };

  if (absl::c_none_of(algorithms, algo_predicate)) {
    return xla::FailedPrecondition(
        "Cannot construct BestIpuSchedule when none of the inputs are valid");
  }

  // Iteratively apply the binary `BestIpuSchedule`.
  // TODO(T9495) Consider building a balanced tree for parallel execution
  IpuSchedulerAlgorithm result;
  for (auto& algo : algorithms) {
    TF_ASSIGN_OR_RETURN(result, BestIpuSchedule(result, algo));
  }

  return result;
}

IpuScheduler::IpuScheduler(const LogicalBuffer::SizeFunction& size_function,
                           const IpuSchedulerAlgorithm& algorithm)
    : size_function_(size_function), algorithm_(algorithm) {}

StatusOr<bool> IpuScheduler::Run(HloModule* module) {
  if (!algorithm_) {
    return xla::FailedPrecondition("No IpuSchedulerAlgorithm provided");
  }

  TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                      IpuScheduleModule(module, size_function_, algorithm_));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
