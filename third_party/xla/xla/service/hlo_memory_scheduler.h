/* Copyright 2016 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_MEMORY_SCHEDULER_H_
#define XLA_SERVICE_HLO_MEMORY_SCHEDULER_H_

#include <cstdint>
#include <functional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/tuple_points_to_analysis.h"

namespace xla {

// Postprocessor of the HloInstructionSequence. This is an opt-in postprocessing
// function to MemorySchedulerAlgorithm to enforce certain hlo schedule
// constraints desired for custom-calls.
using MemorySchedulerPostprocessor =
    std::function<HloInstructionSequence(const HloInstructionSequence&)>;

// A memory scheduler computes an execution sequence for the HLO instructions in
// 'computation' that minimizes peak memory (or finds a balance between memory
// and available concurrency), given a points-to analysis result that describes
// buffer aliasing, together with a target-specific size function that maps a
// tensor's logical size to its padded size. peak_memory (may be nullptr) is set
// to the peak memory of the resulting schedule according to the HeapSimulator.
//
// TODO(yunxing): Cleanup usage of TuplePointsToAnalysis.
using MemorySchedulerAlgorithm =
    std::function<absl::StatusOr<HloInstructionSequence>(
        HloComputation*, const TuplePointsToAnalysis&, const HloAliasAnalysis&,
        const LogicalBuffer::SizeFunction&,
        const MemorySchedulerPostprocessor&,
        /*peak_memory*/ int64_t*)>;

// Scheduler for the entire module.
using ModuleSchedulerAlgorithm = std::function<absl::StatusOr<HloSchedule>(
    const HloModule*, const TuplePointsToAnalysis&, const HloAliasAnalysis&,
    const LogicalBuffer::SizeFunction&,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    /*peak_memory*/ int64_t*)>;

// Lift a computation scheduler into a module scheduler by calling the
// computation scheduler on all computations in a module.
ModuleSchedulerAlgorithm ComputationSchedulerToModuleScheduler(
    const MemorySchedulerAlgorithm&, const MemorySchedulerPostprocessor& = {});

// List scheduler
absl::StatusOr<HloInstructionSequence> ListMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory);

// DFS-order scheduler
absl::StatusOr<HloInstructionSequence> DFSMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory);

// BFS-order scheduler
//
// BFS-order scheduler is a simple memory scheduler that schedules instructions
// in a breadth-first order, which maximizes the available concurrency at the
// cost of increased memory usage (HLO operations that do not have buffer
// conflicts can be executed in parallel).
//
// This is the most trivial scheduling optimized for maximum concurrency. In
// practice it is only useful for CPU backend where memory is cheap and we have
// a lot of available compute cores, and cheap concurrency primitives.
absl::StatusOr<HloInstructionSequence> BFSMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory);

// Naive Post Order scheduler
absl::StatusOr<HloInstructionSequence> PostOrderMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory);

// The default scheduling algorithm. Runs the list scheduler, the DFS scheduler,
// and the post-order scheduler and chooses whichever returns a lower min-
// memory, not accounting for fragmentation. peak_memory (may be nullptr) is set
// to the peak memory of the resulting schedule according to the HeapSimulator.
absl::StatusOr<HloInstructionSequence> DefaultMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory);

absl::StatusOr<HloSchedule> DefaultModuleScheduler(
    const HloModule* module, const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    int64_t* peak_memory);

// Returns an HloSchedule which seeks to minimize the memory required for the
// module. size_function is the function returning the number of bytes required
// for a LogicalBuffer. peak_memory (if not nullptr) is set to the largest peak
// memory (according to the HeapSimulator) of all computations in the module.
absl::StatusOr<HloSchedule> ScheduleModule(
    const HloModule* module, const LogicalBuffer::SizeFunction& size_function,
    const ModuleSchedulerAlgorithm& algorithm = {},
    const absl::flat_hash_set<absl::string_view>& execution_threads = {},
    int64_t* peak_memory = nullptr);

// A pass which schedules the HLO instructions in a module. The HloModule's
// schedule field is set to the resulting HloSchedule using
// HloModule::set_schedule.
class HloMemoryScheduler : public HloModulePass {
 public:
  // size_function is the function returning the number of bytes required for a
  // LogicalBuffer. algorithm is the memory scheduling algorithm to use. If not
  // specified, then DefaultMemoryScheduler is used.
  explicit HloMemoryScheduler(const LogicalBuffer::SizeFunction& size_function,
                              const ModuleSchedulerAlgorithm& algorithm = {});

  ~HloMemoryScheduler() override = default;

  absl::string_view name() const override { return "hlo-memory-scheduler"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  LogicalBuffer::SizeFunction size_function_;

  ModuleSchedulerAlgorithm algorithm_;
};

// A pass which produces a naive, but correct schedule. The schedule is produced
// using a DFS traversal of the graph with no attempt to minimize memory use.
class HloTrivialScheduler : public HloModulePass {
 public:
  absl::string_view name() const override { return "hlo-trivial-scheduler"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

// A trivial pass which clears the schedule currently set on the
// HloModule. After this pass runs HloModule::has_schedule will return false.
class HloDescheduler : public HloModulePass {
 public:
  HloDescheduler() = default;
  ~HloDescheduler() override = default;
  absl::string_view name() const override { return "hlo-descheduler"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_MEMORY_SCHEDULER_H_
