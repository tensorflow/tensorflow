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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_MEMORY_SCHEDULER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_MEMORY_SCHEDULER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/buffer_value.h"

namespace xla {

// A module scheduler computes an execution sequence for the HLO instructions in
// 'module' given a points-to analysis result that describes buffer aliasing.
// peak_memory (may be nullptr) is set to the peak memory of the resulting
// schedule according to the HeapSimulator.
class ModuleSchedulerAlgorithm {
 public:
  explicit ModuleSchedulerAlgorithm(const AliasInfo* alias_info)
      : alias_info_(alias_info) {}
  virtual ~ModuleSchedulerAlgorithm() = default;
  virtual absl::StatusOr<HloSchedule> Run(
      const HloModule* module, const HloAliasAnalysis& alias_analysis,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      int64_t* peak_memory) const = 0;

  const AliasInfo* alias_info() const { return alias_info_; }

 protected:
  const AliasInfo* alias_info_;
};

// Postprocessor of the HloInstructionSequence. This is an opt-in postprocessing
// function to ComputationSchedulerAlgorithm to enforce certain hlo schedule
// constraints desired for custom-calls.
using SchedulerPostprocessor =
    std::function<HloInstructionSequence(const HloInstructionSequence&)>;

// Lift a computation scheduler into a module scheduler by calling the
// computation scheduler on all computations in a module.
// size_function is a target-specific size function that maps a tensor's logical
// size to its padded size.
class ComputationSchedulerAlgorithm : public ModuleSchedulerAlgorithm {
 public:
  virtual absl::StatusOr<HloInstructionSequence> Run(
      HloComputation* computation,
      const HloAliasAnalysis& alias_analysis) const = 0;
  absl::StatusOr<HloSchedule> Run(
      const HloModule* module, const HloAliasAnalysis& alias_analysis,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      int64_t* peak_memory) const override;

 protected:
  ComputationSchedulerAlgorithm(const AliasInfo* alias_info,
                                BufferValue::SizeFunction size_function,
                                SchedulerPostprocessor postprocessor)
      : ModuleSchedulerAlgorithm(alias_info),
        size_function_(std::move(size_function)),
        postprocessor_(std::move(postprocessor)) {}

  BufferValue::SizeFunction size_function_;
  SchedulerPostprocessor postprocessor_;
};

// Class implementing a list scheduler of HLO instructions which produces a
// sequence which minimizes memory usage by preferring to schedule the node that
// frees bigger buffer and defines smaller outputs.
class ListMemoryScheduler : public ComputationSchedulerAlgorithm {
 public:
  ListMemoryScheduler(const AliasInfo* alias_info,
                      BufferValue::SizeFunction size_function,
                      SchedulerPostprocessor postprocessor = {})
      : ComputationSchedulerAlgorithm(alias_info, std::move(size_function),
                                      std::move(postprocessor)) {}
  using ModuleSchedulerAlgorithm::Run;
  absl::StatusOr<HloInstructionSequence> Run(
      HloComputation* computation,
      const HloAliasAnalysis& alias_analysis) const override;
};

// DFS-order scheduler with a heuristic to decide which operand to visit first.
class DFSMemoryScheduler : public ComputationSchedulerAlgorithm {
 public:
  DFSMemoryScheduler(const AliasInfo* alias_info,
                     BufferValue::SizeFunction size_function,
                     SchedulerPostprocessor postprocessor = {})
      : ComputationSchedulerAlgorithm(alias_info, std::move(size_function),
                                      std::move(postprocessor)) {}
  using ModuleSchedulerAlgorithm::Run;
  absl::StatusOr<HloInstructionSequence> Run(
      HloComputation* computation,
      const HloAliasAnalysis& alias_analysis) const override;
};

// BFS-order scheduler
//
// BFS-order scheduler is a simple scheduler that schedules instructions  in a
// breadth-first order, which maximizes the available concurrency at the cost of
// increased memory usage (HLO operations that do not have buffer conflicts can
// be executed in parallel).
//
// This is the most trivial scheduling optimized for maximum concurrency. In
// practice it is only useful for CPU backend where memory is cheap and we have
// a lot of available compute cores, and cheap concurrency primitives.
class BFScheduler : public ComputationSchedulerAlgorithm {
 public:
  BFScheduler(const AliasInfo* alias_info,
              BufferValue::SizeFunction size_function,
              SchedulerPostprocessor postprocessor = {})
      : ComputationSchedulerAlgorithm(alias_info, std::move(size_function),
                                      std::move(postprocessor)) {}
  absl::StatusOr<HloInstructionSequence> Run(
      HloComputation* computation,
      const HloAliasAnalysis& alias_analysis) const override;
};

// Naive Post Order scheduler
class PostOrderScheduler : public ComputationSchedulerAlgorithm {
 public:
  explicit PostOrderScheduler(const AliasInfo* alias_info,
                              BufferValue::SizeFunction size_function,
                              SchedulerPostprocessor postprocessor = {})
      : ComputationSchedulerAlgorithm(alias_info, std::move(size_function),
                                      std::move(postprocessor)) {}
  using ModuleSchedulerAlgorithm::Run;
  absl::StatusOr<HloInstructionSequence> Run(
      HloComputation* computation,
      const HloAliasAnalysis& alias_analysis) const override;
};

// The default scheduling algorithm. Runs the list scheduler, the DFS scheduler,
// and the post-order scheduler and chooses whichever returns a lower min-
// memory, not accounting for fragmentation. peak_memory (may be nullptr) is set
// to the peak memory of the resulting schedule according to the HeapSimulator.
class DefaultMemoryScheduler : public ModuleSchedulerAlgorithm {
 public:
  DefaultMemoryScheduler(const AliasInfo* alias_info,
                         const BufferValue::SizeFunction& size_function,
                         const SchedulerPostprocessor& postprocessor = {})
      : ModuleSchedulerAlgorithm(alias_info),
        list_scheduler_(alias_info, size_function, postprocessor),
        dfs_scheduler_(alias_info, size_function, postprocessor),
        post_order_scheduler_(alias_info, size_function, postprocessor) {}
  absl::StatusOr<HloSchedule> Run(
      const HloModule* module, const HloAliasAnalysis& alias_analysis,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      int64_t* peak_memory) const override;

 private:
  ListMemoryScheduler list_scheduler_;
  DFSMemoryScheduler dfs_scheduler_;
  PostOrderScheduler post_order_scheduler_;
};

// Returns an HloSchedule which seeks to minimize the memory required for the
// module. size_function is the function returning the number of bytes required
// for a LogicalBuffer. peak_memory (if not nullptr) is set to the largest peak
// memory (according to the HeapSimulator) of all computations in the module.
absl::StatusOr<HloSchedule> ScheduleModule(
    const HloModule* module, const ModuleSchedulerAlgorithm& algorithm,
    const absl::flat_hash_set<absl::string_view>& execution_threads = {},
    int64_t* peak_memory = nullptr);
// Schedule the module using the DefaultMemoryScheduler algorithm.
absl::StatusOr<HloSchedule> ScheduleModule(
    const HloModule* module, const AliasInfo* alias_info,
    const BufferValue::SizeFunction& size_function,
    const absl::flat_hash_set<absl::string_view>& execution_threads = {},
    int64_t* peak_memory = nullptr);

// A pass which schedules the HLO instructions in a module. The HloModule's
// schedule field is set to the resulting HloSchedule using
// HloModule::set_schedule.
class HloMemoryScheduler : public HloModulePass {
 public:
  // algorithm is the memory scheduling algorithm to use. If not specified, then
  // DefaultMemoryScheduler is used.
  explicit HloMemoryScheduler(
      std::unique_ptr<ModuleSchedulerAlgorithm> algorithm)
      : algorithm_(std::move(algorithm)) {}
  HloMemoryScheduler(const AliasInfo* alias_info,
                     const BufferValue::SizeFunction& size_function)
      : algorithm_(std::make_unique<DefaultMemoryScheduler>(alias_info,
                                                            size_function)) {}

  absl::string_view name() const override { return "hlo-memory-scheduler"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  std::unique_ptr<ModuleSchedulerAlgorithm> algorithm_;
};

// A pass which produces a naive, but correct schedule. The schedule is produced
// using a DFS traversal of the graph with no attempt to minimize memory use.
class HloTrivialScheduler : public HloModulePass {
 public:
  HloTrivialScheduler() = default;
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
  absl::string_view name() const override { return "hlo-descheduler"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_MEMORY_SCHEDULER_H_
