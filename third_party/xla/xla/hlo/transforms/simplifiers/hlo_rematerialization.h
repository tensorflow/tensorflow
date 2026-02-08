/* Copyright 2017 The OpenXLA Authors.

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
#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/tuple_points_to_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_data_structures.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_memory_tracker.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_options.h"
#include "xla/service/call_graph.h"

namespace xla {

using RematAlgorithmFunction = std::function<absl::StatusOr<bool>(
    HloComputation* computation, HloSchedule* schedule,
    int64_t memory_limit_bytes, int64_t min_remat_size,
    const absl::flat_hash_set<absl::string_view>& execution_threads)>;

// HLO pass which rematerializes instructions to reduce peak memory use, where
// memory use is defined as the total size of all live HLO instruction
// values. Parameters and constants are included in memory use estimates.
//
// CSE will undo the effects of this optimization and should not be run after
// this pass. In general, this pass should be run very late, immediately before
// code generation.
class HloRematerialization : public HloPassInterface {
 public:
  // The minimum cost estimate memory limit in bytes for a computation to be
  // considered for rematerialization. Only in use for peak priority
  // rematerialization.
  constexpr static int64_t kMinimumCostEstimateMemoryLimitBytes =
      1073741824;  // 1 GiB

  // Helper struct that communicates the before / after sizes for the
  // rematerialization process.
  struct RematerializationSizes {
    int64_t before_bytes = -1;
    int64_t after_bytes = -1;
  };

  explicit HloRematerialization(
      HloRematerializationOptions options, RematerializationSizes& sizes,
      absl::AnyInvocable<absl::Status(HloInstruction*, HloInstruction*)>
          on_rematerialized = nullptr)
      : options_(std::move(options)),
        sizes_(sizes),
        on_rematerialized_(std::move(on_rematerialized)) {}

  ~HloRematerialization() override = default;

  absl::string_view name() const override { return "rematerialization"; }

  // Get the next available channel id and increment count.
  int64_t NextChannelId() { return next_channel_id_++; }

  // Performs the appropriate peak memory lookup on either
  // computation_peak_memory_ or computation_peak_memory_tracker_ depending on
  // the remat algorithm.
  absl::StatusOr<int64_t> GetComputationPeakMemory(
      const HloComputation* computation);

  HloRematerializationOptions::RematAlgorithm remat_algorithm() const {
    return options_.remat_algorithm;
  }

  void UpdateMaxRematerializedBlockSize(int new_rematerialized_block_size) {
    max_rematerialized_block_size_ =
        std::max(max_rematerialized_block_size_, new_rematerialized_block_size);
  }

  int64_t GetBlockSizeLimit() const { return options_.block_size_limit; }

  // Holds references to data structures and some constants that are used during
  // rematerialization. This struct is used to avoid long function signatures.
  struct RematerializationStateData {
    HloRematInstructionList* instruction_list;
    HloComputation* computation;
    HloSchedule* schedule;
    const int64_t memory_limit_bytes;
    const int64_t cost_estimate_memory_limit_bytes;
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map;
    absl::flat_hash_set<const HloInstruction*>* remat_move_instructions;
    const absl::flat_hash_set<absl::string_view>* execution_threads;
  };

  // Holds the result of a single rematerialization step. The module_changed
  // field indicates whether the module was changed by the rematerialization
  // step. The net_instructions_added field indicates the net number of
  // instructions of any kind added to the module by the rematerialization step.
  // The remat_instructions_count field indicates the number of instructions
  // that were rematerialized in the rematerialization step.
  struct RematerializationStepResult {
    bool module_changed;
    int64_t net_instructions_added;
    int64_t remat_instructions_count;
  };

  enum class RematSubpassStatus : char {
    kUnchanged,
    kChangedButOverMemoryLimit,
    kChangedAndUnderMemoryLimit,
  };

  struct RematSubpassResult {
    RematSubpassStatus status = RematSubpassStatus::kUnchanged;
    int64_t peak_memory_during_remat = 0;
    const HloInstruction* peak_memory_instruction = nullptr;
  };

  absl::Status on_rematerialized(HloInstruction* original,
                                 HloInstruction* remat) {
    if (on_rematerialized_ != nullptr) {
      return on_rematerialized_(original, remat);
    }
    return absl::OkStatus();
  }

 protected:
  // Updates the schedule to mirror the provided instruction sequence. This is
  // used to update the schedule after each rematerialization due to the memory
  // tracker requiring the schedule to be in sync with the instruction sequence
  // and computation.
  absl::Status UpdateScheduleFromSequence(
      HloComputation* computation, HloSchedule* schedule,
      const HloInstructionSequence& sequence,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Cleans up dead rematerialized instructions out of the module. Basically
  // runs DCE and updates the schedule.
  static absl::StatusOr<bool> CleanupRematerializedInstructions(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Updates the peak memory and instruction variables based on the new
  // instruction list and schedule, this includes updating the schedule to
  // reflect the new instructions, updating the instruction list to reflect
  // the new schedule, and computing the new peak memory and instruction.
  absl::StatusOr<MemoryUsageAndInstruction> PeakPriorityUpdateVariables(
      const HloRematInstructionList& instruction_list,
      HloComputation* computation, HloSchedule* schedule,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Rematerializes instructions within the given computation. 'schedule'
  // constrains the order in which the computation's instructions will be
  // emitted in the backend. Rematerialized instructions will be added to the
  // HLO computation and inserted into 'schedule'.
  virtual absl::StatusOr<bool> RematerializeComputation(
      HloComputation* computation, HloSchedule* schedule,
      int64_t memory_limit_bytes, int64_t min_remat_size,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Calls the peak priority rematerialization
  // algorithm recursively on the callee computations and propagates the
  // information about whether the module was changed.
  absl::StatusOr<bool> RematerializeCalledComputationsPeakPriority(
      const CallSite* callsite, int64_t memory_tracker_memory_usage,
      HloSchedule* schedule, int64_t memory_limit_bytes, int64_t min_remat_size,
      int64_t cost_estimate_memory_limit_bytes,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Alternative rematerialization algorithm that prioritizes rematerializing
  // the highest peaks first. Can offer better memory utilization than the
  // default algorithm but is usually slower.
  virtual absl::StatusOr<bool> RematerializeComputationPeakPriority(
      HloComputation* computation, HloSchedule* schedule,
      int64_t memory_limit_bytes, int64_t min_remat_size,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Runs a single sub-pass of the peak priority rematerialization algorithm.
  // Returns whether the module was changed and whether the rematerialization
  // should be stopped.
  absl::StatusOr<RematSubpassResult> PeakPrioritySubPass(
      HloRematerialization::RematerializationStateData& state,
      HloComputation* computation, const CallGraphNode& call_graph_node,
      int64_t min_remat_size, int64_t peak_memory_during_remat,
      int64_t memory_limit_bytes,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Returns the rematerialization algorithm function corresponding to the given
  // rematerialization algorithm enum.
  virtual absl::StatusOr<RematAlgorithmFunction> GetRematAlgorithmFunction(
      HloRematerializationOptions::RematAlgorithm remat_algorithm);

  // Computes and returns the peak memory used by the given computation. The
  // peak memory is the maximum total size of all live HLO instruction values at
  // any program point. 'order' is the order in which the HLO instructions will
  // be emitted which is used to determine lifespans of HLO values.
  absl::StatusOr<int64_t> ComputePeakMemory(
      const HloComputation* computation, const HloInstructionSequence& order,
      const absl::flat_hash_set<absl::string_view>& execution_threads) const;

  // Computes and returns the peak memory used by the given computation and the
  // instruction live at that point in the program.
  absl::StatusOr<MemoryUsageAndInstruction> ComputePeakMemoryAndInstruction(
      const HloComputation* computation, const HloInstructionSequence& order,
      const absl::flat_hash_set<absl::string_view>& execution_threads) const;

  // Returns the peak memory usage of the called computations for the given
  // instruction. Zero is returned if the instruction calls no computations.
  absl::StatusOr<int64_t> CalledComputationsMemoryUsage(
      const HloInstruction* instruction,
      const absl::flat_hash_set<absl::string_view>& execution_threads) const;

  const HloRematerializationOptions options_;

  // Reference to data structure which records the peak memory usage of the HLO
  // module before/after rematerialization.
  RematerializationSizes& sizes_;

  // Call graph of the hlo_module.
  std::unique_ptr<CallGraph> call_graph_;

  // When the kAlways remat algorithm is used, we directly maintain the peak
  // memory usage of each computation. This map contains only those computations
  // called from sequential context (CallContext::kSequential). These values
  // are updated as rematerialization occurs by constructing new
  // HloRematerializationSweepMemoryTracker's.
  absl::flat_hash_map<const HloComputation*, int64_t> computation_peak_memory_;

  // When the kPeakPriority remat algorithm is used, we indirectly maintain the
  // peak memory usage of each computation by keeping instruction lists and
  // memory trackers. Again, this map contains only those computations called
  // from sequential context (CallContext::kSequential). These values are kept
  // updated via calls to PeakPriorityUpdateVariables() and
  // HloRematerializationPeakMemoryTracker::CalleeComputationWasUpdated() as
  // rematerialization occurs.
  absl::flat_hash_map<const HloComputation*,
                      std::unique_ptr<HloRematInstructionList>>
      computation_instruction_list_;
  absl::flat_hash_map<const HloComputation*,
                      std::unique_ptr<HloRematerializationPeakMemoryTracker>>
      computation_peak_memory_tracker_;

  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;

  // Set of computations which have had rematerialization
  // applied. Rematerialization is only applied once per computation.
  absl::flat_hash_set<const HloComputation*> rematerialized_computations_;

  // Count of the total instructions rematerialized.
  int64_t instructions_rematerialized_ = 0;

  // Count of the net instructions added to the HLO module by
  // rematerialization. This can be different than instructions_rematerialized_
  // because some rematerializations are effectively moves in the HLO
  // schedule. In these cases, the rematerialization instruction replaces all
  // uses of the original instruction and the original instruction is
  // dead. Hence, no net instructions were added.
  int64_t net_instructions_added_ = 0;

  // Size of the largest block that has been rematerialized. This is actually an
  // upper bound (within a factor of 2) on the block size.
  int max_rematerialized_block_size_ = 0;

  // Tracking available channel id numbers to use to apply to rematerialized
  // channel instructions
  int64_t next_channel_id_;

  // Callback set by the user to be called when an instruction is
  // rematerialized.
  absl::AnyInvocable<absl::Status(HloInstruction*, HloInstruction*)>
      on_rematerialized_;

  // Runs rematerialization on the given module. Returns whether the module was
  // changed. Requires that the module has a schedule set
  // (HloModule::has_schedule() is true) before running. Returns whether any
  // instructions were rematerialized. If memory use is already below the limit
  // specified in the constructor then no instructions are rematerialized and
  // false is returned.
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_H_
