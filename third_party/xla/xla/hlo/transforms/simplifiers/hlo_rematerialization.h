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
#include "absl/functional/any_invocable.h"
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
#include "xla/service/call_graph.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"

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
  using ShapeSizeFunction = std::function<int64_t(const Shape&)>;

  using CompactShapeFunction =
      std::function<absl::StatusOr<Shape>(const Shape&)>;

  // Helper struct that communicates the before / after sizes for the
  // rematerialization process.
  struct RematerializationSizes {
    int64_t before_bytes = -1;
    int64_t after_bytes = -1;
  };

  // The high-level rematerialization algorithm to use. Will significantly
  // affect the runtime performance of the pass and the memory utilization of
  // the HLO module.
  enum class RematAlgorithm {
    kAlwaysRemat,   // Rematerializes anything it can at any point the HLO is
                    // above the memory limit. Default rematerialization
                    // algorithm.
    kPeakPriority,  // Prioritize rematerializing the highest peak in the module
                    // at any given step. Much slower than the default algorithm
                    // but can offer better memory utilization in certain cases.
  };

  // Mode in which the rematerialization algorithm should be run.
  struct RematerializationModeConfig {
    RematerializationModeConfig(bool recompute, bool compress,
                                bool host_offload)
        : recompute(recompute),
          compress(compress),
          host_offload(host_offload) {}
    bool recompute;     // Enables the kRecompute RematStrategy.
    bool compress;      // Enables the kCompress RematStrategy.
    bool host_offload;  // Enables the kHostOffload RematStrategy.
  };

  // This is a struct containing configuration options that are specific to the
  // Host Memory Offload strategy.
  struct HostMemoryOffloadConfig {
    explicit HostMemoryOffloadConfig(int64_t host_memory_space,
                                     float bandwidth_to_host_bytes_per_second,
                                     float bandwidth_from_host_bytes_per_second)
        : host_memory_space(host_memory_space),
          bandwidth_to_host_bytes_per_second(
              bandwidth_to_host_bytes_per_second),
          bandwidth_from_host_bytes_per_second(
              bandwidth_from_host_bytes_per_second) {}

    // The host memory space, which is used during the host offload strategy.
    int64_t host_memory_space;

    float bandwidth_to_host_bytes_per_second;

    float bandwidth_from_host_bytes_per_second;
  };

  static Shape DefaultCompactShapeFunction(const Shape& shape) { return shape; }

  struct Options {
    explicit Options(
        HloCostAnalysis& hlo_cost_analysis,
        const RematerializationModeConfig& remat_mode_config,
        int64_t memory_limit_bytes, int block_size_limit,
        float block_rematerialization_factor, int64_t min_remat_size,
        CompactShapeFunction compact_shape_function,
        std::optional<HostMemoryOffloadConfig> host_memory_offload_config =
            std::nullopt,
        absl::flat_hash_map<HloComputation*, int64_t>
            async_computation_parallelism = {},
        RematAlgorithm remat_algorithm = RematAlgorithm::kAlwaysRemat)
        : hlo_cost_analysis(hlo_cost_analysis),
          remat_mode_config(remat_mode_config),
          memory_limit_bytes(memory_limit_bytes),
          block_size_limit(block_size_limit),
          block_rematerialization_factor(block_rematerialization_factor),
          min_remat_size(min_remat_size),
          compact_shape_function(compact_shape_function == nullptr
                                     ? DefaultCompactShapeFunction
                                     : std::move(compact_shape_function)),
          host_memory_offload_config(host_memory_offload_config),
          async_computation_parallelism(
              std::move(async_computation_parallelism)),
          remat_algorithm(remat_algorithm) {}

    // The cost model used for decisions during rematerialization for host
    // memory offload. It is also used for getting Shape size.
    HloCostAnalysis& hlo_cost_analysis;

    // Holds the rematerialization strategy configuration to be used by the
    // pass.
    RematerializationModeConfig remat_mode_config;

    // Function which computes the size of the top-level buffer of a shape.
    const ShapeSizeFunction size_function;

    // The threshold number of bytes to reduce memory use to via
    // rematerialization. This limit is adjusted in the pass by subtracting the
    // size of all module outputs. Callers should consider reducing the amount
    // of available memory by also subtracting the size of module parameters,
    // and to add the size of aliased outputs to avoid subtracting twice for
    // parameter and output.
    int64_t memory_limit_bytes;

    // Maximum number of consecutive instructions to consider for
    // rematerialization.
    int block_size_limit;

    // Controls the amount of effort spent trying to find large blocks for
    // rematerialization. Larger values leads to longer compilation times in
    // return for potentially reduced memory consumption.
    float block_rematerialization_factor;

    // The minimum size, in bytes, of a tensor to be considered for
    // rematerialization. All tensors smaller than this size will be skipped
    // over.
    int64_t min_remat_size;

    // Converts a shape into compact form, returns the same shape if a shape is
    // already considered compact.
    CompactShapeFunction compact_shape_function;

    std::optional<HostMemoryOffloadConfig> host_memory_offload_config;

    // Collection of async entry computations and their number of parallel
    // invocations.
    absl::flat_hash_map<HloComputation*, int64_t> async_computation_parallelism;

    // The high-level rematerialization algorithm to be used.
    RematAlgorithm remat_algorithm;
  };

  explicit HloRematerialization(
      Options options, RematerializationSizes& sizes,
      absl::AnyInvocable<absl::Status(HloInstruction*, HloInstruction*)>
          on_rematerialized = nullptr)
      : options_(std::move(options)),
        sizes_(sizes),
        on_rematerialized_(std::move(on_rematerialized)) {}

  ~HloRematerialization() override = default;

  absl::string_view name() const override { return "rematerialization"; }

  // Get the next available channel id and increment count.
  int64_t NextChannelId() { return next_channel_id_++; }

  // Get the peak memory for the computation.
  int64_t ComputationPeakMemory(const HloComputation* computation) const {
    return computation_peak_memory_.at(computation);
  }

  HloRematerialization::RematAlgorithm remat_algorithm() const {
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

  // Holds the memory usage and instruction at a given program point (usually
  // the peak memory).
  struct MemoryUsageAndInstruction {
    int64_t memory_usage;
    const HloInstruction* instruction;
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
      HloRematInstructionList& instruction_list, HloComputation* computation,
      HloSchedule* schedule,
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
      const HloInstruction* peak_memory_instruction,
      HloRematerialization::RematerializationStateData& state,
      HloComputation* computation, const CallGraphNode& call_graph_node,
      int64_t min_remat_size, int64_t peak_memory_during_remat,
      int64_t memory_limit_bytes,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Returns the rematerialization algorithm function corresponding to the given
  // rematerialization algorithm enum.
  virtual absl::StatusOr<RematAlgorithmFunction> GetRematAlgorithmFunction(
      RematAlgorithm remat_algorithm);

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

  const Options options_;

  // Reference to data structure which records the peak memory usage of the HLO
  // module before/after rematerialization.
  RematerializationSizes& sizes_;

  // Call graph of the hlo_module.
  std::unique_ptr<CallGraph> call_graph_;

  // The peak memory usage of each computation. The map contains only those
  // computations called from sequential context (CallContext::kSequential).
  // These values are updated as rematerialization occurs.
  absl::flat_hash_map<const HloComputation*, int64_t> computation_peak_memory_;

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
