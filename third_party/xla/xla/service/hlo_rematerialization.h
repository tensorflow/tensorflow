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
#ifndef XLA_SERVICE_HLO_REMATERIALIZATION_H_
#define XLA_SERVICE_HLO_REMATERIALIZATION_H_

#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/tuple_points_to_analysis.h"
#include "xla/shape.h"

namespace xla {

// HLO pass which rematerializes instructions to reduce peak memory use, where
// memory use is defined as the total size of all live HLO instruction
// values. Parameters and constants are included in memory use estimates.
//
// CSE will undo the effects of this optimization and should not be run after
// this pass. In general, this pass should be run very late, immediately before
// code generation.
class HloRematerialization : public HloModulePass {
 public:
  using ShapeSizeFunction = std::function<int64_t(const Shape&)>;

  using CompactShapeFunction =
      std::function<absl::StatusOr<Shape>(const Shape&)>;

  // Helper struct that communicates the before / after sizes for the
  // rematerialization process.
  struct RematerializationSizes {
    int64_t before_bytes = -1;
    int64_t after_bytes = -1;
  };

  // Mode in which the rematerialization algorithm should be run.
  struct RematerializationModeConfig {
    RematerializationModeConfig(bool recompute, bool compress,
                                bool host_offload)
        : recompute(recompute),
          compress(compress),
          host_offload(host_offload) {}
    bool recompute;     // Enables the kCompress RematStrategy.
    bool compress;      // Enables the kRecompute RematStrategy.
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
    explicit Options(HloCostAnalysis& hlo_cost_analysis,
                     const RematerializationModeConfig& remat_mode_config,
                     int64_t memory_limit_bytes, int block_size_limit,
                     int block_rematerialization_factor, int64_t min_remat_size,
                     CompactShapeFunction compact_shape_function,
                     std::optional<HostMemoryOffloadConfig>
                         host_memory_offload_config = std::nullopt,
                     absl::flat_hash_map<HloComputation*, int64_t>
                         async_computation_parallelism = {})
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
          async_computation_parallelism(async_computation_parallelism) {}

    // The cost model used for decisions during rematerialization for host
    // memory offload. It is also used for getting Shape size.
    HloCostAnalysis& hlo_cost_analysis;

    // Holds the rematerialization strategy configuration to be used by the
    // pass.
    RematerializationModeConfig remat_mode_config;

    // Function which computes the size of the top-level buffer of a shape.
    const ShapeSizeFunction size_function;

    // The threshold number of bytes to reduce memory use to via
    // rematerialization. Size of aliased outputs should be subtracted
    // from this.
    int64_t memory_limit_bytes;

    // Maximum number of consecutive instructions to consider for
    // rematerialization.
    int block_size_limit;

    // Controls the amount of effort spent trying to find large blocks for
    // rematerialization. Larger values leads to longer compilation times in
    // return for potentially reduced memory consumption.
    int block_rematerialization_factor;

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
  };

  explicit HloRematerialization(Options options, RematerializationSizes& sizes)
      : options_(std::move(options)), sizes_(sizes) {}

  ~HloRematerialization() override = default;

  absl::string_view name() const override { return "rematerialization"; }

  // Get the next available channel id and increment count.
  int64_t NextChannelId() { return next_channel_id_++; }

  // Get the peak memory for the computation.
  int64_t ComputationPeakMemory(const HloComputation* computation) const {
    return computation_peak_memory_.at(computation);
  }

  // Runs rematerialization on the given module. Returns whether the module was
  // changed. Requires that the module has a schedule set
  // (HloModule::has_schedule() is true) before running. Returns whether any
  // instructions were rematerialized. If memory use is already below the limit
  // specified in the constructor then no instructions are rematerialized and
  // false is returned.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 protected:
  // Rematerializes instructions within the given computation. 'schedule'
  // constains the order in which the computation's instructions will be emitted
  // in the backend. Rematerialized instructions will be added to the HLO
  // computation and inserted into 'schedule'.
  virtual absl::StatusOr<bool> RematerializeComputation(
      HloComputation* computation, HloSchedule* schedule,
      int64_t memory_limit_bytes, int64_t min_remat_size,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Computes and returns the peak memory used by the given computation. The
  // peak memory is the maximum total size of all live HLO instruction values at
  // any program point. 'order' is the order in which the HLO instructions will
  // be emitted which is used to determine lifespans of HLO values.
  absl::StatusOr<int64_t> ComputePeakMemory(
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
  // computations called from sequential context
  // (CallContext::kSequential). These values are updated as rematerialization
  // occurs.
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
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_REMATERIALIZATION_H_
