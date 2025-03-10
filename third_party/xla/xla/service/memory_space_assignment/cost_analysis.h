/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_COST_ANALYSIS_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_COST_ANALYSIS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/call_graph.h"
#include "xla/service/cost_modelling/op_cost.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace memory_space_assignment {

// Options to be passed to the CostAnalysis.
struct CostAnalysisOptions {
  // This variable is used by the cost analysis in estimating how many times
  // each while loop will execute. Nested loops will be assumed to have
  // executed pow(while_execution_count, nesting_level) times.
  uint64_t xla_tpu_memory_space_assignment_while_execution_count = 5ULL;

  // This variable is used to scale the alternate memory benefit factor for
  // large buffers. The default scaling function is sqrt.
  std::string
      xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers =
          "SQRT";

  // The window size used to calculate the pipeline overhead when HLO accesses
  // the default memory, in MiB.
  float pipeline_overhead_window_size_mib = 0;

  double alternate_mem_read_bandwidth_bytes_per_second = 0.0f;
  double alternate_mem_write_bandwidth_bytes_per_second = 0.0f;

  double default_mem_bandwidth_bytes_per_second = 0.0f;

  // Scales effective bandwidth for async copies. Valid range is (0, 1].
  float async_copy_bandwidth_scaling_factor = 1.0;

  // Used to get the layout size of a shape in bytes.
  std::function<int64_t(const Shape&)> shape_size_bytes_fn =
      [](const Shape& shape) { return ShapeUtil::ByteSizeOf(shape); };
};

// A wrapper class around BaseCosts with additional knowledge about the
// bandwidths of different memory spaces.
class CostAnalysis {
 public:
  // An optional Cache object may be provided to some of the methods below to
  // speed up the lookup.
  struct Cache {
    // TODO(hanruobing): This map assumes the nested while loops have the same
    // hard-coded trip count. We plan to replace it with a more accurate
    // estimation provided by 'while_nest_trip_count'.
    absl::flat_hash_map<const HloInstruction*, float> while_nest_multiplier;
    absl::flat_hash_map<const HloComputation*, float> computation_trip_count;
    absl::flat_hash_map<HloPosition, float> memory_boundedness;
  };

  // Function type that can be used to indicate which input/output values are in
  // the alternate memory.
  using IsInAlternateMemoryFun = absl::FunctionRef<bool(
      std::optional<int> /*operand_num*/, const ShapeIndex& /*index*/,
      const Shape& /*shape*/)>;

  virtual ~CostAnalysis() = default;

  static absl::StatusOr<std::unique_ptr<CostAnalysis>> Create(
      OpCostManager& op_cost_manager, const CostAnalysisOptions& options,
      const HloModule& module);

  int64_t GetShapeSizeBytes(const Shape& shape) const;

  float OperandBytesAccessed(const HloInstruction& instruction,
                             int64_t operand_num,
                             const ShapeIndex& shape_index) const;

  float OutputBytesAccessed(const HloInstruction& instruction,
                            const ShapeIndex& shape_index) const;

  double DefaultMemBandwidthBytesPerSecond(
      bool use_scaling_factor = false) const;

  // Returns a heuristic value that captures how much putting this tensor to the
  // alternate memory would help if the op is memory bound, or otherwise how far
  // off is the op to memory boundedness. The larger this number, the higher
  // priority it will be placed in the alternate memory.
  float GetAlternateMemoryBenefit(const HloInstruction& instruction,
                                  float elapsed_time_due_to_alternate_mem,
                                  Cache* cache = nullptr) const;
  // Like above, return the benefit of putting the output tensor in the
  // alternate memory.
  float GetAlternateMemoryBenefit(const HloPosition& position,
                                  Cache* cache = nullptr) const;
  // Like above, return the benefit of putting the input tensor in the alternate
  // memory.
  float GetAlternateMemoryBenefit(const HloUse& use,
                                  Cache* cache = nullptr) const;

  // Returns a heuristic value of memory boundedness for the given
  // BufferInterval.  The larger this number, the higher priority it will be
  // placed in the alternate memory.
  float GetMemoryBoundedness(
      const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval,
      Cache* cache = nullptr) const;

  // If enabled in CostAnalysisOptions::pipeline_overhead_window_size_mib,
  // returns the overhead of accessing the default memory, in seconds. The
  // source of the overhead is the software pipelining ovehead. The lowering of
  // the operations typically use tiling to copy one window at a time from
  // default memory, and perform compute:
  //
  // Pipeline overhead:                          <->
  //                        +----+----+----+----+
  // Copy from default mem: |    |    |    |    |
  //                        +----+----+----+----+
  //                            \    \    \    \
  //                             \    \    \    \
  //                              V    V    V    V
  //                             +--+ +--+ +--+ +--+
  // Compute:                    |  | |  | |  | |  |
  //                             +--+ +--+ +--+ +--+
  float GetDefaultMemoryAccessOverhead(
      const HloInstruction& instruction,
      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_mem = {},
      absl::Span<const ShapeIndex> outputs_in_alternate_mem = {}) const;

  // Returns the amount of time the default memory bandwidth is idle, while
  // executing this instruction, in seconds.  This value can be multiplied with
  // the default memory bandwidth to get the amount of bytes that are available
  // to be copied to/from default memory during the execution of this
  // instruction.
  float GetDefaultMemoryBandwidthIdleTime(
      const HloInstruction& instruction,
      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_mem = {},
      absl::Span<const ShapeIndex> outputs_in_alternate_mem = {}) const;

  // Returns the bytes accessed from alternate memory.
  float GetBytesAccessedFromAlternateMemory(
      const HloInstruction& instruction,
      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_mem = {},
      absl::Span<const ShapeIndex> outputs_in_alternate_mem = {}) const;

  // Returns the elapsed time in seconds due to compute only.
  float GetInstructionElapsedDueToCompute(
      const HloInstruction& instruction) const;

  // Returns the elapsed time in seconds due to memory only. If
  // operands_in_alternate_mem or outputs_in_alternate_mem is provided, it will
  // assume that the corresponding operands or output will be in the alternate
  // memory space. This is useful for calculating the benefit of placing the
  // buffer in alternate memory.
  float GetInstructionElapsedDueToMemory(
      const HloInstruction& instruction,
      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_mem = {},
      absl::Span<const ShapeIndex> outputs_in_alternate_mem = {}) const;

  // Like above, only the inputs/outputs indicated by is_in_alternate_mem are in
  // the alternate memory.
  float GetInstructionElapsedDueToMemory(
      const HloInstruction& instruction,
      IsInAlternateMemoryFun is_in_alternate_mem) const;

  // Returns the estimated elapsed duration of the instruction in seconds.  It
  // assumes all operands and outputs of the instruction are in the default
  // memory.
  virtual float GetInstructionElapsed(const HloInstruction& instruction) const;

  // Returns the estimated elapsed duration of the instruction in seconds.  It
  // assumes all operands and outputs of the instruction are in the default
  // memory, except for the operands and outputs specified to be in the
  // alternate memory.
  virtual float GetInstructionElapsedInAlternateMemory(
      const HloInstruction& instruction,
      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_mem,
      absl::Span<const ShapeIndex> outputs_in_alternate_mem) const;

  // Like above, only the inputs/outputs indicated by is_in_alternate_mem are in
  // the alternate memory.
  float GetInstructionElapsedInAlternateMemory(
      const HloInstruction& instruction,
      IsInAlternateMemoryFun is_in_alternate_mem) const;

  // Returns the elapsed time it would take to asynchronously copy the shape
  // from default to alternate memory space (or vice versa).
  virtual float GetAsyncCopyElapsed(const Shape& shape) const;

  int64_t GetScheduleEndTime() const;

  // Returns the number of nested computation levels this instruction resides
  // in. If while_only is true, it returns the while loop nest level and 0
  // means the instruction is not in a while loop.
  int CalculateComputationNestLevel(const HloInstruction* instruction,
                                    bool while_only) const;

  // Returns the number of times the instruction will be executed.
  // For instructions in nested loops, this is the product of the number of
  // trip counts of outer loops.
  float CalculateNestTripCount(const HloInstruction* instruction,
                               Cache* cache = nullptr) const;

  float GetWhileNestMultiplier(int while_nest_level) const;

  const HloLiveRange& hlo_live_range() const { return *hlo_live_range_; }

 protected:
  CostAnalysis(OpCostManager& op_cost_manager,
               const CostAnalysisOptions& options,
               std::unique_ptr<HloAliasAnalysis> alias_analysis,
               std::unique_ptr<HloLiveRange> hlo_live_range,
               std::unique_ptr<CallGraph> call_graph)
      : op_cost_manager_(op_cost_manager),
        options_(options),
        alias_analysis_(std::move(alias_analysis)),
        hlo_live_range_(std::move(hlo_live_range)),
        call_graph_(std::move(call_graph)) {}

 private:
  // A manager responsible for return basic cost metrics.
  OpCostManager& op_cost_manager_;
  const CostAnalysisOptions options_;
  std::unique_ptr<HloAliasAnalysis> alias_analysis_;
  std::unique_ptr<HloLiveRange> hlo_live_range_;
  std::unique_ptr<CallGraph> call_graph_;
};

}  // namespace memory_space_assignment
}  // namespace xla
#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_COST_ANALYSIS_H_
