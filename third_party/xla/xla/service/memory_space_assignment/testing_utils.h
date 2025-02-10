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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_TESTING_UTILS_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_TESTING_UTILS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace memory_space_assignment {

// For testing purposes, we define a cost analysis where we can control the
// elapsed times of each HLO and asynchronous copy.
class FakeCostAnalysis : public CostAnalysis {
 public:
  static absl::StatusOr<std::unique_ptr<FakeCostAnalysis>> Create(
      HloCostAnalysisCosts& cost_analysis_costs, const HloModule& module,
      const CostAnalysisOptions& options) {
    TF_ASSIGN_OR_RETURN(auto alias_analysis, HloAliasAnalysis::Run(&module));
    TF_ASSIGN_OR_RETURN(auto hlo_live_range,
                        HloLiveRange::Run(module.schedule(), *alias_analysis,
                                          module.entry_computation()));
    auto call_graph = CallGraph::Build(&module);
    return absl::WrapUnique(new FakeCostAnalysis(
        cost_analysis_costs, options, std::move(alias_analysis),
        std::move(hlo_live_range), std::move(call_graph)));
  }

  float GetInstructionElapsed(
      const HloInstruction& instruction) const override {
    if (get_instruction_elapsed_override_) {
      return get_instruction_elapsed_override_(instruction);
    }
    return 1.0;
  }

  float GetInstructionElapsedInAlternateMemory(
      const HloInstruction& instruction,
      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_mem,
      absl::Span<const ShapeIndex> outputs_in_alternate_mem) const override {
    if (get_instruction_elapsed_in_alternate_memory_override_) {
      return get_instruction_elapsed_in_alternate_memory_override_(
          instruction, operands_in_alternate_mem, outputs_in_alternate_mem);
    }
    if (!operands_in_alternate_mem.empty()) {
      return 0.5;
    } else {
      return 1.0;
    }
  }

  float GetAsyncCopyElapsed(const Shape& shape) const override {
    if (get_async_copy_elapsed_override_) {
      return get_async_copy_elapsed_override_(shape);
    }
    return 3.0;
  }

  // The following methods can be used to override what the above API calls
  // return.
  void SetOverrideForGetInstructionElapsed(
      std::function<float(const HloInstruction&)> function) {
    get_instruction_elapsed_override_ = function;
  }
  void SetOverrideForGetInstructionElapsedInAlternateMemory(
      std::function<float(const HloInstruction&,
                          absl::Span<const std::pair<int64_t, ShapeIndex>>,
                          absl::Span<const ShapeIndex>)>
          function) {
    get_instruction_elapsed_in_alternate_memory_override_ = function;
  }
  void SetOverrideForGetAsyncCopyElapsed(
      std::function<float(const Shape&)> function) {
    get_async_copy_elapsed_override_ = function;
  }

 protected:
  FakeCostAnalysis(HloCostAnalysisCosts& cost_analysis_costs,
                   const CostAnalysisOptions& options,
                   std::unique_ptr<HloAliasAnalysis> alias_analysis,
                   std::unique_ptr<HloLiveRange> hlo_live_range,
                   std::unique_ptr<CallGraph> call_graph)
      : CostAnalysis(cost_analysis_costs, options, std::move(alias_analysis),
                     std::move(hlo_live_range), std::move(call_graph)) {}

 private:
  std::function<float(const HloInstruction&)>
      get_instruction_elapsed_override_ = nullptr;
  std::function<float(const HloInstruction&,
                      absl::Span<const std::pair<int64_t, ShapeIndex>>,
                      absl::Span<const ShapeIndex>)>
      get_instruction_elapsed_in_alternate_memory_override_ = nullptr;
  std::function<float(const Shape&)> get_async_copy_elapsed_override_ = nullptr;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_TESTING_UTILS_H_
