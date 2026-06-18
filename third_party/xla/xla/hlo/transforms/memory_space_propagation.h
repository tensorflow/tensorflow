/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_MEMORY_SPACE_PROPAGATION_H_
#define XLA_HLO_TRANSFORMS_MEMORY_SPACE_PROPAGATION_H_

#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/shape.h"

namespace xla {

// This is a legalization pass that propagates the memory space (and associated
// split config) in the layout to the fusion computations.
class MemorySpacePropagation : public HloModulePass {
 public:
  explicit MemorySpacePropagation(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis = nullptr)
      : dataflow_analysis_(std::move(dataflow_analysis)) {}
  ~MemorySpacePropagation() override = default;
  absl::string_view name() const override { return "memory-space-propagation"; }

  // Propagates the memory space (and associated split config) in the layout to
  // a given fusion computation. Returns true if the computation is modified.
  bool RunOnComputation(HloComputation* computation);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Given the shape index (operand or output) and its corresponding instruction
  // in the fused computation (parameter or root), propagates the memory space
  // (and associated split config) in the callee side. Returns true if the
  // module is modified.
  bool Propagate(ShapeIndexView index, const HloInstruction* callee_instruction,
                 const Shape& src_shape,
                 absl::flat_hash_set<const HloValue*>& visited) const;

  std::unique_ptr<HloDataflowAnalysis> dataflow_analysis_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_MEMORY_SPACE_PROPAGATION_H_
