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

#ifndef XLA_SERVICE_ALL_GATHER_DECOMPOSER_H_
#define XLA_SERVICE_ALL_GATHER_DECOMPOSER_H_

#include <cstdint>
#include <functional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/shape.h"

namespace xla {

// AllGatherDecomposer is a pass which converts unsupported all-gathers into
// dynamic-update-slices and all-reduces.
class AllGatherDecomposer : public HloModulePass {
 public:
  explicit AllGatherDecomposer(
      std::function<bool(const HloAllGatherInstruction&)> should_decompose)
      : should_decompose_(std::move(should_decompose)) {}
  AllGatherDecomposer()
      : should_decompose_(
            [](const HloAllGatherInstruction& ag) { return true; }) {}
  absl::string_view name() const override { return "all_gather_decomposer"; }

  // Run AllGatherDecomposer pass on computations in 'module'.
  // Returns whether the 'module' was changed.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 protected:
  virtual HloInstruction* TranslateAllGatherToAllReducePerOperand(
      CollectiveOpGroupMode group_mode, const HloAllGatherInstruction& ag,
      const Shape& output_shape, HloInstruction* operand, HloComputation* comp,
      int64_t ag_dim);

  virtual bool ShouldDecompose(const HloAllGatherInstruction& ag) const {
    return should_decompose_(ag);
  }

  absl::Status DecomposeAllGather(HloAllGatherInstruction* ag,
                                  HloComputation* comp);

 private:
  std::function<bool(const HloAllGatherInstruction&)> should_decompose_;
};

}  // namespace xla

#endif  // XLA_SERVICE_ALL_GATHER_DECOMPOSER_H_
