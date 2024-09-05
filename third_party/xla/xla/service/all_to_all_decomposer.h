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

#ifndef XLA_SERVICE_ALL_TO_ALL_DECOMPOSER_H_
#define XLA_SERVICE_ALL_TO_ALL_DECOMPOSER_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/op_expander_pass.h"

namespace xla {

// AllToAllDecomposer is a pass which converts unsupported array all_to_all
// into tuple all_to_all or array all_to_all with a minimum rank where the split
// dimension is the size of the replica_groups.
class AllToAllDecomposer : public OpExpanderPass {
 public:
  explicit AllToAllDecomposer(bool decompose_to_tuple = true,
                              int64_t min_array_rank = 0)
      : decompose_to_tuple_(decompose_to_tuple),
        min_array_rank_(min_array_rank) {}
  ~AllToAllDecomposer() override = default;
  absl::string_view name() const override { return "all_to_all_decomposer"; }

 private:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;
  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
  bool decompose_to_tuple_;
  int64_t min_array_rank_;
};

}  // namespace xla

#endif  // XLA_SERVICE_ALL_TO_ALL_DECOMPOSER_H_
