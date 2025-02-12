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

#ifndef XLA_SERVICE_SCATTER_DETERMINISM_EXPANDER_H_
#define XLA_SERVICE_SCATTER_DETERMINISM_EXPANDER_H_

#include "xla/hlo/transforms/expanders/op_expander_pass.h"

namespace xla {

// This pass rewrites scatter operations into a prefix-scan based algorithm that
// ensures the scatter results to be determininstic. Note that the computation
// after the expansion still contains a scatter operation, but it does not have
// duplicated indices and hence the results are guaranteed to be deterministic.
class ScatterDeterminismExpander : public OpExpanderPass {
 public:
  explicit ScatterDeterminismExpander() = default;

  absl::string_view name() const override {
    return "scatter_determinism_expander";
  }

 protected:
  bool InstructionMatchesPattern(HloInstruction* inst) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* inst) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_SCATTER_DETERMINISM_EXPANDER_H_
