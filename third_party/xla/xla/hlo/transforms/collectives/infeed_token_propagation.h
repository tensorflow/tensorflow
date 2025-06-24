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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_INFEED_TOKEN_PROPAGATION_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_INFEED_TOKEN_PROPAGATION_H_

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/call_graph.h"

namespace xla {
// Finds dangling infeed/outfeed tokens inside nested computations and bubbles
// them up through callers until they reach the entry computation. This is
// needed to prepare these computations to be inlined, otherwise the previous
// computation boundaries won't be there to stop infeeds/outfeeds from being
// reordered during scheduling.
//
// This pass assumes the HLO graph is flattened.
class InfeedTokenPropagation : public HloModulePass {
 public:
  absl::string_view name() const override { return "infeed-token-propagation"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::Status PropagateToken(const HloOrdering& ordering);
  absl::Status PropagateTokenThroughWhileBody();
  absl::Status PropagateTokenThroughConditionalBranch();

  std::unique_ptr<CallGraph> call_graph_;

  HloInstruction* dangling_instruction_ = nullptr;
  HloOpcode original_opcode_;
  HloInstruction* input_token_ = nullptr;
  HloInstruction* output_token_ = nullptr;
};
}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_INFEED_TOKEN_PROPAGATION_H_
