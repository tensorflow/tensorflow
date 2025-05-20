/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/hlo_domain_remover.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_domain_metadata.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/service/hlo_domain_verifier.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"

namespace xla {

class HloDomainRemover::RunContext {
 public:
  RunContext(HloModule* module, HloDomainRemover* remover)
      : module_(module), remover_(remover) {}

  absl::StatusOr<bool> Run(
      const absl::flat_hash_set<absl::string_view>& execution_threads);

 private:
  // Verifies the consistency of the domain, and normalizes the instructions
  // within it.
  absl::Status VerifyAndNormalizeDomain(const DomainMetadata::Domain& domain);

  HloModule* module_;
  HloDomainRemover* remover_;
};

absl::Status HloDomainRemover::RunContext::VerifyAndNormalizeDomain(
    const DomainMetadata::Domain& domain) {
  TF_ASSIGN_OR_RETURN(const DomainMetadata* ref_metadata,
                      HloDomainVerifier::VerifyDomain(domain));
  if (ref_metadata != nullptr) {
    VLOG(4) << "Applying domain normalization: " << ref_metadata->ToString();
    TF_RETURN_IF_ERROR(remover_->normalizer_(domain, ref_metadata));
  } else {
    // No kDomain instruction was present within this domain, so call the
    // generic normalization functions and have them apply their heuristic.
    VLOG(2) << "Applying domain-less normalization";
    TF_RETURN_IF_ERROR(remover_->normalizer_(domain, nullptr));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> HloDomainRemover::RunContext::Run(
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(4) << "Processing metadata domain: '" << remover_->kind_ << "'";
  int64_t removed_domains = 0;
  for (HloComputation* computation : module_->computations(execution_threads)) {
    // First create the domain instruction sets. A domain instruction set is
    // the set of instructions whose edges never cross a kDomain instruction.
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloDomainMap> domain_map,
                        HloDomainMap::Create(computation, remover_->kind_));
    // Verify and normalize every domain populated within the map.
    for (auto& domain : domain_map->GetDomains()) {
      TF_RETURN_IF_ERROR(VerifyAndNormalizeDomain(*domain));
    }

    // Now remove all the kDomain instructions of the kind specified by the
    // remover, that are within the currently processed computation from the
    // graph.
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      for (HloInstruction* operand : instruction->unique_operands()) {
        if (domain_map->IsDomainInstruction(operand)) {
          VLOG(5) << "Removing " << operand->name();
          TF_RETURN_IF_ERROR(
              operand->ReplaceAllUsesWith(operand->mutable_operand(0)));
          TF_RETURN_IF_ERROR(computation->RemoveInstruction(operand));
          ++removed_domains;
        }
      }
    }
    HloInstruction* root = computation->root_instruction();
    if (root != nullptr && domain_map->IsDomainInstruction(root)) {
      VLOG(5) << "Removing " << root->name();
      computation->set_root_instruction(root->mutable_operand(0));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(root));
      ++removed_domains;
    }
  }
  VLOG(3) << "Removed " << removed_domains << " kDomain instructions of '"
          << remover_->kind_ << "' kind";
  return removed_domains > 0;
}

absl::StatusOr<int64_t> HloDomainRemover::RemoveExitDomains(
    HloInstruction* instruction, absl::string_view domain_kind) {
  int64_t removed_domains = 0;
  HloComputation* computation = instruction->parent();
  // Make a const copy of instruction's users to loop through later, as the
  // users vector could be changed during the loop(e.g. ReplaceAllUsesWith).
  const std::vector<HloInstruction*> users(instruction->users());
  for (HloInstruction* user : users) {
    if (user->opcode() == HloOpcode::kDomain &&
        user->user_side_metadata().Kind() == domain_kind &&
        user->operand_side_metadata().Kind() == domain_kind) {
      VLOG(5) << "Removing exit domain " << user->name();
      TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(instruction));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(user));
      ++removed_domains;
    }
  }
  return removed_domains;
}

absl::StatusOr<bool> HloDomainRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  RunContext run_context(module, this);
  return run_context.Run(execution_threads);
}

}  // namespace xla
