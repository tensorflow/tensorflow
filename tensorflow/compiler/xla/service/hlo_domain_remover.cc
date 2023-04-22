/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_domain_remover.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_domain_verifier.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class HloDomainRemover::RunContext {
 public:
  RunContext(HloModule* module, HloDomainRemover* remover)
      : module_(module), remover_(remover) {}

  StatusOr<bool> Run();

 private:
  // Verifies the consistency of the domain, and normalizes the instructions
  // within it.
  Status VerifyAndNormalizeDomain(const DomainMetadata::Domain& domain);

  HloModule* module_;
  HloDomainRemover* remover_;
};

Status HloDomainRemover::RunContext::VerifyAndNormalizeDomain(
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
  return Status::OK();
}

StatusOr<bool> HloDomainRemover::RunContext::Run() {
  VLOG(4) << "Processing metadata domain: '" << remover_->kind_ << "'";
  int64 removed_domains = 0;
  for (HloComputation* computation : module_->computations()) {
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

StatusOr<bool> HloDomainRemover::Run(HloModule* module) {
  RunContext run_context(module, this);
  return run_context.Run();
}

}  // namespace xla
