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

#include "tensorflow/compiler/xla/service/hlo_domain_verifier.h"

#include <set>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class HloDomainVerifier::RunContext {
 public:
  RunContext(HloModule* module, HloDomainVerifier* verifier)
      : module_(module), verifier_(verifier) {}

  Status Run(const absl::flat_hash_set<absl::string_view>& execution_threads);

 private:
  // If the verifier caller passed an empty vector for kinds, we collect all the
  // available domain types.
  Status PopulateDomainKinds(
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  HloModule* module_;
  HloDomainVerifier* verifier_;
};

Status HloDomainVerifier::RunContext::PopulateDomainKinds(
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (verifier_->kinds_.empty()) {
    // The caller specified no domain kinds, collect all the ones available.
    std::set<std::string> kinds;
    for (HloComputation* computation :
         module_->computations(execution_threads)) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kDomain) {
          TF_RET_CHECK(instruction->user_side_metadata().Kind() ==
                       instruction->operand_side_metadata().Kind())
              << instruction->ToString();
          kinds.insert(std::string(instruction->user_side_metadata().Kind()));
        }
      }
    }
    verifier_->kinds_.insert(verifier_->kinds_.end(), kinds.begin(),
                             kinds.end());
  }
  return OkStatus();
}

Status HloDomainVerifier::RunContext::Run(
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(4) << "Running HLO Domain Verifier";
  TF_RETURN_IF_ERROR(PopulateDomainKinds(execution_threads));
  for (HloComputation* computation : module_->computations(execution_threads)) {
    for (auto& kind : verifier_->kinds_) {
      // First create the domain instruction sets. A domain instruction set is
      // the set of instructions whose edges never cross a kDomain instruction.
      TF_ASSIGN_OR_RETURN(std::unique_ptr<HloDomainMap> domain_map,
                          HloDomainMap::Create(computation, kind));
      // Verify every domain populated within the map.
      for (auto& domain : domain_map->GetDomains()) {
        TF_RETURN_IF_ERROR(VerifyDomain(*domain).status());
      }
    }
  }
  return OkStatus();
}

StatusOr<bool> HloDomainVerifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  RunContext run_context(module, this);
  TF_RETURN_IF_ERROR(run_context.Run(execution_threads));
  return false;
}

StatusOr<const DomainMetadata*> HloDomainVerifier::VerifyDomain(
    const DomainMetadata::Domain& domain) {
  const DomainMetadata* ref_metadata = nullptr;
  VLOG(4) << "Reach set:";
  for (HloInstruction* instruction : domain.instructions) {
    VLOG(4) << "  " << instruction->name();
  }
  VLOG(4) << "  Domains:";
  for (HloInstruction* instruction : domain.enter_domains) {
    const DomainMetadata& meta = instruction->user_side_metadata();
    VLOG(4) << "    User side: " << instruction->name();
    VLOG(4) << "      " << meta.ToString();
    if (ref_metadata == nullptr) {
      ref_metadata = &meta;
    } else {
      TF_RET_CHECK(meta.Matches(*ref_metadata))
          << "Metadata mismatch at instruction " << instruction->name() << " : "
          << meta.ToString() << " vs " << ref_metadata->ToString();
    }
  }
  for (HloInstruction* instruction : domain.exit_domains) {
    const DomainMetadata& meta = instruction->operand_side_metadata();
    VLOG(4) << "    Operand side: " << instruction->name();
    VLOG(4) << "      " << meta.ToString();
    if (ref_metadata == nullptr) {
      ref_metadata = &meta;
    } else {
      TF_RET_CHECK(meta.Matches(*ref_metadata))
          << "Metadata mismatch at instruction " << instruction->name() << " : "
          << meta.ToString() << " vs " << ref_metadata->ToString();
    }
  }
  return ref_metadata;
}

}  // namespace xla
