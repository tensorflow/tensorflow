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

#ifndef XLA_SERVICE_HLO_DOMAIN_REMOVER_H_
#define XLA_SERVICE_HLO_DOMAIN_REMOVER_H_

#include "xla/hlo/ir/hlo_domain_metadata.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "tsl/platform/status.h"

namespace xla {

// Removes all the kDomain instructions of a given kind from the input module,
// and calls the normalizer to propagate the properties on the possibly new born
// instructions.
class HloDomainRemover : public HloModulePass {
 public:
  // Creates a new HloDomainRemover object tasked at removing all the kDomain
  // instructions of a given kind.
  // In case a reachable set (the set of instructions within a computation,
  // which are mutually reachable via operand/user pathways) has all the
  // instructions in it with the same attributes (ie, sharding), a normalizer
  // function is tasked at applying attribute normalization on the instructions
  // within such domain.
  HloDomainRemover(absl::string_view kind,
                   std::function<absl::Status(const DomainMetadata::Domain&,
                                              const DomainMetadata* metadata)>
                       normalizer)
      : kind_(kind), normalizer_(std::move(normalizer)) {}

  absl::string_view name() const override { return "domain_remover"; }

  // Remove domains of a given kind which are used as users of a specific
  // instruction.
  static absl::StatusOr<int64_t> RemoveExitDomains(
      HloInstruction* instruction, absl::string_view domain_kind);

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  class RunContext;

  std::string kind_;
  std::function<absl::Status(const DomainMetadata::Domain&,
                             const DomainMetadata* metadata)>
      normalizer_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_DOMAIN_REMOVER_H_
