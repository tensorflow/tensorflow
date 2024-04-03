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

#ifndef XLA_SERVICE_HLO_DOMAIN_ISOLATOR_H_
#define XLA_SERVICE_HLO_DOMAIN_ISOLATOR_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Domain isolation is the task of placing kDomain instructions between HLO
// instructions having different sharding. A kDomain instruction is essentially
// used to break an HLO graph edge connecting two instructions with different
// sharding. If a set of connected instructions have all the same sharding, no
// kDomain instruction will be placed.
class HloDomainIsolator : public HloModulePass {
 public:
  // Creates a new kDomain instruction for the edge between the use instruction
  // (the first HloInstruction argument), and the operand instruction (the
  // third HloInstruction argument) if the interesting attribute of the
  // instruction differences from the attribute of the root (the second
  // HloInstruction argument).
  // Returns nullptr in case no domain separation is necessary.
  using DomainCreator = std::function<HloInstruction*(
      HloInstruction*, HloInstruction*, HloInstruction*)>;
  using DomainCreatorFactory = std::function<DomainCreator()>;
  explicit HloDomainIsolator(DomainCreatorFactory creator_factory_);

  absl::string_view name() const override { return "domain_isolator"; }

  // Update domains for an instruction.
  absl::StatusOr<bool> UpdateDomains(HloInstruction* instruction);

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  DomainCreatorFactory creator_factory_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_DOMAIN_ISOLATOR_H_
