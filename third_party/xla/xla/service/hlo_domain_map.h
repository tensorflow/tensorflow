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

#ifndef XLA_SERVICE_HLO_DOMAIN_MAP_H_
#define XLA_SERVICE_HLO_DOMAIN_MAP_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_domain_metadata.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/statusor.h"
#include "tsl/platform/status.h"

namespace xla {

// The HloDomainMap splits a set of instructions within a module or computation,
// into different domains, separated by kDomain instructions.
// A domain is composed by a set of instructions which can reach each other via
// operand/user edges, without crossing a kDomain insutrction of a given kind.
// A domain never crosses computation boundaries.
class HloDomainMap {
 public:
  // Creates a new HloDomainMap, creating all the domains within the input
  // computation, of the given kind. If domain_kind is not empty, only the
  // kDomain instructions of domain_kind will be considered as separators.
  // Otherwise every kDomain instruction will be splitting domains.
  static absl::StatusOr<std::unique_ptr<HloDomainMap>> Create(
      HloComputation* computation, std::string domain_kind);

  // Creates a new HloDomainMap, creating all the domains within the input
  // module, of the given kind. If domain_kind is not empty, only the
  // kDomain instructions of domain_kind will be considered as separators.
  // Otherwise every kDomain instruction will be splitting domains.
  static absl::StatusOr<std::unique_ptr<HloDomainMap>> Create(
      HloModule* module, std::string domain_kind);

  // Retrieves all the domains the input module or computation are composed by.
  const std::vector<std::unique_ptr<DomainMetadata::Domain>>& GetDomains()
      const {
    return instruction_domains_;
  }

  // Checks whether two instructions are within the same domain.
  bool InSameDomain(const HloInstruction* instruction1,
                    const HloInstruction* instruction2) const;

  // Checks whether instruction is a kDomain instruction of the kind we are
  // currently processing.
  bool IsDomainInstruction(const HloInstruction* instruction) const;

  // Retrieves the domain identifier of the instruction, or -1 in case
  // instruction is not found within any domain.
  int64_t GetDomainId(const HloInstruction* instruction) const;

  // Returns the unique id of the domain metadata for the domain the given
  // instruction belongs to. The given instruction must not be a kDomain
  // instruction since each domain instruction is associated with 2 domains.
  int64_t GetDomainMetadataId(const HloInstruction* instruction) const;

 private:
  // Map used for representing instruction ordering, i.e.
  // order_map[a] < order_map[b] means a must be ordered before b.
  using InstructionOrderMap =
      absl::flat_hash_map<const HloInstruction*, int64_t>;

  HloDomainMap(std::string domain_kind)
      : domain_kind_(std::move(domain_kind)) {}

  // Check if the kDomain instruction is facing (via its operand link) another
  // kDomain instruction of the same kind, hence defining an empty domain.
  // If that is the case, create the empty domain and call the proper
  // normalizer.
  absl::Status TryProcessEmptyDomain(HloInstruction* instruction);

  absl::Status Populate(HloComputation* computation);

  // Inserts the provided domain into the ones tracked by this object,
  // creating a new domain ID.
  absl::Status InsertDomain(std::unique_ptr<DomainMetadata::Domain> domain);

  // From the given instruction, expands operand and user wise, the set of
  // instructions which can be reached without crossing a kDomain instruction
  // of the kind specified by domain_kind_.
  // The domain data structure will be populated with all the reached
  // instructions, and the boundaries of the domain, with the kDomain
  // instructions encountered while expanding the reach.
  absl::Status ExpandDomain(HloInstruction* instruction,
                            DomainMetadata::Domain* domain) const;

  // Creates a domain data structure using the ExpandDomain() API.
  absl::StatusOr<std::unique_ptr<DomainMetadata::Domain>> CreateDomain(
      HloInstruction* instruction,
      const InstructionOrderMap& instructions_order) const;

  // Out of an instruction set, returns a vector of all the ones which are not
  // a kDomain kind.
  static std::vector<HloInstruction*> MakeNonDomainInstructions(
      const absl::flat_hash_set<HloInstruction*>& instruction_set,
      const InstructionOrderMap& instructions_order);

  // Populates domain_metadata_id_ that maps each HloInstruction to the unique
  // ID of its associated domain metatadata.
  absl::Status PopulateDomainMetadataMap();

  std::string domain_kind_;
  std::vector<std::unique_ptr<DomainMetadata::Domain>> instruction_domains_;
  absl::flat_hash_map<const HloInstruction*, int64_t> instruction_to_domain_;
  absl::flat_hash_map<const HloInstruction*, int64_t> domain_metadata_id_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_DOMAIN_MAP_H_
