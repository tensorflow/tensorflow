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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_METADATA_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_METADATA_H_

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_domain_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// A DomainMetadata implementation that internally wraps a sharding attribute.
class ShardingMetadata : public DomainMetadata {
 public:
  explicit ShardingMetadata(std::shared_ptr<const HloSharding> sharding)
      : sharding_(std::move(sharding)) {}

  std::unique_ptr<DomainMetadata> Clone() const override;

  absl::string_view Kind() const override { return KindName(); }

  bool Matches(const DomainMetadata& other) const override;

  string ToString() const override;

  const HloSharding* sharding() const { return sharding_.get(); }

  static absl::string_view KindName() { return "sharding"; }

  static StatusOr<const ShardingMetadata*> ToShardingMetadata(
      const DomainMetadata* metadata);

  // Apply the specified domain metadata onto the specified domain. If no
  // metadata is specified then apply sharding heuristics and normalize the
  // instructions whose sharding deviates from the one which is inferred as to
  // be the original one. Policy wise, HLO passes are allowed to create new
  // unassigned instructions, but if they do create assigned ones, they have to
  // conform to the ones around.
  static Status NormalizeShardingDomain(const DomainMetadata::Domain& domain,
                                        const DomainMetadata* metadata);

 private:
  std::shared_ptr<const HloSharding> sharding_;
};

// If the sharding between root and instruction changes then returns a
// ShardingMetadata based kDomain instruction what can be used to separate
// operand and instruction.
// Returns nullptr if there is no need for a domain separation.
class ShardingDomainCreator {
 public:
  HloInstruction* operator()(HloInstruction* instruction, HloInstruction* root,
                             HloInstruction* operand);

 private:
  // Map from instruction and user sharding to domain users to CSE identical
  // domains.
  struct DomainCseMapKey {
    const HloInstruction* instruction;
    std::shared_ptr<const HloSharding> sharding;

    bool operator==(const DomainCseMapKey& other) const;
  };
  struct DomainCseMapHasher {
    size_t operator()(const DomainCseMapKey& key) const;
  };
  std::unordered_map<DomainCseMapKey, HloInstruction*, DomainCseMapHasher>
      domain_cse_map_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_METADATA_H_
