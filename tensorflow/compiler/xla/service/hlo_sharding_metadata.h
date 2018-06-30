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

#include "tensorflow/compiler/xla/service/hlo_domain_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {

// A DomainMetadata implementation that internally wraps a sharding attribute.
class ShardingMetadata : public DomainMetadata {
 public:
  explicit ShardingMetadata(std::unique_ptr<HloSharding> sharding)
      : sharding_(std::move(sharding)) {}

  std::unique_ptr<DomainMetadata> Clone() const override;

  tensorflow::StringPiece Kind() const override { return KindName(); }

  bool Matches(const DomainMetadata& other) const override;

  string ToString() const override;

  Status NormalizeInstructions(
      const DomainMetadata::Domain& domain) const override;

  static tensorflow::StringPiece KindName() { return "sharding"; }

 private:
  std::unique_ptr<HloSharding> sharding_;
};

// Within a set of instructions which had common sharding attributes before
// entring the HLO passes pipeline, apply sharding heuristics and normalize the
// instructions whose sharding deviates from the one which is inferred as to be
// the original one.
// Policy wise, HLO passes are allowed to create new unassigned instructions,
// but if they do create assigned ones, they have to conform to the ones around.
Status NormalizeShardingDomain(const DomainMetadata::Domain& domain);

// Given an HLO graph edge between instruction and one of its operands, creates
// a ShardingMetadata based kDomain instruction if the sharding between
// instruction and operand changes. Returns nullptr if there is no need for a
// domain separation.
std::unique_ptr<HloInstruction> CreateShardingDomain(
    HloInstruction* instruction, HloInstruction* operand);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_METADATA_H_
