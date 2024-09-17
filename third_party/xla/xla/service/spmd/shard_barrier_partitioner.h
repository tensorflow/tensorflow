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

#ifndef XLA_SERVICE_SPMD_SHARD_BARRIER_PARTITIONER_H_
#define XLA_SERVICE_SPMD_SHARD_BARRIER_PARTITIONER_H_

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/custom_call_sharding_helper.h"

namespace xla {
namespace spmd {

constexpr char kShardBarrierFrom[] = "ShardBarrierFrom";
constexpr char kShardBarrierTo[] = "ShardBarrierTo";

// Custom-call partitioner shard barrier.
class ShardBarrierPartitioner : public CustomCallPartitioner {
 public:
  // This allows ShardBarrier related custom-call ops to be propagated during
  // sharding propagation.
  bool IsCustomCallShardable(const HloInstruction* instruction) const override {
    return true;
  }

  // This allows replicated sharding on custom-call op to pass checks at spmd
  // partitioner preprocess stage.
  bool CanSideEffectingHaveReplicatedSharding() const override { return true; }
};

// Custom-call partitioner for ShardBarrierFrom.
class ShardBarrierFromPartitioner : public ShardBarrierPartitioner {
 public:
  // Always do not allow forward propagation with ShardBarrierFrom.
  std::optional<HloSharding> InferShardingFromOperands(
      const HloInstruction* instruction) const override {
    return std::nullopt;
  }

  // Always let backward propagation run through with ShardBarrierFrom.
  HloSharding PropagateUserSharding(
      const HloInstruction* instruction, const HloInstruction* user,
      const HloSharding& sharding) const override {
    return sharding;
  }
};

// Custom-call partitioner for TPU logger.
class ShardBarrierToPartitioner : public ShardBarrierPartitioner {
 public:
  // Always let forward propagation run through with ShardBarrierTo.
  std::optional<HloSharding> InferShardingFromOperands(
      const HloInstruction* instruction) const override {
    if (instruction->operand(0)->has_sharding()) {
      return instruction->operand(0)->sharding();
    }
    return std::nullopt;
  }

  // Always do not allow backward propagation with ShardBarrierTo.
  HloSharding PropagateUserSharding(
      const HloInstruction* instruction, const HloInstruction* user,
      const HloSharding& sharding) const override {
    return HloSharding::Replicate();
  }
};

}  // namespace spmd
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARD_BARRIER_PARTITIONER_H_
