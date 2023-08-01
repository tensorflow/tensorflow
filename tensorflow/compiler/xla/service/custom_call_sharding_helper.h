/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CUSTOM_CALL_SHARDING_HELPER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CUSTOM_CALL_SHARDING_HELPER_H_

namespace xla {

// Helper class that helps implement sharding propagation policies for
// CustomCalls. It is called and used by the ShardingPropagation pass. Meant to
// be overridden by targets.
class CustomCallShardingHelper {
 public:
  // Function that manipulates an instruction sharding based on a user wanting
  // to update the sharding of an instruction.
  virtual HloSharding PropagateUserSharding(const HloInstruction* instruction,
                                            const HloInstruction* user,
                                            const HloSharding& sharding) const;
  // Infer sharding from the operands of an instruction.
  virtual std::optional<HloSharding> InferShardingFromOperands(
      const HloInstruction* instruction) const;
  // Returns if the instruction passed as parameter is a supported custom-call
  // for which the functions of this class are implemented.
  virtual bool IsCustomCallShardable(const HloInstruction* instruction) const;
  // Returns the list of instructions in sub-computations that must be sharded
  // in the same way as `instruction`.
  virtual std::vector<HloInstruction*> GetRelatedInstructions(
      HloInstruction* instruction) const {
    return {};
  }
  virtual ~CustomCallShardingHelper() = default;
};

namespace spmd {
class SpmdPartitioningVisitor;
}  // namespace spmd

// Helper class that provides a partitioning function in addition to sharding
// policies.
class CustomCallPartitioner : public CustomCallShardingHelper {
 public:
  virtual xla::Status Partition(spmd::SpmdPartitioningVisitor* partitioner,
                                HloInstruction* hlo) const;

  // Returns if the given side-effecting custom-call is allowed to have
  // replicated sharding.
  virtual bool CanSideEffectingHaveReplicatedSharding() const { return false; }
};

// Fetch partitioning overrides on a per-custom_call_target basis.
const CustomCallPartitioner* GetCustomCallPartitioner(
    const std::string& custom_call_target);
// Register partitioning overrides on a per-custom_call_target basis.
void RegisterCustomCallPartitioner(
    const std::string& custom_call_target,
    std::unique_ptr<CustomCallPartitioner> partitioner);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CUSTOM_CALL_SHARDING_HELPER_H__
