/* Copyright 2022 The OpenXLA Authors.

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
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/shape.h"

#ifndef XLA_SERVICE_CUSTOM_CALL_SHARDING_HELPER_H_
#define XLA_SERVICE_CUSTOM_CALL_SHARDING_HELPER_H_

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
  // Returns if the given custom-call instruction can propagate sharding to its
  // operands.
  virtual bool CanPropagateShardingToOperands(
      const HloInstruction* instruction) const;
  virtual ~CustomCallShardingHelper() = default;
};

namespace spmd {
class SpmdPartitioningVisitor;
}  // namespace spmd

// Holds the result of invoking a partitioner-agnostic custom call partitioning
// callback.
struct CustomCallPartitionResult {
  std::string mlir_module;
  std::vector<HloSharding> arg_shardings;
  HloSharding result_sharding = HloSharding::Replicate();
};

// Helper class that provides a partitioning function in addition to sharding
// policies.
class CustomCallPartitioner : public CustomCallShardingHelper {
 public:
  virtual absl::Status Partition(spmd::SpmdPartitioningVisitor* partitioner,
                                 HloInstruction* hlo) const;

  // Returns true if this partitioner implements `PartitionToMlir()`. Subclasses
  // overriding `PartitionToMlir()` must also override this to return true so
  // MLIR partitioners (such as Shardy) dispatch to it.
  virtual bool SupportsMlirPartitioning() const { return false; }

  // Invokes a partitioner-agnostic custom call partitioning callback (such as
  // JAX's `@custom_partitioning` C-API callback) using unpadded global shapes,
  // shardings, and raw `backend_config` bytes to produce a shard-local MLIR
  // module and desired operand/result shardings. Returns `std::nullopt` if this
  // partitioner only implements the GSPMD `Partition(SpmdPartitioningVisitor*,
  // HloInstruction*)` method. Subclasses overriding this must also override
  // `SupportsMlirPartitioning()` to return true.
  virtual absl::StatusOr<std::optional<CustomCallPartitionResult>>
  PartitionToMlir(absl::Span<const Shape> arg_shapes,
                  absl::Span<const std::optional<HloSharding>> arg_shardings,
                  const Shape& result_shape,
                  const std::optional<HloSharding>& result_sharding,
                  absl::string_view backend_config) const;

  // Invokes a sharding-inspection callback for custom calls such as
  // `InspectSharding` without requiring an HloInstruction or GSPMD visitor.
  // `backend_config` may contain raw binary bytes (including embedded null
  // bytes).
  virtual absl::Status InspectSharding(const HloSharding& sharding,
                                       absl::string_view backend_config) const;

  // Returns if the given side-effecting custom-call is allowed to have
  // replicated sharding.
  virtual bool CanSideEffectingHaveReplicatedSharding() const { return false; }
};

// Fetch partitioning overrides on a per-custom_call_target basis.
const CustomCallPartitioner* GetCustomCallPartitioner(
    absl::string_view custom_call_target);
// Register partitioning overrides on a per-custom_call_target basis.
void RegisterCustomCallPartitioner(
    absl::string_view custom_call_target,
    std::unique_ptr<CustomCallPartitioner> partitioner);

}  // namespace xla

#endif  // XLA_SERVICE_CUSTOM_CALL_SHARDING_HELPER_H__
