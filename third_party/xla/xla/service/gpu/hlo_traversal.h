/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_SERVICE_GPU_HLO_TRAVERSAL_H_
#define XLA_SERVICE_GPU_HLO_TRAVERSAL_H_

#include <functional>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {

enum class TraversalResult {
  // Visit the operands of this node.
  kVisitOperands,
  // Do not visit any more nodes.
  kAbortTraversal,
  // Do not visit the operands of this node (but continue the traversal
  // otherwise). If the node visitation function returns this, the `boundary`
  // condition will not be evaluated.
  kDoNotVisitOperands,
};

using FusionBoundaryFn = std::function<bool(const HloInstruction& producer,
                                            const HloInstruction& consumer)>;

// Boundary function for HloFusionInstructions.
bool DefaultFusionBoundaryFn(const HloInstruction& producer,
                             const HloInstruction& consumer);

// Creates a fusion boundary function for fusing the given producer and
// consumer. `fused_consumer` must be a consumer of `fused_producer`.
FusionBoundaryFn MakeProducerConsumerFusion(
    const HloInstruction& fused_producer, const HloInstruction& fused_consumer);

// Creates a fusion boundary function for a fusion consisting only of `root`. If
// `root` is a fusion, the result is the same as `DefaultFusionBuondaryFn`. If
// `root` is the root of a fusion, the result is just that root, not the entire
// computation.
FusionBoundaryFn MakeSingleInstructionFusion(const HloInstruction& root);

// Visit the HLO nodes starting from `roots` in BFS order (consumers before
// producers). Each node will be visited exactly once. The graph is not
// traversed along edges for which `boundary` returns true.
void HloBfsConsumersFirstTraversal(
    absl::Span<const HloInstruction* const> roots,
    const FusionBoundaryFn& boundary,
    const std::function<TraversalResult(const HloInstruction& node)>& visit);

// Visit the HLO nodes starting from `roots`, returning true if the return value
// of `visit` for any of nodes is true. Uses the same order as
// `HloBfsConsumersFirstTraversal`.
bool HloAnyOf(absl::Span<const HloInstruction* const> roots,
              const FusionBoundaryFn& boundary,
              const std::function<bool(const HloInstruction& node)>& visit);

// Visit the HLO nodes stating from `roots`, returning the first
// node for which `visit` returns true, or `nullptr` if no node matches. Uses
// the same order as `HloBfsConsumersFirstTraversal`.
const HloInstruction* HloFindIf(
    absl::Span<const HloInstruction* const> roots,
    const FusionBoundaryFn& boundary,
    const std::function<bool(const HloInstruction& node)>& visit);

// Visit the producers of all parameters that are needed by the fusion.
void FindFusionArguments(
    absl::Span<const HloInstruction* const> roots,
    const FusionBoundaryFn& boundary,
    const std::function<void(const HloInstruction& producer)>& visit);

// Returns all predecessors of node that lie within the boundary.
absl::InlinedVector<const HloInstruction*, 2> FindPredecessors(
    const HloInstruction& node, const FusionBoundaryFn& boundary);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_HLO_TRAVERSAL_H_
