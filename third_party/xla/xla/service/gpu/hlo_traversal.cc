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
#include "xla/service/gpu/hlo_traversal.h"

#include <functional>
#include <queue>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace gpu {

bool DefaultFusionBoundaryFn(const HloInstruction&,
                             const HloInstruction& consumer) {
  return consumer.opcode() == HloOpcode::kParameter;
}

FusionBoundaryFn MakeProducerConsumerFusion(
    const HloInstruction& fused_producer,
    const HloInstruction& fused_consumer) {
  if (fused_consumer.opcode() == HloOpcode::kFusion &&
      fused_producer.opcode() == HloOpcode::kFusion) {
    // fusion -> fusion.
    return [&](const HloInstruction& producer, const HloInstruction& consumer) {
      return DefaultFusionBoundaryFn(producer, consumer) &&
             &producer != &fused_producer;
    };
  }
  if (fused_consumer.opcode() == HloOpcode::kFusion) {
    // non-fusion -> fusion.
    return [&](const HloInstruction& producer, const HloInstruction& consumer) {
      if (DefaultFusionBoundaryFn(producer, consumer)) {
        return &producer != &fused_producer;
      }
      // Otherwise, don't follow edges above the fused producer.
      return &consumer == &fused_producer;
    };
  }
  // anything -> non-fusion.
  return [&](const HloInstruction& producer, const HloInstruction& consumer) {
    if (&consumer == &fused_consumer) {
      // If the consumer is the fused user, only follow edges to the fused
      // producer.
      return &fused_producer != &producer;
    }

    // Otherwise, fall back to the default; we're already in the fused
    // producer.
    return DefaultFusionBoundaryFn(producer, consumer);
  };
}

FusionBoundaryFn MakeSingleInstructionFusion(const HloInstruction& root) {
  if (root.opcode() == HloOpcode::kFusion) {
    return DefaultFusionBoundaryFn;
  }
  return [](const HloInstruction&, const HloInstruction&) { return true; };
}

void HloBfsConsumersFirstTraversal(
    absl::Span<const HloInstruction* const> roots,
    const FusionBoundaryFn& boundary,
    const std::function<TraversalResult(const HloInstruction& node)>& visit) {
  absl::flat_hash_set<const HloInstruction*> visited;
  std::queue<const HloInstruction*> q;
  auto enqueue_operands = [&](const HloInstruction& node) {
    for (const auto* predecessor : FindPredecessors(node, boundary)) {
      if (visited.insert(predecessor).second) {
        q.push(predecessor);
      }
    }
  };

  for (auto* root : roots) {
    q.push(root);
  }
  while (!q.empty()) {
    const HloInstruction* node = q.front();
    q.pop();
    switch (visit(*node)) {
      case TraversalResult::kVisitOperands:
        enqueue_operands(*node);
        break;
      case TraversalResult::kAbortTraversal:
        return;
      case TraversalResult::kDoNotVisitOperands:
        break;
    }
  }
}

void FindFusionArguments(
    absl::Span<const HloInstruction* const> roots,
    const FusionBoundaryFn& boundary,
    const std::function<void(const HloInstruction& param)>& visit) {
  absl::flat_hash_set<const HloInstruction*> visited;
  HloBfsConsumersFirstTraversal(
      roots,
      [&](const HloInstruction& producer, const HloInstruction& consumer) {
        auto is_boundary = boundary(producer, consumer);
        if (is_boundary) {
          if (visited.insert(&producer).second) {
            visit(producer);
          }
        }
        return is_boundary;
      },
      [&](const HloInstruction&) { return TraversalResult::kVisitOperands; });
}

bool HloAnyOf(absl::Span<const HloInstruction* const> roots,
              const FusionBoundaryFn& boundary,
              const std::function<bool(const HloInstruction& node)>& visit) {
  return HloFindIf(roots, boundary, visit) != nullptr;
}

const HloInstruction* HloFindIf(
    absl::Span<const HloInstruction* const> roots,
    const FusionBoundaryFn& boundary,
    const std::function<bool(const HloInstruction& node)>& visit) {
  const HloInstruction* result = nullptr;
  HloBfsConsumersFirstTraversal(roots, boundary,
                                [&](const HloInstruction& node) {
                                  if (visit(node)) {
                                    result = &node;
                                    return TraversalResult::kAbortTraversal;
                                  }
                                  return TraversalResult::kVisitOperands;
                                });
  return result;
}

absl::InlinedVector<const HloInstruction*, 2> FindPredecessors(
    const HloInstruction& node, const FusionBoundaryFn& boundary) {
  absl::InlinedVector<const HloInstruction*, 2> predecessors;
  auto visit = [&](const HloInstruction& predecessor) {
    if (!boundary(predecessor, node)) {
      predecessors.push_back(&predecessor);
    }
  };

  switch (node.opcode()) {
    case HloOpcode::kParameter:
      if (auto* fusion = node.parent()->FusionInstruction()) {
        // If the parent is the entry computation, there's no predecessor.
        visit(*fusion->operand(node.parameter_number()));
      }
      break;
    case HloOpcode::kFusion:
      visit(*node.fused_expression_root());
      break;
    default:
      for (HloInstruction* operand : node.operands()) {
        visit(*operand);
      }
  }
  return predecessors;
}

}  // namespace gpu
}  // namespace xla
