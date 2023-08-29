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
#include "tensorflow/compiler/xla/service/gpu/hlo_traversal.h"

#include <functional>
#include <queue>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace gpu {

void HloBfsConsumersFirstTraversal(
    const HloInstruction& root, const FusionBoundaryFn& boundary,
    const std::function<TraversalResult(const HloInstruction& node)>& visit) {
  absl::flat_hash_set<const HloInstruction*> visited;
  std::queue<const HloInstruction*> q;
  auto enqueue_operands = [&](const HloInstruction& node) {
    if (node.opcode() == HloOpcode::kParameter) {
      auto* fusion = node.parent()->FusionInstruction();
      // If the parent is the entry computation, there's no producer.
      if (!fusion) {
        return;
      }
      auto* operand = fusion->operand(node.parameter_number());
      if (!boundary(*operand, node) && visited.insert(operand).second) {
        q.push(operand);
      }
      return;
    }

    if (node.opcode() == HloOpcode::kFusion) {
      const auto* fusion_root = node.fused_expression_root();
      if (!boundary(*fusion_root, node) && visited.insert(fusion_root).second) {
        q.push(fusion_root);
      }
      return;
    }

    for (HloInstruction* operand : node.operands()) {
      if (!boundary(*operand, node) && visited.insert(operand).second) {
        q.push(operand);
      }
    }
  };

  q.push(&root);
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

void FindFusionParameters(
    const HloInstruction& root, const FusionBoundaryFn& boundary,
    const std::function<void(const HloInstruction& param)>& visit) {
  absl::flat_hash_set<const HloInstruction*> visited;
  HloBfsConsumersFirstTraversal(
      root,
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

}  // namespace gpu
}  // namespace xla
