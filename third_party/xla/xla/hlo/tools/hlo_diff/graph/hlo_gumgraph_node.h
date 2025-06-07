/*
 * Copyright 2025 The OpenXLA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_HLO_TOOLS_HLO_DIFF_GRAPH_HLO_GUMGRAPH_NODE_H_
#define XLA_HLO_TOOLS_HLO_DIFF_GRAPH_HLO_GUMGRAPH_NODE_H_

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/call_graph.h"

namespace xla {
namespace hlo_diff {

// Position of a node in a container of siblings.
struct ListPosition {
  int64_t index = 0;
  int64_t size = 0;
};

// Properties of a instruction node in a HloGumgraph such as generation etc.
struct HloInstructionNodeProps {
  int64_t generation = 0;
  int64_t height = 0;
  uint64_t subgraph_fingerprint = 0;
  uint64_t fingerprint = 0;
  uint64_t canonical_fingerprint = 0;
  ListPosition sibling_position;
  ListPosition pre_order_graph_position;
};

// Properties of a computation node in a HloGumgraph.
struct CallGraphNodeProps {
  const CallGraphNode* call_graph_node;
  uint64_t fingerprint = 0;
  absl::string_view GetName() const {
    return call_graph_node->computation()->name();
  }
};

// A node in a HloGumgraph representing a HLO instruction.
// Only root nodes can have no instruction.
struct HloInstructionNode {
  const HloInstruction* instruction;
  int unique_node_index = 0;
  std::vector<HloInstructionNode*> children;
  std::vector<HloInstructionNode*> parents;
  HloInstructionNodeProps props;
  bool is_root = false;
  absl::string_view GetName() const {
    return is_root ? "root" : instruction->name();
  }
};

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_GRAPH_HLO_GUMGRAPH_NODE_H_
