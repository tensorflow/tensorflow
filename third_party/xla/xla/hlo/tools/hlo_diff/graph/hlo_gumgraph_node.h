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
#include "xla/service/hlo_value.h"

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
  // This fingerprint represents the structure and content of the entire
  // subgraph rooted at this node. It is computed recursively in a bottom-up
  // manner: a node's subgraph fingerprint is derived by combining its own
  // instruction fingerprint with the subgraph fingerprints of all its children
  // nodes. The combination of fingerprints is order-dependent, meaning it
  // accounts for the order of child nodes (e.g., operands). The
  // `GreedySubGraphExactMatcher` utilizes this fingerprint to efficiently
  // identify and match structurally identical subgraphs between two HLO graphs.
  uint64_t subgraph_fingerprint = 0;
  // fingerprint is used to determine if two instructions should be matched.
  uint64_t fingerprint = 0;
  // canonical_fingerprint is used to determine if two mapped instructions are
  // changed.
  uint64_t canonical_fingerprint = 0;
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
  std::vector<int> i_th_parents;
  std::vector<HloInstructionNode*> parents;
  std::vector<int> i_th_children;
  HloInstructionNodeProps props;
  bool is_root = false;
  // All HloValues that this instruction consumes as input.
  std::vector<const HloValue*> used_values;
  // All uses of the HloValues that are present in this instruction's output.
  std::vector<HloUse> value_uses;
  absl::string_view GetName() const {
    return is_root ? "root" : instruction->name();
  }
};

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_GRAPH_HLO_GUMGRAPH_NODE_H_
