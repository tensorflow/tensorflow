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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_GRAPH_HLO_GUMGRAPH_H_
#define XLA_HLO_TOOLS_HLO_DIFF_GRAPH_HLO_GUMGRAPH_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/analysis/hlo_value_tracing.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/service/call_graph.h"

namespace xla {
namespace hlo_diff {

// Options for computing the per instruction/node fingerprint of an HloGumgraph.
struct HloGumgraphFingerprintOptions {
  // Ignore shape when computing the instruction fingerprint.
  bool ignore_shape = false;
};

// A directed acyclic graph representation of an HloModule with all called
// computations inlined i.e. the calling instructions is connected to the
// called computation's root instruction.
class HloGumgraph {
 public:
  // Instantiates a HloGumgraph from a HloModule, pre-processing and caching
  // various graph properties such as height, siblings per node etc.
  static absl::StatusOr<std::unique_ptr<const HloGumgraph>> Create(
      const HloModule* absl_nonnull hlo_module,
      const HloGumgraphFingerprintOptions& fingerprint_options = {});

  // HloGumgraph is neither copyable nor movable as it can be really large.
  HloGumgraph(const HloGumgraph&) = delete;
  HloGumgraph& operator=(const HloGumgraph&) = delete;

  // Returns the dummy root node which is connected to all zero-indegree nodes
  // in the graph. The dummy root is always connected to the entry computation's
  // root instruction but additionally might be connected to other unreachable
  // roots in the entry computation.
  inline const HloInstructionNode& GetRoot() const { return root_; }

  // Returns graph node corresponding to the given HloInstruction. Returns
  // nullptr if the instruction is not in the graph.
  inline HloInstructionNode* GetNode(
      const HloInstruction* absl_nonnull instruction) const {
    if (auto it = instruction_to_node_.find(instruction);
        it != instruction_to_node_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  // Returns all nodes in the graph excluding the dummy root node.
  inline std::vector<HloInstructionNode*> AllNodes() const {
    std::vector<HloInstructionNode*> nodes;
    for (const auto& [_, node] : instruction_to_node_) {
      nodes.push_back(node.get());
    }
    return nodes;
  }

  // Returns the number of nodes in the graph including the dummy root node.
  inline int GetNodeCount() const { return instruction_to_node_.size() + 1; }

  // Returns all properties of computations in the graph.
  inline const absl::flat_hash_map<const HloComputation*, CallGraphNodeProps>&
  AllComputationProps() const {
    return computation_to_props_;
  }

  // Returns the call graph of the HloModule.
  const CallGraph& GetCallGraph() const { return *call_graph_; }

  // Returns the HloValueTracing used to trace the HloValues used by
  // instructions.
  const HloValueTracing& GetHloValueTracing() const {
    return *hlo_value_tracing_;
  }

  // Returns the backing HloModule of the HloGumgraph.
  const HloModule& GetHloModule() const { return hlo_module_; }

 private:
  explicit HloGumgraph(const HloModule& hlo_module,
                       const HloGumgraphFingerprintOptions& fingerprint_options,
                       std::unique_ptr<CallGraph> call_graph,
                       std::unique_ptr<HloValueTracing> hlo_value_tracing)
      : hlo_module_(hlo_module),
        fingerprint_options_(fingerprint_options),
        root_(
            {.instruction = nullptr, .unique_node_index = 0, .is_root = true}),
        call_graph_(std::move(call_graph)),
        hlo_value_tracing_(std::move(hlo_value_tracing)) {}

  // Adds a HloInstructionNode for the given HloInstruction to the graph.
  // Returns a pair of the node and a boolean indicating whether the node was
  // already in the graph.
  std::pair<HloInstructionNode*, bool> AddNode(
      const HloInstruction& instruction, int unique_node_index);

  // Constructs the HloGumgraph from the given HloModule connecting Instruction
  // operands and called computations.
  absl::Status ConstructGraph(const HloModule& hlo_module);

  // Precomputes the generation of each node in the graph. Generation of a node
  // is simply the longest distance of a node from the root node. The generation
  // of the root node is 0. Additionally it returns all zero-indegree nodes.
  absl::StatusOr<std::vector<HloInstructionNode*>> PrecomputeGenerations();

  // Precomputes the size and height of each node in the graph.
  void PrecomputeSizeAndHeight();

  // Precomputes the fingerprint of each computation in the graph, all
  // instructions in the computation are hashed to compute the fingerprint.
  absl::Status PrecomputeComputationFingerprint();

  // Precomputes the index of each node in a pre-order DFS traversal of the
  // graph.
  void PrecomputeDfsPosition();

  const HloModule& hlo_module_;
  const HloGumgraphFingerprintOptions& fingerprint_options_;
  HloInstructionNode root_;
  absl::flat_hash_map<const HloInstruction*,
                      std::unique_ptr<HloInstructionNode>>
      instruction_to_node_;
  absl::flat_hash_map<const HloComputation*, CallGraphNodeProps>
      computation_to_props_;
  std::vector<std::vector<HloInstructionNode*>> nodes_by_generation_;
  const std::unique_ptr<CallGraph> call_graph_;
  const std::unique_ptr<HloValueTracing> hlo_value_tracing_;
};

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_GRAPH_HLO_GUMGRAPH_H_
