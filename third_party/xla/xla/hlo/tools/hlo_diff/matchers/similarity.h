/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_SIMILARITY_H_
#define XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_SIMILARITY_H_

#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"

namespace xla {
namespace hlo_diff {

// Function to compute property match score between two instructions.
// Compares various properties of the instructions and returns a double score.
// Higher the score, more similar the instructions are.
using InstructionSimilarityFn = absl::FunctionRef<double(
    const HloInstructionNode*, const HloInstructionNode*)>;

// A heuristic score based on the node attributes. Calculated by comparing the
// fingerprint, name and generation of the nodes. This set of parameters
// together with min_similarity threshold = 0.75 works the best so far, and
// might need to be tuned later.
double NodeAttributesSimilarity(const HloInstructionNode* absl_nonnull left,
                                const HloInstructionNode* absl_nonnull right);

// A heuristic score based on the ancestor subgraphs of the given nodes.
// Calculated by comparing the fingerprints of the ancestors of the nodes.
double AncestorSubGraphSimilarity(const HloInstructionNode* left,
                                  const HloInstructionNode* right,
                                  int candidate_traversal_limit,
                                  int min_bfs_distance, int left_graph_size,
                                  int right_graph_size);

// Returns similarity of properties between two instructions.
double NodePropertySimilarity(const HloInstructionNode* left,
                              const HloInstructionNode* right);

// Returns a similarity score of two parameters based on their
// sharding, layout, name and users.
double ParamPropertySimilarity(const HloInstructionNode* left,
                               const HloInstructionNode* right);

// Returns a similarity score of two constants based on their
// sharding, layout, name and users.
double ConstantPropertySimilarity(const HloInstructionNode* left,
                                  const HloInstructionNode* right);

inline InstructionSimilarityFn MatchFnForOpcode(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kParameter:
      return ParamPropertySimilarity;
    case HloOpcode::kConstant:
      return ConstantPropertySimilarity;
    default:
      return NodePropertySimilarity;
  }
};

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_SIMILARITY_H_
