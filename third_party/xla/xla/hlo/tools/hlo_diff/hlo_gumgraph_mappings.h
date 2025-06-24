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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_HLO_GUMGRAPH_MAPPINGS_H_
#define XLA_HLO_TOOLS_HLO_DIFF_HLO_GUMGRAPH_MAPPINGS_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "boost/bimap.hpp"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/service/call_graph.h"

namespace xla::hlo_diff {

// Type of matcher that matched two HloInstructionNodes.
enum class MatcherType : std::uint8_t {
  kNotSet,
  kManual,
  kComputationGraphExactFingerprintMatcher,
  kComputationGraphExactSignatureMatcher,
  kGreedySubGraphExactMatcher,
  kGreedyLimitedCandidatesBottomUpMatcher,
  kStrictGreedyTopDownMatcher,
  kGreedyTopDownMatcher,
};

// Computations with matching input parameters and output result are classified
// as kSignature matches. kExact matches on the other hand are kSignature
// matches that additionally have identical instructions in the computation
// graph, i.e. same computation fingerprint.
enum class ComputationMatchType : std::uint8_t { kExact, kSignature };

// Aggregated match characteristics of a mapped HloInstructionNode.
struct HloInstructionNodeMappingProps {
  bool unchanged = false;
  MatcherType matcher_type = MatcherType::kNotSet;
  std::string matcher_debug_info;
};

// Aggregated match characteristics of a mapped CallGraphNode.
struct HloCallGraphNodeMappingProps {
  ComputationMatchType computation_match_type = ComputationMatchType::kExact;
};

using InstructionPair = boost::bimap<
    const HloInstructionNode*, const HloInstructionNode*,
    boost::bimaps::with_info<HloInstructionNodeMappingProps>>::value_type;

using CallGraphNodePair = boost::bimap<
    const CallGraphNode*, const CallGraphNode*,
    boost::bimaps::with_info<HloCallGraphNodeMappingProps>>::value_type;

// Mapped nodes between two HloGumgraphs.
struct HloGumgraphMappings {
  // Map between the left and right CallGraphNodes.
  boost::bimap<const CallGraphNode*, const CallGraphNode*,
               boost::bimaps::with_info<HloCallGraphNodeMappingProps>>
      left_to_right_computation_map;

  // A bi-directional map between the left and right HloInstructionNodes along
  // with additional information about the mapping. Check out
  // https://www.boost.org/doc/libs/1_79_0/libs/bimap/doc/html/boost_bimap/the_tutorial/additional_information.html
  // for more details on the bimap API.
  boost::bimap<const HloInstructionNode*, const HloInstructionNode*,
               boost::bimaps::with_info<HloInstructionNodeMappingProps>>
      left_to_right_instruction_map;

  // Maps two nodes if they are not already mapped. Returns true if mapping
  // was performed.
  inline bool MapInstructionsIfAbsent(
      const HloInstructionNode* left, const HloInstructionNode* right,
      const MatcherType matcher_type,
      absl::string_view matcher_debug_info = "") {
    HloInstructionNodeMappingProps props;
    props.matcher_type = matcher_type;
    props.matcher_debug_info = std::string(matcher_debug_info);
    auto [it, inserted] = left_to_right_instruction_map.insert(
        InstructionPair(left, right, props));

    return inserted;
  }

  // Maps two CallGraphNodes if they are not already mapped. Returns true if
  // mapping was performed.
  inline bool MapComputationsIfAbsent(
      const CallGraphNode& left, const CallGraphNode& right,
      const ComputationMatchType computation_match_type) {
    HloCallGraphNodeMappingProps props;
    props.computation_match_type = computation_match_type;
    auto [it, inserted] = left_to_right_computation_map.insert(
        CallGraphNodePair(&left, &right, props));

    return inserted;
  }

  // Returns true if the left node is mapped to a right node.
  inline bool InstructionMapContainsLeft(
      const HloInstructionNode* left_node) const {
    return left_to_right_instruction_map.left.find(left_node) !=
           left_to_right_instruction_map.left.end();
  }

  // Returns true if the right node is mapped to a left node.
  inline bool InstructionMapContainsRight(
      const HloInstructionNode* right_node) const {
    return left_to_right_instruction_map.right.find(right_node) !=
           left_to_right_instruction_map.right.end();
  }
};

}  // namespace xla::hlo_diff

#endif  // XLA_HLO_TOOLS_HLO_DIFF_HLO_GUMGRAPH_MAPPINGS_H_
