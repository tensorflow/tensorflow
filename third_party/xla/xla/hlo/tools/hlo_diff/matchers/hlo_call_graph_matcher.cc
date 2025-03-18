// Copyright 2025 The OpenXLA Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/hlo/tools/hlo_diff/matchers/hlo_call_graph_matcher.h"

#include <cstdint>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/service/call_graph.h"
#include "xla/shape.h"

namespace xla::hlo_diff {
namespace {

using VisitorFunction = absl::FunctionRef<void(const CallGraphNode&)>;

// Sort the callees of the caller computation in order of the caller
// computation's post order instructions.
std::vector<std::pair<const HloInstruction*, const CallGraphNode*>>
SortCalleesByCallerComputationPostOrder(
    const absl::flat_hash_set<const CallGraphNode*>& callees,
    HloComputation::CachingPostOrder& cpo, const HloGumgraph& gumgraph) {
  std::vector<std::pair<const HloInstruction*, const CallGraphNode*>> callsites;
  for (auto* instruction : cpo.PostOrder()) {
    for (const auto* computation : instruction->called_computations()) {
      const CallGraphNode* callee =
          &gumgraph.GetCallGraph().GetNode(computation);
      if (callees.contains(callee)) {
        callsites.push_back(std::make_pair(instruction, callee));
      }
    }
  }

  return callsites;
}

// Match left and right callee computations based on call_site instruction
// attributes: ex. op_code and instruction position in the caller computation
// post-order.
void MapComputationCalleesWithSameFingerprintOrProgramShape(
    const absl::flat_hash_set<const CallGraphNode*>& left_callees,
    const absl::flat_hash_set<const CallGraphNode*>& right_callees,
    HloComputation::CachingPostOrder& left_cpo,
    HloComputation::CachingPostOrder& right_cpo, const HloGumgraph& left,
    const HloGumgraph& right, HloGumgraphMappings& mappings,
    const ComputationMatchType computation_match_type) {
  // Don't attempt to match if there are different number of callee
  // computations as its difficult to disambiguate between the callees.
  if (left_callees.size() != right_callees.size()) {
    return;
  }

  // Match computations if there is exactly one callee on both sides.
  if (left_callees.size() == 1 && right_callees.size() == 1) {
    mappings.MapComputationsIfAbsent(*(*left_callees.begin()),
                                     *(*right_callees.begin()),
                                     computation_match_type);
    return;
  }

  // For multiple callees, match them in order of the caller computation's post
  // order instructions.
  std::vector<std::pair<const HloInstruction*, const CallGraphNode*>>
      left_callsites =
          SortCalleesByCallerComputationPostOrder(left_callees, left_cpo, left);
  std::vector<std::pair<const HloInstruction*, const CallGraphNode*>>
      right_callsites = SortCalleesByCallerComputationPostOrder(
          right_callees, right_cpo, right);

  // Don't attempt to match if there are different number of call sites as its
  // difficult to disambiguate between the call sites.
  if (left_callsites.size() != right_callsites.size()) {
    return;
  }

  // Verify that all call sites instruction op codes and metadata op_name match.
  for (int i = 0; i < left_callsites.size(); ++i) {
    if (left_callsites[i].first->opcode() !=
        right_callsites[i].first->opcode()) {
      return;
    }

    if (left_callsites[i].first->metadata().op_name() !=
        right_callsites[i].first->metadata().op_name()) {
      return;
    }
  }

  for (int i = 0; i < left_callsites.size(); ++i) {
    mappings.MapComputationsIfAbsent(*left_callsites[i].second,
                                     *right_callsites[i].second,
                                     computation_match_type);
  }
}

void MapCalledComputations(const HloInstruction* left_instruction,
                           const HloInstruction* right_instruction,
                           const HloGumgraph& left, const HloGumgraph& right,
                           HloGumgraphMappings& mappings) {
  for (int i = 0; i < left_instruction->called_computations().size(); ++i) {
    mappings.MapComputationsIfAbsent(
        left.GetCallGraph().GetNode(
            left_instruction->called_computations().at(i)),
        right.GetCallGraph().GetNode(
            right_instruction->called_computations().at(i)),
        ComputationMatchType::kSignature);
  }
}

// Process a single computation (CallGraphNode) in the left call graph. For each
// called computation in this computation, attempt to find a matching
// computation on the right call graph.
void ProcessCallGraphNode(const CallGraphNode& left_computation,
                          const HloGumgraph& left, const HloGumgraph& right,
                          HloGumgraphMappings& mappings) {
  // Only match called computations if current computation is already matched.
  auto it = mappings.left_to_right_computation_map.left.find(&left_computation);
  if (it == mappings.left_to_right_computation_map.left.end() ||
      left_computation.callees().empty()) {
    return;
  }

  const CallGraphNode* right_computation = it->second;
  HloComputation::CachingPostOrder left_cpo(left_computation.computation());
  HloComputation::CachingPostOrder right_cpo(right_computation->computation());

  // Phase 1: Match called computations to computations with matching
  // computation fingerprints, i.e. exact matches.
  absl::flat_hash_map<uint64_t, absl::flat_hash_set<const CallGraphNode*>>
      left_callees_by_fingerprint, right_callees_by_fingerprint;
  for (const HloComputation* callee : left_computation.callees()) {
    CallGraphNodeProps left_props = left.AllComputationProps().at(callee);
    left_callees_by_fingerprint[left_props.fingerprint].insert(
        left_props.call_graph_node);
  }
  for (const HloComputation* callee : right_computation->callees()) {
    CallGraphNodeProps right_props = right.AllComputationProps().at(callee);
    right_callees_by_fingerprint[right_props.fingerprint].insert(
        right_props.call_graph_node);
  }

  for (const auto& [fingerprint, left_callees] : left_callees_by_fingerprint) {
    if (auto right_it = right_callees_by_fingerprint.find(fingerprint);
        right_it != right_callees_by_fingerprint.end()) {
      const absl::flat_hash_set<const CallGraphNode*>& right_callees =
          right_it->second;
      MapComputationCalleesWithSameFingerprintOrProgramShape(
          left_callees, right_callees, left_cpo, right_cpo, left, right,
          mappings, ComputationMatchType::kExact);
    }
  }

  // Phase2: Match left called computations to right computations if their
  // callsite instructions have matching opcodes and metadata op_name.
  absl::flat_hash_map<std::pair<HloOpcode, std::string>,
                      std::vector<const HloInstruction*>>
      left_instructions_by_op, right_instructions_by_op;
  // First we filter out instructions whose called computations are already
  // matched.
  for (const HloInstruction* instruction : left_cpo.PostOrder()) {
    bool all_called_computations_matched = true;
    for (const HloComputation* callee : instruction->called_computations()) {
      if (auto left_it = mappings.left_to_right_computation_map.left.find(
              &left.GetCallGraph().GetNode(callee));
          left_it == mappings.left_to_right_computation_map.left.end()) {
        all_called_computations_matched = false;
        break;
      }
    }

    if (!all_called_computations_matched) {
      std::pair op_code_and_name = std::make_pair(
          instruction->opcode(), instruction->metadata().op_name());
      left_instructions_by_op[op_code_and_name].push_back(instruction);
    }
  }

  for (const HloInstruction* instruction : right_cpo.PostOrder()) {
    bool all_called_computations_matched = true;
    for (const HloComputation* callee : instruction->called_computations()) {
      if (auto right_it = mappings.left_to_right_computation_map.right.find(
              &right.GetCallGraph().GetNode(callee));
          right_it == mappings.left_to_right_computation_map.right.end()) {
        all_called_computations_matched = false;
        break;
      }
    }

    if (!all_called_computations_matched) {
      std::pair op_code_and_name = std::make_pair(
          instruction->opcode(), instruction->metadata().op_name());
      right_instructions_by_op[op_code_and_name].push_back(instruction);
    }
  }

  // Match called computations if their callsite instructions have matching
  // opcodes and metadata op_name and there is exactly one called computation
  // on both sides.
  for (const auto& [op, left_instructions] : left_instructions_by_op) {
    auto right_it = right_instructions_by_op.find(op);
    if (right_it == right_instructions_by_op.end()) {
      continue;
    }

    std::vector<const HloInstruction*> right_instructions = right_it->second;
    if (left_instructions.size() == 1 && right_instructions.size() == 1) {
      MapCalledComputations(left_instructions[0], right_instructions[0], left,
                            right, mappings);
    } else {
      // Even if there are multiple call sites with matching opcodes and
      // metadata op_name, we still attempt to match the called computations if
      // they are of the same size, but only for While opcodes.
      switch (op.first) {
        case HloOpcode::kWhile: {
          if (left_instructions.size() != right_instructions.size()) {
            break;
          }

          for (int i = 0; i < left_instructions.size(); ++i) {
            MapCalledComputations(left_instructions[i], right_instructions[i],
                                  left, right, mappings);
          }
          break;
        }
        default:
          break;
      }
    }
  }

  // Phase 3: Match children computations with matching opcode, metadata op-name
  // and program shapes as signature matches.
  absl::flat_hash_map<std::string, absl::flat_hash_set<const CallGraphNode*>>
      unmatched_left_callees, unmatched_right_callees;
  for (const HloComputation* callee : left_computation.callees()) {
    if (auto left_it = mappings.left_to_right_computation_map.left.find(
            &left.GetCallGraph().GetNode(callee));
        left_it == mappings.left_to_right_computation_map.left.end()) {
      const CallGraphNode& callee_node = left.GetCallGraph().GetNode(callee);
      std::string opcode_and_name;
      if (!callee_node.caller_callsites().empty()) {
        const HloInstruction* caller_instruction =
            callee_node.caller_callsites()[0].instruction();
        opcode_and_name =
            absl::StrCat(caller_instruction->opcode(),
                         "::", caller_instruction->metadata().op_name());
      } else {
        LOG(WARNING) << "Callee node " << callee_node.computation()->name()
                     << " has no caller callsites";
      }
      std::string opcode_name_shape = absl::StrCat(
          opcode_and_name,
          "::", callee->ComputeProgramShape(/*include ids=*/false).ToString());
      unmatched_left_callees[opcode_name_shape].insert(&callee_node);
    }
  }
  for (const HloComputation* callee : right_computation->callees()) {
    if (auto right_it = mappings.left_to_right_computation_map.right.find(
            &right.GetCallGraph().GetNode(callee));
        right_it == mappings.left_to_right_computation_map.right.end()) {
      const CallGraphNode& callee_node = right.GetCallGraph().GetNode(callee);
      std::string opcode_and_name;
      if (callee_node.caller_callsites().size() == 1) {
        const HloInstruction* caller_instruction =
            callee_node.caller_callsites()[0].instruction();
        opcode_and_name =
            absl::StrCat(caller_instruction->opcode(),
                         "::", caller_instruction->metadata().op_name());
      }
      std::string key = absl::StrCat(
          opcode_and_name,
          "::", callee->ComputeProgramShape(/*include ids=*/false).ToString());
      unmatched_right_callees[key].insert(&callee_node);
    }
  }

  for (const auto& [shape, left_calleees] : unmatched_left_callees) {
    if (auto right_it = unmatched_right_callees.find(shape);
        right_it != unmatched_right_callees.end()) {
      const absl::flat_hash_set<const CallGraphNode*>&
          program_shape_matched_right_calleees = right_it->second;
      MapComputationCalleesWithSameFingerprintOrProgramShape(
          left_calleees, program_shape_matched_right_calleees, left_cpo,
          right_cpo, left, right, mappings, ComputationMatchType::kSignature);
    }
  }
}

// Visits all CallGraphNodes in the call graph in BFS order.
void VisitCallGraphNodesBfs(const CallGraph& call_graph,
                            const CallGraphNode& root,
                            VisitorFunction visit_fn) {
  absl::flat_hash_set<const CallGraphNode*> visited;
  std::queue<const CallGraphNode*> queue;
  queue.push(&root);

  while (!queue.empty()) {
    const CallGraphNode* current_node = queue.front();
    queue.pop();

    if (!visited.insert(current_node).second) {
      continue;
    }

    visit_fn(*current_node);

    for (const HloComputation* callee : current_node->callees()) {
      queue.push(&call_graph.GetNode(callee));
    }
  }
}

}  // namespace

void MatchCallGraphs(const HloGumgraph& left, const HloGumgraph& right,
                     HloGumgraphMappings& mappings) {
  // Match the entry computations as signature matches. This optimizes for the
  // common case, i.e. users comparing similar programs whose input/output
  // parameters are often identical or very similar.
  ComputationMatchType entry_computation_match_type =
      ComputationMatchType::kSignature;
  if (left.AllComputationProps()
          .at(left.GetHloModule().entry_computation())
          .fingerprint == right.AllComputationProps()
                              .at(right.GetHloModule().entry_computation())
                              .fingerprint) {
    entry_computation_match_type = ComputationMatchType::kExact;
  }
  mappings.MapComputationsIfAbsent(
      left.GetCallGraph().GetNode(left.GetHloModule().entry_computation()),
      right.GetCallGraph().GetNode(right.GetHloModule().entry_computation()),
      entry_computation_match_type);

  // Traverse the call graph of the left HloGumgraph in BFS order. For each
  // visited computation node in the left call graph we attempt to find a
  // matching computation node on the right call graph. Two computation nodes
  // are only matched if their parent computations are already matched.
  VisitCallGraphNodesBfs(
      left.GetCallGraph(),
      left.GetCallGraph().GetNode(left.GetHloModule().entry_computation()),
      [&](const CallGraphNode& left_node) {
        return ProcessCallGraphNode(left_node, left, right, mappings);
      });

  int signature_match_count = 0, exact_match_count = 0;
  for (auto it = mappings.left_to_right_computation_map.left.begin();
       it != mappings.left_to_right_computation_map.left.end(); ++it) {
    if (it->info.computation_match_type == ComputationMatchType::kSignature) {
      ++signature_match_count;
    } else {
      ++exact_match_count;
    }
  }
  LOG(INFO) << "Finished matching call graphs for "
            << left.GetHloModule().name() << ": "
            << left.GetCallGraph().nodes().size() << " and "
            << right.GetHloModule().name() << ": "
            << right.GetCallGraph().nodes().size()
            << ". Total signature matched computations: "
            << signature_match_count
            << ". Total exact matched computations: " << exact_match_count;
}

}  // namespace xla::hlo_diff
