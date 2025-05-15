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

#include "xla/hlo/tools/hlo_diff/hlo_diff_eval.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla::hlo_diff {
namespace {

// Counts the number of split allegiance in the diff result.
// Split allegiance is defined as:
// Two computation nodes share the same fingerprint and some instructions are
// matched, but other instructions in the left computation are matched to
// instructions in a different computation node in the right graph.
// Returns a pair of the number of computations that are split allegiance, and
// the accumulated number of minimum instructions that are mismatched inside.
std::pair<int64_t, int64_t> CountSplitAllegiance(
    const HloGumgraph& left, const HloGumgraph& right,
    const DiffSummary& diff_summary) {
  int64_t split_allegiance_computation_count = 0;
  int64_t split_allegiance_instruction_count = 0;
  for (auto const& [computation, computation_props] :
       left.AllComputationProps()) {
    if (auto it = diff_summary.computation_summary.find(computation);
        it != diff_summary.computation_summary.end()) {
      const ComputationSummary& cmi = it->second;
      if (cmi.split_allegiance_instruction_count > 0 &&
          right.AllComputationProps()
                  .at(cmi.main_matched_computation)
                  .fingerprint == computation_props.fingerprint) {
        ++split_allegiance_computation_count;
        split_allegiance_instruction_count +=
            cmi.split_allegiance_instruction_count;
      }
    }
  }
  return std::make_pair(split_allegiance_computation_count,
                        split_allegiance_instruction_count);
}

// Counts the number of split allegiance parental in the diff result.
// Split allegiance parental is defined as:
// Two nodes are matched, they share the same number of children and children
// opcodes, but some of their children are not matched.
int64_t CountSplitAllegianceParental(const HloGumgraph& left,
                                     const HloGumgraph& right,
                                     const HloGumgraphMappings& mappings) {
  int64_t count = 0;
  for (const auto it : mappings.left_to_right_instruction_map.left) {
    if (it.first->children.size() != it.second->children.size()) {
      continue;
    }
    bool children_opcode_mismatch = false;
    for (int i = 0; i < it.first->children.size(); ++i) {
      if (it.first->children[i]->instruction->opcode() !=
          it.second->children[i]->instruction->opcode()) {
        children_opcode_mismatch = true;
        break;
      }
    }
    if (children_opcode_mismatch) {
      continue;
    }
    for (int i = 0; i < it.first->children.size(); ++i) {
      if (auto cit = mappings.left_to_right_instruction_map.left.find(
              it.first->children[i]);
          cit == mappings.left_to_right_instruction_map.left.end() ||
          cit->second != it.second->children[i]) {
        count++;
        // LOG(INFO) << it.first->instruction->name() << " has split child: "
        //           << it.first->children[i]->instruction->name();
      }
    }
  }
  return count;
}

}  // namespace

std::unique_ptr<const DiffEval> ComputeDiffEval(
    const HloGumgraph& left, const HloGumgraph& right,
    const HloGumgraphMappings& mappings, const DiffResult& diff_result,
    const DiffSummary& diff_summary) {
  LOG(INFO) << "Evaluating diff result";
  auto eval = std::make_unique<DiffEval>();
  auto [split_allegiance_computation_count,
        split_allegiance_instruction_count] =
      CountSplitAllegiance(left, right, diff_summary);
  eval->num_split_allegiance_computation = split_allegiance_computation_count;
  eval->num_split_allegiance_instruction = split_allegiance_instruction_count;
  eval->num_split_allegiance_parental =
      CountSplitAllegianceParental(left, right, mappings);

  eval->len_left_unmatched =
      diff_result.left_module_unmatched_instructions.size();
  eval->len_right_unmatched =
      diff_result.right_module_unmatched_instructions.size();
  eval->len_changed = diff_result.changed_instructions.size();
  eval->len_unchanged = diff_result.unchanged_instructions.size();

  eval->left_node_count = left.GetNodeCount();
  eval->right_node_count = right.GetNodeCount();

  return eval;
}

void LogDiffEval(const DiffEval& diff_eval) {
  LOG(INFO) << "Split Allegiance Computation: "
            << diff_eval.num_split_allegiance_computation;
  LOG(INFO) << "Split Allegiance Instruction: "
            << diff_eval.num_split_allegiance_instruction;
  LOG(INFO) << "Split Allegiance Parental: "
            << diff_eval.num_split_allegiance_parental;
}

}  // namespace xla::hlo_diff
