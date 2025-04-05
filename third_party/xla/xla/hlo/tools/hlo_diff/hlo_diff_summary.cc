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

#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/utils/connected_components.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace hlo_diff {
namespace {

// Returns the mapped instruction node of the given instruction in the given
// direction. Returns nullptr if the instruction is not mapped.
const HloInstructionNode* FindMappedInstructionNode(
    const HloGumgraphMappings& mappings, const HloInstructionNode* instruction,
    ComputationMappingDirection direction) {
  switch (direction) {
    case ComputationMappingDirection::kLeftToRight: {
      auto it = mappings.left_to_right_instruction_map.left.find(instruction);
      if (it != mappings.left_to_right_instruction_map.left.end()) {
        return it->second;
      }
      break;
    }
    case ComputationMappingDirection::kRightToLeft: {
      auto it = mappings.left_to_right_instruction_map.right.find(instruction);
      if (it != mappings.left_to_right_instruction_map.right.end()) {
        return it->second;
      }
      break;
    }
  }
  return nullptr;
}

// Result of finding the main matched computation.
struct MainMatchedComputationResult {
  const HloComputation* main_matched_computation = nullptr;
  const int max_matched_instruction_count = 0;
  const int split_allegiance_instruction_count = 0;
};

// Returns the main matched computation of the given computation in the given
// direction. A computation is considered as the main matched computation if it
// has the most matched instructions.
MainMatchedComputationResult FindMainMatchedComputation(
    const HloComputation* computation, const HloGumgraph& graph,
    const HloGumgraphMappings& mappings,
    ComputationMappingDirection direction) {
  ComputationSummary result;
  absl::flat_hash_map<const HloComputation*, int> matched_instruction_count;
  int max_count = 0;
  int mapped_instruction_count = 0;
  const HloComputation* main_matched_computation = nullptr;
  for (const HloInstruction* instruction : computation->instructions()) {
    if (const HloInstructionNode* const mapped_instruction_node =
            FindMappedInstructionNode(mappings, graph.GetNode(instruction),
                                      direction);
        mapped_instruction_node != nullptr) {
      ++mapped_instruction_count;
      const HloComputation* right_computation =
          mapped_instruction_node->instruction->parent();
      const int count = ++matched_instruction_count[right_computation];
      if (count > max_count) {
        max_count = count;
        main_matched_computation = right_computation;
      }
    }
  }
  return {.main_matched_computation = main_matched_computation,
          .max_matched_instruction_count = max_count,
          .split_allegiance_instruction_count =
              mapped_instruction_count - max_count};
}

uint64_t GetDiffTypeFingerprint(
    const HloInstruction* instruction,
    const absl::flat_hash_set<const HloInstruction*>& changed_instructions,
    const absl::flat_hash_set<const HloInstruction*>& unmatched_instructions) {
  if (changed_instructions.contains(instruction)) {
    return DiffCode::kChanged;
  }
  if (unmatched_instructions.contains(instruction)) {
    return DiffCode::kUnmatched;
  }
  return DiffCode::kUnchanged;
}

struct DiffFingerprint {
  bool all_unchanged;
  uint64_t diff_fingerprint;
};

DiffFingerprint ComputationDiffFingerprint(
    const xla::HloComputation* computation,
    const absl::flat_hash_set<const HloInstruction*>& changed_instructions,
    const absl::flat_hash_set<const HloInstruction*>& unmatched_instructions) {
  absl::flat_hash_map<const HloInstruction*, uint64_t> subgraph_fingerprint;
  DiffFingerprint result;
  bool all_unchanged = true;
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    uint64_t fp = static_cast<uint64_t>(instruction->opcode());
    uint64_t diff_type_fp = GetDiffTypeFingerprint(
        instruction, changed_instructions, unmatched_instructions);
    all_unchanged = all_unchanged && (diff_type_fp == DiffCode::kUnchanged);
    fp = tsl::FingerprintCat64(fp, diff_type_fp);
    for (const HloInstruction* operand : instruction->operands()) {
      fp = tsl::FingerprintCat64(fp, subgraph_fingerprint.at(operand));
    }
    // TODO(b/394201811): Make sure no fingerprint collision.
    subgraph_fingerprint[instruction] = fp;
  }
  result.all_unchanged = all_unchanged;
  result.diff_fingerprint =
      subgraph_fingerprint.at(computation->root_instruction());
  return result;
}

// Split the computations into left and right computations.
ComputationGroup SplitComputations(
    const std::vector<const HloComputation*>& computations,
    const absl::flat_hash_map<const HloComputation*, const ComputationSummary>&
        computation_summaries) {
  ComputationGroup result;
  for (const HloComputation* computation : computations) {
    if (auto it = computation_summaries.find(computation);
        it != computation_summaries.end()) {
      if (it->second.direction == ComputationMappingDirection::kLeftToRight) {
        result.left_computations.push_back(computation);
      } else {
        result.right_computations.push_back(computation);
      }
    }
  }
  return result;
}

// Returns the connected components of the given computation summary.
absl::flat_hash_map<uint64_t, std::vector<ComputationGroup>>
FindConnectedComponents(
    absl::flat_hash_map<const HloComputation*, const ComputationSummary>
        computation_summary) {
  ConnectedComponentsFinder cc;
  absl::flat_hash_map<uint64_t, std::vector<ComputationGroup>> result;
  for (const auto& [computation, computation_match_info] :
       computation_summary) {
    if (computation_match_info.main_matched_computation != nullptr) {
      cc.AddEdge(computation, computation_match_info.main_matched_computation);
    }
  }
  std::vector<std::vector<const HloComputation*>> connected_component_groups =
      cc.FindConnectedComponents();

  for (const auto& component_group : connected_component_groups) {
    bool all_unchanged = true;
    for (const auto& computation : component_group) {
      all_unchanged =
          all_unchanged && computation_summary.at(computation).all_unchanged;
    }
    if (all_unchanged) {
      continue;
    }
    std::vector<const HloComputation*> sorted_component_group(component_group);
    std::sort(sorted_component_group.begin(), sorted_component_group.end(),
              [&](const HloComputation* a, const HloComputation* b) {
                return computation_summary.at(a).diff_fingerprint <
                       computation_summary.at(b).diff_fingerprint;
              });
    uint64_t fingerprint = 0;
    for (const auto& computation : sorted_component_group) {
      fingerprint = tsl::FingerprintCat64(
          fingerprint, computation_summary.at(computation).diff_fingerprint);
    }
    result[fingerprint].push_back(
        SplitComputations(sorted_component_group, computation_summary));
  }
  return result;
}

absl::flat_hash_map<const HloComputation*, const ComputationSummary>
SummarizeAllComputationsInGraph(
    const HloGumgraph& graph, const HloGumgraphMappings& mappings,
    const DiffResult& diff_result,
    const absl::flat_hash_set<const HloInstruction*>& changed_instructions,
    const absl::flat_hash_set<const HloInstruction*>& unmatched_instructions,
    ComputationMappingDirection direction) {
  absl::flat_hash_map<const HloComputation*, const ComputationSummary> result;
  for (auto const& [computation, _] : graph.AllComputationProps()) {
    const MainMatchedComputationResult mmc =
        FindMainMatchedComputation(computation, graph, mappings, direction);
    DiffFingerprint dfp = ComputationDiffFingerprint(
        computation, changed_instructions, unmatched_instructions);
    result.insert(
        {computation,
         {
             .direction = direction,
             .main_matched_computation = mmc.main_matched_computation,
             .max_matched_instruction_count = mmc.max_matched_instruction_count,
             .split_allegiance_instruction_count =
                 mmc.split_allegiance_instruction_count,
             .diff_fingerprint = dfp.diff_fingerprint,
             .all_unchanged = dfp.all_unchanged,
         }});
  }
  return result;
}
}  // namespace

std::unique_ptr<const DiffSummary> ConstructDiffSummary(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const HloGumgraphMappings& mappings, const DiffResult& diff_result) {
  auto summary = std::make_unique<DiffSummary>();
  absl::flat_hash_set<const HloInstruction*> left_changed_instructions;
  absl::flat_hash_set<const HloInstruction*> right_changed_instructions;
  absl::flat_hash_set<const HloInstruction*> left_unmatched_instructions;
  absl::flat_hash_set<const HloInstruction*> right_unmatched_instructions;
  for (auto const& [left, right] : diff_result.changed_instructions) {
    left_changed_instructions.insert(left);
    right_changed_instructions.insert(right);
  }
  left_unmatched_instructions.insert(
      diff_result.left_module_unmatched_instructions.begin(),
      diff_result.left_module_unmatched_instructions.end());
  right_unmatched_instructions.insert(
      diff_result.right_module_unmatched_instructions.begin(),
      diff_result.right_module_unmatched_instructions.end());
  summary->computation_summary.merge(SummarizeAllComputationsInGraph(
      left_graph, mappings, diff_result, left_changed_instructions,
      left_unmatched_instructions, ComputationMappingDirection::kLeftToRight));
  summary->computation_summary.merge(SummarizeAllComputationsInGraph(
      right_graph, mappings, diff_result, right_changed_instructions,
      right_unmatched_instructions, ComputationMappingDirection::kRightToLeft));

  // Group the computations by their diff fingerprint.
  summary->grouped_computations =
      FindConnectedComponents(summary->computation_summary);

  return summary;
}

void LogDiffSummary(const DiffSummary& diff_summary) {
  // Log the connected components repeated more than 3 times.
  LOG(INFO) << "Find Repeated Connected Components: ";
  for (const auto& [fingerprint, computation_groups] :
       diff_summary.grouped_computations) {
    if (computation_groups.size() < 3) {
      continue;
    }
    LOG(INFO) << computation_groups.size()
              << " Repeated Connected Components Fingerprint: " << fingerprint;
    int i = 0;
    for (const auto& computation_group : computation_groups) {
      ++i;
      std::string computations_str;
      for (const HloComputation* computation :
           computation_group.left_computations) {
        absl::StrAppend(&computations_str,
                        absl::StrFormat("L: %s, ", computation->name()));
      }
      for (const HloComputation* computation :
           computation_group.right_computations) {
        absl::StrAppend(&computations_str,
                        absl::StrFormat("R: %s, ", computation->name()));
      }
      LOG(INFO) << computations_str;
      if (i >= 5) {
        LOG(INFO) << "...";
        break;
      }
    }
  }
}

}  // namespace hlo_diff
}  // namespace xla
