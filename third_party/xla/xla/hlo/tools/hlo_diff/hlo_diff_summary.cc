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
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "boost/bimap.hpp"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/utils/connected_components.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace hlo_diff {
namespace {

using InstructionBimap =
    boost::bimap<const HloInstruction*, const HloInstruction*>;

InstructionBimap ConstructInstructionBimap(const DiffResult& diff_result) {
  InstructionBimap mapping;
  for (const auto& [left, right] : diff_result.unchanged_instructions) {
    mapping.insert({left, right});
  }
  for (const auto& [left, right] : diff_result.changed_instructions) {
    mapping.insert({left, right});
  }
  return mapping;
}

// Returns the mapped instruction node of the given instruction in the given
// direction. Returns nullptr if the instruction is not mapped.
const HloInstruction* FindMappedInstruction(const InstructionBimap& mapping,
                                            const HloInstruction* instruction,
                                            DiffSide side) {
  switch (side) {
    case DiffSide::kLeft: {
      auto it = mapping.left.find(instruction);
      if (it != mapping.left.end()) {
        return it->second;
      }
      break;
    }
    case DiffSide::kRight: {
      auto it = mapping.right.find(instruction);
      if (it != mapping.right.end()) {
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
  int max_matched_instruction_count = 0;
  int split_allegiance_instruction_count = 0;
};

// Returns the main matched computation of the given computation in the given
// direction. A computation is considered as the main matched computation if it
// has the most matched instructions.
MainMatchedComputationResult FindMainMatchedComputation(
    const HloComputation* computation, const InstructionBimap& mapping,
    DiffSide side) {
  absl::flat_hash_map<const HloComputation*, int> matched_instruction_count;
  int max_count = 0;
  int mapped_instruction_count = 0;
  const HloComputation* main_matched_computation = nullptr;
  for (const HloInstruction* instruction : computation->instructions()) {
    if (const HloInstruction* const mapped_instruction =
            FindMappedInstruction(mapping, instruction, side);
        mapped_instruction != nullptr) {
      ++mapped_instruction_count;
      const HloComputation* right_computation = mapped_instruction->parent();
      const int count = ++matched_instruction_count[right_computation];
      if (count > max_count) {
        max_count = count;
        main_matched_computation = right_computation;
      }
    }
  }
  MainMatchedComputationResult result;
  result.main_matched_computation = main_matched_computation;
  result.max_matched_instruction_count = max_count;
  result.split_allegiance_instruction_count =
      mapped_instruction_count - max_count;
  return result;
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
      if (it->second.side == DiffSide::kLeft) {
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

DiffMetrics GetDiffMetrics(const ComputationGroup& computation_group,
                           const DiffResult& diff_result) {
  DiffMetrics result;
  for (const HloComputation* computation :
       computation_group.left_computations) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (diff_result.changed_instructions.contains(instruction)) {
        ++result.changed_instruction_count;
      } else if (diff_result.left_module_unmatched_instructions.contains(
                     instruction)) {
        ++result.left_unmatched_instruction_count;
      }
    }
  }
  for (const HloComputation* computation :
       computation_group.right_computations) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (diff_result.changed_instructions.contains(instruction)) {
        ++result.changed_instruction_count;
      } else if (diff_result.right_module_unmatched_instructions.contains(
                     instruction)) {
        ++result.right_unmatched_instruction_count;
      }
    }
  }
  return result;
}

std::vector<ComputationDiffPattern> FindComputationDiffPatterns(
    const absl::flat_hash_map<const HloComputation*, const ComputationSummary>&
        computation_summary,
    const DiffResult& diff_result) {
  std::vector<ComputationDiffPattern> result;
  absl::flat_hash_map<uint64_t, std::vector<ComputationGroup>>
      connected_components = FindConnectedComponents(computation_summary);
  for (const auto& [fingerprint, computation_groups] : connected_components) {
    ComputationDiffPattern diff_pattern;
    diff_pattern.fingerprint = fingerprint;
    diff_pattern.computation_groups = computation_groups;
    diff_pattern.diff_metrics =
        GetDiffMetrics(computation_groups[0], diff_result);
    result.push_back(std::move(diff_pattern));
  }
  return result;
}

// Summarizes all computations in the given graph.
absl::flat_hash_map<const HloComputation*, const ComputationSummary>
SummarizeAllComputationsInGraph(
    const HloModule& module, const InstructionBimap& mapping,
    const absl::flat_hash_set<const HloInstruction*>& changed_instructions,
    const absl::flat_hash_set<const HloInstruction*>& unmatched_instructions,
    DiffSide side) {
  absl::flat_hash_map<const HloComputation*, const ComputationSummary> result;
  for (const HloComputation* computation : module.computations()) {
    const MainMatchedComputationResult mmc =
        FindMainMatchedComputation(computation, mapping, side);
    DiffFingerprint dfp = ComputationDiffFingerprint(
        computation, changed_instructions, unmatched_instructions);
    ComputationSummary summary;
    summary.side = side;
    summary.main_matched_computation = mmc.main_matched_computation;
    summary.max_matched_instruction_count = mmc.max_matched_instruction_count;
    summary.split_allegiance_instruction_count =
        mmc.split_allegiance_instruction_count;
    summary.diff_fingerprint = dfp.diff_fingerprint;
    summary.all_unchanged = dfp.all_unchanged;
    result.insert({computation, summary});
  }
  return result;
}
}  // namespace

std::unique_ptr<const DiffSummary> ConstructDiffSummary(
    const HloModule& left_module, const HloModule& right_module,
    const DiffResult& diff_result) {
  auto summary = std::make_unique<DiffSummary>();
  absl::flat_hash_set<const HloInstruction*> left_changed_instructions;
  absl::flat_hash_set<const HloInstruction*> right_changed_instructions;
  absl::flat_hash_set<const HloInstruction*> left_unmatched_instructions;
  absl::flat_hash_set<const HloInstruction*> right_unmatched_instructions;
  for (auto const& [left, right] : diff_result.changed_instructions) {
    left_changed_instructions.insert(left);
    right_changed_instructions.insert(right);
  }
  InstructionBimap mapping = ConstructInstructionBimap(diff_result);
  left_unmatched_instructions.insert(
      diff_result.left_module_unmatched_instructions.begin(),
      diff_result.left_module_unmatched_instructions.end());
  right_unmatched_instructions.insert(
      diff_result.right_module_unmatched_instructions.begin(),
      diff_result.right_module_unmatched_instructions.end());
  summary->computation_summary.merge(SummarizeAllComputationsInGraph(
      left_module, mapping, left_changed_instructions,
      left_unmatched_instructions, DiffSide::kLeft));
  summary->computation_summary.merge(SummarizeAllComputationsInGraph(
      right_module, mapping, right_changed_instructions,
      right_unmatched_instructions, DiffSide::kRight));

  // Group the computations by their diff fingerprint.
  summary->computation_diff_patterns =
      FindComputationDiffPatterns(summary->computation_summary, diff_result);

  return summary;
}

void LogDiffSummary(const DiffSummary& diff_summary) {
  // Log the connected components repeated more than 3 times.
  LOG(INFO) << "Find Repeated Diff Patterns: ";
  for (const ComputationDiffPattern& diff_pattern :
       diff_summary.computation_diff_patterns) {
    if (diff_pattern.computation_groups.size() < 3) {
      continue;
    }
    LOG(INFO) << diff_pattern.computation_groups.size()
              << " Repeated Diff Pattern Fingerprint: "
              << diff_pattern.fingerprint;
    int i = 0;
    for (const auto& computation_group : diff_pattern.computation_groups) {
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

void PrintTo(const ComputationDiffPattern& diff_pattern, std::ostream* os) {
  *os << "{ fingerprint: " << diff_pattern.fingerprint;
  for (const auto& computation_group : diff_pattern.computation_groups) {
    *os << ", computation_groups: "
        << "{ L: ";
    for (const HloComputation* computation :
         computation_group.left_computations) {
      *os << absl::StrFormat("%s ", computation->name());
    }
    *os << ", R: ";
    for (const HloComputation* computation :
         computation_group.right_computations) {
      *os << absl::StrFormat("%s ", computation->name());
    }
    *os << " }";
  }
  *os << ", diff_metrics: {"
      << diff_pattern.diff_metrics.changed_instruction_count << " changed, "
      << diff_pattern.diff_metrics.left_unmatched_instruction_count
      << " left unmatched, "
      << diff_pattern.diff_metrics.right_unmatched_instruction_count
      << " right unmatched }";
  *os << " }";
}

}  // namespace hlo_diff
}  // namespace xla
