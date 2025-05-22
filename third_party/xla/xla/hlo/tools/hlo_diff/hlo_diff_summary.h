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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_HLO_DIFF_SUMMARY_H_
#define XLA_HLO_TOOLS_HLO_DIFF_HLO_DIFF_SUMMARY_H_

#include <cstdint>
#include <memory>
#include <ostream>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"

namespace xla {
namespace hlo_diff {

enum DiffCode : uint8_t {
  kUnchanged,
  kChanged,
  kUnmatched,
};

enum class DiffSide : std::uint8_t { kLeft, kRight };

struct ComputationSummary {
  DiffSide side;

  // Computation in the other graph that has most instructions matched.
  // Can be nullptr if no instructions are matched.
  const HloComputation* main_matched_computation = nullptr;

  // Number of instructions that are mapped to instructions in the main matched
  // computation.
  int64_t max_matched_instruction_count = 0;

  // Number of instructions that are mapped to instructions in a different
  // computation.
  int64_t split_allegiance_instruction_count = 0;

  // Fingerprint of the computation including diff.
  uint64_t diff_fingerprint = 0;

  // Whether all instructions in the computation are unchanged.
  bool all_unchanged = true;
};

// A group of left and right computations that form a diff pattern.
struct ComputationGroup {
  std::vector<const HloComputation*> left_computations;
  std::vector<const HloComputation*> right_computations;
};

// Metrics of the diff pattern.
struct DiffMetrics {
  int64_t changed_instruction_count = 0;
  int64_t left_unmatched_instruction_count = 0;
  int64_t right_unmatched_instruction_count = 0;
};

// A computation diff pattern is multiple groups of computations that have the
// same diff.
struct ComputationDiffPattern {
  uint64_t fingerprint = 0;
  std::vector<ComputationGroup> computation_groups;
  DiffMetrics diff_metrics;
};

// Teach the gunit to print the diff pattern.
void PrintTo(const ComputationDiffPattern& diff_pattern, std::ostream* os);

//  Summary of the diff result of the left and right HLO modules.
struct DiffSummary {
  // The computation diff patterns found in the diff result.
  std::vector<ComputationDiffPattern> computation_diff_patterns;

  // Summary of each computation.
  absl::flat_hash_map<const HloComputation*, const ComputationSummary>
      computation_summary;
};

// Constructs the diff summary from the diff result.
// `left_module` and `right_module` are the original HLO modules.
// `diff_result` contains the edit script(insert/delete/change/move) created
// from the node mappings.
std::unique_ptr<const DiffSummary> ConstructDiffSummary(
    const HloModule& left_module, const HloModule& right_module,
    const DiffResult& diff_result);

// Logs the diff summary.
void LogDiffSummary(const DiffSummary& diff_summary);

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_HLO_DIFF_SUMMARY_H_
