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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_HLO_DIFF_EVAL_H_
#define XLA_HLO_TOOLS_HLO_DIFF_HLO_DIFF_EVAL_H_

#include <cstdint>
#include <memory>

#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla::hlo_diff {

// Evaluation metrics for the diff result.
struct DiffEval {
  // Split allegiance is defined as:
  // Two computation nodes share the same fingerprint and some instructions are
  // matched, but other instructions in the left computation are matched to
  // instructions in a different computation node in the right graph.
  int64_t num_split_allegiance_computation = 0;
  int64_t num_split_allegiance_instruction = 0;
  // Split allegiance parental is defined as:
  // Two nodes are matched, they share the same number of children and children
  // opcodes, but some of their children are not matched.
  int64_t num_split_allegiance_parental = 0;

  // Size of the diff result.
  int64_t len_left_unmatched = 0;
  int64_t len_right_unmatched = 0;
  int64_t len_changed = 0;
  int64_t len_unchanged = 0;

  // Graph node counts.
  int64_t left_node_count = 0;
  int64_t right_node_count = 0;
};

// Computes the diff evaluation metrics.
// left and right are the original graphs.
// mappings are the node mappings between the two graphs.
// diff_result contains the edit script(insert/delete/change/move) created from
// the node mappings. diff_summary summarizes the computation-based repeated
// diff patterns.
std::unique_ptr<const DiffEval> ComputeDiffEval(
    const HloGumgraph& left, const HloGumgraph& right,
    const HloGumgraphMappings& mappings, const DiffResult& diff_result,
    const DiffSummary& diff_summary);

// Logs the diff evaluation metrics.
void LogDiffEval(const DiffEval& diff_eval);

}  // namespace xla::hlo_diff

#endif  // XLA_HLO_TOOLS_HLO_DIFF_HLO_DIFF_EVAL_H_
