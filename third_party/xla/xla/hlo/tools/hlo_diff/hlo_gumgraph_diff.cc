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

#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_eval.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/matchers/hlo_call_graph_matcher.h"
#include "xla/hlo/tools/hlo_diff/matchers/hlo_computation_graph_matcher.h"
#include "xla/hlo/tools/hlo_diff/matchers/hlo_gumgraph_matcher.h"
#include "xla/service/call_graph.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

absl::StatusOr<std::unique_ptr<const HloGumgraphMappings>> FindMappings(
    const HloGumgraph& left, const HloGumgraph& right,
    const MatchOptions& options = {}) {
  LOG(INFO) << "Running Matchers";
  auto mappings = std::make_unique<HloGumgraphMappings>();
  mappings->MapInstructionsIfAbsent(&left.GetRoot(), &right.GetRoot(),
                                    MatcherType::kManual);

  MatchCallGraphs(left, right, *mappings);

  TF_RETURN_IF_ERROR(left.GetCallGraph().VisitNodes(
      [&](const CallGraphNode& node) {
        if (auto it = mappings->left_to_right_computation_map.left.find(&node);
            it != mappings->left_to_right_computation_map.left.end()) {
          MatchComputationGraphs(left, right, node, *it->second, *mappings);
        }
        return absl::OkStatus();
      },
      /*visit_unreachable_nodes=*/true));

  std::vector<std::unique_ptr<HloGumgraphMatcher>> matchers;
  matchers.push_back(
      std::make_unique<GreedySubGraphExactMatcher>(&left, &right));
  matchers.push_back(std::make_unique<GreedyTopDownMatcher>(
      &left, &right, /*require_same_children=*/true));
  matchers.push_back(
      std::make_unique<GreedyLimitedCandidatesBottomUpMatcher>(&left, &right));
  if (options.use_top_down_matcher) {
    matchers.push_back(std::make_unique<GreedyTopDownMatcher>(
        &left, &right, /*require_same_children=*/true));
    matchers.push_back(std::make_unique<GreedyTopDownMatcher>(&left, &right));
  }

  for (auto& matcher : matchers) {
    matcher->Match(*mappings);
  }

  return mappings;
}
}  // namespace

absl::StatusOr<HloGumgraphDiffResults> ComputeDiff(const HloModule& left,
                                                   const HloModule& right,
                                                   const DiffOptions& options,
                                                   bool run_eval) {
  LOG(INFO) << "Initializing left module graph";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<const HloGumgraph> left_graph,
                      HloGumgraph::Create(&left, options.fingerprint_options));
  LOG(INFO) << "Initialized left module graph of size: "
            << left_graph->GetNodeCount()
            << " and height: " << left_graph->GetRoot().props.height;

  LOG(INFO) << "Initializing right module graph";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<const HloGumgraph> right_graph,
                      HloGumgraph::Create(&right, options.fingerprint_options));
  LOG(INFO) << "Initialized right module graph of size: "
            << right_graph->GetNodeCount()
            << " and height: " << right_graph->GetRoot().props.height;

  TF_ASSIGN_OR_RETURN(std::unique_ptr<const HloGumgraphMappings> mappings,
                      FindMappings(*left_graph, *right_graph));

  std::unique_ptr<const DiffResult> diff_result =
      ConstructDiffResult(*left_graph, *right_graph, *mappings);
  std::unique_ptr<const DiffSummary> diff_summary =
      ConstructDiffSummary(left, right, *diff_result);
  std::unique_ptr<const DiffEval> diff_eval = nullptr;
  if (run_eval) {
    diff_eval = ComputeDiffEval(*left_graph, *right_graph, *mappings,
                                *diff_result, *diff_summary);
  }

  return HloGumgraphDiffResults(
      {std::move(diff_result), std::move(diff_summary), std::move(diff_eval)});
}

}  // namespace hlo_diff
}  // namespace xla
