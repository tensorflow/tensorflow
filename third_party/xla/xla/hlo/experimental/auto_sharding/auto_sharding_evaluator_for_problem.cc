/* Copyright 2022 The OpenXLA Authors.

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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_iopddl.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/iopddl.h"

namespace xla {
namespace spmd {

iopddl::Cost MinCost(const iopddl::Node& node) {
  iopddl::Cost min_cost = kInfinityInt;
  for (const auto& strategy : node.strategies) {
    min_cost = std::min(min_cost, strategy.cost);
  }
  return min_cost;
}

iopddl::Cost MinCost(const iopddl::Edge& edge) {
  iopddl::Cost min_cost = kInfinityInt;
  for (const auto& strategy : edge.strategies) {
    min_cost = std::min(min_cost, strategy.cost);
  }
  return min_cost;
}

AutoShardingEvaluation Evaluate(const iopddl::Problem& problem,
                                const AutoShardingSolverOutput& result,
                                const AutoShardingSolverParams& params) {
  const std::vector<int64_t> s_follows = GetFollowers(problem);
  const std::vector<iopddl::Edge> aliases = GetAliases(problem);
  const std::vector<iopddl::Edge> deduplicated_edges =
      GetDeduplicatedEdges(problem);
  const std::vector<NodeStrategyIdx>& s_val = result.s_val;
  const auto e_val = [&](EdgeIdx edge_idx) {
    const auto& edge = deduplicated_edges[edge_idx];
    const auto num_strat = problem.nodes[edge.nodes[1]].strategies.size();
    return s_val[edge.nodes[0]] * num_strat + s_val[edge.nodes[1]];
  };
  AutoShardingEvaluation evaluation;
  // Compute violations.
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    NodeIdx s_follow = s_follows[node_idx];
    if (s_follow >= 0 && s_val[node_idx] != s_val[s_follow]) {
      evaluation.violation_codes.insert(kFollowerViolationCode);
    }
  }
  for (auto alias_idx = 0; alias_idx < aliases.size(); ++alias_idx) {
    const auto& alias = aliases[alias_idx];
    NodeStrategyIdx p = s_val[alias.nodes[0]], q = s_val[alias.nodes[1]];
    auto strat_idx = p * problem.nodes[alias.nodes[1]].strategies.size() + q;
    if (alias.strategies[strat_idx].cost > 0) {
      evaluation.violation_codes.insert(kAliasViolationCode);
    }
  }
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    NodeStrategyIdx strat_idx = s_val[node_idx];
    const double node_cost = problem.nodes[node_idx].strategies[strat_idx].cost;
    if (node_cost >= kInfinityInt) {
      evaluation.violation_codes.insert(kInfiniteCostViolationCode);
    }
  }
  for (EdgeIdx edge_idx = 0; edge_idx < deduplicated_edges.size(); ++edge_idx) {
    const auto& edge = deduplicated_edges[edge_idx];
    if (edge.strategies[e_val(edge_idx)].cost >= kInfinityInt) {
      evaluation.violation_codes.insert(kInfiniteCostViolationCode);
    }
  }
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    if (params.departure_costs.empty()) {
      continue;
    }
    evaluation.total_departures +=
        params.departure_costs[node_idx][s_val[node_idx]];

    if (params.max_departures.has_value() &&
        evaluation.total_departures > *params.max_departures) {
      evaluation.violation_codes.insert(kMaxDeparturesViolationCode);
    }
  }
  if (problem.usage_limit.has_value()) {
    std::vector<iopddl::TotalUsage> total_usages, lower_bound_usages;
    for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
      const auto& interval = problem.nodes[node_idx].interval;
      if (interval.first > interval.second) {
        continue;
      }
      // Expand cost vectors if needed to cover the range of this interval.
      while (total_usages.size() < interval.second) {
        total_usages.push_back(0);
        lower_bound_usages.push_back(0);
      }
      iopddl::TotalUsage total_usage = 0, lower_bound_usage = kInfinityInt;
      const auto& node = problem.nodes[node_idx];
      for (const auto& strategy : node.strategies) {
        lower_bound_usage =
            std::min(lower_bound_usage, iopddl::TotalUsage(strategy.usage));
      }
      total_usage = node.strategies[s_val[node_idx]].usage;
      for (LivenessIdx time_idx = interval.first; time_idx < interval.second;
           ++time_idx) {
        total_usages[time_idx] += total_usage;
        lower_bound_usages[time_idx] += lower_bound_usage;
      }
    }
    iopddl::TotalUsage total_overbudget = 0, lower_bound_overbudget = 0;
    for (LivenessIdx time_idx = 0; time_idx < total_usages.size(); ++time_idx) {
      evaluation.total.max_usage =
          std::max(evaluation.total.max_usage, total_usages[time_idx]);
      evaluation.lower_bound.max_usage = std::max(
          evaluation.lower_bound.max_usage, lower_bound_usages[time_idx]);
      if (params.overbudget_coeff.has_value()) {
        total_overbudget = std::max(
            total_overbudget, total_usages[time_idx] - *problem.usage_limit);
        lower_bound_overbudget =
            std::max(lower_bound_overbudget,
                     lower_bound_usages[time_idx] - *problem.usage_limit);
      } else if (total_usages[time_idx] > *problem.usage_limit) {
        evaluation.violation_codes.insert(kMemoryViolationCode);
      }
    }
    if (params.overbudget_coeff.has_value()) {
      evaluation.total.overbudget_usage =
          static_cast<iopddl::Cost>(*params.overbudget_coeff) *
          total_overbudget;
      evaluation.lower_bound.overbudget_usage =
          static_cast<iopddl::Cost>(*params.overbudget_coeff) *
          lower_bound_overbudget;
    }
  }
  // Compute metrics and lower bounds.
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    const iopddl::Node& node = problem.nodes[node_idx];
    evaluation.total.node_cost += node.strategies[s_val[node_idx]].cost;
    evaluation.lower_bound.node_cost += MinCost(node);
  }
  for (EdgeIdx edge_idx = 0; edge_idx < deduplicated_edges.size(); ++edge_idx) {
    const iopddl::Edge& edge = deduplicated_edges[edge_idx];
    evaluation.total.edge_cost += edge.strategies[e_val(edge_idx)].cost;
    evaluation.lower_bound.edge_cost += MinCost(edge);
  }
  return evaluation;
}

}  // namespace spmd
}  // namespace xla
