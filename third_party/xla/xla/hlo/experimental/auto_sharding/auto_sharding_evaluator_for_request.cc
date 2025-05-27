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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"

namespace xla {
namespace spmd {

AutoShardingEvaluation Evaluate(const AutoShardingSolverRequest& request,
                                const AutoShardingSolverOutput& result,
                                const AutoShardingSolverParams& params) {
  const auto& c = request.computation_costs();
  const auto& d = request.communication_costs();
  const auto& r = request.resharding_costs();
  const auto& v = request.value_costs();
  const auto& p = params.departure_costs;
  const std::vector<NodeStrategyIdx>& s_val = result.s_val;
  const auto e_val = [&](EdgeIdx edge_idx) {
    const auto& edge = request.edges(edge_idx);
    return s_val[edge.first()] * request.s_len(edge.second()) +
           s_val[edge.second()];
  };
  AutoShardingEvaluation evaluation;
  // Compute violations.
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    NodeIdx s_follow = request.s_follow(node_idx);
    if (s_follow >= 0 && s_val[node_idx] != s_val[s_follow]) {
      evaluation.violation_codes.insert(kFollowerViolationCode);
    }
  }
  for (auto alias_idx = 0; alias_idx < request.aliases_size(); ++alias_idx) {
    const auto& alias = request.aliases(alias_idx);
    NodeStrategyIdx p = s_val[alias.first()], q = s_val[alias.second()];
    if (v.at(alias_idx).costs(p * request.s_len(alias.second()) + q) > 0.5) {
      evaluation.violation_codes.insert(kAliasViolationCode);
    }
  }
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    NodeStrategyIdx strat_idx = s_val[node_idx];
    const double node_cost =
        c.at(node_idx).costs(strat_idx) + d.at(node_idx).costs(strat_idx);
    if (node_cost >= kInfinityCost) {
      evaluation.violation_codes.insert(kInfiniteCostViolationCode);
    }
  }
  for (EdgeIdx edge_idx = 0; edge_idx < request.edges_size(); ++edge_idx) {
    if (r.at(edge_idx).costs(e_val(edge_idx)) >= kInfinityCost) {
      evaluation.violation_codes.insert(kInfiniteCostViolationCode);
    }
  }
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    if (p.empty()) {
      continue;
    }
    evaluation.total_departures += p[node_idx][s_val[node_idx]];

    if (params.max_departures.has_value() &&
        evaluation.total_departures > *params.max_departures) {
      evaluation.violation_codes.insert(kMaxDeparturesViolationCode);
    }
  }
  if (request.memory_budget() > 0) {
    std::vector<double> total_memory_costs, lower_bound_memory_costs;
    for (NodeIdx node_idx = 0; node_idx < request.node_intervals_size();
         ++node_idx) {
      const auto& interval = request.node_intervals(node_idx);
      if (interval.first() > interval.second()) {
        continue;
      }
      // Expand cost vectors if needed to cover the range of this interval.
      while (total_memory_costs.size() <= interval.second()) {
        total_memory_costs.push_back(0.0);
        lower_bound_memory_costs.push_back(0.0);
      }
      double total_memory_cost = 0.0, lower_bound_memory_cost = 0.0;
      const auto& m = request.memory_costs(node_idx).costs();
      total_memory_cost = m[s_val[node_idx]];
      lower_bound_memory_cost = *std::min_element(m.begin(), m.end());
      for (LivenessIdx time_idx = interval.first();
           time_idx <= interval.second(); ++time_idx) {
        total_memory_costs[time_idx] += total_memory_cost;
        lower_bound_memory_costs[time_idx] += lower_bound_memory_cost;
      }
    }
    double total_overbudget = 0.0;
    double lower_bound_overbudget = 0.0;
    for (LivenessIdx time_idx = 0; time_idx < total_memory_costs.size();
         ++time_idx) {
      evaluation.total.max_memory =
          std::max(evaluation.total.max_memory, total_memory_costs[time_idx]);
      evaluation.lower_bound.max_memory =
          std::max(evaluation.lower_bound.max_memory,
                   lower_bound_memory_costs[time_idx]);
      if (request.has_overbudget_coeff()) {
        total_overbudget =
            std::max(total_overbudget,
                     total_memory_costs[time_idx] - request.memory_budget());
        lower_bound_overbudget = std::max(
            lower_bound_overbudget,
            lower_bound_memory_costs[time_idx] - request.memory_budget());
      } else if (total_memory_costs[time_idx] > request.memory_budget()) {
        evaluation.violation_codes.insert(kMemoryViolationCode);
      }
    }
    if (params.overbudget_coeff.has_value()) {
      evaluation.total.overbudget_cost =
          *params.overbudget_coeff * total_overbudget;
      evaluation.lower_bound.overbudget_cost =
          *params.overbudget_coeff * lower_bound_overbudget;
    }
  }
  // Compute metrics and lower bounds.
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    evaluation.total.communication_cost +=
        d.at(node_idx).costs(s_val[node_idx]);
    evaluation.total.computation_cost += c.at(node_idx).costs(s_val[node_idx]);
    evaluation.lower_bound.communication_cost += *std::min_element(
        d.at(node_idx).costs().begin(), d.at(node_idx).costs().end());
    evaluation.lower_bound.computation_cost += *std::min_element(
        c.at(node_idx).costs().begin(), c.at(node_idx).costs().end());
  }
  for (EdgeIdx edge_idx = 0; edge_idx < request.edges_size(); ++edge_idx) {
    evaluation.total.resharding_cost += r.at(edge_idx).costs(e_val(edge_idx));
    evaluation.lower_bound.resharding_cost += *std::min_element(
        r.at(edge_idx).costs().begin(), r.at(edge_idx).costs().end());
  }
  return evaluation;
}

}  // namespace spmd
}  // namespace xla
