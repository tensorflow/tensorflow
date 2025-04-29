/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_iopddl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/iopddl.h"

namespace xla {
namespace spmd {

iopddl::Cost ConvertCost(const double cost) {
  CHECK_GE(cost, 0);  // Contest problems shouldn't include any negative costs.
  if (cost >= kInfinityInt) {
    return kInfinityInt;
  }
  return static_cast<int64_t>(cost);
}

iopddl::Problem ConvertToProblem(const AutoShardingSolverRequest& request) {
  CHECK(request.node_groups().empty());  // Contest files don't support groups.
  iopddl::Problem problem = {.name = request.request_name()};
  std::vector<iopddl::Interval> node_intervals;
  for (int64_t node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    iopddl::Interval node_interval = {kInfinityInt, -1};
    if (request.live().empty()) {
      CHECK_LT(node_idx, request.node_intervals_size());
      const auto& interval = request.node_intervals(node_idx);
      if (interval.first() <= interval.second()) {
        node_interval = {interval.first(), interval.second() + 1};
      }
    }
    node_intervals.push_back(node_interval);
  }
  for (LivenessIdx t = 0; t < request.live_size(); ++t) {
    for (int64_t node_idx : request.live(t).nodes()) {
      node_intervals[node_idx] = {
          std::min(node_intervals[node_idx].first, t),
          std::max(node_intervals[node_idx].second, t + 1)};
    }
  }
  for (int64_t node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    const auto& interval = node_intervals[node_idx];
    iopddl::Interval node_interval = {0, 0};
    if (interval.first <= interval.second) {
      node_interval = {interval.first, interval.second};
    }
    problem.nodes.push_back({node_interval});
    CHECK_LT(node_idx, request.s_len_size());
    CHECK_LT(node_idx, request.computation_costs_size());
    CHECK_LT(node_idx, request.communication_costs_size());
    CHECK_LT(node_idx, request.memory_costs_size());
    for (int64_t j = 0; j < request.s_len(node_idx); ++j) {
      CHECK_LT(j, request.computation_costs(node_idx).costs_size());
      CHECK_LT(j, request.communication_costs(node_idx).costs_size());
      CHECK_LT(j, request.memory_costs(node_idx).costs_size());
      const double node_cost = request.computation_costs(node_idx).costs(j) +
                               request.communication_costs(node_idx).costs(j);
      const iopddl::Cost cost = ConvertCost(node_cost);
      const iopddl::Usage usage =
          ConvertCost(request.memory_costs(node_idx).costs(j));
      problem.nodes.back().strategies.push_back({cost, usage});
    }
  }
  // The first kind of edges come from request.edges
  for (int64_t edge_idx = 0; edge_idx < request.edges_size(); ++edge_idx) {
    const auto& edge = request.edges(edge_idx);
    CHECK_LT(edge.first(), request.s_len_size());
    CHECK_LT(edge.second(), request.s_len_size());
    CHECK_LT(edge_idx, request.resharding_costs_size());
    problem.edges.push_back({{edge.first(), edge.second()}});
    for (int64_t i = 0; i < request.s_len(edge.first()); ++i) {
      for (int64_t j = 0; j < request.s_len(edge.second()); ++j) {
        const int64_t k = i * request.s_len(edge.second()) + j;
        CHECK_LT(k, request.resharding_costs(edge_idx).costs_size());
        const iopddl::Cost cost =
            ConvertCost(request.resharding_costs(edge_idx).costs(k));
        problem.edges.back().strategies.push_back({cost});
      }
    }
  }
  // The second kind of edges come from request.aliases
  for (int64_t alias_idx = 0; alias_idx < request.aliases_size(); ++alias_idx) {
    const auto& alias = request.aliases(alias_idx);
    problem.edges.push_back({{alias.first(), alias.second()}});
    CHECK_LT(alias.first(), request.s_len_size());
    CHECK_LT(alias.second(), request.s_len_size());
    CHECK_LT(alias_idx, request.value_costs_size());
    for (int64_t i = 0; i < request.s_len(alias.first()); ++i) {
      for (int64_t j = 0; j < request.s_len(alias.second()); ++j) {
        const int64_t k = i * request.s_len(alias.second()) + j;
        CHECK_LT(k, request.value_costs(alias_idx).costs_size());
        const iopddl::Cost cost =
            ConvertCost(request.value_costs(alias_idx).costs(k) * kInfinityInt);
        problem.edges.back().strategies.push_back({cost});
      }
    }
  }
  // The third kind of edges come from request.s_follow
  for (int64_t node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    CHECK_LT(node_idx, request.s_follow_size());
    if (request.s_follow(node_idx) < 0) {
      continue;
    }
    problem.edges.push_back({{request.s_follow(node_idx), node_idx}});
    CHECK_LT(node_idx, request.s_len_size());
    for (int64_t i = 0; i < request.s_len(node_idx); ++i) {
      for (int64_t j = 0; j < request.s_len(node_idx); ++j) {
        const iopddl::Cost cost = (i == j) ? 0 : kInfinityInt;
        problem.edges.back().strategies.push_back({cost});
      }
    }
  }
  if (request.memory_budget() > 0) {
    problem.usage_limit = request.memory_budget();
  }
  return problem;
}

static bool IsEdgeFollower(const iopddl::Problem& problem,
                           const iopddl::Edge& edge) {
  int strategies0 = problem.nodes[edge.nodes[0]].strategies.size();
  int strategies1 = problem.nodes[edge.nodes[1]].strategies.size();
  if (strategies0 != strategies1) {
    return false;
  }
  for (iopddl::StrategyIdx idx0 = 0; idx0 < strategies0; ++idx0) {
    for (iopddl::StrategyIdx idx1 = 0; idx1 < strategies1; ++idx1) {
      const auto strategy = edge.strategies[idx0 * strategies1 + idx1];
      if (idx0 == idx1 && strategy.cost != 0) {
        return false;
      }
      if (idx0 != idx1 && strategy.cost != kInfinityInt) {
        return false;
      }
    }
  }
  return true;
}

static bool IsEdgeAlias(const iopddl::Edge& edge) {
  for (const iopddl::Strategy& strategy : edge.strategies) {
    if (strategy.cost == kInfinityInt) {
      return true;
    }
  }
  return false;
}

AutoShardingSolverRequest ConvertToSolverRequest(
    const iopddl::Problem& problem) {
  AutoShardingSolverRequest request;
  request.set_request_name(problem.name);
  request.set_num_nodes(problem.nodes.size());
  request.set_memory_budget(problem.usage_limit.value_or(-1));
  for (iopddl::NodeIdx node_idx = 0; node_idx < problem.nodes.size();
       ++node_idx) {
    const iopddl::Node& node = problem.nodes[node_idx];
    request.add_s_len(node.strategies.size());
    request.add_s_follow(-1);
    request.add_communication_costs();
    request.add_computation_costs();
    request.add_memory_costs();
    for (const iopddl::Strategy& strategy : node.strategies) {
      double strategy_cost = (strategy.cost == kInfinityInt)
                                 ? kInfinityCost
                                 : static_cast<double>(strategy.cost);
      request.mutable_computation_costs()->rbegin()->add_costs(
          static_cast<double>(strategy_cost));
      request.mutable_communication_costs()->rbegin()->add_costs(0.0);
      request.mutable_memory_costs()->rbegin()->add_costs(
          static_cast<double>(strategy.usage));
    }
    request.add_node_intervals();
    bool empty_interval = (node.interval.first == node.interval.second);
    request.mutable_node_intervals()->rbegin()->set_first(
        empty_interval ? 100 : node.interval.first);
    request.mutable_node_intervals()->rbegin()->set_second(
        empty_interval ? -1 : node.interval.second - 1);
  }
  for (iopddl::EdgeIdx edge_idx = 0; edge_idx < problem.edges.size();
       ++edge_idx) {
    const iopddl::Edge& edge = problem.edges[edge_idx];
    if (IsEdgeFollower(problem, edge)) {
      request.mutable_s_follow()->Set(edge.nodes[1], edge.nodes[0]);
      continue;
    }
    if (IsEdgeAlias(edge)) {
      auto* alias = request.add_aliases();
      alias->set_first(edge.nodes[0]);
      alias->set_second(edge.nodes[1]);
      request.add_value_costs();
      for (const iopddl::Strategy& strategy : edge.strategies) {
        request.mutable_value_costs()->rbegin()->add_costs(
            strategy.cost == kInfinityInt ? 1.0 : 0.0);
      }
      continue;
    }
    auto* edge_proto = request.add_edges();
    edge_proto->set_first(edge.nodes[0]);
    edge_proto->set_second(edge.nodes[1]);
    request.add_resharding_costs();
    for (const iopddl::Strategy& strategy : edge.strategies) {
      request.mutable_resharding_costs()->rbegin()->add_costs(
          static_cast<double>(strategy.cost));
    }
  }
  return request;
}

void RandomizeCosts(iopddl::Problem& problem) {
  unsigned int seed = 2025;
  auto get_multiplier = [&]() {  // Returns a value between 1/16 and 16.0
    return std::pow(2.0, (rand_r(&seed) % 9) - 4);
  };
  auto randomize = [&](iopddl::Cost& cost, const double multiplier) {
    if (cost != kInfinityInt) {
      cost = static_cast<iopddl::Cost>(static_cast<double>(cost) * multiplier);
    }
  };
  for (iopddl::Node& node : problem.nodes) {
    const double multiplier = get_multiplier();
    for (iopddl::Strategy& strategy : node.strategies) {
      randomize(strategy.cost, multiplier);
    }
  }
  for (iopddl::Edge& edge : problem.edges) {
    const double multiplier = get_multiplier();
    for (iopddl::Strategy& strategy : edge.strategies) {
      randomize(strategy.cost, multiplier);
    }
  }
}

}  // namespace spmd
}  // namespace xla
