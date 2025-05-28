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
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/iopddl.h"
#include "xla/status_macros.h"
#include "ortools/linear_solver/linear_solver.pb.h"
#ifdef PLATFORM_GOOGLE
#include "util/task/status.pb.h"
#endif

namespace xla {
namespace spmd {

using EdgeAdjacency = std::vector<std::vector<EdgeIdx>>;

namespace {

// Computes the edge resharding index from ths terminal node sharding indices.
EdgeStrategyIdx GetEdgeStrategy(
    const AutoShardingSolverRequest& request,
    const std::vector<NodeStrategyIdx>& node_strategies, const EdgeIdx edge) {
  int u = request.edges(edge).first();
  int v = request.edges(edge).second();
  int64_t num_v_strategies = request.computation_costs(v).costs_size();
  return node_strategies[u] * num_v_strategies + node_strategies[v];
}

// Stores the active times for each node.
std::vector<std::vector<LivenessIdx>> GetNodeToActiveTimes(
    const AutoShardingSolverRequest& request) {
  std::vector<std::vector<LivenessIdx>> node_to_active_times(
      request.num_nodes());
  for (LivenessIdx t = 0; t < request.live_size(); ++t) {
    for (NodeIdx node : request.live(t).nodes()) {
      node_to_active_times[node].push_back(t);
    }
  }
  return node_to_active_times;
}

// Computes the memory slack for each time (i.e., budget - live memory at t)
std::vector<double> TrackMemorySlack(
    const AutoShardingSolverRequest& request,
    const std::vector<NodeStrategyIdx>& node_strategies) {
  std::vector<double> memory_slack(request.live_size(), 0.0);
  for (LivenessIdx t = 0; t < request.live_size(); ++t) {
    double live_memory = 0.0;
    for (NodeIdx node : request.live(t).nodes()) {
      live_memory += request.memory_costs(node).costs(node_strategies[node]);
    }
    memory_slack[t] = request.memory_budget() - live_memory;
  }
  return memory_slack;
}

std::pair<EdgeAdjacency, EdgeAdjacency> GetAdjacencyMatrix(
    const AutoShardingSolverRequest& request) {
  // outward_edges: i-th vector is the edges of the form (i-th node)->v.
  // inward_edges: i-th vector is the edges of the form v->(i-th node).
  EdgeAdjacency outward_edges(request.num_nodes());
  EdgeAdjacency inward_edges(request.num_nodes());
  for (EdgeIdx edge_idx = 0; edge_idx < request.edges_size(); ++edge_idx) {
    const auto& edge = request.edges(edge_idx);
    outward_edges[edge.first()].push_back(edge_idx);
    inward_edges[edge.second()].push_back(edge_idx);
  }
  return {outward_edges, inward_edges};
}

// Store the edges within the path.
std::vector<EdgeIdx> GetEdgesWithinPath(
    const AutoShardingSolverRequest& request, const std::vector<NodeIdx>& path,
    const EdgeAdjacency& outward_edges) {
  std::vector<EdgeIdx> edges_within_path;
  for (const NodeIdx& node : path) {
    for (const EdgeIdx& edge : outward_edges[node]) {
      auto it =
          std::find(path.begin(), path.end(), request.edges(edge).second());
      if (it != path.end()) {
        edges_within_path.push_back(edge);
      }
    }
  }
  return edges_within_path;
}

// Sample a random path of length `path_length'.
std::vector<NodeIdx> SamplePath(const AutoShardingSolverRequest& request,
                                const EdgeAdjacency& outward_edges,
                                const int path_length, std::mt19937_64& rng) {
  std::vector<NodeIdx> path;
  path.reserve(path_length + 1);
  if (path_length == 0) {  // Sample a random node.
    std::uniform_int_distribution<> dist(0, request.num_nodes() - 1);
    path.push_back(dist(rng));
  } else if (path_length == 1) {  // Sample a random edge.
    std::uniform_int_distribution<> dist(0, request.edges_size() - 1);
    EdgeIdx random_edge_idx = dist(rng);
    path.push_back(request.edges(random_edge_idx).first());
    path.push_back(request.edges(random_edge_idx).second());
  } else {  // Path-sampling by concatenating nodes.
    int scanned_length = 0;
    std::uniform_int_distribution<> dist(0, request.edges_size() - 1);
    NodeIdx u = request.edges(dist(rng)).first();
    path.push_back(u);
    while (scanned_length < path_length) {
      // Sample edges from the outward edges of u.
      if (outward_edges[u].empty()) {
        break;
      }
      scanned_length++;
      std::uniform_int_distribution<> dist(0, outward_edges[u].size() - 1);
      EdgeIdx edge_idx = outward_edges[u][dist(rng)];
      u = request.edges(edge_idx).second();
      path.push_back(u);
    }
  }
  return path;
}

// Computes the cost induced by a node and its adjacent edges.
double AggregateCostAroundNode(
    const AutoShardingSolverRequest& request,
    const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency,
    const std::vector<NodeStrategyIdx>& node_strategies, const NodeIdx& node) {
  const EdgeAdjacency& outward_edges = adjacency.first;
  const EdgeAdjacency& inward_edges = adjacency.second;
  double cost = 0.0;
  // Node cost
  cost += request.computation_costs(node).costs(node_strategies[node]) +
          request.communication_costs(node).costs(node_strategies[node]);

  // Edge cost
  for (const EdgeIdx& outward_edge : outward_edges[node]) {
    cost += request.resharding_costs(outward_edge)
                .costs(GetEdgeStrategy(request, node_strategies, outward_edge));
  }
  for (const EdgeIdx& inward_edge : inward_edges[node]) {
    cost += request.resharding_costs(inward_edge)
                .costs(GetEdgeStrategy(request, node_strategies, inward_edge));
  }
  return cost;
}

// Computes the cost induced by a path (cost of nodes and adjacent edges).
double ComputePathCost(const AutoShardingSolverRequest& request,
                       const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency,
                       const std::vector<NodeIdx>& path,
                       const std::vector<EdgeIdx>& edges_within_path,
                       std::vector<NodeStrategyIdx>& node_strategies) {
  double cost = 0.0;
  for (const NodeIdx& node : path) {
    cost += AggregateCostAroundNode(request, adjacency, node_strategies, node);
  }
  // Subtracting the overcounted edge costs within the path.
  for (const EdgeIdx& edge : edges_within_path) {
    EdgeStrategyIdx edge_strategy =
        GetEdgeStrategy(request, node_strategies, edge);
    cost -= request.resharding_costs(edge).costs(edge_strategy);
  }
  return cost;
}

// Recursively optimizes over the path.
std::pair<double, std::vector<NodeStrategyIdx>> _OptimizeOverPath(
    const AutoShardingSolverRequest& request, const std::vector<NodeIdx>& path,
    const std::vector<EdgeIdx>& edges_within_path,
    std::vector<NodeStrategyIdx>& node_strategies,
    const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency,
    int num_remaining_nodes) {
  double best_cost = std::numeric_limits<double>::infinity();
  std::vector<NodeStrategyIdx> best_strategy(path.size(), 0);
  for (int i = 0; i < path.size(); ++i) {
    best_strategy[i] = node_strategies[path[i]];
  }

  if (num_remaining_nodes == 1) {  // Base case of the recursion.
    NodeIdx last_node = path[path.size() - 1];
    for (NodeStrategyIdx node_strategy = 0;
         node_strategy < request.computation_costs(last_node).costs_size();
         ++node_strategy) {
      node_strategies[last_node] = node_strategy;
      double path_cost = ComputePathCost(request, adjacency, path,
                                         edges_within_path, node_strategies);
      if (path_cost < best_cost) {
        best_cost = path_cost;
        best_strategy[best_strategy.size() - 1] = node_strategy;
      }
    }
  } else {
    NodeIdx current_node = path[path.size() - num_remaining_nodes];
    for (NodeStrategyIdx node_strategy = 0;
         node_strategy < request.computation_costs(current_node).costs_size();
         ++node_strategy) {
      node_strategies[current_node] = node_strategy;
      auto [path_cost, path_strategy] =
          _OptimizeOverPath(request, path, edges_within_path, node_strategies,
                            adjacency, num_remaining_nodes - 1);
      if (path_cost < best_cost) {
        best_cost = path_cost;
        best_strategy = path_strategy;
      }
    }
  }
  return {best_cost, best_strategy};
}

// A wrapper function for `_OptimizeOverPath`, which is a recursive
// function to find (1) the best sharding strategies for the path and (2) the
// the improvement in cost (w.r.t. the current strategies).
std::pair<double, std::vector<NodeStrategyIdx>> OptimizeOverPath(
    const AutoShardingSolverRequest& request, const std::vector<NodeIdx>& path,
    std::vector<NodeStrategyIdx>& node_strategies,
    const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency) {
  std::vector<NodeStrategyIdx> old_strategies(path.size(), 0);
  for (int i = 0; i < path.size(); ++i) {
    old_strategies[i] = node_strategies[path[i]];
  }
  std::vector<EdgeIdx> edges_within_path =
      GetEdgesWithinPath(request, path, /*outward_edges=*/adjacency.first);

  double original_path_cost = ComputePathCost(
      request, adjacency, path, edges_within_path, node_strategies);
  auto [new_path_cost, best_path_strategies] =
      _OptimizeOverPath(request, path, edges_within_path, node_strategies,
                        adjacency, path.size());

  // node_strategies could change within _OptimizeOverPath, so we restore the
  // original sharding strategies for the nodes on the path.
  for (int i = 0; i < path.size(); ++i) {
    node_strategies[path[i]] = old_strategies[i];
  }
  double cost_delta = new_path_cost - original_path_cost;
  CHECK_LE(cost_delta, 0.0);
  return {cost_delta, best_path_strategies};
}

// Check if a path's new configuration satisfies the memory constraints.
absl::flat_hash_map<LivenessIdx, double> GetNewMemorySlack(
    const AutoShardingSolverRequest& request, const std::vector<NodeIdx>& path,
    const std::vector<NodeStrategyIdx>& path_strategies,
    const std::vector<NodeStrategyIdx>& node_strategies,
    const std::vector<std::vector<LivenessIdx>>& node_to_active_times,
    const std::vector<double>& memory_slack) {
  absl::flat_hash_map<LivenessIdx, double> new_memory_slack;
  for (int i = 0; i < path.size(); ++i) {
    NodeIdx node = path[i];
    if (!node_to_active_times[node].empty()) {
      for (LivenessIdx t : node_to_active_times[node]) {
        if (!new_memory_slack.contains(t)) {
          new_memory_slack[t] = memory_slack[t];
        }
        new_memory_slack[t] -=
            (request.memory_costs(node).costs(path_strategies[i]) -
             request.memory_costs(node).costs(node_strategies[node]));
      }
    }
  }
  return new_memory_slack;
}

// Update `node_strategies` for the nodes in `path` if `new_path_strategies` is
// a feasible set of improving changes. Returns true iff the update is accepted.
bool UpdateNodeStrategies(
    const AutoShardingSolverRequest& request, const std::vector<NodeIdx>& path,
    const std::vector<NodeStrategyIdx>& new_path_strategies,
    std::vector<NodeStrategyIdx>& node_strategies,
    const std::string& memory_mode, std::vector<double>& memory_slack,
    const std::vector<std::vector<LivenessIdx>>& node_to_active_times) {
  if (memory_mode == "inactive") {
    for (int i = 0; i < path.size(); ++i) {
      node_strategies[path[i]] = new_path_strategies[i];
    }
  } else if (memory_mode == "active") {
    // Check: the new strategy satisfies the memory constraints.
    const auto new_memory_slack_at_times =
        GetNewMemorySlack(request, path, new_path_strategies, node_strategies,
                          node_to_active_times, memory_slack);
    for (const auto& [time_step, new_slack] : new_memory_slack_at_times) {
      if (new_slack < 0.0) {
        return false;
      }
    }
    // If feasible, update the sharding strategies and memory slack.
    for (const auto& [time_step, new_slack] : new_memory_slack_at_times) {
      memory_slack[time_step] = new_slack;
    }
    for (int i = 0; i < path.size(); ++i) {
      node_strategies[path[i]] = new_path_strategies[i];
    }
  }
  return true;
}

std::tuple<double, std::vector<NodeIdx>, std::vector<NodeStrategyIdx>>
SampleAndOptimizePath(const AutoShardingSolverRequest& request,
                      std::vector<NodeStrategyIdx>& node_strategies,
                      const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency,
                      const int path_length, std::mt19937_64& rng) {
  std::vector<NodeIdx> path =
      SamplePath(request, adjacency.first, path_length, rng);
  if (path.size() != path_length + 1) {
    return {0.0, {}, {}};
  }
  const auto [cost_delta, new_path_strategies] =
      OptimizeOverPath(request, path, node_strategies, adjacency);
  // Check that the new path strategy improves the cost.
  if (cost_delta == 0.0) {
    return {0.0, {}, {}};
  }
  return {cost_delta, path, new_path_strategies};
}

// Checks if the node-sharding strategy has a finite cost and satisfies the
// peak-memory constraint.
std::optional<AutoShardingViolationCode> ShardingStrategyHasViolation(
    const AutoShardingSolverRequest& request,
    const std::vector<NodeStrategyIdx>& node_strategies) {
  const int num_nodes = request.num_nodes();
  const int num_edges = request.edges_size();
  // Check for infinite coefficients in the objective function.
  for (NodeIdx v = 0; v < num_nodes; ++v) {
    NodeStrategyIdx strategy = node_strategies[v];
    if (request.computation_costs(v).costs(strategy) >= kInfinityCost ||
        request.communication_costs(v).costs(strategy) >= kInfinityCost) {
      return AutoShardingViolationCode::kInfiniteCostViolationCode;
    }
    double combined_cost = request.computation_costs(v).costs(strategy) +
                           request.communication_costs(v).costs(strategy);
    if (combined_cost < 0.0 || combined_cost >= kInfinityCost) {
      return AutoShardingViolationCode::kInfiniteCostViolationCode;
    }
  }
  for (EdgeIdx e = 0; e < num_edges; ++e) {
    EdgeStrategyIdx strategy = GetEdgeStrategy(request, node_strategies, e);
    if (request.resharding_costs(e).costs(strategy) >= kInfinityCost) {
      return AutoShardingViolationCode::kInfiniteCostViolationCode;
    }
  }
  // Check that the peak-memory constraint is satisfied at each time step t.
  for (LivenessIdx t = 0; t < request.live_size(); ++t) {
    double live_memory = 0.0;
    for (NodeIdx v : request.live(t).nodes()) {
      live_memory += request.memory_costs(v).costs(node_strategies[v]);
      if (live_memory > request.memory_budget()) {
        return AutoShardingViolationCode::kMemoryViolationCode;
      }
    }
  }
  return std::nullopt;
}

// Assigns all nodes to their first sharding configuration. If the assignment is
// infeasible, the output cost is negative and encodes the violation code.
AutoShardingSolverOutput SolveTrivial(
    const AutoShardingSolverRequest& request) {
  std::vector<NodeStrategyIdx> node_strategies(request.num_nodes(), 0);

  AutoShardingSolverOutput output;
  output.s_val = node_strategies;
  output.cost = ComputeShardingCost(request, node_strategies);
  return output;
}

AutoShardingSolverOutput SolveRandom(const AutoShardingSolverRequest& request,
                                     const int num_trials) {
  std::mt19937_64 rng(0);
  const int num_nodes = request.num_nodes();
  std::vector<NodeStrategyIdx> best_node_strategies(num_nodes, -1);
  double best_cost = -std::numeric_limits<double>::infinity();

  for (int trial = 0; trial < num_trials; ++trial) {
    std::vector<NodeStrategyIdx> node_strategies(num_nodes, -1);
    for (NodeIdx v = 0; v < num_nodes; ++v) {
      int num_strategies = request.computation_costs(v).costs_size();
      std::uniform_int_distribution<> dist(0, num_strategies - 1);
      NodeStrategyIdx strategy = dist(rng);
      node_strategies[v] = strategy;
    }
    double cost = ComputeShardingCost(request, node_strategies);

    bool have_feasible_solution = (best_cost >= 0.0);
    bool candidate_is_feasible = (cost >= 0.0);
    if (have_feasible_solution && !candidate_is_feasible) {
      continue;
    }
    if (have_feasible_solution && candidate_is_feasible) {
      if (cost < best_cost) {
        best_node_strategies = node_strategies;
        best_cost = cost;
      }
    } else if (!have_feasible_solution && candidate_is_feasible) {
      best_node_strategies = node_strategies;
      best_cost = cost;
    } else {  // Don't have feasible solution and candidate is also infeasible.
      if (cost > best_cost) {
        best_node_strategies = node_strategies;
        best_cost = cost;  // Track encoded reason for infeasibility.
      }
    }
  }

  AutoShardingSolverOutput output;
  output.s_val = best_node_strategies;
  output.cost = best_cost;
  return output;
}

// Greedily selects the node sharding strategies. Valid modes:
// - "node-cost"
// - "node-memory"
AutoShardingSolverOutput SolveGreedy(const AutoShardingSolverRequest& request,
                                     const std::string& mode) {
  const int num_nodes = request.num_nodes();
  std::vector<NodeStrategyIdx> node_strategies(num_nodes, -1);

  for (NodeIdx v = 0; v < num_nodes; ++v) {
    int num_strategies = request.computation_costs(v).costs_size();
    NodeStrategyIdx best_strategy = -1;
    double best_cost = -std::numeric_limits<double>::infinity();
    for (NodeStrategyIdx strategy = 0; strategy < num_strategies; ++strategy) {
      double cost = 0.0;
      if (mode == "node-cost") {
        cost = request.computation_costs(v).costs(strategy) +
               request.communication_costs(v).costs(strategy);
      } else if (mode == "node-memory") {
        cost = request.memory_costs(v).costs(strategy);
      } else {
        CHECK(false) << absl::Substitute(
            "SolveGreedy mode $0 is not implemented.", mode);
      }
      if (best_strategy == -1 || cost < best_cost) {
        best_strategy = strategy;
        best_cost = cost;
      }
    }
    CHECK_NE(best_strategy, -1);
    node_strategies[v] = best_strategy;
  }

  AutoShardingSolverOutput output;
  output.s_val = node_strategies;
  output.cost = ComputeShardingCost(request, node_strategies);
  return output;
}

// A local search algorithm that iteratively picks a random path of length
// `path_length` and computes the best sharding configuration for the path.
// - `path_length = 0` corresponds to a random node.
// - `path_length = 1` corresponds to a random edge.
// It has two `memory_mode` options for how it handles peak-memory constraints:
// - "inactive": ignores peak-memory constraints
// - "active": treats the peak-memory usage as a hard constraint
// `tolerance` in [0, 1] is a threshold for the relative cost decrease
// (out of the previous cost) to continue iterations.
AutoShardingSolverOutput SolveRandomPathGreedy(
    const AutoShardingSolverRequest& request, const int path_length,
    const int num_trials, const double tolerance,
    const std::string& memory_mode) {
  std::mt19937_64 rng(0);
  if (memory_mode != "inactive" && memory_mode != "active") {
    CHECK(false) << absl::Substitute("Memory mode $0 is not implemented.",
                                     memory_mode);
  }

  // Initialize each node's sharding strategy with the least-memory usage.
  AutoShardingSolverOutput output;
  std::vector<NodeStrategyIdx> node_strategies =
      SolveGreedy(request, "node-memory").s_val;
  const std::pair<EdgeAdjacency, EdgeAdjacency> adjacency =
      GetAdjacencyMatrix(request);
  std::vector<std::vector<LivenessIdx>> node_to_active_times;
  std::vector<double> memory_slack;
  double current_cost = ComputeShardingCost(
      request, node_strategies, /*use_negative_violation_codes=*/false);
  if (memory_mode == "active") {
    node_to_active_times = GetNodeToActiveTimes(request);
    memory_slack = TrackMemorySlack(request, node_strategies);
  }
  // Phase 0: Return if minimum possible memory usage already exceeds budget.
  // This makes sense because we're initializing the solution w/
  // greedy-node-memory (which has minimum possible peak memory).
  std::optional<AutoShardingViolationCode> violation_code =
      ShardingStrategyHasViolation(request, node_strategies);
  if (violation_code.has_value() && *violation_code == kMemoryViolationCode) {
    output.s_val = node_strategies;
    output.cost = current_cost;
    return output;
  }

  // Phase 1: Store the sharding costs of the last `window_size` trials.
  CHECK_GE(num_trials, 20);
  int window_size = std::min(static_cast<int>(0.05 * num_trials), 100000);
  std::vector<double> cost_window(window_size, -1.0);
  cost_window[0] = current_cost;
  for (int window_idx = 1; window_idx < window_size; ++window_idx) {
    auto [cost_delta, path, new_path_strategies] = SampleAndOptimizePath(
        request, node_strategies, adjacency, path_length, rng);
    if (cost_delta < 0.0 &&
        UpdateNodeStrategies(request, path, new_path_strategies,
                             node_strategies, memory_mode, memory_slack,
                             node_to_active_times)) {
      current_cost += cost_delta;
    }
    cost_window[window_idx] = current_cost;
  }
  // Phase 2: Optimize the sharding cost with an early-stopping feature.
  for (int trial = window_size; trial < num_trials; ++trial) {
    auto [cost_delta, path, new_path_strategies] = SampleAndOptimizePath(
        request, node_strategies, adjacency, path_length, rng);
    if (cost_delta < 0.0 &&
        UpdateNodeStrategies(request, path, new_path_strategies,
                             node_strategies, memory_mode, memory_slack,
                             node_to_active_times)) {
      current_cost += cost_delta;
    }
    if (1.0 - current_cost / cost_window[trial % window_size] < tolerance) {
      break;
    }
    cost_window[trial % window_size] = current_cost;
  }

  output.s_val = node_strategies;
  output.cost = ComputeShardingCost(request, node_strategies);
  return output;
}

}  // namespace

absl::StatusOr<AutoShardingSolverOutput> RunHeuristicSolver(
    const AutoShardingSolverRequest& request, const std::string& algorithm) {
  absl::Time start_time = absl::Now();
  AutoShardingSolverOutput output;
  if (algorithm == "trivial") {
    output = SolveTrivial(request);
  } else if (algorithm == "random") {
    output = SolveRandom(request, 10000);
  } else if (algorithm == "greedy-node-cost") {
    output = SolveGreedy(request, "node-cost");
  } else if (algorithm == "greedy-node-memory") {
    output = SolveGreedy(request, "node-memory");
  } else if (algorithm == "random-path-greedy") {
    const int num_trials =
        2 * request.edges_size() * std::log(request.edges_size());
    output = SolveRandomPathGreedy(request, /*path_length=*/2, num_trials,
                                   /*tolerance=*/0.001,
                                   /*memory_mode=*/"active");
  } else if (algorithm == "brkga") {
    output = SolveBrkga(request);
  } else {
    CHECK(false) << absl::Substitute("Algorithm $0 is not implemented.",
                                     algorithm);
  }
  auto duration = absl::Now() - start_time;
  LOG(INFO) << "Solver took " << absl::ToInt64Milliseconds(duration) << " ms";
  LOG(INFO) << "Objective value: " << output.cost;
  LOG(INFO) << "Total Cost: " << ComputeShardingCost(request, output.s_val);
  return output;
}

bool CostComponents::operator==(const CostComponents& other) const {
  return communication_cost == other.communication_cost &&
         computation_cost == other.computation_cost &&
         resharding_cost == other.resharding_cost &&
         overbudget_cost == other.overbudget_cost &&
         max_memory == other.max_memory && node_cost == other.node_cost &&
         edge_cost == other.edge_cost &&
         overbudget_usage == other.overbudget_usage &&
         max_usage == other.max_usage;
}

double CostComponents::cost() const {
  return communication_cost + computation_cost + resharding_cost +
         overbudget_cost;
}

iopddl::TotalCost CostComponents::total_cost() const {
  return node_cost + edge_cost + overbudget_usage;
}

bool AutoShardingEvaluation::operator==(
    const AutoShardingEvaluation& other) const {
  return violation_codes == other.violation_codes && total == other.total &&
         lower_bound == other.lower_bound &&
         total_departures == other.total_departures;
}

double ComputeShardingCost(const AutoShardingSolverRequest& request,
                           const std::vector<NodeStrategyIdx>& node_strategies,
                           const bool use_negative_violation_codes) {
  double cost = 0.0;
  for (NodeIdx v = 0; v < request.num_nodes(); ++v) {
    NodeStrategyIdx strategy = node_strategies[v];
    cost += request.computation_costs(v).costs(strategy) +
            request.communication_costs(v).costs(strategy);
  }
  for (EdgeIdx e = 0; e < request.edges_size(); ++e) {
    EdgeStrategyIdx strategy = GetEdgeStrategy(request, node_strategies, e);
    cost += request.resharding_costs(e).costs(strategy);
  }
  if (use_negative_violation_codes) {
    std::optional<AutoShardingViolationCode> violation_code =
        ShardingStrategyHasViolation(request, node_strategies);
    if (violation_code.has_value()) {
      cost = -1 * (*violation_code);
    }
  }
  return cost;
}

absl::Status ValidateRequest(const AutoShardingSolverRequest& request) {
  const int num_nodes = request.num_nodes();
  const int num_edges = request.edges_size();
  TF_RET_CHECK(num_nodes == request.computation_costs_size());
  TF_RET_CHECK(num_nodes == request.communication_costs_size());
  TF_RET_CHECK(num_nodes == request.memory_costs_size());
  TF_RET_CHECK(num_edges == request.resharding_costs_size());

  for (NodeIdx u = 0; u < num_nodes; ++u) {
    const int num_strategies = request.computation_costs(u).costs_size();
    TF_RET_CHECK(num_strategies >= 1);
    TF_RET_CHECK(num_strategies == request.communication_costs(u).costs_size());
    TF_RET_CHECK(num_strategies == request.memory_costs(u).costs_size());
    for (NodeStrategyIdx strategy = 0; strategy < num_strategies; ++strategy) {
      TF_RET_CHECK(request.computation_costs(u).costs(strategy) >= 0.0);
      TF_RET_CHECK(request.communication_costs(u).costs(strategy) >= 0.0);
      TF_RET_CHECK(request.memory_costs(u).costs(strategy) >= 0.0);
    }
  }

  absl::btree_set<std::pair<int, int>> edges_seen;
  for (EdgeIdx e = 0; e < num_edges; ++e) {
    const int u = request.edges(e).first();
    const int v = request.edges(e).second();
    TF_RET_CHECK(u >= 0);
    TF_RET_CHECK(u < num_nodes);
    TF_RET_CHECK(v >= 0);
    TF_RET_CHECK(v < num_nodes);
    TF_RET_CHECK(u < v);
    TF_RET_CHECK(edges_seen.count({u, v}) == 0);
    edges_seen.insert({u, v});

    const int num_strategies = request.resharding_costs(e).costs_size();
    const int num_u_strategies = request.computation_costs(u).costs_size();
    const int num_v_strategies = request.computation_costs(v).costs_size();
    CHECK_EQ(num_strategies, num_u_strategies * num_v_strategies);
  }
  return absl::OkStatus();
}

}  // namespace spmd
}  // namespace xla
