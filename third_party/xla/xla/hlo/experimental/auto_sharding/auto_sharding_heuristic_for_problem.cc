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
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"
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
    const iopddl::Problem& problem,
    const std::vector<NodeStrategyIdx>& node_strategies, const EdgeIdx edge) {
  int u = problem.edges[edge].nodes[0];
  int v = problem.edges[edge].nodes[1];
  int64_t num_v_strategies = problem.nodes[v].strategies.size();
  return node_strategies[u] * num_v_strategies + node_strategies[v];
}

// Stores the active times for each node.
std::vector<std::vector<LivenessIdx>> GetNodeToActiveTimes(
    const iopddl::Problem& problem) {
  std::vector<std::vector<LivenessIdx>> node_to_active_times(
      problem.nodes.size());
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    const iopddl::Node& node = problem.nodes[node_idx];
    for (LivenessIdx t = node.interval.first; t < node.interval.second; ++t) {
      node_to_active_times[node_idx].push_back(t);
    }
  }
  return node_to_active_times;
}

// Computes the memory slack for each time (i.e., budget - live memory at t)
std::vector<iopddl::TotalUsage> TrackMemorySlack(
    const iopddl::Problem& problem,
    const std::vector<NodeStrategyIdx>& node_strategies) {
  LivenessIdx max_time = 0;
  for (const iopddl::Node& node : problem.nodes) {
    max_time = std::max(max_time, node.interval.second);
  }
  std::vector<iopddl::TotalUsage> memory_slack(max_time, *problem.usage_limit);
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    const iopddl::Node& node = problem.nodes[node_idx];
    for (LivenessIdx t = node.interval.first; t < node.interval.second; ++t) {
      memory_slack[t] -= node.strategies[node_strategies[node_idx]].usage;
    }
  }
  return memory_slack;
}

std::pair<EdgeAdjacency, EdgeAdjacency> GetAdjacencyMatrix(
    const iopddl::Problem& problem) {
  // outward_edges: i-th vector is the edges of the form (i-th node)->v.
  // inward_edges: i-th vector is the edges of the form v->(i-th node).
  EdgeAdjacency outward_edges(problem.nodes.size());
  EdgeAdjacency inward_edges(problem.nodes.size());
  for (EdgeIdx edge_idx = 0; edge_idx < problem.edges.size(); ++edge_idx) {
    const auto& edge = problem.edges[edge_idx];
    outward_edges[edge.nodes[0]].push_back(edge_idx);
    inward_edges[edge.nodes[1]].push_back(edge_idx);
  }
  return {outward_edges, inward_edges};
}

// Store the edges within the path.
std::vector<EdgeIdx> GetEdgesWithinPath(const iopddl::Problem& problem,
                                        const std::vector<NodeIdx>& path,
                                        const EdgeAdjacency& outward_edges) {
  std::vector<EdgeIdx> edges_within_path;
  for (const NodeIdx& node : path) {
    for (const EdgeIdx& edge : outward_edges[node]) {
      auto it =
          std::find(path.begin(), path.end(), problem.edges[edge].nodes[1]);
      if (it != path.end()) {
        edges_within_path.push_back(edge);
      }
    }
  }
  return edges_within_path;
}

// Sample a random path of length `path_length'.
std::vector<NodeIdx> SamplePath(const iopddl::Problem& problem,
                                const EdgeAdjacency& outward_edges,
                                const int path_length, std::mt19937_64& rng) {
  std::vector<NodeIdx> path;
  path.reserve(path_length + 1);
  if (path_length == 0) {  // Sample a random node.
    std::uniform_int_distribution<> dist(0, problem.nodes.size() - 1);
    path.push_back(dist(rng));
  } else if (path_length == 1) {  // Sample a random edge.
    std::uniform_int_distribution<> dist(0, problem.edges.size() - 1);
    EdgeIdx random_edge_idx = dist(rng);
    path.push_back(problem.edges[random_edge_idx].nodes[0]);
    path.push_back(problem.edges[random_edge_idx].nodes[1]);
  } else {  // Path-sampling by concatenating nodes.
    int scanned_length = 0;
    std::uniform_int_distribution<> dist(0, problem.edges.size() - 1);
    NodeIdx u = problem.edges[dist(rng)].nodes[0];
    path.push_back(u);
    while (scanned_length < path_length) {
      // Sample edges from the outward edges of u.
      if (outward_edges[u].empty()) {
        break;
      }
      scanned_length++;
      std::uniform_int_distribution<> dist(0, outward_edges[u].size() - 1);
      EdgeIdx edge_idx = outward_edges[u][dist(rng)];
      u = problem.edges[edge_idx].nodes[1];
      path.push_back(u);
    }
  }
  return path;
}

// Computes the cost induced by a node and its adjacent edges.
iopddl::TotalCost AggregateCostAroundNode(
    const iopddl::Problem& problem,
    const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency,
    const std::vector<NodeStrategyIdx>& node_strategies, const NodeIdx& node) {
  const EdgeAdjacency& outward_edges = adjacency.first;
  const EdgeAdjacency& inward_edges = adjacency.second;
  iopddl::TotalCost cost = 0;
  // Node cost
  cost += problem.nodes[node].strategies[node_strategies[node]].cost;

  // Edge cost
  for (const EdgeIdx& outward_edge : outward_edges[node]) {
    cost +=
        problem.edges[outward_edge]
            .strategies[GetEdgeStrategy(problem, node_strategies, outward_edge)]
            .cost;
  }
  for (const EdgeIdx& inward_edge : inward_edges[node]) {
    cost +=
        problem.edges[inward_edge]
            .strategies[GetEdgeStrategy(problem, node_strategies, inward_edge)]
            .cost;
  }
  return cost;
}

// Computes the cost induced by a path (cost of nodes and adjacent edges).
iopddl::TotalCost ComputePathCost(
    const iopddl::Problem& problem,
    const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency,
    const std::vector<NodeIdx>& path,
    const std::vector<EdgeIdx>& edges_within_path,
    std::vector<NodeStrategyIdx>& node_strategies) {
  iopddl::TotalCost cost = 0;
  for (const NodeIdx& node : path) {
    cost += AggregateCostAroundNode(problem, adjacency, node_strategies, node);
  }
  // Subtracting the overcounted edge costs within the path.
  for (const EdgeIdx& edge : edges_within_path) {
    EdgeStrategyIdx edge_strategy =
        GetEdgeStrategy(problem, node_strategies, edge);
    cost -= problem.edges[edge].strategies[edge_strategy].cost;
  }
  return cost;
}

// Recursively optimizes over the path.
std::pair<iopddl::TotalCost, std::vector<NodeStrategyIdx>> _OptimizeOverPath(
    const iopddl::Problem& problem, const std::vector<NodeIdx>& path,
    const std::vector<EdgeIdx>& edges_within_path,
    std::vector<NodeStrategyIdx>& node_strategies,
    const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency,
    int num_remaining_nodes) {
  iopddl::TotalCost best_cost = absl::Int128Max();
  std::vector<NodeStrategyIdx> best_strategy(path.size(), 0);
  for (int i = 0; i < path.size(); ++i) {
    best_strategy[i] = node_strategies[path[i]];
  }

  if (num_remaining_nodes == 1) {  // Base case of the recursion.
    NodeIdx last_node = path[path.size() - 1];
    for (NodeStrategyIdx node_strategy = 0;
         node_strategy < problem.nodes[last_node].strategies.size();
         ++node_strategy) {
      node_strategies[last_node] = node_strategy;
      iopddl::TotalCost path_cost = ComputePathCost(
          problem, adjacency, path, edges_within_path, node_strategies);
      if (path_cost < best_cost) {
        best_cost = path_cost;
        best_strategy[best_strategy.size() - 1] = node_strategy;
      }
    }
  } else {
    NodeIdx current_node = path[path.size() - num_remaining_nodes];
    for (NodeStrategyIdx node_strategy = 0;
         node_strategy < problem.nodes[current_node].strategies.size();
         ++node_strategy) {
      node_strategies[current_node] = node_strategy;
      auto [path_cost, path_strategy] =
          _OptimizeOverPath(problem, path, edges_within_path, node_strategies,
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
std::pair<iopddl::TotalCost, std::vector<NodeStrategyIdx>> OptimizeOverPath(
    const iopddl::Problem& problem, const std::vector<NodeIdx>& path,
    std::vector<NodeStrategyIdx>& node_strategies,
    const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency) {
  std::vector<NodeStrategyIdx> old_strategies(path.size(), 0);
  for (int i = 0; i < path.size(); ++i) {
    old_strategies[i] = node_strategies[path[i]];
  }
  std::vector<EdgeIdx> edges_within_path =
      GetEdgesWithinPath(problem, path, /*outward_edges=*/adjacency.first);

  iopddl::TotalCost original_path_cost = ComputePathCost(
      problem, adjacency, path, edges_within_path, node_strategies);
  auto [new_path_cost, best_path_strategies] =
      _OptimizeOverPath(problem, path, edges_within_path, node_strategies,
                        adjacency, path.size());

  // node_strategies could change within _OptimizeOverPath, so we restore the
  // original sharding strategies for the nodes on the path.
  for (int i = 0; i < path.size(); ++i) {
    node_strategies[path[i]] = old_strategies[i];
  }
  iopddl::TotalCost cost_delta = new_path_cost - original_path_cost;
  CHECK_LE(cost_delta, 0);
  return {cost_delta, best_path_strategies};
}

// Check if a path's new configuration satisfies the memory constraints.
absl::flat_hash_map<LivenessIdx, iopddl::TotalUsage> GetNewMemorySlack(
    const iopddl::Problem& problem, const std::vector<NodeIdx>& path,
    const std::vector<NodeStrategyIdx>& path_strategies,
    const std::vector<NodeStrategyIdx>& node_strategies,
    const std::vector<std::vector<LivenessIdx>>& node_to_active_times,
    const std::vector<iopddl::TotalUsage>& memory_slack) {
  absl::flat_hash_map<LivenessIdx, iopddl::TotalUsage> new_memory_slack;
  for (int i = 0; i < path.size(); ++i) {
    NodeIdx node = path[i];
    if (!node_to_active_times[node].empty()) {
      for (LivenessIdx t : node_to_active_times[node]) {
        if (!new_memory_slack.contains(t)) {
          new_memory_slack[t] = memory_slack[t];
        }
        new_memory_slack[t] -=
            (problem.nodes[node].strategies[path_strategies[i]].usage -
             problem.nodes[node].strategies[node_strategies[node]].usage);
      }
    }
  }
  return new_memory_slack;
}

// Update `node_strategies` for the nodes in `path` if `new_path_strategies` is
// a feasible set of improving changes. Returns true iff the update is accepted.
bool UpdateNodeStrategies(
    const iopddl::Problem& problem, const std::vector<NodeIdx>& path,
    const std::vector<NodeStrategyIdx>& new_path_strategies,
    std::vector<NodeStrategyIdx>& node_strategies,
    const std::string& memory_mode,
    std::vector<iopddl::TotalUsage>& memory_slack,
    const std::vector<std::vector<LivenessIdx>>& node_to_active_times) {
  if (memory_mode == "inactive") {
    for (int i = 0; i < path.size(); ++i) {
      node_strategies[path[i]] = new_path_strategies[i];
    }
  } else if (memory_mode == "active") {
    // Check: the new strategy satisfies the memory constraints.
    const auto new_memory_slack_at_times =
        GetNewMemorySlack(problem, path, new_path_strategies, node_strategies,
                          node_to_active_times, memory_slack);
    for (const auto& [time_step, new_slack] : new_memory_slack_at_times) {
      if (new_slack < 0) {
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

std::tuple<iopddl::TotalCost, std::vector<NodeIdx>,
           std::vector<NodeStrategyIdx>>
SampleAndOptimizePath(const iopddl::Problem& problem,
                      std::vector<NodeStrategyIdx>& node_strategies,
                      const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency,
                      const int path_length, std::mt19937_64& rng) {
  std::vector<NodeIdx> path =
      SamplePath(problem, adjacency.first, path_length, rng);
  if (path.size() != path_length + 1) {
    return {0, {}, {}};
  }
  const auto [cost_delta, new_path_strategies] =
      OptimizeOverPath(problem, path, node_strategies, adjacency);
  // Check that the new path strategy improves the cost.
  if (cost_delta == 0) {
    return {0, {}, {}};
  }
  return {cost_delta, path, new_path_strategies};
}

iopddl::TotalCost RunPathOptimization(
    const iopddl::Problem& problem,
    std::vector<NodeStrategyIdx>& node_strategies,
    const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency,
    const int path_length, const std::string& memory_mode,
    std::vector<iopddl::TotalUsage>& memory_slack,
    const std::vector<std::vector<LivenessIdx>>& node_to_active_times,
    std::mt19937_64& rng) {
  auto [cost_delta, path, new_path_strategies] = SampleAndOptimizePath(
      problem, node_strategies, adjacency, path_length, rng);
  if (cost_delta < 0 &&
      UpdateNodeStrategies(problem, path, new_path_strategies, node_strategies,
                           memory_mode, memory_slack, node_to_active_times)) {
    return cost_delta;
  }
  return 0;
}

// Compare optimized values under `path_length` and `path_length + 1`, and
// then stick to the better path length for the remaining iterations.
int LearnPathLength(
    const iopddl::Problem& problem,
    std::vector<NodeStrategyIdx>& node_strategies,
    const std::pair<EdgeAdjacency, EdgeAdjacency>& adjacency,
    const int path_length, const std::string& memory_mode,
    std::vector<iopddl::TotalUsage>& memory_slack,
    const std::vector<std::vector<LivenessIdx>>& node_to_active_times,
    std::vector<iopddl::TotalCost>& cost_window, std::mt19937_64& rng) {
  CHECK_GE(path_length, 0);
  const int window_size = cost_window.size();
  std::vector<iopddl::TotalCost> cost_window_other = cost_window;
  std::vector<NodeStrategyIdx> node_strategies_other = node_strategies;
  std::vector<iopddl::TotalUsage> memory_slack_other = memory_slack;
  iopddl::TotalCost cost_delta = 0;
  for (int trial = 1; trial < window_size; ++trial) {
    cost_delta = RunPathOptimization(problem, node_strategies, adjacency,
                                     path_length, memory_mode, memory_slack,
                                     node_to_active_times, rng);
    cost_window[trial] = cost_window[trial - 1] + cost_delta;
  }
  for (int trial = 1; trial < window_size; ++trial) {
    cost_delta = RunPathOptimization(
        problem, node_strategies_other, adjacency, path_length + 1, memory_mode,
        memory_slack_other, node_to_active_times, rng);
    cost_window_other[trial] = cost_window_other[trial - 1] + cost_delta;
  }

  if (cost_window_other[window_size - 1] < cost_window[window_size - 1]) {
    memory_slack = std::move(memory_slack_other);
    node_strategies = std::move(node_strategies_other);
    cost_window = std::move(cost_window_other);
    return path_length + 1;
  }
  return path_length;
}

// Checks if the node-sharding strategy has a finite cost and satisfies the
// peak-memory constraint.
std::optional<AutoShardingViolationCode> ShardingStrategyHasViolation(
    const iopddl::Problem& problem,
    const std::vector<NodeStrategyIdx>& node_strategies) {
  const int num_nodes = problem.nodes.size();
  const int num_edges = problem.edges.size();
  // Check for infinite coefficients in the objective function.
  for (NodeIdx v = 0; v < num_nodes; ++v) {
    NodeStrategyIdx strategy = node_strategies[v];
    if (problem.nodes[v].strategies[strategy].cost >= absl::Int128Max()) {
      return AutoShardingViolationCode::kInfiniteCostViolationCode;
    }
    iopddl::TotalCost combined_cost =
        problem.nodes[v].strategies[strategy].cost;
    if (combined_cost < 0 || combined_cost >= absl::Int128Max()) {
      return AutoShardingViolationCode::kInfiniteCostViolationCode;
    }
  }
  for (EdgeIdx e = 0; e < num_edges; ++e) {
    EdgeStrategyIdx strategy = GetEdgeStrategy(problem, node_strategies, e);
    if (problem.edges[e].strategies[strategy].cost >= absl::Int128Max()) {
      return AutoShardingViolationCode::kInfiniteCostViolationCode;
    }
  }
  // Check that the peak-memory constraint is satisfied at each time step t.
  std::vector<iopddl::TotalUsage> live_memory;
  for (NodeIdx v = 0; v < num_nodes; ++v) {
    const iopddl::Node& node = problem.nodes[v];
    while (live_memory.size() < node.interval.second) {
      live_memory.push_back(0);
    }
    for (LivenessIdx t = node.interval.first; t < node.interval.second; ++t) {
      live_memory[t] += problem.nodes[v].strategies[node_strategies[v]].usage;
    }
  }
  for (LivenessIdx t = 0; t < live_memory.size(); ++t) {
    if (live_memory[t] > *problem.usage_limit) {
      return AutoShardingViolationCode::kMemoryViolationCode;
    }
  }
  return std::nullopt;
}

// Assigns all nodes to their first sharding configuration. If the assignment is
// infeasible, the output cost is negative and encodes the violation code.
AutoShardingSolverOutput SolveTrivial(const iopddl::Problem& problem) {
  std::vector<NodeStrategyIdx> node_strategies(problem.nodes.size(), 0);

  AutoShardingSolverOutput output;
  output.s_val = node_strategies;
  output.cost =
      static_cast<double>(ComputeShardingCost(problem, node_strategies));
  return output;
}

AutoShardingSolverOutput SolveRandom(const iopddl::Problem& problem,
                                     const int num_trials) {
  std::mt19937_64 rng(0);
  const int num_nodes = problem.nodes.size();
  std::vector<NodeStrategyIdx> best_node_strategies(num_nodes, -1);
  iopddl::TotalCost best_cost = absl::Int128Min();

  for (int trial = 0; trial < num_trials; ++trial) {
    std::vector<NodeStrategyIdx> node_strategies(num_nodes, -1);
    for (NodeIdx v = 0; v < num_nodes; ++v) {
      int num_strategies = problem.nodes[v].strategies.size();
      std::uniform_int_distribution<> dist(0, num_strategies - 1);
      NodeStrategyIdx strategy = dist(rng);
      node_strategies[v] = strategy;
    }
    iopddl::TotalCost cost = ComputeShardingCost(problem, node_strategies);

    bool have_feasible_solution = (best_cost >= 0);
    bool candidate_is_feasible = (cost >= 0);
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
  output.cost = static_cast<double>(best_cost);
  return output;
}

// Greedily selects the node sharding strategies. Valid modes:
// - "node-cost"
// - "node-memory"
AutoShardingSolverOutput SolveGreedy(const iopddl::Problem& problem,
                                     const std::string& mode) {
  const int num_nodes = problem.nodes.size();
  std::vector<NodeStrategyIdx> node_strategies(num_nodes, -1);

  for (NodeIdx v = 0; v < num_nodes; ++v) {
    int num_strategies = problem.nodes[v].strategies.size();
    NodeStrategyIdx best_strategy = -1;
    absl::int128 best_cost_or_usage = absl::Int128Min();
    for (NodeStrategyIdx strategy = 0; strategy < num_strategies; ++strategy) {
      absl::int128 cost_or_usage = 0;
      if (mode == "node-cost") {
        cost_or_usage = problem.nodes[v].strategies[strategy].cost;
      } else if (mode == "node-memory") {
        cost_or_usage = problem.nodes[v].strategies[strategy].usage;
      } else {
        CHECK(false) << absl::Substitute(
            "SolveGreedy mode $0 is not implemented.", mode);
      }
      if (best_strategy == -1 || cost_or_usage < best_cost_or_usage) {
        best_strategy = strategy;
        best_cost_or_usage = cost_or_usage;
      }
    }
    CHECK_NE(best_strategy, -1);
    node_strategies[v] = best_strategy;
  }

  AutoShardingSolverOutput output;
  output.s_val = node_strategies;
  output.cost =
      static_cast<double>(ComputeShardingCost(problem, node_strategies));
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
AutoShardingSolverOutput SolveRandomPathGreedy(const iopddl::Problem& problem,
                                               int path_length,
                                               const bool learn_path_length,
                                               const int num_trials,
                                               const double tolerance,
                                               const std::string& memory_mode) {
  std::mt19937_64 rng(0);
  if (memory_mode != "inactive" && memory_mode != "active") {
    CHECK(false) << absl::Substitute("Memory mode $0 is not implemented.",
                                     memory_mode);
  }

  // Initialize each node's sharding strategy with the least-memory usage.
  AutoShardingSolverOutput output;
  std::vector<NodeStrategyIdx> node_strategies =
      SolveGreedy(problem, "node-memory").s_val;
  const std::pair<EdgeAdjacency, EdgeAdjacency> adjacency =
      GetAdjacencyMatrix(problem);
  std::vector<std::vector<LivenessIdx>> node_to_active_times;
  std::vector<iopddl::TotalUsage> memory_slack;
  iopddl::TotalCost current_cost = ComputeShardingCost(
      problem, node_strategies, /*use_negative_violation_codes=*/false);
  if (memory_mode == "active") {
    node_to_active_times = GetNodeToActiveTimes(problem);
    memory_slack = TrackMemorySlack(problem, node_strategies);
  }
  // Phase 0: Return if minimum possible memory usage already exceeds budget.
  // This makes sense because we're initializing the solution w/
  // greedy-node-memory (which has minimum possible peak memory).
  std::optional<AutoShardingViolationCode> violation_code =
      ShardingStrategyHasViolation(problem, node_strategies);
  if (violation_code.has_value() && *violation_code == kMemoryViolationCode) {
    output.s_val = node_strategies;
    output.cost = static_cast<double>(current_cost);
    return output;
  }

  // Phase 1: Store the sharding costs of the last `window_size` trials.
  CHECK_GE(num_trials, 20);
  int window_size = std::min(static_cast<int>(0.15 * num_trials), 100000);
  std::vector<iopddl::TotalCost> cost_window(window_size, -1);
  cost_window[0] = current_cost;
  if (learn_path_length) {
    path_length = LearnPathLength(problem, node_strategies, adjacency,
                                  path_length, memory_mode, memory_slack,
                                  node_to_active_times, cost_window, rng);
  } else {
    for (int window_idx = 1; window_idx < window_size; ++window_idx) {
      current_cost += RunPathOptimization(
          problem, node_strategies, adjacency, path_length, memory_mode,
          memory_slack, node_to_active_times, rng);
      cost_window[window_idx] = current_cost;
    }
  }
  // Phase 2: Optimize the sharding cost with an early-stopping feature.
  current_cost = cost_window[window_size - 1];
  for (int trial = window_size; trial < num_trials; ++trial) {
    current_cost += RunPathOptimization(problem, node_strategies, adjacency,
                                        path_length, memory_mode, memory_slack,
                                        node_to_active_times, rng);
    if (1.0 - static_cast<double>(current_cost) /
                  static_cast<double>(cost_window[trial % window_size]) <
        tolerance) {
      break;
    }
    cost_window[trial % window_size] = current_cost;
  }

  output.s_val = node_strategies;
  output.cost =
      static_cast<double>(ComputeShardingCost(problem, node_strategies));
  return output;
}

}  // namespace

absl::StatusOr<AutoShardingSolverOutput> RunHeuristicSolver(
    const iopddl::Problem& problem, const std::string& algorithm) {
  absl::Time start_time = absl::Now();
  AutoShardingSolverOutput output;
  if (algorithm == "trivial") {
    output = SolveTrivial(problem);
  } else if (algorithm == "random") {
    output = SolveRandom(problem, 10000);
  } else if (algorithm == "greedy-node-cost") {
    output = SolveGreedy(problem, "node-cost");
  } else if (algorithm == "greedy-node-memory") {
    output = SolveGreedy(problem, "node-memory");
  } else if (algorithm == "random-path-greedy") {
    const int num_trials =
        2 * problem.edges.size() * std::log(problem.edges.size());
    output = SolveRandomPathGreedy(problem, /*path_length=*/1,
                                   /*learn_path_length=*/true, num_trials,
                                   /*tolerance=*/0.0,
                                   /*memory_mode=*/"active");
  } else {
    CHECK(false) << absl::Substitute("Algorithm $0 is not implemented.",
                                     algorithm);
  }
  auto duration = absl::Now() - start_time;
  LOG(INFO) << "Solver took " << absl::ToInt64Milliseconds(duration) << " ms";
  LOG(INFO) << "Objective value: " << output.cost;
  LOG(INFO) << "Total Cost: " << ComputeShardingCost(problem, output.s_val);
  return output;
}

iopddl::TotalCost ComputeShardingCost(
    const iopddl::Problem& problem,
    const std::vector<NodeStrategyIdx>& node_strategies,
    const bool use_negative_violation_codes) {
  iopddl::TotalCost cost = 0;
  for (NodeIdx v = 0; v < problem.nodes.size(); ++v) {
    NodeStrategyIdx strategy = node_strategies[v];
    cost += problem.nodes[v].strategies[strategy].cost;
  }
  for (EdgeIdx e = 0; e < problem.edges.size(); ++e) {
    EdgeStrategyIdx strategy = GetEdgeStrategy(problem, node_strategies, e);
    cost += problem.edges[e].strategies[strategy].cost;
  }
  if (use_negative_violation_codes) {
    std::optional<AutoShardingViolationCode> violation_code =
        ShardingStrategyHasViolation(problem, node_strategies);
    if (violation_code.has_value()) {
      cost = -1 * (*violation_code);
    }
  }
  return cost;
}

absl::Status ValidateProblem(const iopddl::Problem& problem) {
  const int num_nodes = problem.nodes.size();
  const int num_edges = problem.edges.size();

  for (NodeIdx u = 0; u < num_nodes; ++u) {
    const int num_strategies = problem.nodes[u].strategies.size();
    TF_RET_CHECK(num_strategies >= 1);
    for (NodeStrategyIdx strategy = 0; strategy < num_strategies; ++strategy) {
      TF_RET_CHECK(problem.nodes[u].strategies[strategy].cost >= 0);
      TF_RET_CHECK(problem.nodes[u].strategies[strategy].usage >= 0);
    }
  }

  absl::btree_set<std::pair<int, int>> edges_seen;
  for (EdgeIdx e = 0; e < num_edges; ++e) {
    const int u = problem.edges[e].nodes[0];
    const int v = problem.edges[e].nodes[1];
    TF_RET_CHECK(u >= 0);
    TF_RET_CHECK(u < num_nodes);
    TF_RET_CHECK(v >= 0);
    TF_RET_CHECK(v < num_nodes);
    TF_RET_CHECK(u < v);
    TF_RET_CHECK(edges_seen.count({u, v}) == 0);
    edges_seen.insert({u, v});

    const int num_strategies = problem.edges[e].strategies.size();
    const int num_u_strategies = problem.nodes[u].strategies.size();
    const int num_v_strategies = problem.nodes[v].strategies.size();
    CHECK_EQ(num_strategies, num_u_strategies * num_v_strategies);
  }
  return absl::OkStatus();
}

}  // namespace spmd
}  // namespace xla
