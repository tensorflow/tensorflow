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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "ortools/linear_solver/linear_solver.h"

namespace xla {
namespace spmd {

struct AutoShardingSolverOutput {
  std::vector<NodeStrategyIdx> s_val;
  double cost = -1.0;
  bool is_optimal = true;
  absl::flat_hash_set<LivenessIdx> peak_times;

  bool operator==(const AutoShardingSolverOutput& other) const;
};

// Scales down values to reduce the range of costs & coefficients in the solver.
AutoShardingSolverRequest ScaleRequest(
    const AutoShardingSolverRequest& request);

absl::StatusOr<AutoShardingSolverOutput> FormulateAndSolveMIPFromSolverRequest(
    const AutoShardingSolverRequest& request);

// TODO(fahrbach): Create AutoShardingHeuristicOptions proto with a oneof field.
// Runs a heuristic specified by one of the following values of `algorithm`:
// - "trivial"
// - "random"
// - "greedy-node-cost"
// - "greedy-node-memory"
// - "brkga"
absl::StatusOr<AutoShardingSolverOutput> RunHeuristicSolver(
    const AutoShardingSolverRequest& request, const std::string& algorithm);

enum AutoShardingViolationCode {
  kAliasViolationCode,     // Some node's strategy does not match its alias
  kFollowerViolationCode,  // Some node's strategy does not match its follower
  kInfiniteCostViolationCode,   // Some node or edge incurs infinite cost
  kMemoryViolationCode,         // The solution eclipses the memory budget
  kMaxDeparturesViolationCode,  // The solution has too many sharding departures
};

struct CostComponents {
  double communication_cost = 0.0;
  double computation_cost = 0.0;
  double resharding_cost = 0.0;
  double overbudget_cost = 0.0;
  double makespan_cost = 0.0;
  double max_memory = 0.0;

  double cost() const;

  bool operator==(const CostComponents& other) const;
};

// Captures the metrics, lower bounds, and constraint violations for the
// sharding result.
struct AutoShardingEvaluation {
  // A set of constraint violations; should be empty for any viable solution.
  absl::flat_hash_set<AutoShardingViolationCode> violation_codes;

  // A breakdown and lower bound for each individual cost component.
  CostComponents total;
  CostComponents lower_bound;

  // How many instructions departed from the "default" sharding strategy.
  double total_departures = 0.0;

  // The (raw) total makespan, i.e., not scaled by the makespan coefficient.
  double total_makespan = 0.0;

  bool operator==(const AutoShardingEvaluation& other) const;
};

// Evaluates the given solver result w.r.t. the input request, computing various
// solution quality metrics and validating the consistency of hard constraints.
AutoShardingEvaluation Evaluate(const AutoShardingSolverRequest& request,
                                const AutoShardingSolverOutput& result);

// Computes the objective value of the sharding strategy. If the objective value
// is infinite or the sharding is infeasible (e.g., violates the peak-memory
// constraint), then a negated `AutoShardingViolationCode` value is returned.
// This function is used instead of `Evaluate` for faster iteration loops in the
// heuristic solver library.
double ComputeShardingStrategyCost(
    const AutoShardingSolverRequest& request,
    const std::vector<NodeStrategyIdx>& node_strategies);

// Creates and returns a variable for makespan.
operations_research::MPVariable* CreateMakespanVar(
    const AutoShardingSolverRequest& request,
    const std::vector<std::vector<operations_research::MPVariable*>>& e,
    operations_research::MPSolver& solver);

double EvaluateMakespan(const AutoShardingSolverRequest& request,
                        const AutoShardingSolverOutput& result,
                        AutoShardingEvaluation& evaluation);

// Determines if strategy 'first' is dominated by strategy 'second' (i.e., its
// costs are all equal or worse, and it has identical alias mappings).
bool CheckDominance(const AutoShardingSolverRequest& request,
                    const std::vector<EdgeIdx>& src_edges,
                    const std::vector<EdgeIdx>& dst_edges,
                    const std::vector<AliasIdx>& src_aliases,
                    const std::vector<AliasIdx>& dst_aliases, NodeIdx node_idx,
                    NodeStrategyIdx first, NodeStrategyIdx second);

class StrategyShaver {
 public:
  explicit StrategyShaver(const AutoShardingSolverRequest& request);

  // For every node, examine each sharding strategy to see if it is dominated by
  // another.
  NodeStrategies FindShavedStrategies() const;

 private:
  const AutoShardingSolverRequest& request_;  // NOLINT
  std::vector<std::vector<EdgeIdx>> src_edge_map_;
  std::vector<std::vector<EdgeIdx>> dst_edge_map_;
  std::vector<std::vector<AliasIdx>> src_alias_map_;
  std::vector<std::vector<AliasIdx>> dst_alias_map_;
  std::vector<std::vector<NodeIdx>> followers_;
};

// Check fail if `request` is invalid (e.g., because of negative node costs).
// Note: This does not include checks for valid variable aliasing yet.
absl::Status ValidateRequest(const AutoShardingSolverRequest& request);

void SolverRequestCallback(const AutoShardingSolverRequest& request);

AutoShardingSolverOutput SolveBrkga(const AutoShardingSolverRequest& request);

}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_H_
