/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/statusor.h"
#include "ortools/linear_solver/linear_solver.h"

using MPSolver = operations_research::MPSolver;
using MPVariable = operations_research::MPVariable;

namespace xla {
namespace spmd {

struct AutoShardingSolverRequest {
  int64_t num_nodes = 0;
  int64_t memory_budget = -1;
  std::vector<int> s_len;
  std::vector<NodeIdx> s_follow;
  std::vector<NodeStrategyIdx> s_hint;
  std::vector<std::pair<NodeIdx, NodeIdx>> e;
  std::vector<std::vector<NodeIdx>> live;
  std::vector<std::vector<double>> c;
  std::vector<std::vector<double>> d;
  std::vector<std::vector<double>> m;
  std::vector<std::vector<double>> p;
  std::vector<std::vector<double>> r;
  std::vector<std::vector<double>> t;
  std::vector<std::pair<NodeIdx, NodeIdx>> a;
  std::vector<std::vector<double>> v;
  std::vector<std::string> instruction_names;
  std::optional<int64_t> solver_timeout_in_seconds;
  std::optional<double> overbudget_coeff = 1e6;
  std::optional<double> makespan_coeff;
  std::optional<double> max_departures;
  bool crash_at_infinity_costs_check = false;
  bool compute_iis = true;
  double saltiplier = 0.001;  // Modifies each objective term by at most 0.1%
};

struct AutoShardingSolverResult {
 public:
  AutoShardingSolverResult(
      StatusOr<std::tuple<std::vector<NodeStrategyIdx>,
                          std::vector<EdgeStrategyIdx>, double>>
          status,
      bool skip_auto_sharding)
      : status(status), skip_auto_sharding(skip_auto_sharding) {}
  bool operator==(const AutoShardingSolverResult& other) const;
  StatusOr<std::tuple<std::vector<int64_t>, std::vector<int64_t>, double>>
      status;
  bool skip_auto_sharding;
};

AutoShardingSolverResult CallORToolsSolver(
    const AutoShardingSolverRequest& request);

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

  double cost() const;

  bool operator==(const CostComponents& other) const;
};

// Captures the metrics, lower bounds, and constraint violations for the
// sharding result.
struct AutoShardingEvaluation {
  // A set of constraint violations; should be empty for any viable solution.
  absl::flat_hash_set<AutoShardingViolationCode> violation_codes;

  // A breakdown & lower bound for each individual cost component.
  CostComponents total;
  CostComponents lower_bound;

  // How many instructions departed from the "default" sharding strategy.
  double total_departures = 0.0;

  // The (raw) total makespan, i.e. not scaled by the makespan coefficient.
  double total_makespan = 0.0;

  bool operator==(const AutoShardingEvaluation& other) const;
};

// Evaluates the given solver result w.r.t. the input request, computing various
// solution quality metrics and validating the consistency of hard constraints.
AutoShardingEvaluation Evaluate(const AutoShardingSolverRequest& request,
                                const AutoShardingSolverResult& result);

// Produces a list of rationales for why an alternate result may be suboptimal.
std::vector<std::string> Rationalize(const AutoShardingSolverRequest& request,
                                     const AutoShardingSolverResult& result,
                                     const AutoShardingSolverResult& subopt);

// Creates and returns a variable for makespan.
MPVariable* CreateMakespanVar(const AutoShardingSolverRequest& request,
                              const std::vector<std::vector<MPVariable*>>& e,
                              MPSolver& solver);

double EvaluateMakespan(const AutoShardingSolverRequest& request,
                        const AutoShardingSolverResult& result,
                        AutoShardingEvaluation& evaluation);

}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_H_
