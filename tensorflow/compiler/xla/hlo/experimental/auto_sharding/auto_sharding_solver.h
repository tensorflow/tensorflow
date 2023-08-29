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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_H_
#define TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/statusor.h"

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
  std::vector<std::vector<double>> r;
  std::vector<std::pair<NodeIdx, NodeIdx>> a;
  std::vector<std::vector<double>> v;
  std::vector<std::string> instruction_names;
  std::optional<int64_t> solver_timeout_in_seconds;
  bool crash_at_infinity_costs_check = false;
  bool compute_iis = true;
  double saltiplier = 0.0001;  // Modifies each objective term by at most 0.01%
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
  kInfiniteCostViolationCode,  // Some node or edge incurs infinite cost
  kMemoryViolationCode,        // The solution eclipses the memory budget
};

// Captures the metrics, lower bounds, and constraint violations for the
// sharding result.
struct AutoShardingEvaluation {
  // A set of constraint violations; should be empty for any viable solution.
  absl::flat_hash_set<AutoShardingViolationCode> violation_codes;

  // A breakdown of each individual cost component.
  double total_communication_cost = 0.0;
  double total_computation_cost = 0.0;
  double total_resharding_cost = 0.0;

  // The total (global) objective cost.
  double total_cost = 0.0;

  // A lower bound for each individual cost component.
  double lower_bound_communication_cost = 0.0;
  double lower_bound_computation_cost = 0.0;
  double lower_bound_resharding_cost = 0.0;

  // A lower bound on the total (global) objective cost.
  double lower_bound_cost = 0.0;

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

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_H_
