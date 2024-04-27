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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"

#ifdef PLATFORM_GOOGLE
#include "file/base/options.h"
#endif
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_memory.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/hash.h"
#include "tsl/platform/types.h"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_solver.pb.h"
#ifdef PLATFORM_GOOGLE
#include "file/base/helpers.h"
#include "util/task/status.pb.h"
#endif

namespace xla {
namespace spmd {

using ::operations_research::MPConstraint;
using ::operations_research::MPSolver;
using ::operations_research::MPVariable;

// We need to nudge the maximum cost (if present) slightly, since the constraint
// solver cannot guarantee exact numerical precision.
constexpr double kMaxCostEpsilon = 1.0001;

// In the Mixed ILP, we model all memory-related terms (i.e., coefficients,
// bounds, etc.) using smaller absolute values, due to limitations on precision.
// To compensate, the overbudget objective coefficient must be amplified by the
// same amount.
constexpr double kMemoryMultiplier = 1e-5;

bool AutoShardingSolverOutput::operator==(
    const AutoShardingSolverOutput& other) const {
  return s_val == other.s_val && e_val == other.e_val && cost == other.cost &&
         peak_times == other.peak_times;
}

bool AutoShardingSolverResult::operator==(
    const AutoShardingSolverResult& other) const {
  return status == other.status &&
         skip_auto_sharding == other.skip_auto_sharding;
}

void PrintLargestInstructions(
    const std::vector<NodeStrategyIdx>& chosen_strategy,
    const AutoShardingSolverRequest& request) {
  if (!request.node_intervals().empty()) return;  // TODO(moffitt): Handle this.
  // This memory consumption computation is different from that in
  // PrintAutoShardingSolution() because L and m are created to be different
  // from liveness_set and strategy.memory_cost.
  std::vector<std::pair<LivenessIdx, double>> time_memory_usage;
  for (LivenessIdx time_idx = 0; time_idx < request.live_size(); ++time_idx) {
    double mem = 0.0;
    for (NodeIdx node_idx : request.live(time_idx).nodes()) {
      mem += request.memory_costs(node_idx).costs(chosen_strategy[node_idx]);
    }
    time_memory_usage.push_back({time_idx, mem});
  }
  struct {
    bool operator()(std::pair<LivenessIdx, double> a,
                    std::pair<LivenessIdx, double> b) const {
      return a.second > b.second;
    }
  } MemLarger;
  std::sort(time_memory_usage.begin(), time_memory_usage.end(), MemLarger);

  LOG(INFO) << "using m[] and L[], max memory usage: "
            << time_memory_usage.front().second / (1024 * 1024 * 1024)
            << " GB at time " << time_memory_usage.front().first;
  // Gets largest tensors in top k time steps.
  size_t k = 3;
  k = std::min(k, time_memory_usage.size());
  std::vector<std::pair<NodeIdx, double>> instruction_mem;
  absl::flat_hash_set<NodeIdx> instruction_set;
  for (auto usage_idx = 0; usage_idx < k; ++usage_idx) {
    LivenessIdx time_idx = time_memory_usage.at(usage_idx).first;
    for (NodeIdx node_idx : request.live(time_idx).nodes()) {
      double mem =
          request.memory_costs(node_idx).costs(chosen_strategy[node_idx]);
      if (mem > 100 * 1024 * 1024 &&
          instruction_set.find(node_idx) == instruction_set.end()) {
        instruction_mem.push_back({node_idx, mem});
        instruction_set.insert(node_idx);
      }
    }
  }
  std::sort(instruction_mem.begin(), instruction_mem.end(), MemLarger);

  size_t top_tensors = 10;
  top_tensors = std::min(top_tensors, instruction_mem.size());
  VLOG(1) << "Top " << top_tensors << " largest tensors:";
  for (size_t i = 0; i < top_tensors; ++i) {
    VLOG(1) << "instruction name: "
            << request.instruction_names(instruction_mem.at(i).first)
            << " memory usage: "
            << instruction_mem.at(i).second / (1024 * 1024 * 1024) << "GB";
  }
}

// Applies deterministic noise to the coefficient using `name` and `saltiplier`
// so that ties between equal terms can be broken in the solver's objective
// function. We include both a multiplicative term (in case the coefficient is
// large) and an additive term (in case the coefficient is zero).
void AddSalt(const std::string& name, const double saltiplier, double* coeff) {
  if (saltiplier <= 0.0) return;
  const tsl::uint64 hash = tsl::Hash64(name);  // stable across runs & platforms
  double salt = saltiplier * hash / std::numeric_limits<tsl::uint64>::max();
  *coeff = *coeff * (1.0 + salt) + salt;
}

AutoShardingSolverResult SolveAndExtractSolution(
    const AutoShardingSolverRequest& request,
    const std::vector<std::vector<MPVariable*>>& s,
    const std::vector<std::vector<MPVariable*>>& e,
    const MPVariable* overbudget_var, const MPVariable* makespan_var,
    MPSolver& solver);

double MinimumMemoryBudgetRequired(const AutoShardingSolverRequest& request) {
  double min_memory_budget_required_estimate = 0.0;
  for (LivenessIdx time_idx = 0; time_idx < request.live_size(); ++time_idx) {
    double min_memory_budget_required_estimate_local = 0.0;
    for (NodeIdx node_idx : request.live(time_idx).nodes()) {
      const auto& m = request.memory_costs(node_idx).costs();
      const double fixed_memory_cost = *std::min_element(m.begin(), m.end());
      min_memory_budget_required_estimate_local += fixed_memory_cost;
    }
    min_memory_budget_required_estimate =
        std::max(min_memory_budget_required_estimate,
                 min_memory_budget_required_estimate_local);
  }
  return min_memory_budget_required_estimate;
}

double MaxCoeff(
    const tsl::protobuf::RepeatedPtrField<AutoShardingSolverRequest_Costs>&
        cost_mat) {
  double max_coeff = 0.0;
  for (auto& costs : cost_mat) {
    for (auto& cost : costs.costs()) {
      if (cost < kInfinityCost) {
        max_coeff = std::max(max_coeff, cost);
      }
    }
  }
  return max_coeff;
}

void ScaleCoeffs(
    double scaling_factor,
    tsl::protobuf::RepeatedPtrField<AutoShardingSolverRequest_Costs>*
        cost_mat) {
  for (auto& costs : *cost_mat) {
    for (auto& cost : *costs.mutable_costs()) {
      if (cost < kInfinityCost) {
        cost = floor(cost * scaling_factor);
      }
    }
  }
}

AutoShardingSolverRequest ScaleRequest(
    const AutoShardingSolverRequest& request) {
  if (!request.has_coeff_limit()) return request;
  VLOG(0) << "Scaling request by coefficient limit: "
          << request.coeff_limit().coeff();
  double max_coeff = 0.0;
  max_coeff = std::max(max_coeff, MaxCoeff(request.communication_costs()));
  max_coeff = std::max(max_coeff, MaxCoeff(request.computation_costs()));
  max_coeff = std::max(max_coeff, MaxCoeff(request.resharding_costs()));
  if (max_coeff <= request.coeff_limit().coeff()) return request;
  const double scaling_factor = request.coeff_limit().coeff() / max_coeff;
  AutoShardingSolverRequest scaled_request = request;
  ScaleCoeffs(scaling_factor, scaled_request.mutable_communication_costs());
  ScaleCoeffs(scaling_factor, scaled_request.mutable_computation_costs());
  ScaleCoeffs(scaling_factor, scaled_request.mutable_resharding_costs());
  return scaled_request;
}

// Given the live matrix and memory costs (for nodes or edges), reduce terms and
// create constrained variables for the subsequent groups.
std::pair<int64_t, int64_t> ReduceMemoryTerms(
    MPSolver& solver, int64_t num_lives, int64_t num_primitives,
    const std::function<
        tsl::protobuf::RepeatedField<int64_t>(int64_t)>&  // NOLINT
        live,
    const tsl::protobuf::RepeatedPtrField<  // NOLINT
        AutoShardingSolverRequest_Pair>& intervals,
    const tsl::protobuf::RepeatedPtrField<  // NOLINT
        AutoShardingSolverRequest_Costs>& memory_costs,
    std::string_view prim_type,
    std::vector<std::vector<MPVariable*>>& prim_vars,
    std::vector<std::pair<int64_t, int64_t>>& reduced_intervals,
    std::vector<MPVariable*>& group_vars) {
  // If we've been given primitive intervals instead of a liveness matrix, we
  // need to update the # of lives in order to use the memory term reducer.
  for (const auto& interval : intervals) {
    if (interval.first() > interval.second()) continue;  // No interval defined.
    num_lives = std::max(num_lives, interval.second() + 1);
  }
  auto Intervals =
      [intervals](int64_t prim_idx) -> std::pair<int64_t, int64_t> {
    return {intervals.at(prim_idx).first(), intervals.at(prim_idx).second()};
  };
  MemoryTermReducer reducer;
  auto num_terms =
      intervals.empty()
          ? reducer.Reduce(num_lives, num_primitives, live)
          : reducer.Reduce(num_lives, num_primitives, std::move(Intervals));
  reduced_intervals = reducer.GetReducedIntervals();
  const auto& reduced_groups = reducer.GetReducedGroups();
  solver.MakeIntVarArray(reduced_groups.size(), 0.0, MPSolver::infinity(),
                         absl::StrCat("group_", prim_type), &group_vars);
  for (int64_t group_idx = 0; group_idx < group_vars.size(); ++group_idx) {
    MPConstraint* constraint = solver.MakeRowConstraint(
        -MPSolver::infinity(), 0.0,
        absl::StrCat("group_", prim_type, "[", group_idx, "]"));
    constraint->SetCoefficient(group_vars[group_idx], -1.0);
    for (const int64_t prim_idx : reduced_groups[group_idx]) {
      for (int64_t j = 0; j < prim_vars[prim_idx].size(); ++j) {
        double memory_cost = memory_costs.at(prim_idx).costs(j);
        memory_cost *= kMemoryMultiplier;
        const double accumulated_coefficient =
            constraint->GetCoefficient(prim_vars[prim_idx][j]);
        constraint->SetCoefficient(prim_vars[prim_idx][j],
                                   accumulated_coefficient + memory_cost);
      }
    }
  }
  return num_terms;
}

// Adds the appropriate memory terms (for nodes or edges) at the given time.
void AddMemoryTerms(
    const AutoShardingSolverRequest& request, MPSolver& solver,
    int64_t num_primitives,
    const std::vector<std::pair<int64_t, int64_t>>& intervals,
    const tsl::protobuf::RepeatedPtrField<  // NOLINT
        AutoShardingSolverRequest_Costs>& memory_costs,
    const MPVariable* overbudget_var,
    std::vector<std::vector<MPVariable*>>& prim_vars,
    std::vector<MPVariable*>& group_vars,
    absl::flat_hash_map<LivenessIdx, MPConstraint*>& constraints) {
  for (int64_t prim_idx = 0; prim_idx < intervals.size(); ++prim_idx) {
    for (int64_t time_idx = intervals[prim_idx].first;
         time_idx <= intervals[prim_idx].second; ++time_idx) {
      if (!constraints.contains(time_idx)) {
        MPConstraint* constraint = solver.MakeRowConstraint(
            -MPSolver::infinity(), request.memory_budget() * kMemoryMultiplier,
            absl::StrCat("mem[", time_idx, "]"));
        if (overbudget_var) constraint->SetCoefficient(overbudget_var, -1.0);
        constraints[time_idx] = constraint;
      }
      MPConstraint* constraint = constraints[time_idx];
      if (prim_idx >= num_primitives) {
        constraint->SetCoefficient(group_vars[prim_idx - num_primitives], 1.0);
        continue;
      }
      for (int64_t j = 0; j < prim_vars[prim_idx].size(); ++j) {
        double memory_cost = memory_costs.at(prim_idx).costs(j);
        memory_cost *= kMemoryMultiplier;
        const double accumulated_coefficient =
            constraint->GetCoefficient(prim_vars[prim_idx][j]);
        constraint->SetCoefficient(prim_vars[prim_idx][j],
                                   accumulated_coefficient + memory_cost);
      }
    }
  }
}

// Taking an auto-sharding problem (`request`) as an input, calls the OR tools
// CP-SAT solver and outputs a solution to the input problem.
//
// We formulate the auto-sharding process as the following ILP problem
// (correspondences to the fields of the request parameter are specified in
// parenthesis):
// Variables:
//   s[i]: Sharding strategy one-hot vector.
//         dim(s[i]) == # sharding strategies of the i-th XLA op
//         s_len[i] := dim(s[i]) in the arguments
//   e[i, j]: Strategy one-hot vector of edge i -> j.
//            dim(e[i, j]) == dim(s[i]) * dim(s[j])
// Constants:
//   N: Number of total XLA ops (request.num_nodes)
//   M: Memory budget (request.memory_budget)
//   E: Edge set {(i, j)} (request.edges)
//   L[t]: Index of live instructions at time t (request.live)
//   c[i]: Computation cost vector of instruction i (request.computation_costs)
//   d[i]: Communication cost vector of instruction i
//         (request.communication_costs)
//   m[i]: Memory cost vector of instruction i (request.memory_costs)
//         dim(c[i]) == dim(d[i]) == dim(m[i]) == dim(s[i])
//   r[i, j]: The resharding cost vector of edge i -> j
//            (request.resharding_costs)
//            dim(e[i, j]) == dim(r[i, j])
//   A: Alias set {(i, j)} (request.aliases)
//   v[i, j]: v[i, j](p, q) == 1 if strategy p is different than q, otherwise
//            v[i, j](p, q) == 0
//            (request.value_costs)
//            dim(e[i, j]) == dim(v[i, j])
// Problem:
//   Minimize sum_{0 <= i < N} s[i]^T * (c[i] + d[i])
//            + sum_{(i, j) in E} e[i, j]^T * r[i, j]
//   s.t.
//       Make sure s is one-hot:
//     0. Do not choose solutions with infinity cost (b/238210866).
//     a. For 0 <= i < N, s[i] in {0, 1} ^ dim(s[i])
//     b. For 0 <= i < N, s[i]^T * 1 == 1
//       Memory constraint:
//     c. For all t: sum_{i in L[t]} s[i]^T * m[i] <= M
//       Make sure e is one-hot:
//     d. For all (i, j) in E, e[i, j] in {0, 1} ^ dim(e[i, j])
//     e. For all (i, j) in E, e[i, j]^T * 1 == 1
//       Make sure s[i] and s[j] align with e[i, j]:
//     f. For all (i, j) in E, 0 <= p < dim(s[i]),
//        sum_{0 <= q < dim(s[j])} e[i, j](p * dim(s[j]) + q) <= s[i](p)
//     g. For all (i, j) in E, 0 <= q < dim(s[j]),
//        sum_{0 <= p < dim(s[i])} e[i, j](p * dim(s[j]) + q) <= s[j](q)
//     h. For all (i, j) in A and all (p, q),
//        s[i][p] + s[j][q] <= 1 if v[p, q] == 1.0
// Serialize parameters of the ILP problem as numpy arrays and call the python
// solver.

// Beyond what is described, note the following:
// 1. We also enforce that certain HLO ops have the same sharding as some other
//    HLO ops (think elementwise ops, for example). This information stored in
//    request.s_follow, where if s_follow[i] >= 0, then instruction i is forced
//    the share same sharding as s_follow[i].
// 2. If request.overbudget_coeff is present, we turn the hard memory budget
//    constraint into a soft constraint instead.
// 3. If request.makespan_coeff is present, the objective additionally includes
//    a makespan term. This is experimental and turned off by default.
// 4. request.max_departures is used only for debugging and can be ignored.
// 5. Note that due to our modeling of XLA's AllReduceReassociate optimization
//    (more details in CostGraph::CostGraph() in auto_sharding_cost_graph.cc,
//    and in CreateElementwiseOperatorStrategies() in auto_sharding.cc), there
//    can be a few (usually < 10) edges in the problem with negative costs. This
//    is guaranteed to never produce a negative overall cost for the graph,
//    however.
AutoShardingSolverResult CallORToolsSolver(
    const AutoShardingSolverRequest& unscaled_request) {
  const absl::Time start_time = absl::Now();
  const AutoShardingSolverRequest& request = ScaleRequest(unscaled_request);
  const size_t num_edges = request.edges_size();
  const int num_workers = 32;
  // SAT or SCIP
  std::unique_ptr<MPSolver> solver(std::make_unique<MPSolver>("", MPSolver::SAT_INTEGER_PROGRAMMING));
  CHECK(solver);
  solver->MutableObjective()->SetMinimization();
  std::string solver_parameter_str;
#ifdef PLATFORM_GOOGLE
  if (solver->ProblemType() ==
      operations_research::MPSolver::SAT_INTEGER_PROGRAMMING) {
    // Set random_seed, interleave_search and share_binary_clauses for
    // determinism, mip_max_bound (to handle large costs), and num_workers for
    // parallelism.
    solver_parameter_str =
        request.deterministic_mode()
            ? absl::StrCat(
                  "share_binary_clauses:false,random_seed:1,interleave_"
                  "search:true,mip_max_bound:1e9,num_workers:",
                  num_workers)
            : absl::StrCat("mip_max_bound:1e9,num_workers:", num_workers);
    solver->SetSolverSpecificParametersAsString(solver_parameter_str);
  }
#endif
  // Create variables
  std::vector<std::vector<MPVariable*>> s(request.num_nodes());
  std::vector<std::vector<MPVariable*>> e(num_edges);
  MPVariable* overbudget_var = nullptr;
  MPVariable* makespan_var = nullptr;

  size_t unique_nodes = 0;
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    if (request.s_follow(node_idx) < 0) {
      unique_nodes += 1;
      // Creates variables for instructions that do not follow others.
      solver->MakeBoolVarArray(request.s_len(node_idx),
                               absl::StrCat("s[", node_idx, "]"), &s[node_idx]);
    }
  }

  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    if (request.s_follow(node_idx) >= 0) {
      CHECK_EQ(request.s_len(node_idx),
               request.s_len(request.s_follow(node_idx)));
      // Copies the variable of followed instruction to the following
      // instruction.
      s[node_idx] = s[request.s_follow(node_idx)];
    }
  }

  size_t unique_edges = 0;
  std::vector<EdgeIdx> e_follow(num_edges, -1);
  absl::flat_hash_map<std::pair<NodeIdx, NodeIdx>, EdgeIdx> edge_map;
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    const auto& raw_edge = request.edges(edge_idx);
    const std::pair<NodeIdx, NodeIdx> edge(raw_edge.first(), raw_edge.second());
    auto followed_edge = edge;
    if (int f = request.s_follow(edge.first); f >= 0) followed_edge.first = f;
    if (int f = request.s_follow(edge.second); f >= 0) followed_edge.second = f;
    if (const auto& it = edge_map.find(followed_edge); it != edge_map.end()) {
      e[edge_idx] = e[it->second];  // Copy variable of followed edge
      e_follow[edge_idx] = it->second;
      continue;
    }
    unique_edges += 1;
    solver->MakeBoolVarArray(
        request.s_len(edge.first) * request.s_len(edge.second),
        absl::StrCat("e[", edge.first, ",", edge.second, "]"), &e[edge_idx]);
    edge_map.insert({followed_edge, edge_idx});
  }

  if (request.memory_budget() > 0 && request.has_overbudget_coeff()) {
    overbudget_var =
        solver->MakeNumVar(0.0, MPSolver::infinity(), "overbudget");
  }

  if (request.has_makespan_coeff()) {
    makespan_var = CreateMakespanVar(request, e, *solver);
  }

  // Construct objective function.
  // Node costs
  absl::flat_hash_set<MPVariable*> infinity_vars;
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
      double accumulated_coefficient =
          solver->MutableObjective()->GetCoefficient(s[node_idx][j]);
      double coefficient = request.computation_costs(node_idx).costs(j) +
                           request.communication_costs(node_idx).costs(j);
      if (coefficient >= kInfinityCost) {
        infinity_vars.insert(s[node_idx][j]);
        continue;
      }
      AddSalt(absl::StrCat(node_idx, "S", j), request.saltiplier(),
              &coefficient);
      solver->MutableObjective()->SetCoefficient(
          s[node_idx][j], accumulated_coefficient + coefficient);
    }
  }
  // Edge costs
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    for (EdgeStrategyIdx j = 0; j < e[edge_idx].size(); ++j) {
      double accumulated_coefficient =
          solver->MutableObjective()->GetCoefficient(e[edge_idx][j]);
      double coefficient = request.resharding_costs(edge_idx).costs(j);
      if (coefficient >= kInfinityCost) {
        infinity_vars.insert(e[edge_idx][j]);
        continue;
      }
      AddSalt(absl::StrCat(edge_idx, "E", j), request.saltiplier(),
              &coefficient);
      solver->MutableObjective()->SetCoefficient(
          e[edge_idx][j], accumulated_coefficient + coefficient);
    }
  }
  LOG(INFO) << "Number of infinity terms: " << infinity_vars.size();

  // Add constraints.
  // 0. Do not choose solutions with infinity costs, as it will make the
  // objective value so large that other solution choices do not matter anymore.
  // Also eliminate strategies that are known to be dominated by others.
  const NodeStrategies shaved_strategies =
      StrategyShaver(request).FindShavedStrategies();
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    if (s[node_idx].empty() || request.s_follow(node_idx) >= 0) continue;
    bool all_infinity = true;
    for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
      if (infinity_vars.contains(s[node_idx][j]) ||
          shaved_strategies.contains({node_idx, j})) {
        MPConstraint* constraint = solver->MakeRowConstraint(
            0.0, 0.0,
            absl::StrCat("infinitycost: s[", node_idx, "][", j, "] = 0"));
        constraint->SetCoefficient(s[node_idx][j], 1.0);
      } else {
        all_infinity = false;
      }
    }
    if (all_infinity) {
      LOG(FATAL) << "All of s[" << node_idx << "][*] have infinity costs";
    }
  }
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    if (e[edge_idx].empty() || e_follow[edge_idx] >= 0) continue;
    bool all_infinity = true;
    for (EdgeStrategyIdx j = 0; j < e[edge_idx].size(); ++j) {
      if (infinity_vars.contains(e[edge_idx][j])) {
        MPConstraint* constraint = solver->MakeRowConstraint(
            0.0, 0.0,
            absl::StrCat("infinitycost: e[", edge_idx, "][", j, "] = 0"));
        constraint->SetCoefficient(e[edge_idx][j], 1.0);
      } else {
        all_infinity = false;
      }
    }
    if (all_infinity) {
      auto err_msg = absl::StrCat("All of e[", request.edges(edge_idx).first(),
                                  "][", request.edges(edge_idx).second(),
                                  "][*] have infinity costs");
      if (request.crash_at_infinity_costs_check()) {
        LOG(FATAL) << err_msg;
      } else {
        LOG(WARNING) << err_msg;
        return AutoShardingSolverResult(absl::InternalError(err_msg), false);
      }
    }
  }

  // a. specified via "BoolVarArray"
  // b.
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    if (request.s_follow(node_idx) >= 0) continue;
    MPConstraint* constraint = solver->MakeRowConstraint(
        1.0, 1.0,
        absl::StrCat("sum(s[", node_idx, "][j] for j = [0 .. ",
                     s[node_idx].size(), ")) = 1"));
    for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
      constraint->SetCoefficient(s[node_idx][j], 1.0);
    }
  }
  // c.
  if (request.memory_budget() > 0) {
    auto LiveNodes =
        [request](int64_t live_idx) -> tsl::protobuf::RepeatedField<int64_t> {
      return request.live(live_idx).nodes();
    };
    auto LiveEdges =
        [request](int64_t live_idx) -> tsl::protobuf::RepeatedField<int64_t> {
      return request.live_edges(live_idx).edges();
    };
    std::vector<std::pair<int64_t, int64_t>> reduced_intervals_nodes,
        reduced_intervals_edges;
    std::vector<MPVariable*> group_node_vars, group_edge_vars;
    const absl::Time term_reduction_start_time = absl::Now();
    auto num_node_terms = ReduceMemoryTerms(
        *solver, request.live_size(), request.num_nodes(), std::move(LiveNodes),
        request.node_intervals(), request.memory_costs(), "node", s,
        reduced_intervals_nodes, group_node_vars);
    auto num_edge_terms =
        ReduceMemoryTerms(*solver, request.live_edges_size(),
                          request.edges_size(), std::move(LiveEdges),
                          request.edge_intervals(), request.memory_edge_costs(),
                          "edge", e, reduced_intervals_edges, group_edge_vars);
    const absl::Time term_reduction_end_time = absl::Now();
    const auto term_reduction_duration =
        term_reduction_end_time - term_reduction_start_time;
    LOG(INFO) << "Memory Term Reducer took "
              << absl::ToInt64Milliseconds(term_reduction_duration)
              << " ms and reduced the number of terms from "
              << num_node_terms.first + num_edge_terms.first << " to "
              << num_node_terms.second + num_edge_terms.second;
    absl::flat_hash_map<LivenessIdx, MPConstraint*> constraints;
    AddMemoryTerms(request, *solver, request.num_nodes(),
                   reduced_intervals_nodes, request.memory_costs(),
                   overbudget_var, s, group_node_vars, constraints);
    if (request.enable_memory_edge_costs()) {
      AddMemoryTerms(request, *solver, request.edges_size(),
                     reduced_intervals_edges, request.memory_edge_costs(),
                     overbudget_var, e, group_edge_vars, constraints);
    }
    if (overbudget_var) {
      solver->MutableObjective()->SetCoefficient(
          overbudget_var,
          request.overbudget_coeff().coeff() / kMemoryMultiplier);
    }
    LOG(INFO) << "Minimum memory budget estimate: "
              << MinimumMemoryBudgetRequired(request);
    LOG(INFO) << "Using memory budget: "
              << static_cast<double>(request.memory_budget());
  }

  // d. specified via "BoolVarArray"
  // e.
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    if (e_follow[edge_idx] >= 0) continue;
    const auto& edge = request.edges(edge_idx);
    MPConstraint* constraint = solver->MakeRowConstraint(
        1.0, 1.0,
        absl::StrCat("sum(e[", edge.first(), "][", edge.second(), "][*]) = 1"));
    for (EdgeStrategyIdx j = 0; j < e[edge_idx].size(); ++j) {
      constraint->SetCoefficient(e[edge_idx][j], 1.0);
    }
  }
  // f.
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    if (e_follow[edge_idx] >= 0) continue;
    const auto& edge = request.edges(edge_idx);
    for (NodeStrategyIdx p = 0; p < s[edge.first()].size(); ++p) {
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), 0,
          absl::StrCat("f for i = ", edge_idx, ", p = ", p));
      constraint->SetCoefficient(s[edge.first()][p], -1.0);
      for (NodeStrategyIdx q = 0; q < s[edge.second()].size(); ++q) {
        const EdgeStrategyIdx j = p * s[edge.second()].size() + q;
        constraint->SetCoefficient(e[edge_idx][j], 1.0);
      }
    }
  }
  // g.
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    if (e_follow[edge_idx] >= 0) continue;
    const auto& edge = request.edges(edge_idx);
    for (NodeStrategyIdx q = 0; q < s[edge.second()].size(); ++q) {
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), 0,
          absl::StrCat("g for i = ", edge_idx, ", q = ", q));
      constraint->SetCoefficient(s[edge.second()][q], -1.0);
      for (NodeStrategyIdx p = 0; p < s[edge.first()].size(); ++p) {
        const EdgeStrategyIdx j = p * s[edge.second()].size() + q;
        constraint->SetCoefficient(e[edge_idx][j], 1.0);
      }
    }
  }
  // h.
  absl::flat_hash_set<std::pair<NodeIdx, NodeIdx>> alias_set;
  for (auto alias_idx = 0; alias_idx < request.aliases_size(); ++alias_idx) {
    const auto& raw_alias = request.aliases(alias_idx);
    const std::pair<NodeIdx, NodeIdx> alias(raw_alias.first(),
                                            raw_alias.second());
    if (alias_set.contains(alias)) continue;
    alias_set.insert(alias);
    const auto& value_costs = request.value_costs(alias_idx).costs();
    for (NodeStrategyIdx p = 0; p < s[alias.first].size(); ++p) {
      for (NodeStrategyIdx q = 0; q < s[alias.second].size(); ++q) {
        // if lhs == 1
        if (value_costs[p * s[alias.second].size() + q] > 0.5) {
          MPConstraint* constraint = solver->MakeRowConstraint(
              -MPSolver::infinity(), 1,
              absl::StrCat("s[", alias.first, "][", p, "] + s[", alias.second,
                           "][", q, "] <= 1"));
          constraint->SetCoefficient(s[alias.first][p], 1.0);
          constraint->SetCoefficient(s[alias.second][q], 1.0);
        }
      }
    }
  }
  if (request.has_max_departures()) {
    MPConstraint* constraint = solver->MakeRowConstraint(
        0, request.max_departures().coeff(),
        absl::StrCat("departures <= ", request.max_departures().coeff()));
    for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
      for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
        double accumulated_coefficient =
            constraint->GetCoefficient(s[node_idx][j]);
        double departure_cost = request.departure_costs(node_idx).costs(j);
        constraint->SetCoefficient(s[node_idx][j],
                                   accumulated_coefficient + departure_cost);
      }
    }
  }
  if (request.has_max_cost()) {
    double max_cost = kMaxCostEpsilon * request.max_cost().coeff();
    max_cost -= solver->Objective().offset();
    MPConstraint* cost_constraint = solver->MakeRowConstraint(
        -MPSolver::infinity(), max_cost, "cost_constraint");
    for (const auto [var, coeff] : solver->Objective().terms()) {
      cost_constraint->SetCoefficient(var, coeff);
    }
  }

  if (!request.s_hint().empty() && !request.deterministic_mode()) {
    std::vector<std::pair<const MPVariable*, double>> hint;
    for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
      if (request.s_follow(node_idx) >= 0) continue;
      for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
        double hint_val = (request.s_hint(node_idx) == j) ? 1.0 : 0.0;
        hint.push_back({s[node_idx][j], hint_val});
      }
    }
    solver->SetHint(hint);
  }

#ifdef PLATFORM_GOOGLE
  // Exports the model for debugging.
  bool dump_model = false;
  if (dump_model) {
    operations_research::MPModelProto model_proto;
    solver->ExportModelToProto(&model_proto);
    auto write_status = file::SetTextProto(
        // Modify this file path if needed.
        absl::StrCat("/tmp/model_", solver->NumVariables(), ".proto"),
        model_proto, file::Defaults());
    if (!write_status.ok()) {
      LOG(ERROR) << write_status.message();
    }
  }
  // Exports the *unscaled* solver request proto for debugging.
  bool dump_solver_request = false;
  if (dump_solver_request) {
    uint64_t solver_request_fprint =
        tsl::Fingerprint64(unscaled_request.SerializeAsString());
    std::string request_dump_path =
        absl::StrCat("/tmp/solver_request_", unscaled_request.request_name(),
                     "_", solver_request_fprint, ".proto");
    auto write_status = file::SetBinaryProto(
        // Modify this file path if needed.
        request_dump_path, unscaled_request, file::Defaults());
    VLOG(5) << "Dumped solver request to " << request_dump_path;
    if (!write_status.ok()) {
      LOG(ERROR) << write_status.message();
    }
  }
#endif
  if (request.has_solver_timeout()) {
    solver->SetTimeLimit(
        absl::Seconds(request.solver_timeout().solver_timeout_in_seconds()));
  }
  if (request.enable_output()) {
    solver->EnableOutput();
  }
  VLOG(0) << "Starting solver " << solver->ProblemType() << "\n"
          << "Solver parameter string: " << solver_parameter_str << "\n"
          << "Number of workers: " << num_workers << "\n"
          << "Number of threads: " << solver->GetNumThreads() << "\n"
          << "Time limit: " << solver->time_limit() << "\n"
          << "Request valid: " << ValidateRequest(request).ok() << "\n"
          << "Aliases: " << request.aliases_size() << "\n"
          << "Unique nodes: " << unique_nodes << "\n"
          << "Unique edges: " << unique_edges << "\n"
          << "Total instructions: " << request.num_nodes() << "\n"
          << "Total edges: " << request.edges_size() << "\n"
          << "Memory budget: " << request.memory_budget() / (1024 * 1024 * 1024)
          << "GB\n"
          << "Number variables for ILP: " << solver->NumVariables() << "\n"
          << "Number of ILP constraints: " << solver->NumConstraints() << "\n"
          << "Deterministic mode: " << request.deterministic_mode() << "\n"
          << "Module name: " << request.module_name();
  if (request.has_max_cost()) {
    VLOG(0) << "Max cost: " << request.max_cost().coeff();
  }
  auto result = SolveAndExtractSolution(request, s, e, overbudget_var,
                                        makespan_var, *solver);
  if (result.status.ok()) {
    const AutoShardingEvaluation evaluation =
        Evaluate(unscaled_request, result);
    LOG(INFO) << "*** Total costs for the (unscaled) solver request ***";
    LOG(INFO) << "Total Communication Cost: "
              << evaluation.total.communication_cost
              << " (lower bound: " << evaluation.lower_bound.communication_cost
              << ")";
    LOG(INFO) << "Total Computation Cost: " << evaluation.total.computation_cost
              << " (lower bound: " << evaluation.lower_bound.computation_cost
              << ")";
    LOG(INFO) << "Total Resharding Cost: " << evaluation.total.resharding_cost
              << " (lower bound: " << evaluation.lower_bound.resharding_cost
              << ")";
    LOG(INFO) << "Total Overbudget Cost: " << evaluation.total.overbudget_cost
              << " (lower bound: " << evaluation.lower_bound.overbudget_cost
              << ")";
    LOG(INFO) << "Total Makespan Cost: " << evaluation.total.makespan_cost
              << " (lower bound: " << evaluation.lower_bound.makespan_cost
              << ")";
    LOG(INFO) << "Total Cost: " << evaluation.total.cost()
              << " (lower bound: " << evaluation.lower_bound.cost() << ")";
    LOG(INFO) << "Total Departures: " << evaluation.total_departures;
    LOG(INFO) << "Total Makespan: " << evaluation.total_makespan;
    LOG(INFO) << "Total Violations: " << evaluation.violation_codes.size();
  }
  const absl::Time end_time = absl::Now();
  const auto duration = end_time - start_time;
  LOG(INFO) << "Solver took " << absl::ToInt64Milliseconds(duration) << " ms";
  return result;
}

std::vector<NodeStrategyIdx> GetChosenNodeStrategy(
    const AutoShardingSolverRequest& request,
    const std::vector<std::vector<MPVariable*>>& s) {
  std::vector<NodeStrategyIdx> chosen_node_strategy(request.num_nodes(), -1);
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
      // if lhs == 1
      if (s[node_idx][j]->solution_value() > 0.5) {
        chosen_node_strategy[node_idx] = j;
        break;
      }
    }
  }
  return chosen_node_strategy;
}

std::vector<EdgeStrategyIdx> GetChosenEdgeStrategy(
    const AutoShardingSolverRequest& request,
    const std::vector<std::vector<MPVariable*>>& e) {
  size_t num_edges = request.edges_size();
  std::vector<NodeStrategyIdx> chosen_edge_strategy(num_edges, -1);
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    for (EdgeStrategyIdx j = 0; j < e[edge_idx].size(); ++j) {
      // if lhs == 1
      if (e[edge_idx][j]->solution_value() > 0.5) {
        chosen_edge_strategy[edge_idx] = j;
        break;
      }
    }
  }
  return chosen_edge_strategy;
}

AutoShardingSolverResult SolveAndExtractSolution(
    const AutoShardingSolverRequest& request,
    const std::vector<std::vector<MPVariable*>>& s,
    const std::vector<std::vector<MPVariable*>>& e,
    const MPVariable* overbudget_var, const MPVariable* makespan_var,
    MPSolver& solver) {
  auto status = solver.Solve();
  LOG(INFO) << "Solver Status: " << status;

  if (status == operations_research::MPSolver::INFEASIBLE) {
    LOG(ERROR) << "MPSolver could not find any feasible solution.";
#ifdef PLATFORM_GOOGLE
    if (request.compute_iis()) {
      operations_research::MPModelRequest model_request;
      solver.ExportModelToProto(model_request.mutable_model());
      if (solver.ProblemType() ==
          operations_research::MPSolver::SAT_INTEGER_PROGRAMMING) {
        model_request.set_solver_type(
            operations_research::MPModelRequest::SAT_INTEGER_PROGRAMMING);
      } else if (solver.ProblemType() == operations_research::MPSolver::
                                             SCIP_MIXED_INTEGER_PROGRAMMING) {
        model_request.set_solver_type(operations_research::MPModelRequest::
                                          SCIP_MIXED_INTEGER_PROGRAMMING);
      }
      model_request.set_solver_time_limit_seconds(100);
      auto iis = MPSolver::ComputeIrreducibleInfeasibleSubset(model_request);
      LOG(INFO) << iis.status().DebugString();
      LOG(INFO) << "Infeasible constraints: ";
      for (int index : iis.constraint_index()) {
        LOG(INFO) << " - " << model_request.model().constraint(index).name();
      }
      for (int index : iis.general_constraint_index()) {
        LOG(INFO)
            << " - "
            << model_request.model().general_constraint(index).DebugString();
      }
    }
#endif

    return AutoShardingSolverResult(
        absl::InternalError("MPSolver could not find any feasible solution."),
        false);
  } else if (status == operations_research::MPSolver::MODEL_INVALID) {
    LOG(FATAL) << "Solver says that the input MIP is invalid. This is most "
                  "likely a bug and should be reported.";
  } else if (status != operations_research::MPSolver::OPTIMAL) {
    auto err_msg = "Solver timed out.";
    return AutoShardingSolverResult(absl::InternalError(err_msg), true);
  }

  // Fingerprint the model & solution (useful when checking for determinism).
  // We use TensorFlow's fingerprint library here, which differs from CP-SAT's.
  operations_research::MPModelProto model_proto;
  solver.ExportModelToProto(&model_proto);
  uint64_t model_fprint = tsl::Fingerprint64(model_proto.SerializeAsString());
  operations_research::MPSolutionResponse response;
  solver.FillSolutionResponseProto(&response);
  response.clear_solve_info();  // Remove for fingerprint; can vary between runs
  uint64_t solution_fprint = tsl::Fingerprint64(response.SerializeAsString());

  LOG(INFO) << "Objective value: " << solver.Objective().Value()
            << " Model fingerprint: " << model_fprint
            << " Solution fingerprint: " << solution_fprint;
  if (solver.Objective().Value() >= kInfinityCost) {
    LOG(WARNING) << "Objective (" << solver.Objective().Value()
                 << ") is larger than kInfinityCost. It means the solver "
                    "chooses a solution with kInfinityCost and there may be "
                    "numerical issues when the solver considering other costs.";
  }
  if (VLOG_IS_ON(10)) {
    // Print solver information for debugging. This hasn't been useful so far,
    // so leave it at VLOG level 10.
    VLOG(10) << "MODEL:";
    XLA_VLOG_LINES(10, model_proto.DebugString());
    VLOG(10) << "RESPONSE:";
    XLA_VLOG_LINES(10, response.DebugString());
  }

  // Return value
  size_t num_edges = request.edges_size();
  double unsalted_objective = 0.0;
  const std::vector<NodeStrategyIdx> chosen_node_strategy =
      GetChosenNodeStrategy(request, s);
  const std::vector<EdgeStrategyIdx> chosen_edge_strategy =
      GetChosenEdgeStrategy(request, e);
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    const NodeStrategyIdx j = chosen_node_strategy[node_idx];
    unsalted_objective += request.computation_costs(node_idx).costs(j) +
                          request.communication_costs(node_idx).costs(j);
  }
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    const EdgeStrategyIdx j = chosen_edge_strategy[edge_idx];
    unsalted_objective += request.resharding_costs(edge_idx).costs(j);
  }
  if (overbudget_var) {
    unsalted_objective += request.overbudget_coeff().coeff() *
                          overbudget_var->solution_value() / kMemoryMultiplier;
  }
  if (makespan_var) {
    unsalted_objective +=
        request.makespan_coeff().coeff() * makespan_var->solution_value();
  }

  LOG(INFO) << "Unsalted objective value: " << unsalted_objective;
  LOG(INFO) << "N = " << request.num_nodes();
  if (request.memory_budget() < 0) {
    LOG(INFO) << "memory budget: -1";
  } else {
    LOG(INFO) << "memory budget: "
              << request.memory_budget() / (1024 * 1024 * 1024) << " GB";
  }
  PrintLargestInstructions(chosen_node_strategy, request);
  const AutoShardingSolverOutput output = {std::move(chosen_node_strategy),
                                           std::move(chosen_edge_strategy),
                                           unsalted_objective};
  return AutoShardingSolverResult(output, false);
}

bool CostComponents::operator==(const CostComponents& other) const {
  return communication_cost == other.communication_cost &&
         computation_cost == other.computation_cost &&
         resharding_cost == other.resharding_cost &&
         overbudget_cost == other.overbudget_cost &&
         makespan_cost == other.makespan_cost;
}

double CostComponents::cost() const {
  return communication_cost + computation_cost + resharding_cost +
         overbudget_cost + makespan_cost;
}

bool AutoShardingEvaluation::operator==(
    const AutoShardingEvaluation& other) const {
  return violation_codes == other.violation_codes && total == other.total &&
         lower_bound == other.lower_bound &&
         total_departures == other.total_departures;
}

AutoShardingEvaluation Evaluate(const AutoShardingSolverRequest& request,
                                const AutoShardingSolverResult& result) {
  const auto& c = request.computation_costs();
  const auto& d = request.communication_costs();
  const auto& r = request.resharding_costs();
  const auto& v = request.value_costs();
  const auto& p = request.departure_costs();
  const std::vector<NodeStrategyIdx>& s_val = result.status->s_val;
  const std::vector<EdgeStrategyIdx>& e_val = result.status->e_val;
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
    if (r.at(edge_idx).costs(e_val[edge_idx]) >= kInfinityCost) {
      evaluation.violation_codes.insert(kInfiniteCostViolationCode);
    }
  }
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    evaluation.total_departures += p.at(node_idx).costs(s_val[node_idx]);
    if (request.has_max_departures() &&
        evaluation.total_departures > request.max_departures().coeff()) {
      evaluation.violation_codes.insert(kMaxDeparturesViolationCode);
    }
  }
  if (request.memory_budget() > 0) {
    double total_overbudget = 0.0;
    double lower_bound_overbudget = 0.0;
    std::vector<double> total_memory_costs, lower_bound_memory_costs;
    if (request.node_intervals().empty()) {  // Handles live matrices.
      total_memory_costs.resize(request.live_size(), 0.0);
      lower_bound_memory_costs.resize(request.live_size(), 0.0);
      for (LivenessIdx time_idx = 0; time_idx < request.live_size();
           ++time_idx) {
        for (NodeIdx node_idx : request.live(time_idx).nodes()) {
          const auto& m = request.memory_costs(node_idx).costs();
          total_memory_costs[time_idx] += m[s_val[node_idx]];
          lower_bound_memory_costs[time_idx] +=
              *std::min_element(m.begin(), m.end());
        }
        if (!request.live_edges().empty() &&
            request.enable_memory_edge_costs()) {
          for (EdgeIdx edge_idx : request.live_edges(time_idx).edges()) {
            const auto& m = request.memory_edge_costs(edge_idx).costs();
            total_memory_costs[time_idx] += m[e_val[edge_idx]];
            lower_bound_memory_costs[time_idx] +=
                *std::min_element(m.begin(), m.end());
          }
        }
      }
    } else {  // Handles the interval-based memory representation.
      for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
        const auto& interval = request.node_intervals(node_idx);
        if (interval.first() > interval.second()) continue;
        // Expand cost vectors if needed to cover the range of this interval.
        while (total_memory_costs.size() <= interval.second()) {
          total_memory_costs.push_back(0.0);
          lower_bound_memory_costs.push_back(0.0);
        }
        for (LivenessIdx time_idx = interval.first();
             time_idx <= interval.second(); ++time_idx) {
          const auto& m = request.memory_costs(node_idx).costs();
          total_memory_costs[time_idx] += m[s_val[node_idx]];
          lower_bound_memory_costs[time_idx] +=
              *std::min_element(m.begin(), m.end());
        }
      }
      if (request.enable_memory_edge_costs()) {
        for (EdgeIdx edge_idx = 0; edge_idx < request.edges_size();
             ++edge_idx) {
          const auto& interval = request.edge_intervals(edge_idx);
          if (interval.first() > interval.second()) continue;
          // Expand cost vectors if needed to cover the range of this interval.
          while (total_memory_costs.size() <= interval.second()) {
            total_memory_costs.push_back(0.0);
            lower_bound_memory_costs.push_back(0.0);
          }
          for (LivenessIdx time_idx = interval.first();
               time_idx <= interval.second(); ++time_idx) {
            const auto& m = request.memory_edge_costs(edge_idx).costs();
            total_memory_costs[time_idx] += m[e_val[edge_idx]];
            lower_bound_memory_costs[time_idx] +=
                *std::min_element(m.begin(), m.end());
          }
        }
      }
    }
    for (LivenessIdx time_idx = 0; time_idx < total_memory_costs.size();
         ++time_idx) {
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
    if (request.has_overbudget_coeff()) {
      evaluation.total.overbudget_cost =
          request.overbudget_coeff().coeff() * total_overbudget;
      evaluation.lower_bound.overbudget_cost =
          request.overbudget_coeff().coeff() * lower_bound_overbudget;
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
    evaluation.total.resharding_cost += r.at(edge_idx).costs(e_val[edge_idx]);
    evaluation.lower_bound.resharding_cost += *std::min_element(
        r.at(edge_idx).costs().begin(), r.at(edge_idx).costs().end());
  }
  evaluation.total_makespan = EvaluateMakespan(request, result, evaluation);
  return evaluation;
}

std::vector<std::string> Rationalize(const AutoShardingSolverRequest& request,
                                     const AutoShardingSolverResult& result,
                                     const AutoShardingSolverResult& subopt) {
  std::vector<std::string> rationales;
  const auto& names = request.instruction_names();

  const std::vector<NodeStrategyIdx>& s_result = result.status->s_val;
  const std::vector<NodeStrategyIdx>& s_subopt = subopt.status->s_val;
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    const NodeStrategyIdx j = s_result[node_idx], k = s_subopt[node_idx];
    if (j != k) {
      rationales.push_back(absl::StrCat(
          "strategy changes for ", names[node_idx], " (", j, " -> ", k, ")"));
    }
    const double dj = request.communication_costs(node_idx).costs(j);
    const double dk = request.communication_costs(node_idx).costs(k);
    if (dj < dk) {
      rationales.push_back(absl::StrCat("communication cost increases for ",
                                        names[node_idx], " (", dj, " -> ", dk,
                                        ")"));
    }
    const double cj = request.computation_costs(node_idx).costs(j);
    const double ck = request.computation_costs(node_idx).costs(k);
    if (cj < ck) {
      rationales.push_back(absl::StrCat("computation cost increases for ",
                                        names[node_idx], " (", cj, " -> ", ck,
                                        ")"));
    }
  }

  const std::vector<EdgeStrategyIdx>& e_result = result.status->e_val;
  const std::vector<EdgeStrategyIdx>& e_subopt = subopt.status->e_val;
  for (EdgeIdx edge_idx = 0; edge_idx < request.edges_size(); ++edge_idx) {
    const auto& edge = request.edges(edge_idx);
    const EdgeStrategyIdx j = e_result[edge_idx], k = e_subopt[edge_idx];
    const double rj = request.resharding_costs(edge_idx).costs(j);
    const double rk = request.resharding_costs(edge_idx).costs(k);
    if (rj < rk) {
      const std::string edge_name =
          absl::StrCat(names[edge.first()], " and ", names[edge.second()]);
      rationales.push_back(absl::StrCat("resharding cost increases for ",
                                        edge_name, " (", rj, " -> ", rk, ")"));
    }
  }

  return rationales;
}

Status ValidateRequest(const AutoShardingSolverRequest& request) {
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
  return OkStatus();
}

}  // namespace spmd
}  // namespace xla
