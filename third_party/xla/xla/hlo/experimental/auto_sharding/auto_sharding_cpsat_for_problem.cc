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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"

#ifdef PLATFORM_GOOGLE
#include "file/base/options.h"
#endif
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_iopddl.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_memory.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/iopddl.h"
#include "xla/util.h"
#include "tsl/platform/fingerprint.h"
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

// Memory contributions in the Mixed ILP are converted to units in this range;
// beware that significantly larger / smaller values can cause numerical issues.
constexpr double kMemoryMultiplier = 1e6;

// Maximum costs above this threshold can lead to Invalid MIPs.
// TODO(moffitt): Handle hints properly for problems with high overbudget costs.
constexpr double kMaxCostValue = 1e18;

namespace {

std::vector<NodeStrategyIdx> GetChosenNodeStrategy(
    const iopddl::Problem& problem,
    const std::vector<std::vector<MPVariable*>>& s) {
  std::vector<NodeStrategyIdx> chosen_node_strategy(problem.nodes.size(), -1);
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
      // if lhs == 1
      if (s[node_idx][j]->solution_value() > 0.5) {
        chosen_node_strategy[node_idx] = j;
        break;
      }
    }
    if (chosen_node_strategy[node_idx] == -1) {
      LOG(WARNING) << "No strategy chosen for node " << node_idx
                   << ", replacing with zero.";
      chosen_node_strategy[node_idx] = 0;
    }
  }
  return chosen_node_strategy;
}

absl::StatusOr<AutoShardingSolverOutput> SolveAndExtractSolution(
    const iopddl::Problem& problem,
    const std::vector<iopddl::Edge>& deduplicated_edges,
    const AutoShardingSolverParams& params,
    const std::vector<std::vector<MPVariable*>>& s,
    const std::vector<std::vector<MPVariable*>>& e,
    const MPVariable* overbudget_var, MPSolver& solver) {
  auto status = solver.Solve();
  LOG(INFO) << "Solver absl::Status: " << status;

  bool is_optimal = false;
  if (status == operations_research::MPSolver::INFEASIBLE) {
    LOG(ERROR) << "MPSolver could not find any feasible solution.";
#ifdef PLATFORM_GOOGLE
    if (params.compute_iis) {
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
    return absl::InternalError(
        "MPSolver could not find any feasible solution.");
  }
  if (status == operations_research::MPSolver::MODEL_INVALID) {
    LOG(FATAL) << "The MIP fed to the solver is invalid. This is most likely a "
                  "bug and should be reported.";
    return absl::InternalError("Invalid MIP.");
  }
  if (status == operations_research::MPSolver::NOT_SOLVED) {
    LOG(WARNING) << "Solver timeout; no solution was produced";
    return absl::InternalError("Solver timed out.");
  }
  if (status != operations_research::MPSolver::OPTIMAL) {
    LOG(WARNING) << "Solver timeout; moving forward with a suboptimal solution";
  } else {
    is_optimal = true;
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
  size_t num_edges = deduplicated_edges.size();
  double unsalted_objective = 0.0;
  const std::vector<NodeStrategyIdx> chosen_node_strategy =
      GetChosenNodeStrategy(problem, s);
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    const NodeStrategyIdx j = chosen_node_strategy[node_idx];
    unsalted_objective += problem.nodes[node_idx].strategies[j].cost;
  }
  const auto chosen_edge_strategy = [&](EdgeIdx edge_idx) {
    const auto& edge = deduplicated_edges[edge_idx];
    const auto& node = problem.nodes[edge.nodes[1]];
    return chosen_node_strategy[edge.nodes[0]] * node.strategies.size() +
           chosen_node_strategy[edge.nodes[1]];
  };
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    const EdgeStrategyIdx j = chosen_edge_strategy(edge_idx);
    unsalted_objective += deduplicated_edges[edge_idx].strategies[j].cost;
  }
  if (overbudget_var) {
    unsalted_objective += *params.overbudget_coeff *
                          overbudget_var->solution_value() *
                          (*problem.usage_limit);
  }

  LOG(INFO) << "Unsalted objective value: " << unsalted_objective;
  LOG(INFO) << "N = " << problem.nodes.size();
  if (!problem.usage_limit.has_value()) {
    LOG(INFO) << "memory budget: -1";
  } else {
    LOG(INFO) << "memory budget: "
              << *problem.usage_limit / (1024 * 1024 * 1024) << " GB";
  }
  return AutoShardingSolverOutput{.s_val = std::move(chosen_node_strategy),
                                  .cost = solver.Objective().Value(),
                                  .is_optimal = is_optimal};
}

// Given the live matrix and memory costs (for nodes or edges), reduce terms and
// create constrained variables for the subsequent groups.
std::optional<std::pair<int64_t, int64_t>> ReduceMemoryTerms(
    const iopddl::Problem& problem, MPSolver& solver,
    std::vector<std::vector<MPVariable*>>& prim_vars,
    std::vector<std::pair<int64_t, int64_t>>& reduced_intervals,
    std::vector<MPVariable*>& group_vars,
    absl::flat_hash_set<int64_t>& reduced_times) {
  const absl::Time term_reduction_start_time = absl::Now();
  std::optional<std::pair<int64_t, int64_t>> num_terms = std::nullopt;
  std::vector<absl::btree_set<int64_t>> reduced_groups;
  // We need to compute the number of lives in order to use the memory term
  // reducer.
  int64_t num_lives = 0;
  for (const auto& node : problem.nodes) {
    if (node.interval.first >= node.interval.second) {
      continue;  // undefined
    }
    num_lives = std::max(num_lives, node.interval.second);
  }
  auto Intervals = [&problem](int64_t prim_idx) -> std::pair<int64_t, int64_t> {
    return {problem.nodes[prim_idx].interval.first,
            problem.nodes[prim_idx].interval.second - 1};
  };
  MemoryTermReducer reducer;
  num_terms =
      reducer.Reduce(num_lives, problem.nodes.size(), std::move(Intervals));
  reduced_intervals = reducer.GetReducedIntervals();
  reduced_groups = reducer.GetReducedGroups();
  solver.MakeNumVarArray(reduced_groups.size(), 0.0, MPSolver::infinity(),
                         "group", &group_vars);
  for (int64_t group_idx = 0; group_idx < group_vars.size(); ++group_idx) {
    MPConstraint* constraint = solver.MakeRowConstraint(
        -MPSolver::infinity(), 0.0, absl::StrCat("group", "[", group_idx, "]"));
    constraint->SetCoefficient(group_vars[group_idx], -1.0);
    for (const int64_t prim_idx : reduced_groups[group_idx]) {
      for (int64_t j = 0; j < prim_vars[prim_idx].size(); ++j) {
        double memory_cost = problem.nodes[prim_idx].strategies[j].usage;
        memory_cost /= *problem.usage_limit / kMemoryMultiplier;
        const double accumulated_coefficient =
            constraint->GetCoefficient(prim_vars[prim_idx][j]);
        constraint->SetCoefficient(prim_vars[prim_idx][j],
                                   accumulated_coefficient + memory_cost);
      }
    }
  }
  const absl::flat_hash_set<int64_t> times = MemoryTermReducer::GetReducedTimes(
      problem.nodes.size(), reduced_intervals, reduced_groups);
  reduced_times.insert(times.begin(), times.end());
  const absl::Time term_reduction_end_time = absl::Now();
  if (num_terms) {
    const auto term_reduction_duration =
        term_reduction_end_time - term_reduction_start_time;
    LOG(INFO) << "Memory Term Reducer took "
              << absl::ToInt64Milliseconds(term_reduction_duration)
              << " ms and reduced the number of terms from " << num_terms->first
              << " to " << num_terms->second;
  }
  return num_terms;
}

// Adds the appropriate memory terms (for nodes or edges) at the given time.
void AddMemoryTerms(
    const iopddl::Problem& problem, MPSolver& solver,
    const std::vector<std::pair<int64_t, int64_t>>& intervals,
    const MPVariable* overbudget_var,
    const absl::flat_hash_set<int64_t>& reduced_times,
    std::vector<std::vector<MPVariable*>>& prim_vars,
    std::vector<MPVariable*>& group_vars,
    absl::flat_hash_map<LivenessIdx, MPConstraint*>& constraints) {
  for (int64_t prim_idx = 0; prim_idx < intervals.size(); ++prim_idx) {
    for (int64_t time_idx = intervals[prim_idx].first;
         time_idx <= intervals[prim_idx].second; ++time_idx) {
      if (!reduced_times.contains(time_idx)) {
        continue;
      }
      if (!constraints.contains(time_idx)) {
        MPConstraint* constraint =
            solver.MakeRowConstraint(-MPSolver::infinity(), kMemoryMultiplier,
                                     absl::StrCat("mem[", time_idx, "]"));
        if (overbudget_var) {
          constraint->SetCoefficient(overbudget_var, -kMemoryMultiplier);
        }
        constraints[time_idx] = constraint;
      }
      MPConstraint* constraint = constraints[time_idx];
      if (prim_idx >= problem.nodes.size()) {
        constraint->SetCoefficient(group_vars[prim_idx - problem.nodes.size()],
                                   1.0);
        continue;
      }
      for (int64_t j = 0; j < prim_vars[prim_idx].size(); ++j) {
        double memory_cost = problem.nodes[prim_idx].strategies[j].usage;
        memory_cost /= *problem.usage_limit / kMemoryMultiplier;
        const double accumulated_coefficient =
            constraint->GetCoefficient(prim_vars[prim_idx][j]);
        constraint->SetCoefficient(prim_vars[prim_idx][j],
                                   accumulated_coefficient + memory_cost);
      }
    }
  }
}

}  // namespace

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
//     f. For all (i, j) in A and all (p, q),
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
// 3. request.max_departures is used only for debugging and can be ignored.
// 4. Note that due to our modeling of XLA's AllReduceReassociate optimization
//    (more details in CostGraph::CostGraph() in auto_sharding_cost_graph.cc,
//    and in CreateElementwiseOperatorStrategies() in auto_sharding.cc), there
//    can be a few (usually < 10) edges in the problem with negative costs. This
//    is guaranteed to never produce a negative overall cost for the graph,
//    however.
absl::StatusOr<AutoShardingSolverOutput> FormulateAndSolveMIPFromProblem(
    const iopddl::Problem& problem,
    const AutoShardingSolverParams& params) {
  const absl::Time start_time = absl::Now();
  const std::vector<int64_t> followers = GetFollowers(problem);
  const std::vector<iopddl::Edge> aliases = GetAliases(problem);
  const std::vector<iopddl::Edge> deduplicated_edges =
      GetDeduplicatedEdges(problem);
  const size_t num_edges = deduplicated_edges.size();
  const int num_workers = 32;
  // SAT or SCIP
#ifdef PLATFORM_GOOGLE
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SAT"));
#else
  std::unique_ptr<MPSolver> solver(
      std::make_unique<MPSolver>("", MPSolver::SAT_INTEGER_PROGRAMMING));
#endif
  CHECK(solver);
  solver->MutableObjective()->SetMinimization();
  std::string solver_parameter_str;
  if (solver->ProblemType() ==
      operations_research::MPSolver::SAT_INTEGER_PROGRAMMING) {
    // Set random_seed, interleave_search and share_binary_clauses for
    // determinism, mip_max_bound (to handle large costs), and num_workers for
    // parallelism.
    solver_parameter_str = absl::StrCat("num_workers:", num_workers);
    if (params.deterministic_mode) {
      absl::StrAppend(
          &solver_parameter_str,
          ",share_binary_clauses:false,random_seed:1,interleave_search:true");
    }
    if (params.solver_timeout != absl::InfiniteDuration()) {
      if (params.deterministic_mode) {
        absl::StrAppend(&solver_parameter_str, ",max_deterministic_time:",
                        absl::ToInt64Seconds(params.solver_timeout));
      } else {
        solver->SetTimeLimit(params.solver_timeout);
      }
    }
    solver->SetSolverSpecificParametersAsString(solver_parameter_str);
  }
  // Create variables
  std::vector<std::vector<MPVariable*>> s(problem.nodes.size());
  std::vector<std::vector<MPVariable*>> e(num_edges);
  MPVariable* overbudget_var = nullptr;

  size_t unique_nodes = 0;
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    if (followers[node_idx] < 0) {
      unique_nodes += 1;
      // Creates variables for instructions that do not follow others.
      solver->MakeBoolVarArray(problem.nodes[node_idx].strategies.size(),
                               absl::StrCat("s[", node_idx, "]"), &s[node_idx]);
    }
  }

  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    if (followers[node_idx] >= 0) {
      CHECK_EQ(problem.nodes[node_idx].strategies.size(),
               problem.nodes[followers[node_idx]].strategies.size());
      // Copies the variable of followed instruction to the following
      // instruction.
      s[node_idx] = s[followers[node_idx]];
    }
  }

  size_t unique_edges = 0;
  std::vector<EdgeIdx> e_follow(num_edges, -1);
  absl::flat_hash_map<std::pair<NodeIdx, NodeIdx>, EdgeIdx> edge_map;
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    const auto& raw_edge = deduplicated_edges[edge_idx];
    const std::pair<NodeIdx, NodeIdx> edge(raw_edge.nodes[0],
                                           raw_edge.nodes[1]);
    auto followed_edge = edge;
    if (int f = followers[edge.first]; f >= 0) {
      followed_edge.first = f;
    }
    if (int f = followers[edge.second]; f >= 0) {
      followed_edge.second = f;
    }
    if (const auto& it = edge_map.find(followed_edge); it != edge_map.end()) {
      e[edge_idx] = e[it->second];  // Copy variable of followed edge
      e_follow[edge_idx] = it->second;
      continue;
    }
    unique_edges += 1;
    solver->MakeBoolVarArray(
        problem.nodes[edge.first].strategies.size() *
            problem.nodes[edge.second].strategies.size(),
        absl::StrCat("e[", edge.first, ",", edge.second, "]"), &e[edge_idx]);
    edge_map.insert({followed_edge, edge_idx});
  }

  if (problem.usage_limit.has_value() && params.overbudget_coeff.has_value()) {
    overbudget_var =
        solver->MakeNumVar(0.0, MPSolver::infinity(), "overbudget");
  }

  // Construct objective function.
  // Node costs
  absl::flat_hash_set<MPVariable*> infinity_vars;
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
      double coefficient = problem.nodes[node_idx].strategies[j].cost;
      if (coefficient >= kInfinityInt) {
        infinity_vars.insert(s[node_idx][j]);
        continue;
      }
      if (params.minimize_departures) {
        continue;
      }
      double accumulated_coefficient =
          solver->MutableObjective()->GetCoefficient(s[node_idx][j]);
      solver->MutableObjective()->SetCoefficient(
          s[node_idx][j], accumulated_coefficient + coefficient);
    }
  }
  // Edge costs
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    for (EdgeStrategyIdx j = 0; j < e[edge_idx].size(); ++j) {
      double coefficient = deduplicated_edges[edge_idx].strategies[j].cost;
      if (coefficient >= kInfinityInt) {
        infinity_vars.insert(e[edge_idx][j]);
        continue;
      }
      if (params.minimize_departures) {
        continue;
      }
      double accumulated_coefficient =
          solver->MutableObjective()->GetCoefficient(e[edge_idx][j]);
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
      params.shave_strategies
          ? StrategyShaverForProblem(params, problem, followers, aliases,
                                     deduplicated_edges)
                .FindShavedStrategies()
          : NodeStrategies();
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    if (s[node_idx].empty() || followers[node_idx] >= 0) {
      continue;
    }
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
    if (e[edge_idx].empty() || e_follow[edge_idx] >= 0) {
      continue;
    }
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
      auto err_msg = absl::StrCat(
          "All of e[", deduplicated_edges[edge_idx].nodes[0], "][",
          deduplicated_edges[edge_idx].nodes[1], "][*] have infinity costs");
      if (params.crash_at_infinity_costs_check) {
        LOG(FATAL) << err_msg;
      } else {
        LOG(WARNING) << err_msg;
        return absl::InternalError(err_msg);
      }
    }
  }

  // a. specified via "BoolVarArray"
  // b.
  for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
    if (followers[node_idx] >= 0) {
      continue;
    }
    MPConstraint* constraint = solver->MakeRowConstraint(
        1.0, 1.0,
        absl::StrCat("sum(s[", node_idx, "][j] for j = [0 .. ",
                     s[node_idx].size(), ")) = 1"));
    for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
      constraint->SetCoefficient(s[node_idx][j], 1.0);
    }
  }
  // c.
  if (problem.usage_limit.has_value()) {
    std::vector<std::pair<int64_t, int64_t>> reduced_intervals_nodes;
    absl::flat_hash_set<int64_t> reduced_times;
    std::vector<MPVariable*> group_node_vars;
    std::optional<std::pair<int64_t, int64_t>> num_node_terms;
    num_node_terms =
        ReduceMemoryTerms(problem, *solver, s, reduced_intervals_nodes,
                          group_node_vars, reduced_times);
    absl::flat_hash_map<LivenessIdx, MPConstraint*> constraints;
    AddMemoryTerms(problem, *solver, reduced_intervals_nodes, overbudget_var,
                   reduced_times, s, group_node_vars, constraints);
    if (overbudget_var && !params.minimize_departures) {
      solver->MutableObjective()->SetCoefficient(
          overbudget_var, *params.overbudget_coeff * (*problem.usage_limit));
    }
    LOG(INFO) << "Using memory budget: "
              << static_cast<double>(*problem.usage_limit);
  }

  // d. specified via "BoolVarArray"
  // e.
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    if (e_follow[edge_idx] >= 0) {
      continue;
    }
    const auto& edge = deduplicated_edges[edge_idx];
    for (NodeStrategyIdx p = 0; p < s[edge.nodes[0]].size(); ++p) {
      for (NodeStrategyIdx q = 0; q < s[edge.nodes[1]].size(); ++q) {
        const EdgeStrategyIdx j = p * s[edge.nodes[1]].size() + q;
        MPConstraint* constraint = solver->MakeRowConstraint(
            -1.0, MPSolver::infinity(),
            absl::StrCat("edge[", edge_idx, "][", j, "]"));
        // In the special case where the source and destination are both
        // represented by the same node variable, its coefficient in this
        // constraint must be doubled.
        double coeff = (s[edge.nodes[0]][p] == s[edge.nodes[1]][q]) ? 2.0 : 1.0;
        constraint->SetCoefficient(s[edge.nodes[0]][p], -coeff);
        constraint->SetCoefficient(s[edge.nodes[1]][q], -coeff);
        constraint->SetCoefficient(e[edge_idx][j], 1.0);
      }
    }
  }
  // f.
  absl::flat_hash_set<std::pair<NodeIdx, NodeIdx>> alias_set;
  for (auto alias_idx = 0; alias_idx < aliases.size(); ++alias_idx) {
    const auto& raw_alias = aliases[alias_idx];
    const std::pair<NodeIdx, NodeIdx> alias(raw_alias.nodes[0],
                                            raw_alias.nodes[1]);
    if (alias_set.contains(alias)) {
      continue;
    }
    alias_set.insert(alias);
    for (NodeStrategyIdx p = 0; p < s[alias.first].size(); ++p) {
      for (NodeStrategyIdx q = 0; q < s[alias.second].size(); ++q) {
        // if lhs == 1
        if (raw_alias.strategies[p * s[alias.second].size() + q].cost > 0.5) {
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
  if (params.max_departures.has_value()) {
    MPConstraint* constraint = solver->MakeRowConstraint(
        0, *params.max_departures,
        absl::StrCat("departures <= ", *params.max_departures));
    for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
      for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
        double accumulated_coefficient =
            constraint->GetCoefficient(s[node_idx][j]);
        double departure_cost = params.departure_costs[node_idx][j];
        constraint->SetCoefficient(s[node_idx][j],
                                   accumulated_coefficient + departure_cost);
      }
    }
  }
  if (params.minimize_departures) {
    for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
      for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
        double accumulated_coefficient =
            solver->MutableObjective()->GetCoefficient(s[node_idx][j]);
        double departure_cost = params.departure_costs[node_idx][j];
        solver->MutableObjective()->SetCoefficient(
            s[node_idx][j], accumulated_coefficient + departure_cost);
      }
    }
  }

  if (!params.s_hint.empty() && !params.deterministic_mode &&
      (!params.max_cost.has_value() || *params.max_cost < kMaxCostValue)) {
    std::vector<std::pair<const MPVariable*, double>> hint;
    for (NodeIdx node_idx = 0; node_idx < problem.nodes.size(); ++node_idx) {
      if (followers[node_idx] >= 0) {
        continue;
      }
      for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
        double hint_val = (params.s_hint[node_idx] == j) ? 1.0 : 0.0;
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
#endif
  if (params.enable_output) {
    solver->EnableOutput();
  }
  VLOG(0) << "Starting solver " << solver->ProblemType() << "\n"
          << "Solver parameter string: " << solver_parameter_str << "\n"
          << "Number of workers: " << num_workers << "\n"
          << "Number of threads: " << solver->GetNumThreads() << "\n"
          << "Time limit: " << params.solver_timeout << "\n"
          << "Aliases: " << aliases.size() << "\n"
          << "Unique nodes: " << unique_nodes << "\n"
          << "Unique edges: " << unique_edges << "\n"
          << "Total instructions: " << problem.nodes.size() << "\n"
          << "Total edges: " << deduplicated_edges.size() << "\n"
          << "Memory budget: " << problem.usage_limit.value_or(-1) << " ("
          << problem.usage_limit.value_or(-1) / (1024 * 1024 * 1024) << "GB)\n"
          << "Number variables for ILP: " << solver->NumVariables() << "\n"
          << "Number of ILP constraints: " << solver->NumConstraints() << "\n"
          << "Deterministic mode: " << params.deterministic_mode << "\n"
          << "Minimize departures: " << params.minimize_departures << "\n"
          << "Module name: " << problem.name;
  if (params.max_cost.has_value()) {
    VLOG(0) << "Max cost: " << *params.max_cost;
  }
  if (params.max_departures.has_value()) {
    VLOG(0) << "Max departures: " << *params.max_departures;
  }
  auto result = SolveAndExtractSolution(problem, deduplicated_edges, params, s,
                                        e, overbudget_var, *solver);
  if (result.ok()) {
    const AutoShardingEvaluation evaluation =
        Evaluate(problem, *result, params);
    LOG(INFO) << "*** Total costs for the solver request ***";
    LOG(INFO) << "Total Node Cost: " << evaluation.total.node_cost
              << " (lower bound: " << evaluation.lower_bound.node_cost << ")";
    LOG(INFO) << "Total Edge Cost: " << evaluation.total.edge_cost
              << " (lower bound: " << evaluation.lower_bound.edge_cost << ")";
    LOG(INFO) << "Total Overbudget Cost: " << evaluation.total.overbudget_usage
              << " (lower bound: " << evaluation.lower_bound.overbudget_usage
              << ")";
    LOG(INFO) << "Total Cost: " << evaluation.total.total_cost()
              << " (lower bound: " << evaluation.lower_bound.total_cost()
              << ")";
    LOG(INFO) << "Total Departures: " << evaluation.total_departures;
    LOG(INFO) << "Total Violations: " << evaluation.violation_codes.size();
    LOG(INFO) << "Total Maximum Memory: " << evaluation.total.max_usage
              << " (lower bound: " << evaluation.lower_bound.max_usage << ")";
  }
  const absl::Time end_time = absl::Now();
  const auto duration = end_time - start_time;
  LOG(INFO) << "Solver took " << absl::ToInt64Milliseconds(duration) << " ms";
  return result;
}

}  // namespace spmd
}  // namespace xla
