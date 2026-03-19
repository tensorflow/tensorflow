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
#include "xla/hlo/experimental/auto_sharding/auto_sharding_memory.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
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

bool AutoShardingSolverOutput::operator==(
    const AutoShardingSolverOutput& other) const {
  return s_val == other.s_val && cost == other.cost &&
         is_optimal == other.is_optimal && peak_times == other.peak_times;
}

double MinimumMemoryBudgetRequired(const AutoShardingSolverRequest& request) {
  std::vector<double> min_memory_required;
  std::vector<double> min_memory_required_group;
  for (NodeIdx node_idx = 0; node_idx < request.node_intervals_size();
       ++node_idx) {
    const auto& interval = request.node_intervals(node_idx);
    if (interval.first() > interval.second()) {
      continue;
    }
    // Expand cost vectors if needed to cover the range of this interval.
    while (min_memory_required.size() <= interval.second()) {
      min_memory_required.push_back(0.0);
    }
    double fixed_memory_cost = 0.0;
    if (node_idx < request.num_nodes()) {
      const auto& m = request.memory_costs(node_idx).costs();
      fixed_memory_cost = *std::min_element(m.begin(), m.end());
    } else {
      int64_t group_idx = node_idx - request.num_nodes();
      fixed_memory_cost = min_memory_required_group[group_idx];
    }
    for (LivenessIdx time_idx = interval.first(); time_idx <= interval.second();
         ++time_idx) {
      min_memory_required[time_idx] += fixed_memory_cost;
    }
  }
  double min_memory_budget_required_estimate = 0.0;
  for (double min_memory_budget_required_estimate_local : min_memory_required) {
    min_memory_budget_required_estimate =
        std::max(min_memory_budget_required_estimate,
                 min_memory_budget_required_estimate_local);
  }
  return min_memory_budget_required_estimate;
}

AutoShardingSolverParams GetParams(const AutoShardingSolverRequest& request) {
  AutoShardingSolverParams params;
  params.compute_iis = request.compute_iis();
  params.crash_at_infinity_costs_check =
      request.crash_at_infinity_costs_check();
  for (const auto& departure_cost : request.departure_costs()) {
    params.departure_costs.push_back(
        {departure_cost.costs().begin(), departure_cost.costs().end()});
  }
  params.deterministic_mode = request.deterministic_mode();
  params.enable_output = request.enable_output();
  params.max_cost = request.has_max_cost()
                        ? std::make_optional(request.max_cost().coeff())
                        : std::nullopt;
  params.max_departures =
      request.has_max_departures()
          ? std::make_optional(request.max_departures().coeff())
          : std::nullopt;
  params.minimize_departures = request.minimize_departures();
  params.overbudget_coeff =
      request.has_overbudget_coeff()
          ? std::make_optional(request.overbudget_coeff().coeff())
          : std::nullopt;
  params.s_hint = {request.s_hint().begin(), request.s_hint().end()};
  params.solver_timeout =
      request.has_solver_timeout()
          ? absl::Seconds(request.solver_timeout().solver_timeout_in_seconds())
          : absl::InfiniteDuration();
  return params;
}

namespace {

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
    if (chosen_node_strategy[node_idx] == -1) {
      LOG(WARNING) << "No strategy chosen for node " << node_idx
                   << ", replacing with zero.";
      chosen_node_strategy[node_idx] = 0;
    }
  }
  return chosen_node_strategy;
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

absl::StatusOr<AutoShardingSolverOutput> SolveAndExtractSolution(
    const AutoShardingSolverRequest& request,
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
  size_t num_edges = request.edges_size();
  double unsalted_objective = 0.0;
  const std::vector<NodeStrategyIdx> chosen_node_strategy =
      GetChosenNodeStrategy(request, s);
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    const NodeStrategyIdx j = chosen_node_strategy[node_idx];
    unsalted_objective += request.computation_costs(node_idx).costs(j) +
                          request.communication_costs(node_idx).costs(j);
  }
  const auto chosen_edge_strategy = [&](EdgeIdx edge_idx) {
    const auto& edge = request.edges(edge_idx);
    return chosen_node_strategy[edge.first()] * request.s_len(edge.second()) +
           chosen_node_strategy[edge.second()];
  };
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    const EdgeStrategyIdx j = chosen_edge_strategy(edge_idx);
    unsalted_objective += request.resharding_costs(edge_idx).costs(j);
  }
  if (overbudget_var) {
    unsalted_objective += *params.overbudget_coeff *
                          overbudget_var->solution_value() *
                          request.memory_budget();
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
  return AutoShardingSolverOutput{.s_val = std::move(chosen_node_strategy),
                                  .cost = solver.Objective().Value(),
                                  .is_optimal = is_optimal};
}

// Given the live matrix and memory costs (for nodes or edges), reduce terms and
// create constrained variables for the subsequent groups.
std::optional<std::pair<int64_t, int64_t>> ReduceMemoryTerms(
    const AutoShardingSolverRequest& request, MPSolver& solver,
    int64_t num_primitives,
    const tsl::protobuf::RepeatedPtrField<  // NOLINT
        AutoShardingSolverRequest_Pair>& intervals,
    const tsl::protobuf::RepeatedPtrField<  // NOLINT
        AutoShardingSolverRequest_Costs>& memory_costs,
    absl::string_view prim_type,
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
  for (const auto& interval : intervals) {
    if (interval.first() > interval.second()) {
      continue;  // Interval undefined
    }
    num_lives = std::max(num_lives, interval.second() + 1);
  }
  auto Intervals =
      [intervals](int64_t prim_idx) -> std::pair<int64_t, int64_t> {
    return {intervals.at(prim_idx).first(), intervals.at(prim_idx).second()};
  };
  MemoryTermReducer reducer;
  num_terms = reducer.Reduce(num_lives, num_primitives, std::move(Intervals));
  reduced_intervals = reducer.GetReducedIntervals();
  reduced_groups = reducer.GetReducedGroups();
  solver.MakeNumVarArray(reduced_groups.size(), 0.0, MPSolver::infinity(),
                         absl::StrCat("group_", prim_type), &group_vars);
  for (int64_t group_idx = 0; group_idx < group_vars.size(); ++group_idx) {
    MPConstraint* constraint = solver.MakeRowConstraint(
        -MPSolver::infinity(), 0.0,
        absl::StrCat("group_", prim_type, "[", group_idx, "]"));
    constraint->SetCoefficient(group_vars[group_idx], -1.0);
    for (const int64_t prim_idx : reduced_groups[group_idx]) {
      for (int64_t j = 0; j < prim_vars[prim_idx].size(); ++j) {
        double memory_cost = memory_costs.at(prim_idx).costs(j);
        memory_cost /= request.memory_budget() / kMemoryMultiplier;
        const double accumulated_coefficient =
            constraint->GetCoefficient(prim_vars[prim_idx][j]);
        constraint->SetCoefficient(prim_vars[prim_idx][j],
                                   accumulated_coefficient + memory_cost);
      }
    }
  }
  const absl::flat_hash_set<int64_t> times = MemoryTermReducer::GetReducedTimes(
      num_primitives, reduced_intervals, reduced_groups);
  reduced_times.insert(times.begin(), times.end());
  const absl::Time term_reduction_end_time = absl::Now();
  if (num_terms) {
    const auto term_reduction_duration =
        term_reduction_end_time - term_reduction_start_time;
    LOG(INFO) << "Memory Term Reducer for " << prim_type << "s took "
              << absl::ToInt64Milliseconds(term_reduction_duration)
              << " ms and reduced the number of terms from " << num_terms->first
              << " to " << num_terms->second;
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
    const absl::flat_hash_set<int64_t>& reduced_times,
    std::vector<std::vector<MPVariable*>>& prim_vars,
    std::vector<MPVariable*>& group_vars,
    absl::flat_hash_map<LivenessIdx, MPConstraint*>& constraints) {
  for (int64_t prim_idx = 0; prim_idx < intervals.size(); ++prim_idx) {
    for (int64_t time_idx = intervals[prim_idx].first;
         time_idx <= intervals[prim_idx].second; ++time_idx) {
      if (!reduced_times.contains(time_idx)) continue;
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
      if (prim_idx >= num_primitives) {
        constraint->SetCoefficient(group_vars[prim_idx - num_primitives], 1.0);
        continue;
      }
      for (int64_t j = 0; j < prim_vars[prim_idx].size(); ++j) {
        double memory_cost = memory_costs.at(prim_idx).costs(j);
        memory_cost /= request.memory_budget() / kMemoryMultiplier;
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
absl::StatusOr<AutoShardingSolverOutput> FormulateAndSolveMIPFromSolverRequest(
    const AutoShardingSolverRequest& request,
    const AutoShardingSolverParams& params) {
  const absl::Time start_time = absl::Now();
  const size_t num_edges = request.edges_size();
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
  std::vector<std::vector<MPVariable*>> s(request.num_nodes());
  std::vector<std::vector<MPVariable*>> e(num_edges);
  MPVariable* overbudget_var = nullptr;

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

  if (request.memory_budget() > 0 && params.overbudget_coeff.has_value()) {
    overbudget_var =
        solver->MakeNumVar(0.0, MPSolver::infinity(), "overbudget");
  }

  // Construct objective function.
  // Node costs
  absl::flat_hash_set<MPVariable*> infinity_vars;
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
      double coefficient = request.computation_costs(node_idx).costs(j) +
                           request.communication_costs(node_idx).costs(j);
      if (coefficient >= kInfinityCost) {
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
      double coefficient = request.resharding_costs(edge_idx).costs(j);
      if (coefficient >= kInfinityCost) {
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
          ? StrategyShaverForRequest(params, request).FindShavedStrategies()
          : NodeStrategies();
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
    std::vector<std::pair<int64_t, int64_t>> reduced_intervals_nodes;
    absl::flat_hash_set<int64_t> reduced_times;
    std::vector<MPVariable*> group_node_vars;
    std::optional<std::pair<int64_t, int64_t>> num_node_terms;
    num_node_terms = ReduceMemoryTerms(
        request, *solver, request.num_nodes(), request.node_intervals(),
        request.memory_costs(), "node", s, reduced_intervals_nodes,
        group_node_vars, reduced_times);
    absl::flat_hash_map<LivenessIdx, MPConstraint*> constraints;
    AddMemoryTerms(request, *solver, request.num_nodes(),
                   reduced_intervals_nodes, request.memory_costs(),
                   overbudget_var, reduced_times, s, group_node_vars,
                   constraints);
    if (overbudget_var && !params.minimize_departures) {
      solver->MutableObjective()->SetCoefficient(
          overbudget_var, *params.overbudget_coeff * request.memory_budget());
    }
    LOG(INFO) << "Minimum memory budget estimate: "
              << MinimumMemoryBudgetRequired(request);
    LOG(INFO) << "Using memory budget: "
              << static_cast<double>(request.memory_budget());
  }

  // d. specified via "BoolVarArray"
  // e.
  for (EdgeIdx edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    if (e_follow[edge_idx] >= 0) {
      continue;
    }
    const auto& edge = request.edges(edge_idx);
    for (NodeStrategyIdx p = 0; p < s[edge.first()].size(); ++p) {
      for (NodeStrategyIdx q = 0; q < s[edge.second()].size(); ++q) {
        const EdgeStrategyIdx j = p * s[edge.second()].size() + q;
        MPConstraint* constraint = solver->MakeRowConstraint(
            -1.0, MPSolver::infinity(),
            absl::StrCat("edge[", edge_idx, "][", j, "]"));
        // In the special case where the source and destination are both
        // represented by the same node variable, its coefficient in this
        // constraint must be doubled.
        double coeff = (s[edge.first()][p] == s[edge.second()][q]) ? 2.0 : 1.0;
        constraint->SetCoefficient(s[edge.first()][p], -coeff);
        constraint->SetCoefficient(s[edge.second()][q], -coeff);
        constraint->SetCoefficient(e[edge_idx][j], 1.0);
      }
    }
  }
  // f.
  absl::flat_hash_set<std::pair<NodeIdx, NodeIdx>> alias_set;
  for (auto alias_idx = 0; alias_idx < request.aliases_size(); ++alias_idx) {
    const auto& raw_alias = request.aliases(alias_idx);
    const std::pair<NodeIdx, NodeIdx> alias(raw_alias.first(),
                                            raw_alias.second());
    if (alias_set.contains(alias)) {
      continue;
    }
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
  if (params.max_departures.has_value()) {
    MPConstraint* constraint = solver->MakeRowConstraint(
        0, *params.max_departures,
        absl::StrCat("departures <= ", *params.max_departures));
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
  if (params.minimize_departures) {
    for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
      for (NodeStrategyIdx j = 0; j < s[node_idx].size(); ++j) {
        double accumulated_coefficient =
            solver->MutableObjective()->GetCoefficient(s[node_idx][j]);
        double departure_cost = request.departure_costs(node_idx).costs(j);
        solver->MutableObjective()->SetCoefficient(
            s[node_idx][j], accumulated_coefficient + departure_cost);
      }
    }
  }

  if (!params.s_hint.empty() && !params.deterministic_mode &&
      (!params.max_cost.has_value() || *params.max_cost < kMaxCostValue)) {
    std::vector<std::pair<const MPVariable*, double>> hint;
    for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
      if (request.s_follow(node_idx) >= 0) {
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
  // Exports the solver request proto for debugging.
  bool dump_solver_request = false;
  if (dump_solver_request) {
    uint64_t solver_request_fprint =
        tsl::Fingerprint64(request.SerializeAsString());
    std::string request_dump_path =
        absl::StrCat("/tmp/solver_request_", request.request_name(), "_",
                     solver_request_fprint, ".textproto");
    auto write_status = file::SetTextProto(
        // Modify this file path if needed.
        request_dump_path, request, file::Defaults());
    LOG(INFO) << "Dumped solver request to " << request_dump_path;
    if (!write_status.ok()) {
      LOG(ERROR) << write_status.message();
    }
  }
  // Invokes the solver request callback for any additional debugging.
  bool solver_request_callback = false;
  if (solver_request_callback) {
    SolverRequestCallback(request);
  }
#endif
  if (params.enable_output) {
    solver->EnableOutput();
  }
  VLOG(0) << "Starting solver " << solver->ProblemType() << "\n"
          << "Solver parameter string: " << solver_parameter_str << "\n"
          << "Number of workers: " << num_workers << "\n"
          << "Number of threads: " << solver->GetNumThreads() << "\n"
          << "Time limit: "
          << request.solver_timeout().solver_timeout_in_seconds()
          << " seconds\n"
          << "Request valid: " << ValidateRequest(request).ok() << "\n"
          << "Aliases: " << request.aliases_size() << "\n"
          << "Unique nodes: " << unique_nodes << "\n"
          << "Unique edges: " << unique_edges << "\n"
          << "Total instructions: " << request.num_nodes() << "\n"
          << "Total edges: " << request.edges_size() << "\n"
          << "Memory budget: " << request.memory_budget() << " ("
          << request.memory_budget() / (1024 * 1024 * 1024) << "GB)\n"
          << "Number variables for ILP: " << solver->NumVariables() << "\n"
          << "Number of ILP constraints: " << solver->NumConstraints() << "\n"
          << "Deterministic mode: " << params.deterministic_mode << "\n"
          << "Minimize departures: " << params.minimize_departures << "\n"
          << "Module name: " << request.module_name();
  if (params.max_cost.has_value()) {
    VLOG(0) << "Max cost: " << *params.max_cost;
  }
  if (params.max_departures.has_value()) {
    VLOG(0) << "Max departures: " << *params.max_departures;
  }
  auto result =
      SolveAndExtractSolution(request, params, s, e, overbudget_var, *solver);
  if (result.ok()) {
    const AutoShardingEvaluation evaluation =
        Evaluate(request, *result, params);
    LOG(INFO) << "*** Total costs for the solver request ***";
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
    LOG(INFO) << "Total Cost: " << evaluation.total.cost()
              << " (lower bound: " << evaluation.lower_bound.cost() << ")";
    LOG(INFO) << "Total Departures: " << evaluation.total_departures;
    LOG(INFO) << "Total Violations: " << evaluation.violation_codes.size();
    LOG(INFO) << "Total Maximum Memory: " << evaluation.total.max_memory
              << " (lower bound: " << evaluation.lower_bound.max_memory << ")";
  }
  const absl::Time end_time = absl::Now();
  const auto duration = end_time - start_time;
  LOG(INFO) << "Solver took " << absl::ToInt64Milliseconds(duration) << " ms";
  return result;
}

}  // namespace spmd
}  // namespace xla
