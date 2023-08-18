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

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef PLATFORM_GOOGLE
#include "file/base/options.h"
#endif
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/hash.h"
#include "tensorflow/tsl/platform/types.h"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_solver.pb.h"
#ifdef PLATFORM_GOOGLE
#include "file/base/helpers.h"
#include "util/task/status.pb.h"
#endif

using MPConstraint = operations_research::MPConstraint;
using MPSolver = operations_research::MPSolver;
using MPSolverParameters = operations_research::MPSolverParameters;
using MPVariable = operations_research::MPVariable;

namespace xla {
namespace spmd {

bool AutoShardingSolverResult::operator==(
    const AutoShardingSolverResult& other) const {
  return status == other.status &&
         skip_auto_sharding == other.skip_auto_sharding;
}

void PrintLargestInstructions(
    const std::vector<NodeStrategyIdx>& chosen_strategy,
    const std::vector<std::vector<double>>& memory_cost,
    const std::vector<std::vector<NodeIdx>>& liveness,
    const std::vector<std::string>& instruction_names) {
  // This memory consumption computation is different from
  // that in PrintAutoShardingSolution() because how L and m are created to be
  // different from liveness_set and strategy.memory_cost.

  std::vector<std::pair<LivenessIdx, double>> time_memory_usage;
  for (LivenessIdx t = 0; t < liveness.size(); ++t) {
    double mem = 0.0;
    for (NodeIdx i : liveness[t]) {
      mem += memory_cost[i][chosen_strategy[i]];
    }
    time_memory_usage.push_back(std::make_pair(t, mem));
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
  for (LivenessIdx t = 0; t < k; t++) {
    for (NodeIdx i : liveness[time_memory_usage.at(t).first]) {
      double mem = memory_cost[i][chosen_strategy[i]];
      if (mem > 100 * 1024 * 1024 &&
          instruction_set.find(i) == instruction_set.end()) {
        instruction_mem.push_back(std::make_pair(i, mem));
        instruction_set.insert(i);
      }
    }
  }
  std::sort(instruction_mem.begin(), instruction_mem.end(), MemLarger);

  size_t top_tensors = 10;
  top_tensors = std::min(top_tensors, instruction_mem.size());
  VLOG(1) << "Top " << top_tensors << " largest tensors:";
  for (size_t i = 0; i < top_tensors; i++) {
    VLOG(1) << "instruction name: "
            << instruction_names.at(instruction_mem.at(i).first)
            << " memory usage: "
            << instruction_mem.at(i).second / (1024 * 1024 * 1024) << "GB";
  }
}

// Adds deterministic noise to the coefficient using the name & salt multiplier.
void AddSalt(const std::string& name, double saltiplier, double* coeff) {
  if (saltiplier <= 0.0) return;
  const tsl::uint64 hash = tsl::Hash64(name);  // stable across runs & platforms
  *coeff *= 1.0 + saltiplier * hash / std::numeric_limits<tsl::uint64>::max();
}

// We formulate the auto sharding process as the following ILP problem:
// Variables:
//   s[i]: Sharding strategy one-hot vector.
//         dim(s[i]) == # sharding strategies of the i-th XLA op
//         s_len[i] := dim(s[i]) in the arguments
//   e[i, j]: Strategy one-hot vector of edge i -> j.
//            dim(e[i, j]) == dim(s[i]) * dim(s[j])
// Constants:
//   N: Number of total XLA ops
//   M: Memory budget
//   E: Edge set {(i, j)}
//   L[t]: Index of live instructions at time t
//   c[i]: Computation cost vector of instruction i
//   d[i]: Communication cost vector of instruction i
//   m[i]: Memory cost vector of instruction i
//         dim(c[i]) == dim(d[i]) == dim(m[i]) == dim(s[i])
//   r[i, j]: The resharding cost vector of edge i -> j
//            dim(e[i, j]) == dim(r[i, j])
//   A: Alias set {(i, j)}
//   v[i, j]: v[i, j](p, q) == 1 if strategy p is different than q, otherwise
//            v[i, j](p, q) == 0
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

AutoShardingSolverResult CallORToolsSolver(
    const AutoShardingSolverRequest& request) {
  size_t num_edges = request.e.size();

  int32_t num_workers = 32;
  // SAT or SCIP
  std::unique_ptr<MPSolver> solver(std::make_unique<MPSolver>("", MPSolver::SAT_INTEGER_PROGRAMMING));
  CHECK(solver);
  solver->MutableObjective()->SetMinimization();
  std::string solver_parameter_str;
#ifdef PLATFORM_GOOGLE
  if (solver->ProblemType() ==
      operations_research::MPSolver::SAT_INTEGER_PROGRAMMING) {
    // Set num_workers for parallelism.
    solver_parameter_str = absl::StrCat("num_workers:", num_workers);
    solver->SetSolverSpecificParametersAsString(solver_parameter_str);
  }
#endif
  // Create variables
  std::vector<std::vector<MPVariable*>> s(request.num_nodes);
  std::vector<std::vector<MPVariable*>> e(num_edges);

  size_t var_vector_cnt = 0;
  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    if (request.s_follow[i] < 0) {
      var_vector_cnt += 1;
      // Creates variables for instructions that do not follow others.
      solver->MakeBoolVarArray(request.s_len[i], absl::StrCat("s[", i, "]"),
                               &s[i]);
    }
  }

  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    if (request.s_follow[i] >= 0) {
      // Copies the variable of followed instruction to the following
      // instruction.
      s[i] = s[request.s_follow[i]];
    }
  }

  std::vector<EdgeIdx> e_follow(num_edges, -1);
  absl::flat_hash_map<std::pair<NodeIdx, NodeIdx>, EdgeIdx> edge_map;
  for (EdgeIdx i = 0; i < num_edges; ++i) {
    const std::pair<NodeIdx, NodeIdx>& edge = request.e[i];
    std::pair<NodeIdx, NodeIdx> followed_edge = edge;
    if (int f = request.s_follow[edge.first]; f >= 0) followed_edge.first = f;
    if (int f = request.s_follow[edge.second]; f >= 0) followed_edge.second = f;
    if (const auto& it = edge_map.find(followed_edge); it != edge_map.end()) {
      e[i] = e[it->second];  // Copy variable of followed edge to following edge
      e_follow[i] = it->second;
      continue;
    }
    solver->MakeBoolVarArray(
        request.s_len[edge.first] * request.s_len[edge.second],
        absl::StrCat("e[", edge.first, ",", edge.second, "]"), &e[i]);
    edge_map.insert({followed_edge, i});
  }

  // Objective
  // Node costs
  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    for (NodeStrategyIdx j = 0; j < s[i].size(); ++j) {
      double accumulated_coefficient =
          solver->Objective().GetCoefficient(s[i][j]);
      double coefficient = request.c[i][j] + request.d[i][j];
      AddSalt(absl::StrCat(i, "S", j), request.saltiplier, &coefficient);
      solver->MutableObjective()->SetCoefficient(
          s[i][j], accumulated_coefficient + coefficient);
    }
  }
  // Edge costs
  for (EdgeIdx i = 0; i < num_edges; ++i) {
    for (EdgeStrategyIdx j = 0; j < e[i].size(); ++j) {
      double accumulated_coefficient =
          solver->Objective().GetCoefficient(e[i][j]);
      double coefficient = request.r[i][j];
      AddSalt(absl::StrCat(i, "E", j), request.saltiplier, &coefficient);
      solver->MutableObjective()->SetCoefficient(
          e[i][j], accumulated_coefficient + coefficient);
    }
  }

  // Constraints
  // 0. Do not choose solutions with infinity costs, as it will make the
  // objective value so large that other solution choices do not matter anymore.
  // Remove these constraints once b/238210866 is done.
  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    if (s[i].empty() || request.s_follow[i] >= 0) continue;
    bool all_infinity = true;
    for (NodeStrategyIdx j = 0; j < s[i].size(); ++j) {
      if (solver->Objective().GetCoefficient(s[i][j]) >= kInfinityCost) {
        MPConstraint* constraint = solver->MakeRowConstraint(
            0.0, 0.0, absl::StrCat("infinitycost: s[", i, "][", j, "] = 0"));
        constraint->SetCoefficient(s[i][j], 1.0);
      } else {
        all_infinity = false;
      }
    }
    if (all_infinity) {
      LOG(FATAL) << "All of s[" << i << "][*] have infinity costs";
    }
  }

  for (EdgeIdx i = 0; i < num_edges; ++i) {
    if (e[i].empty() || e_follow[i] >= 0) continue;
    bool all_infinity = true;
    for (EdgeStrategyIdx j = 0; j < e[i].size(); ++j) {
      if (solver->Objective().GetCoefficient(e[i][j]) >= kInfinityCost) {
        MPConstraint* constraint = solver->MakeRowConstraint(
            0.0, 0.0, absl::StrCat("infinitycost: e[", i, "][", j, "] = 0"));
        constraint->SetCoefficient(e[i][j], 1.0);
      } else {
        all_infinity = false;
      }
    }
    if (all_infinity) {
      auto err_msg =
          absl::StrCat("All of e[", request.e[i].first, "][",
                       request.e[i].second, "][*] have infinity costs");
      if (request.crash_at_infinity_costs_check) {
        LOG(FATAL) << err_msg;
      } else {
        LOG(WARNING) << err_msg;
        return AutoShardingSolverResult(absl::InternalError(err_msg), false);
      }
    }
  }

  // a. specified via "BoolVarArray"
  // b.
  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    if (request.s_follow[i] >= 0) continue;
    MPConstraint* constraint = solver->MakeRowConstraint(
        1.0, 1.0,
        absl::StrCat("sum(s[", i, "][j] for j = [0 .. ", s[i].size(),
                     ")) = 1"));
    for (NodeStrategyIdx j = 0; j < s[i].size(); ++j) {
      constraint->SetCoefficient(s[i][j], 1.0);
    }
  }
  // c.
  if (request.memory_budget > 0) {
    int64_t minimum_memory_budget_required_estimate = 0;
    for (LivenessIdx t = 0; t < request.live.size(); ++t) {
      int64_t minimum_memory_budget_required_estimate_local = 0;
      std::string str = "[";
      double total_fixed_memory_cost = 0.0;  // Amount consumed "no matter what"
      for (NodeIdx i : request.live[t]) {
        absl::StrAppend(&str, i, ", ");
        total_fixed_memory_cost +=
            *std::min_element(request.m[i].begin(), request.m[i].end());
      }
      str += "]";
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(),
          request.memory_budget - total_fixed_memory_cost,
          absl::StrCat("mem[", t, "] = ", str));
      for (NodeIdx i : request.live[t]) {
        auto fixed_memory_cost =
            *std::min_element(request.m[i].begin(), request.m[i].end());
        minimum_memory_budget_required_estimate_local += fixed_memory_cost;
        for (size_t j = 0; j < s[i].size(); ++j) {
          double accumulated_coefficient = constraint->GetCoefficient(s[i][j]);
          constraint->SetCoefficient(
              s[i][j],
              accumulated_coefficient + request.m[i][j] - fixed_memory_cost);
        }
      }
      minimum_memory_budget_required_estimate =
          std::max(minimum_memory_budget_required_estimate,
                   minimum_memory_budget_required_estimate_local);
    }
    LOG(INFO) << "Minimum memory budget estimate: "
              << minimum_memory_budget_required_estimate;
    LOG(INFO) << "Using memory budget: " << request.memory_budget;
  }

  // d. specified via "BoolVarArray"
  // e.
  for (EdgeIdx i = 0; i < num_edges; ++i) {
    if (e_follow[i] >= 0) continue;
    const std::pair<NodeIdx, NodeIdx>& edge = request.e[i];
    MPConstraint* constraint = solver->MakeRowConstraint(
        1.0, 1.0,
        absl::StrCat("sum(e[", edge.first, "][", edge.second, "][*]) = 1"));
    for (EdgeStrategyIdx j = 0; j < e[i].size(); ++j) {
      constraint->SetCoefficient(e[i][j], 1.0);
    }
  }
  // f.
  for (EdgeIdx i = 0; i < num_edges; ++i) {
    if (e_follow[i] >= 0) continue;
    const std::pair<NodeIdx, NodeIdx>& edge = request.e[i];
    for (NodeStrategyIdx p = 0; p < s[edge.first].size(); ++p) {
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), 0, absl::StrCat("f for i = ", i, ", p = ", p));
      constraint->SetCoefficient(s[edge.first][p], -1.0);
      for (NodeStrategyIdx q = 0; q < s[edge.second].size(); ++q) {
        constraint->SetCoefficient(e[i][p * s[edge.second].size() + q], 1.0);
      }
    }
  }
  // g.
  for (EdgeIdx i = 0; i < num_edges; ++i) {
    if (e_follow[i] >= 0) continue;
    const std::pair<NodeIdx, NodeIdx>& edge = request.e[i];
    for (NodeStrategyIdx q = 0; q < s[edge.second].size(); ++q) {
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), 0, absl::StrCat("g for i = ", i, ", q = ", q));
      constraint->SetCoefficient(s[edge.second][q], -1.0);
      for (NodeStrategyIdx p = 0; p < s[edge.first].size(); ++p) {
        constraint->SetCoefficient(e[i][p * s[edge.second].size() + q], 1.0);
      }
    }
  }
  // h.
  for (AliasIdx i = 0; i < request.a.size(); ++i) {
    const std::pair<NodeIdx, NodeIdx>& alias = request.a[i];
    for (NodeStrategyIdx p = 0; p < s[alias.first].size(); ++p) {
      for (NodeStrategyIdx q = 0; q < s[alias.second].size(); ++q) {
        // if lhs == 1
        if (request.v[i][p * s[alias.second].size() + q] > 0.5) {
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

  if (!request.s_hint.empty()) {
    std::vector<std::pair<const MPVariable*, double>> hint;
    for (NodeIdx i = 0; i < request.num_nodes; ++i) {
      if (request.s_follow[i] >= 0) continue;
      for (NodeStrategyIdx j = 0; j < s[i].size(); ++j) {
        hint.push_back({s[i][j], (request.s_hint[i] == j) ? 1.0 : 0.0});
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
  if (request.solver_timeout_in_seconds) {
    solver->SetTimeLimit(absl::Seconds(*request.solver_timeout_in_seconds));
  }
  VLOG(0) << "Starting solver " << solver->ProblemType() << "\n"
          << "Solver parameter string: " << solver_parameter_str << "\n"
          << "Number of workers: " << num_workers << "\n"
          << "Number of threads: " << solver->GetNumThreads() << "\n"
          << "Time limit: " << solver->time_limit() << "\n"
          << "Number variables for ILP: " << solver->NumVariables() << "\n"
          << "Total vector of variables: " << var_vector_cnt << "\n"
          << "Total instructions: " << request.num_nodes << "\n"
          << "Memory budget: " << request.memory_budget / (1024 * 1024 * 1024)
          << "GB\n"
          << "Number of ILP constraints: " << solver->NumConstraints();
  auto status = solver->Solve();

  if (status == operations_research::MPSolver::INFEASIBLE) {
    LOG(ERROR) << "MPSolver could not find any feasible solution.";
#ifdef PLATFORM_GOOGLE
    if (request.compute_iis) {
      operations_research::MPModelRequest model_request;
      solver->ExportModelToProto(model_request.mutable_model());
      if (solver->ProblemType() ==
          operations_research::MPSolver::SAT_INTEGER_PROGRAMMING) {
        model_request.set_solver_type(
            operations_research::MPModelRequest::SAT_INTEGER_PROGRAMMING);
      } else if (solver->ProblemType() == operations_research::MPSolver::
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
  } else if (status != operations_research::MPSolver::OPTIMAL) {
    auto err_msg = "Solver timed out. Will proceed without auto sharding.";
    LOG(WARNING) << err_msg;

    // The solver timed out. We now rely on heuristic-based sharding propagation
    // to degrade gracefully.
    return AutoShardingSolverResult(absl::InternalError(err_msg), true);
  }

  LOG(INFO) << "Solver Status: " << status
            << " Objective value: " << solver->Objective().Value();
  if (solver->Objective().Value() >= kInfinityCost) {
    LOG(WARNING) << "Objective (" << solver->Objective().Value()
                 << ") is larger than kInfinityCost. It means the solver "
                    "chooses a solution with kInfinityCost and there may be "
                    "numerical issues when the solver considering other costs.";
  }
  if (VLOG_IS_ON(10)) {
    // Print solver information for debugging. This hasn't been useful so far,
    // so leave it at VLOG level 10.
    operations_research::MPModelProto model_proto;
    solver->ExportModelToProto(&model_proto);
    VLOG(10) << "MODEL:";
    XLA_VLOG_LINES(10, model_proto.DebugString());
    VLOG(10) << "RESPONSE:";
    operations_research::MPSolutionResponse response;
    solver->FillSolutionResponseProto(&response);
    XLA_VLOG_LINES(10, response.DebugString());
  }

  // Return value
  double unsalted_objective = 0.0;
  std::vector<NodeStrategyIdx> chosen_strategy(request.num_nodes, -1);
  std::vector<EdgeStrategyIdx> e_val(num_edges, -1);
  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    for (NodeStrategyIdx j = 0; j < s[i].size(); ++j) {
      // if lhs == 1
      if (s[i][j]->solution_value() > 0.5) {
        chosen_strategy[i] = j;
        unsalted_objective += request.c[i][j] + request.d[i][j];
        break;
      }
    }
  }
  for (EdgeIdx i = 0; i < num_edges; ++i) {
    for (EdgeStrategyIdx j = 0; j < e[i].size(); ++j) {
      // if lhs == 1
      if (e[i][j]->solution_value() > 0.5) {
        e_val[i] = j;
        unsalted_objective += request.r[i][j];
        break;
      }
    }
  }

  LOG(INFO) << "Unsalted objective value: " << unsalted_objective;
  LOG(INFO) << "N = " << request.num_nodes;
  if (request.memory_budget < 0) {
    LOG(INFO) << "memory budget: -1";
  } else {
    LOG(INFO) << "memory budget: "
              << request.memory_budget / (1024 * 1024 * 1024) << " GB";
  }
  PrintLargestInstructions(chosen_strategy, request.m, request.live,
                           request.instruction_names);
  return AutoShardingSolverResult(
      std::make_tuple(std::move(chosen_strategy), std::move(e_val),
                      unsalted_objective),
      false);
}

bool AutoShardingEvaluation::operator==(
    const AutoShardingEvaluation& other) const {
  return violation_codes == other.violation_codes &&
         total_communication_cost == other.total_communication_cost &&
         total_computation_cost == other.total_computation_cost &&
         total_resharding_cost == other.total_resharding_cost &&
         total_cost == other.total_cost &&
         lower_bound_communication_cost ==
             other.lower_bound_communication_cost &&
         lower_bound_computation_cost == other.lower_bound_computation_cost &&
         lower_bound_resharding_cost == other.lower_bound_resharding_cost &&
         lower_bound_cost == other.lower_bound_cost;
}

AutoShardingEvaluation Evaluate(const AutoShardingSolverRequest& request,
                                const AutoShardingSolverResult& result) {
  const std::vector<NodeStrategyIdx>& s_val = std::get<0>(*result.status);
  const std::vector<EdgeStrategyIdx>& e_val = std::get<1>(*result.status);
  AutoShardingEvaluation evaluation;
  // Compute violations.
  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    if (request.s_follow[i] >= 0 && s_val[i] != s_val[request.s_follow[i]]) {
      evaluation.violation_codes.insert(kFollowerViolationCode);
    }
  }
  for (AliasIdx i = 0; i < request.a.size(); ++i) {
    const std::pair<NodeIdx, NodeIdx>& alias = request.a[i];
    NodeStrategyIdx p = s_val[alias.first], q = s_val[alias.second];
    if (request.v[i][p * request.s_len[alias.second] + q] > 0.5) {
      evaluation.violation_codes.insert(kAliasViolationCode);
    }
  }
  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    if (request.c[i][s_val[i]] + request.d[i][s_val[i]] >= kInfinityCost) {
      evaluation.violation_codes.insert(kInfiniteCostViolationCode);
    }
  }
  for (EdgeIdx i = 0; i < request.e.size(); ++i) {
    if (request.r[i][e_val[i]] >= kInfinityCost) {
      evaluation.violation_codes.insert(kInfiniteCostViolationCode);
    }
  }
  if (request.memory_budget > 0) {
    for (LivenessIdx t = 0; t < request.live.size(); ++t) {
      double total_memory_cost = 0.0;
      for (NodeIdx i : request.live[t]) {
        total_memory_cost += request.m[i][s_val[i]];
      }
      if (total_memory_cost > request.memory_budget) {
        evaluation.violation_codes.insert(kMemoryViolationCode);
      }
    }
  }
  // Compute metrics & lower bounds.
  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    evaluation.total_communication_cost += request.d[i][s_val[i]];
    evaluation.total_computation_cost += request.c[i][s_val[i]];
    evaluation.lower_bound_communication_cost +=
        *std::min_element(request.d[i].begin(), request.d[i].end());
    evaluation.lower_bound_computation_cost +=
        *std::min_element(request.c[i].begin(), request.c[i].end());
  }
  for (EdgeIdx i = 0; i < request.e.size(); ++i) {
    evaluation.total_resharding_cost += request.r[i][e_val[i]];
    evaluation.lower_bound_resharding_cost +=
        *std::min_element(request.r[i].begin(), request.r[i].end());
  }
  evaluation.total_cost += evaluation.total_communication_cost;
  evaluation.total_cost += evaluation.total_computation_cost;
  evaluation.total_cost += evaluation.total_resharding_cost;
  evaluation.lower_bound_cost += evaluation.lower_bound_communication_cost;
  evaluation.lower_bound_cost += evaluation.lower_bound_computation_cost;
  evaluation.lower_bound_cost += evaluation.lower_bound_resharding_cost;
  return evaluation;
}

std::vector<std::string> Rationalize(const AutoShardingSolverRequest& request,
                                     const AutoShardingSolverResult& result,
                                     const AutoShardingSolverResult& subopt) {
  std::vector<std::string> rationales;
  const std::vector<std::string>& names = request.instruction_names;

  const std::vector<NodeStrategyIdx>& s_result = std::get<0>(*result.status);
  const std::vector<NodeStrategyIdx>& s_subopt = std::get<0>(*subopt.status);
  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    const NodeStrategyIdx j = s_result[i], k = s_subopt[i];
    if (j != k) {
      rationales.push_back(absl::StrCat("strategy changes for ", names[i], " (",
                                        j, " -> ", k, ")"));
    }
    const double dj = request.d[i][j], dk = request.d[i][k];
    if (dj < dk) {
      rationales.push_back(absl::StrCat("communication cost increases for ",
                                        names[i], " (", dj, " -> ", dk, ")"));
    }
    const double cj = request.c[i][j], ck = request.c[i][k];
    if (cj < ck) {
      rationales.push_back(absl::StrCat("computation cost increases for ",
                                        names[i], " (", cj, " -> ", ck, ")"));
    }
  }

  const std::vector<EdgeStrategyIdx>& e_result = std::get<1>(*result.status);
  const std::vector<EdgeStrategyIdx>& e_subopt = std::get<1>(*subopt.status);
  for (EdgeIdx i = 0; i < request.e.size(); ++i) {
    const std::pair<NodeIdx, NodeIdx>& edge = request.e[i];
    const EdgeStrategyIdx j = e_result[i], k = e_subopt[i];
    const double rj = request.r[i][j], rk = request.r[i][k];
    if (rj < rk) {
      const std::string edge_name =
          absl::StrCat(names[edge.first], " and ", names[edge.second]);
      rationales.push_back(absl::StrCat("resharding cost increases for ",
                                        edge_name, " (", rj, " -> ", rk, ")"));
    }
  }

  return rationales;
}

}  // namespace spmd
}  // namespace xla
