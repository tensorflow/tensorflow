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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
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

void PrintLargestInstructions(
    const std::vector<int64_t>& chosen_strategy,
    const std::vector<std::vector<double>>& memory_cost,
    const std::vector<std::vector<int>>& liveness,
    const std::vector<std::string>& instruction_names) {
  // This memory consumption computation is different from
  // that in PrintAutoShardingSolution() because how L and m are created to be
  // different from liveness_set and strategy.memory_cost.

  std::vector<int64_t> instruction_ids;
  std::vector<std::pair<size_t, double>> time_memory_usage;
  for (size_t t = 0; t < liveness.size(); ++t) {
    double mem = 0.0;
    for (auto i : liveness[t]) {
      mem += memory_cost[i][chosen_strategy[i]];
    }
    time_memory_usage.push_back(std::make_pair(t, mem));
  }
  struct {
    bool operator()(std::pair<size_t, double> a,
                    std::pair<size_t, double> b) const {
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
  std::vector<std::pair<size_t, double>> instruction_mem;
  absl::flat_hash_set<size_t> instruction_set;
  for (size_t t = 0; t < k; t++) {
    for (auto i : liveness[time_memory_usage.at(t).first]) {
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
    // Set random_seed, interleave_search and share_binary_clauses for
    // determinism, and num_workers for parallelism.
    solver_parameter_str = absl::StrCat(
        "share_binary_clauses:false,random_seed:1,interleave_"
        "search:true,num_workers:",
        num_workers);
    solver->SetSolverSpecificParametersAsString(solver_parameter_str);
  }
#endif
  // Create variables
  std::vector<std::vector<MPVariable*>> s(request.num_nodes);
  std::vector<std::vector<MPVariable*>> e(num_edges);

  size_t var_vector_cnt = 0;
  for (size_t i = 0; i < request.num_nodes; ++i) {
    if (request.s_follow[i] < 0) {
      var_vector_cnt += 1;
      // Creates variables for instructions that do not follow others.
      solver->MakeBoolVarArray(request.s_len[i], absl::StrCat("s[", i, "]"),
                               &s[i]);
    }
  }

  for (size_t i = 0; i < request.num_nodes; ++i) {
    if (request.s_follow[i] >= 0) {
      // Copies the variable of followed instruction to the following
      // instruction.
      s[i] = s[request.s_follow[i]];
    }
  }

  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = request.e[i];
    solver->MakeBoolVarArray(
        request.s_len[edge.first] * request.s_len[edge.second],
        absl::StrCat("e[", edge.first, ",", edge.second, "]"), &e[i]);
  }

  // Objective
  // Node costs
  for (size_t i = 0; i < request.num_nodes; ++i) {
    for (size_t j = 0; j < s[i].size(); ++j) {
      double accumulated_coefficient =
          solver->MutableObjective()->GetCoefficient(s[i][j]);
      solver->MutableObjective()->SetCoefficient(
          s[i][j], accumulated_coefficient + request.c[i][j] + request.d[i][j]);
    }
  }
  // Edge costs
  for (size_t i = 0; i < num_edges; ++i) {
    for (size_t j = 0; j < e[i].size(); ++j) {
      double accumulated_coefficient =
          solver->MutableObjective()->GetCoefficient(e[i][j]);
      solver->MutableObjective()->SetCoefficient(
          e[i][j], accumulated_coefficient + request.r[i][j]);
    }
  }

  // Constraints
  // 0. Do not choose solutions with infinity costs, as it will make the
  // objective value so large that other solution choices do not matter anymore.
  // Remove these constraints once b/238210866 is done.
  for (size_t i = 0; i < request.num_nodes; ++i) {
    if (s[i].empty()) {
      continue;
    }
    bool all_infinity = true;
    for (size_t j = 0; j < s[i].size(); ++j) {
      if (solver->MutableObjective()->GetCoefficient(s[i][j]) >=
          kInfinityCost) {
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

  for (size_t i = 0; i < num_edges; ++i) {
    if (e[i].empty()) {
      continue;
    }
    bool all_infinity = true;
    for (size_t j = 0; j < e[i].size(); ++j) {
      std::pair<int, int> edge = request.e[i];
      solver->MutableObjective()->SetCoefficient(e[i][j], request.r[i][j]);
      if (request.r[i][j] >= kInfinityCost) {
        MPConstraint* constraint = solver->MakeRowConstraint(
            0.0, 0.0,
            absl::StrCat("infinitycost: e[", edge.first, "][", edge.second,
                         "][", j, "] = 0"));
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
  for (size_t i = 0; i < request.num_nodes; ++i) {
    MPConstraint* constraint = solver->MakeRowConstraint(
        1.0, 1.0,
        absl::StrCat("sum(s[", i, "][j] for j = [0 .. ", s[i].size(),
                     ")) = 1"));
    for (size_t j = 0; j < s[i].size(); ++j) {
      constraint->SetCoefficient(s[i][j], 1.0);
    }
  }
  // c.
  if (request.memory_budget > 0) {
    for (size_t t = 0; t < request.live.size(); ++t) {
      std::string str = "[";
      double total_fixed_memory_cost = 0.0;  // Amount consumed "no matter what"
      for (auto i : request.live[t]) {
        absl::StrAppend(&str, i, ", ");
        total_fixed_memory_cost +=
            *std::min_element(request.m[i].begin(), request.m[i].end());
      }
      str += "]";
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(),
          request.memory_budget - total_fixed_memory_cost,
          absl::StrCat("mem[", t, "] = ", str));
      for (auto i : request.live[t]) {
        auto fixed_memory_cost =
            *std::min_element(request.m[i].begin(), request.m[i].end());
        for (size_t j = 0; j < s[i].size(); ++j) {
          double accumulated_coefficient = constraint->GetCoefficient(s[i][j]);
          constraint->SetCoefficient(
              s[i][j],
              accumulated_coefficient + request.m[i][j] - fixed_memory_cost);
        }
      }
    }
  }

  // d. specified via "BoolVarArray"
  // e.
  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = request.e[i];
    MPConstraint* constraint = solver->MakeRowConstraint(
        1.0, 1.0,
        absl::StrCat("sum(e[", edge.first, "][", edge.second, "][*]) = 1"));
    for (size_t j = 0; j < e[i].size(); ++j) {
      constraint->SetCoefficient(e[i][j], 1.0);
    }
  }
  // f.
  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = request.e[i];
    for (size_t p = 0; p < s[edge.first].size(); ++p) {
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), 0, absl::StrCat("f for i = ", i, ", p = ", p));
      constraint->SetCoefficient(s[edge.first][p], -1.0);
      for (size_t q = 0; q < s[edge.second].size(); ++q) {
        constraint->SetCoefficient(e[i][p * s[edge.second].size() + q], 1.0);
      }
    }
  }
  // g.
  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = request.e[i];
    for (size_t q = 0; q < s[edge.second].size(); ++q) {
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), 0, absl::StrCat("g for i = ", i, ", q = ", q));
      constraint->SetCoefficient(s[edge.second][q], -1.0);
      for (size_t p = 0; p < s[edge.first].size(); ++p) {
        constraint->SetCoefficient(e[i][p * s[edge.second].size() + q], 1.0);
      }
    }
  }
  // h.
  for (size_t i = 0; i < request.a.size(); ++i) {
    std::pair<int, int> alias = request.a[i];
    for (size_t p = 0; p < s[alias.first].size(); ++p) {
      for (size_t q = 0; q < s[alias.second].size(); ++q) {
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
    operations_research::MPModelRequest model_request;
    solver->ExportModelToProto(model_request.mutable_model());
    if (solver->ProblemType() ==
        operations_research::MPSolver::SAT_INTEGER_PROGRAMMING) {
      model_request.set_solver_type(
          operations_research::MPModelRequest::SAT_INTEGER_PROGRAMMING);
    } else if (solver->ProblemType() ==
               operations_research::MPSolver::SCIP_MIXED_INTEGER_PROGRAMMING) {
      model_request.set_solver_type(
          operations_research::MPModelRequest::SCIP_MIXED_INTEGER_PROGRAMMING);
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
  std::vector<int64_t> chosen_strategy(request.num_nodes, -1),
      e_val(num_edges, -1);
  for (int i = 0; i < request.num_nodes; ++i) {
    for (int j = 0; j < s[i].size(); ++j) {
      // if lhs == 1
      if (s[i][j]->solution_value() > 0.5) {
        chosen_strategy[i] = j;
        break;
      }
    }
  }
  for (int i = 0; i < num_edges; ++i) {
    for (int j = 0; j < e[i].size(); ++j) {
      // if lhs == 1
      if (e[i][j]->solution_value() > 0.5) {
        e_val[i] = j;
        break;
      }
    }
  }

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
                      solver->Objective().Value()),
      false);
}

bool AutoShardingEvaluation::operator==(
    const AutoShardingEvaluation& other) const {
  return violation_codes == other.violation_codes &&
         total_communication_cost == other.total_communication_cost &&
         total_computation_cost == other.total_computation_cost &&
         total_resharding_cost == other.total_resharding_cost &&
         total_cost == other.total_cost;
}

AutoShardingEvaluation Evaluate(const AutoShardingSolverRequest& request,
                                const AutoShardingSolverResult& result) {
  const std::vector<int64_t>& s_val = std::get<0>(*result.status);
  const std::vector<int64_t>& e_val = std::get<1>(*result.status);
  AutoShardingEvaluation evaluation;
  // Compute violations.
  for (size_t i = 0; i < request.num_nodes; ++i) {
    if (request.s_follow[i] >= 0 && s_val[i] != s_val[request.s_follow[i]]) {
      evaluation.violation_codes.insert(kFollowerViolationCode);
    }
  }
  for (size_t i = 0; i < request.a.size(); ++i) {
    const std::pair<int, int>& alias = request.a[i];
    size_t p = s_val[alias.first], q = s_val[alias.second];
    if (request.v[i][p * request.s_len[alias.second] + q] > 0.5) {
      evaluation.violation_codes.insert(kAliasViolationCode);
    }
  }
  if (request.memory_budget > 0) {
    for (size_t t = 0; t < request.live.size(); ++t) {
      double total_memory_cost = 0.0;
      for (auto i : request.live[t]) {
        total_memory_cost += request.m[i][s_val[i]];
      }
      if (total_memory_cost > request.memory_budget) {
        evaluation.violation_codes.insert(kMemoryViolationCode);
      }
    }
  }
  // Compute metrics.
  for (size_t i = 0; i < request.num_nodes; ++i) {
    const int64_t j = s_val[i];
    evaluation.total_communication_cost += request.d[i][j];
    evaluation.total_computation_cost += request.c[i][j];
  }
  for (size_t i = 0; i < request.e.size(); ++i) {
    const int64_t j = e_val[i];
    evaluation.total_resharding_cost += request.r[i][j];
  }
  evaluation.total_cost += evaluation.total_communication_cost;
  evaluation.total_cost += evaluation.total_computation_cost;
  evaluation.total_cost += evaluation.total_resharding_cost;
  return evaluation;
}

}  // namespace spmd
}  // namespace xla
