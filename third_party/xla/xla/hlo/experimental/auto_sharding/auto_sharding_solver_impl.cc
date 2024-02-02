/* Copyright 2023 The OpenXLA Authors.

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

#include <vector>

#include "absl/log/check.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "ortools/linear_solver/linear_solver.h"

namespace xla {
namespace spmd {

using ::operations_research::MPSolver;
using ::operations_research::MPVariable;

MPVariable* CreateMakespanVar(const AutoShardingSolverRequest& request,
                              const std::vector<std::vector<MPVariable*>>& e,
                              MPSolver& solver) {
  return nullptr;  // TODO(moffitt): Implement this.
}

double EvaluateMakespan(const AutoShardingSolverRequest& request,
                        const AutoShardingSolverResult& result,
                        AutoShardingEvaluation& evaluation) {
  return 0.0;  // TODO(moffitt): Implement this.
}

std::vector<std::vector<NodeStrategyIdx>> StratFollow(
    const AutoShardingSolverRequest& request) {
  CHECK_EQ(request.num_nodes(), request.s_len_size());
  std::vector<std::vector<NodeStrategyIdx>> strat_follow(request.num_nodes());
  for (NodeIdx node_idx = 0; node_idx < request.num_nodes(); ++node_idx) {
    if (request.s_follow(node_idx) >= 0) continue;
    strat_follow[node_idx].resize(request.s_len(node_idx), -1);
  }
  return strat_follow;
}

}  // namespace spmd
}  // namespace xla
