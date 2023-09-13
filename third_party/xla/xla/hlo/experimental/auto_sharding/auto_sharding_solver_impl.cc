/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"
#include "ortools/linear_solver/linear_solver.h"

using MPSolver = operations_research::MPSolver;
using MPVariable = operations_research::MPVariable;

namespace xla {
namespace spmd {

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

}  // namespace spmd
}  // namespace xla
