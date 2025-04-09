/*
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "xla/hlo/experimental/auto_sharding/solver.h"

#include <stdlib.h>

#include <iostream>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/hlo/experimental/auto_sharding/iopddl.h"

namespace iopddl {

////////////////////////////////////////////////////////////////////////////////
//  A simple solver that generates random solutions until the given timeout.  //
//  Contest participants SHOULD replace this implementation with their own!!  //
////////////////////////////////////////////////////////////////////////////////

absl::StatusOr<Solution> Solver::Solve(const Problem& problem,
                                       absl::Duration timeout) {
  const absl::Time start_time = absl::Now();
  std::optional<TotalCost> best_cost;
  std::optional<Solution> best_solution;
  unsigned int seed = 2025;
  while (absl::Now() - start_time < timeout) {
    Solution solution;
    solution.reserve(problem.nodes.size());
    for (const Node& node : problem.nodes) {
      solution.push_back(rand_r(&seed) % node.strategies.size());
    }
    auto cost = Evaluate(problem, solution);
    if (!cost.ok() || (best_cost && *best_cost <= *cost)) {
      continue;
    }
    std::cout << "# Found solution [" << absl::StrJoin(solution, ", ")
              << "] with cost " << *cost << std::endl;;
    best_cost = *cost;
    best_solution = solution;
  }
  if (!best_solution) {
    return absl::NotFoundError("No solution found");
  }
  return *best_solution;
}

}  // namespace iopddl
