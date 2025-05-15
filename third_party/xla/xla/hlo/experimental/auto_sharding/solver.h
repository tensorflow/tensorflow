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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_SOLVER_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_SOLVER_H_

#include <optional>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/hlo/experimental/auto_sharding/iopddl.h"

namespace iopddl {

class Solver {
 public:
  absl::StatusOr<Solution> Solve(
      const Problem& problem,
      absl::Duration timeout = absl::InfiniteDuration());
};

}  // namespace iopddl

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_SOLVER_H_
