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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_WRAPPER_H_
#define TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_WRAPPER_H_

#include <cstdint>
#include <vector>

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_cost_graph.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_live_range.h"

namespace xla {
namespace spmd {

// A wrapper around the solver that converts the given objects into a
// combinatorial optimization problem & solves it.
AutoShardingSolverResult CallSolver(
    const HloLiveRange& hlo_live_range, const LivenessSet& liveness_set,
    const StrategyMap& strategy_map, const LeafStrategies& leaf_strategies,
    const CostGraph& cost_graph, const AliasSet& alias_set,
    const std::vector<NodeStrategyIdx>& s_hint,
    int64_t memory_budget_per_device, bool crash_at_infinity_costs_check,
    bool compute_iis, int64_t solver_timeout_in_seconds,
    bool allow_alias_to_follower_conversion);

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_WRAPPER_H_
