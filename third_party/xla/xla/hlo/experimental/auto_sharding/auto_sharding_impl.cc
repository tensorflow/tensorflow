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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_cost_graph.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_wrapper.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/hlo_cost_analysis.h"

namespace xla {
namespace spmd {

absl::StatusOr<AutoShardingSolverOutput> Solve(
    const HloModule& hlo_module, const HloLiveRange& hlo_live_range,
    const StrategyMap& strategy_map, const StrategyGroups& strategy_groups,
    const CostGraph& cost_graph, const AliasSet& alias_set,
    const AutoShardingOption& option, absl::string_view request_prefix,
    const absl::flat_hash_map<std::string, HloSharding>&
        sharding_propagation_solution) {
  return CreateAutoShardingSolverRequestAndCallSolver(
      hlo_module, hlo_live_range, strategy_map, strategy_groups, cost_graph,
      alias_set, /*s_hint*/ {}, /*compute_iis*/ true,
      option.solver_timeout_in_seconds, option, /*max_cost*/ std::nullopt,
      request_prefix, sharding_propagation_solution,
      /*deterministic mode*/ true);
}

void PopulateTemporalValues(const CostGraph& cost_graph,
                            AutoShardingSolverRequest& request) {
  // TODO(moffitt): Implement this.
}

double GetDotConvReplicationPenalty(const HloInstruction* inst,
                                    size_t instruction_id, size_t window,
                                    const HloInstructionSequence& sequence,
                                    const HloCostAnalysis& hlo_cost_analysis) {
  return 0;
}

}  // namespace spmd
}  // namespace xla
