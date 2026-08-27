/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_GROUPER_H_
#define XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_GROUPER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/utils/hlo_original_value_analysis.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"
#include "xla/hlo/utils/hlo_sharding_reconstruction_util.h"
#include "xla/literal.h"

namespace xla {

// Aggregates recovered original tensor literals for debug log instructions
// (e.g., `xla_debug_log`) across multi-operand calls and execution scopes.
//
// WHAT IT DOES:
// Collects individual recovered operand literals belonging to the same debug
// log call (`callback_id`) within a given execution scope, and fires a single
// `GroupReadyCallback` containing all ordered operand literals once the group
// is complete.
//
// WHY IT IS NEEDED:
// In the original HLO program, a user may insert an `xla_debug_log` call taking
// multiple operands (`op_id = 0, 1, ...`). Compiler optimization passes (e.g.,
// fusion, constant folding, sharding, loop transformations) often cause these
// individual operands to be evaluated or recovered asynchronously at different
// times, or even hoisted across loop boundaries.
// The grouper acts as an aggregation barrier so that downstream consumers
// receive a complete snapshot of all operands for a debug log call in a single
// callback invocation, rather than receiving fragmented piecemeal updates.
//
// HOW IT WORKS:
// - Grouping Key: Groups are uniquely identified by a `GroupKey` consisting of
//   `scope_instructions` (the execution/loop context) and `callback_id`.
// - Initialization & Filtering: When a tensor arrives via
//   `OnOriginalTensorReady`, the grouper initializes a `PendingGroup` for the
//   corresponding scope and callback. It inspects `optimized_module_`'s
//   debug attributes to discover all expected `op_id`s for that callback.
//   Unrecoverable operands (via `HloOriginalValueAnalysis`) or non-addressable
//   logical devices are excluded from the expected set so the group does not
//   wait indefinitely for unrecoverable data.
// - Hoisting Support: Tensors with wildcard iteration indices (e.g., hoisted
//   loop invariants) are cached in `hoisted_values_` and automatically merged
//   into matching pending groups when initialized or updated.
// - Dispatch: When `received_op_ids.size() == expected_op_ids.size()`, the
//   group is complete. The grouper calls `on_group_ready_` with the complete
//   ordered vector of `Literal`s (indexed by `op_id`) and erases the pending
//   group state.
class HloOriginalValueGrouper {
 public:
  using GroupReadyCallback = std::function<void(
      int64_t callback_id, int64_t replica_id, int64_t partition_id,
      absl::Span<std::shared_ptr<Literal> const> literals)>;

  HloOriginalValueGrouper(
      const HloModule* optimized_module,
      std::shared_ptr<const HloOriginalValueAnalysis> analysis,
      GroupReadyCallback on_group_ready, bool skip_recoverability_check = false,
      std::optional<absl::AnyInvocable<bool(int64_t)>>
          logical_device_is_addressable = std::nullopt);

  ~HloOriginalValueGrouper();

  void OnOriginalTensorReady(
      const AbsoluteScopedTensorKey& original_tensor_key,
      const OriginalArray& original_tensor,
      std::shared_ptr<Literal> recovered_data,
      const std::vector<HloModule::DebugAttributes>& debug_attributes,
      int64_t partition_id);

 private:
  const HloModule* optimized_module_;
  std::shared_ptr<const HloOriginalValueAnalysis> analysis_;
  GroupReadyCallback on_group_ready_;
  std::optional<absl::AnyInvocable<bool(int64_t)>>
      logical_device_is_addressable_;
  bool skip_recoverability_check_;

  struct PendingGroup {
    std::vector<std::shared_ptr<Literal>> literals;
    absl::flat_hash_set<int64_t> expected_op_ids;
    absl::flat_hash_set<int64_t> received_op_ids;
  };

  struct GroupKey {
    std::vector<ScopeInstruction> scope_instructions;
    int64_t callback_id;

    bool operator==(const GroupKey& other) const {
      return scope_instructions == other.scope_instructions &&
             callback_id == other.callback_id;
    }

    template <typename H>
    friend H AbslHashValue(H h, const GroupKey& key) {
      return H::combine(std::move(h), key.scope_instructions, key.callback_id);
    }
  };

  absl::flat_hash_map<GroupKey, PendingGroup> pending_groups_;

  // Stores literals for hoisted instructions.
  // Keyed by wildcard scope_instructions, then callback_id, then op_id.
  absl::flat_hash_map<
      std::vector<ScopeInstruction>,
      absl::flat_hash_map<
          int64_t, absl::flat_hash_map<int64_t, std::shared_ptr<Literal>>>>
      hoisted_values_;
};

}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_GROUPER_H_
