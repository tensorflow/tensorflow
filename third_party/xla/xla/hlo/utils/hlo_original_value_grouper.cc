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

#include "xla/hlo/utils/hlo_original_value_grouper.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/utils/hlo_original_value_analysis.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"
#include "xla/hlo/utils/hlo_sharding_reconstruction_util.h"
#include "xla/literal.h"

namespace xla {
namespace {

constexpr int64_t kWildcardIndex = -1;
constexpr int64_t kPlaceholderIndex = -2;

}  // namespace

HloOriginalValueGrouper::HloOriginalValueGrouper(
    const HloModule* optimized_module,
    std::shared_ptr<const HloOriginalValueAnalysis> analysis,
    GroupReadyCallback on_group_ready, bool skip_recoverability_check,
    std::optional<absl::AnyInvocable<bool(int64_t)>>
        logical_device_is_addressable)
    : optimized_module_(optimized_module),
      analysis_(std::move(analysis)),
      on_group_ready_(std::move(on_group_ready)),
      logical_device_is_addressable_(std::move(logical_device_is_addressable)),
      skip_recoverability_check_(skip_recoverability_check) {}

HloOriginalValueGrouper::~HloOriginalValueGrouper() {
  if (!pending_groups_.empty()) {
    LOG(ERROR)
        << "Found pending groups when HloOriginalValueGrouper "
           "is destroyed. This indicates a bug in "
           "HloOriginalValueAnalysis::IsOriginalAbsoluteTensorKeyRecoverable.";
    // NOLINTNEXTLINE
    for (const auto& [key, group] : pending_groups_) {
      LOG(ERROR) << "   callback_id: " << key.callback_id
                 << ", expected size: " << group.expected_op_ids.size()
                 << ", received size: " << group.received_op_ids.size();
      // NOLINTNEXTLINE
      for (int id : group.expected_op_ids) {
        if (!group.received_op_ids.contains(id)) {
          LOG(ERROR) << "     Missing op_id: " << id;
        }
      }
    }
  }
}

void HloOriginalValueGrouper::OnOriginalTensorReady(
    const AbsoluteScopedTensorKey& original_tensor_key,
    const OriginalArray& original_tensor,
    std::shared_ptr<Literal> recovered_data,
    const std::vector<HloModule::DebugAttributes>& debug_attributes,
    int64_t partition_id) {
  LOG(INFO) << "OnOriginalTensorReady called for "
            << original_tensor_key.ToString();
  std::vector<const HloModule::DebugAttributes*> matching_attrs;
  for (const auto& attr : debug_attributes) {
    if (attr.log_mode ==
            HloModule::DebugAttributes::DebugLogMode::kFusionDebugger ||
        attr.log_mode == HloModule::DebugAttributes::DebugLogMode::kDefault) {
      matching_attrs.push_back(&attr);
    }
  }
  if (matching_attrs.empty()) {
    LOG(INFO) << "No matching debug_attributes found";
    return;
  }

  auto scopes_match = [](const ScopeInstruction& cand,
                         const ScopeInstruction& query) {
    if (cand.instruction_name != query.instruction_name) {
      return false;
    }
    if (cand.iteration_index == kWildcardIndex ||
        cand.iteration_index == kPlaceholderIndex) {
      return true;
    }
    return cand.iteration_index == query.iteration_index;
  };

  auto scopes_match_suffix =
      [&](absl::Span<const ScopeInstruction> cand_scopes,
          absl::Span<const ScopeInstruction> query_scopes) {
        if (cand_scopes.size() > query_scopes.size()) {
          return false;
        }
        for (size_t i = 0; i < cand_scopes.size(); ++i) {
          if (!scopes_match(cand_scopes[cand_scopes.size() - 1 - i],
                            query_scopes[query_scopes.size() - 1 - i])) {
            return false;
          }
        }
        return true;
      };

  const auto& current_scopes = original_tensor_key.scope_instructions;

  bool is_wildcard = absl::c_any_of(current_scopes, [](const auto& scope) {
    return scope.MatchesAnyIteration() ||
           scope.iteration_index == kPlaceholderIndex;
  });

  if (is_wildcard) {
    LOG(INFO) << "Processing wildcard tensor";
    for (const auto* attr : matching_attrs) {
      int64_t current_callback_id = attr->callback_id;
      int64_t op_id = attr->op_id;
      if (recovered_data != nullptr) {
        hoisted_values_[current_scopes][current_callback_id][op_id] =
            recovered_data;
      } else {
        hoisted_values_[current_scopes][current_callback_id][op_id] = nullptr;
      }

      // Fill into pending groups that are waiting for this hoisted value
      std::vector<GroupKey> groups_to_complete;
      // NOLINTNEXTLINE
      for (auto& [key, group] : pending_groups_) {
        if (key.callback_id == current_callback_id) {
          if (scopes_match_suffix(current_scopes, key.scope_instructions)) {
            if (group.expected_op_ids.contains(op_id) &&
                group.received_op_ids.insert(op_id).second) {
              LOG(INFO)
                  << "Filling hoisted value into pending group for callback_id="
                  << current_callback_id;
              const auto& stored_literal =
                  hoisted_values_[current_scopes][current_callback_id][op_id];
              if (stored_literal != nullptr) {
                group.literals[op_id] = stored_literal;
              }
              if (group.received_op_ids.size() ==
                  group.expected_op_ids.size()) {
                groups_to_complete.push_back(key);
              }
            }
          }
        }
      }

      for (const auto& key : groups_to_complete) {
        auto& group = pending_groups_[key];
        LOG(INFO)
            << "Invoking group ready callback (hoisted fill) for callback_id="
            << key.callback_id;
        on_group_ready_(key.callback_id, 0, partition_id, group.literals);
        pending_groups_.erase(key);
      }
    }
    return;
  }

  // Concrete tensor case
  LOG(INFO) << "Processing concrete tensor";

  // Find all relevant callback IDs for this scope
  absl::flat_hash_set<int64_t> relevant_callback_ids;
  for (const auto& [orig_array, attrs_vec] :
       // NOLINTNEXTLINE
       optimized_module_->debug_attributes()) {
    for (const auto& attr : attrs_vec) {
      auto cand_relative_key = RelativeScopedTensorKey::FromString(
          orig_array.instruction_name, orig_array.shape_index);
      if (scopes_match_suffix(cand_relative_key.scope_instructions,
                              current_scopes)) {
        relevant_callback_ids.insert(attr.callback_id);
      }
    }
  }

  // NOLINTNEXTLINE
  for (int64_t callback_id : relevant_callback_ids) {
    GroupKey key{current_scopes, callback_id};
    auto& group = pending_groups_[key];

    if (group.expected_op_ids.empty()) {
      LOG(INFO) << "Initializing group for callback_id=" << callback_id;
      int64_t max_op_id = -1;

      for (const auto& [orig_array, attrs_vec] :
           // NOLINTNEXTLINE
           optimized_module_->debug_attributes()) {
        for (const auto& candidate_attr : attrs_vec) {
          if (candidate_attr.callback_id == callback_id) {
            auto cand_relative_key = RelativeScopedTensorKey::FromString(
                orig_array.instruction_name, orig_array.shape_index);

            if (scopes_match_suffix(cand_relative_key.scope_instructions,
                                    current_scopes)) {
              max_op_id = std::max(max_op_id, candidate_attr.op_id);

              AbsoluteScopedTensorKey new_absolute_key =
                  AbsoluteScopedTensorKey::Create(cand_relative_key.tensor_key,
                                                  current_scopes);

              bool is_addressable = true;
              if (logical_device_is_addressable_.has_value()) {
                auto opt_mapping_it =
                    analysis_->original_to_optimized_tensor_map().find(
                        new_absolute_key.tensor_key);
                if (opt_mapping_it !=
                    analysis_->original_to_optimized_tensor_map().end()) {
                  for (const auto& [opt_tensor_key, _] :
                       opt_mapping_it->second) {
                    auto sharding_it =
                        analysis_->optimized_tensor_sharding().find(
                            opt_tensor_key);
                    if (sharding_it !=
                        analysis_->optimized_tensor_sharding().end()) {
                      absl::flat_hash_set<int64_t> logical_devices =
                          GetLogicalDeviceIds(sharding_it->second,
                                              optimized_module_);
                      is_addressable =
                          !absl::c_any_of(logical_devices, [&](int64_t id) {
                            return !(*logical_device_is_addressable_)(id);
                          });
                      if (!is_addressable) {
                        break;
                      }
                    }
                  }
                }
              }

              if (is_addressable &&
                  (skip_recoverability_check_ ||
                   analysis_->IsOriginalAbsoluteTensorKeyRecoverable(
                       new_absolute_key))) {
                group.expected_op_ids.insert(candidate_attr.op_id);
              }
            }
          }
        }
      }
      group.literals.resize(max_op_id + 1);

      // Fill in already available hoisted values
      // NOLINTNEXTLINE
      for (const auto& [hoisted_scope, hlo_map] : hoisted_values_) {
        if (scopes_match_suffix(hoisted_scope, current_scopes)) {
          if (auto it = hlo_map.find(callback_id); it != hlo_map.end()) {
            for (const auto& [hoisted_op_id, literal] :
                 // NOLINTNEXTLINE
                 it->second) {
              if (group.expected_op_ids.contains(hoisted_op_id)) {
                LOG(INFO) << "Filling cached hoisted value op_id="
                          << hoisted_op_id;
                if (literal != nullptr) {
                  group.literals[hoisted_op_id] = literal;
                }
                group.received_op_ids.insert(hoisted_op_id);
              }
            }
          }
        }
      }
    }

    for (const auto* attr : matching_attrs) {
      if (attr->callback_id == callback_id) {
        int64_t op_id = attr->op_id;
        if (op_id >= 0 && op_id < group.literals.size()) {
          LOG(INFO) << "Received op_id: " << op_id
                    << " for current callback_id=" << callback_id;
          if (recovered_data != nullptr) {
            group.literals[op_id] = recovered_data;
          }
          group.received_op_ids.insert(op_id);
          group.expected_op_ids.insert(op_id);
        }
      }
    }

    bool is_matching_callback = absl::c_any_of(
        matching_attrs,
        [&](const auto* a) { return a->callback_id == callback_id; });
    if (!is_matching_callback &&
        group.received_op_ids.size() != group.expected_op_ids.size()) {
      LOG(INFO) << "Skipping incomplete group for different callback_id="
                << callback_id;
      pending_groups_.erase(key);
    }
  }

  // Check completion for all groups with this scope
  std::vector<GroupKey> groups_to_complete;
  // NOLINTNEXTLINE
  for (auto& [key, group] : pending_groups_) {
    if (key.scope_instructions == current_scopes) {
      if (group.received_op_ids.size() == group.expected_op_ids.size() &&
          !group.expected_op_ids.empty()) {
        groups_to_complete.push_back(key);
      }
    }
  }

  absl::c_sort(groups_to_complete, [](const GroupKey& a, const GroupKey& b) {
    return a.callback_id < b.callback_id;
  });

  for (const auto& key : groups_to_complete) {
    auto& group = pending_groups_[key];
    LOG(INFO) << "Invoking group ready callback for callback_id="
              << key.callback_id;
    on_group_ready_(key.callback_id, 0, partition_id, group.literals);
    pending_groups_.erase(key);
  }
}

}  // namespace xla
