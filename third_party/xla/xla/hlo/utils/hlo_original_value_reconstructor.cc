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

#include "xla/hlo/utils/hlo_original_value_reconstructor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_original_value_analysis.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"
#include "xla/hlo/utils/hlo_sharding_reconstruction_util.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

HloOriginalValueReconstructor::HloOriginalValueReconstructor(
    std::shared_ptr<const HloOriginalValueAnalysis> analysis,
    OriginalTensorReadyCallback on_original_tensor_ready,
    std::optional<absl::AnyInvocable<bool(int64_t)>>
        logical_device_is_addressable,
    const HloModule* optimized_module)
    : logical_device_is_addressable_(std::move(logical_device_is_addressable)),
      analysis_(std::move(analysis)),
      on_original_tensor_ready_(std::move(on_original_tensor_ready)) {
  PopulateExpectedShardIds(optimized_module);
}

void HloOriginalValueReconstructor::PopulateExpectedShardIds(
    const HloModule* optimized_module) {
  std::vector<TensorKey> sorted_keys;
  sorted_keys.reserve(
      analysis_->original_tensor_by_optimized_tensor_key().size());
  for (const auto& [opt_key, _] :
       analysis_->original_tensor_by_optimized_tensor_key()) {
    sorted_keys.push_back(opt_key);
  }
  absl::c_sort(sorted_keys);
  for (const auto& opt_key : sorted_keys) {
    const auto& original_infos =
        analysis_->original_tensor_by_optimized_tensor_key().at(opt_key);
    auto sharding_it = analysis_->optimized_tensor_sharding().find(opt_key);
    if (sharding_it == analysis_->optimized_tensor_sharding().end()) {
      continue;
    }
    const HloSharding& sharding = sharding_it->second;

    absl::flat_hash_set<int64_t> logical_devices =
        GetLogicalDeviceIds(sharding, optimized_module);

    if (logical_device_is_addressable_.has_value() &&
        absl::c_any_of(logical_devices, [&](int64_t logical_id) {
          return !(*logical_device_is_addressable_)(logical_id);
        })) {
      non_reconstructible_keys_.insert(opt_key);
      LOG_FIRST_N(WARNING, 100)
          << "Original value reconstruction for tensor '"
          << opt_key.instruction_name
          << "' is aborted because it requires shards from unaddressable "
             "devices.";
      continue;
    }

    for (const auto& info : original_infos) {
      expected_shard_ids_by_corresponding_tensor_key_pair_[{
          opt_key, info.original_scoped_tensor_key}] = logical_devices;
    }
  }
}

absl::StatusOr<AbsoluteScopedTensorKey>
HloOriginalValueReconstructor::ConstructOriginalTensorKey(
    absl::Span<const ScopeInstruction> optimized_root_scopes,
    const RelativeScopedTensorKey& relative_original_key) const {
  return AbsoluteScopedTensorKey::Create(
      optimized_root_scopes, relative_original_key, analysis_->call_map());
}

absl::Status HloOriginalValueReconstructor::ProcessShardTensor(
    const AbsoluteScopedTensorKey& optimized_tensor_position,
    ShardTensor shard_tensor) {
  if (non_reconstructible_keys_.contains(
          optimized_tensor_position.tensor_key)) {
    return absl::OkStatus();
  }
  auto it = analysis_->original_tensor_by_optimized_tensor_key().find(
      optimized_tensor_position.tensor_key);
  if (it == analysis_->original_tensor_by_optimized_tensor_key().end() ||
      it->second.empty()) {
    return absl::OkStatus();
  }

  auto& shards =
      received_tensor_shards_by_optimized_key_[optimized_tensor_position];
  if (absl::c_any_of(shards, [&](const ShardTensor& shard) {
        return shard.logical_shard_id == shard_tensor.logical_shard_id;
      })) {
    return absl::OkStatus();
  }
  shards.push_back(std::move(shard_tensor));

  const absl::InlinedVector<HloOriginalValueAnalysis::OriginalTensorInfo, 1>&
      original_infos = it->second;
  auto sharding_it = analysis_->optimized_tensor_sharding().find(it->first);

  bool all_completed = true;
  for (const auto& original_info : original_infos) {
    ASSIGN_OR_RETURN(
        AbsoluteScopedTensorKey original_tensor_key,
        ConstructOriginalTensorKey(optimized_tensor_position.scope_instructions,
                                   original_info.original_scoped_tensor_key));

    auto it_attr = analysis_->requested_original_arrays().find(
        original_info.original_array);
    std::vector<HloModule::DebugAttributes> debug_attributes;
    if (it_attr != analysis_->requested_original_arrays().end()) {
      debug_attributes = it_attr->second;
    }

    std::vector<HloModule::DebugAttributes> attrs_with_partitioned_false;
    std::vector<HloModule::DebugAttributes> attrs_with_partitioned_true;
    for (const auto& attr : debug_attributes) {
      if (attr.partitioned) {
        attrs_with_partitioned_true.push_back(attr);
      } else {
        attrs_with_partitioned_false.push_back(attr);
      }
    }

    bool unshard_exists = false;
    int unshard_module_index = -1;
    for (int i = 0; i < original_info.recovery_modules.size(); ++i) {
      if (GetShardingFromUnshardRecoveryModule(
              *original_info.recovery_modules[i])
              .has_value()) {
        unshard_exists = true;
        unshard_module_index = i;
        break;
      }
    }

    bool unshard_is_last = true;
    if (unshard_exists) {
      unshard_is_last =
          (unshard_module_index == original_info.recovery_modules.size() - 1);
    }

    auto mark_partitioned_shard_reported = [&](int64_t logical_shard_id) {
      auto& completed = completed_partitioned_shards_[original_tensor_key];
      if (completed.empty()) {
        if (unshard_exists) {
          auto sharding = GetShardingFromUnshardRecoveryModule(
              *original_info.recovery_modules[unshard_module_index]);
          completed.resize(sharding->num_devices(), false);
        } else if (!original_info.recovery_modules.empty()) {
          completed.resize(
              original_info.recovery_modules[0]->config().num_partitions(),
              false);
        } else {
          completed.resize(1, false);
        }
      }

      if (logical_shard_id >= 0 && logical_shard_id < completed.size()) {
        if (completed[logical_shard_id]) {
          return true;
        }
        completed[logical_shard_id] = true;
      }
      return false;
    };

    // Path 2: partitioned = true (Process immediately!)
    if (!attrs_with_partitioned_true.empty()) {
      if (unshard_exists && !unshard_is_last) {
        // Fail recovery for partitioned=true by reporting nullptr.
        int64_t logical_shard_id = shards.back().logical_shard_id;
        if (!mark_partitioned_shard_reported(logical_shard_id)) {
          for (const auto& attr : attrs_with_partitioned_true) {
            on_original_tensor_ready_(original_tensor_key,
                                      original_info.original_array, nullptr,
                                      {attr}, logical_shard_id);
          }
        }
      } else {
        Literal current_shard_data = shards.back().data->Clone();
        int64_t logical_shard_id = shards.back().logical_shard_id;

        for (int i = 0; i < original_info.recovery_modules.size(); ++i) {
          const auto& module = original_info.recovery_modules[i];
          if (unshard_exists && i == unshard_module_index) {
            // Skip the final unshard!
            continue;
          }
          HloEvaluator evaluator;
          ASSIGN_OR_RETURN(current_shard_data,
                           evaluator.Evaluate(*module, {&current_shard_data}));
        }

        if (!mark_partitioned_shard_reported(logical_shard_id)) {
          auto shared_shard_data =
              std::make_shared<Literal>(std::move(current_shard_data));
          for (const auto& attr : attrs_with_partitioned_true) {
            on_original_tensor_ready_(
                original_tensor_key, original_info.original_array,
                shared_shard_data, {attr}, logical_shard_id);
          }
        }
      }
    }

    // Path 1: partitioned = false (Wait for ready!)
    if (!attrs_with_partitioned_false.empty() || debug_attributes.empty()) {
      if (completed_tensor_keys_.contains(original_tensor_key)) {
        continue;
      }

      bool ready = false;
      if (sharding_it == analysis_->optimized_tensor_sharding().end()) {
        ready = true;
      } else {
        auto expected_it =
            expected_shard_ids_by_corresponding_tensor_key_pair_.find(
                {it->first, original_info.original_scoped_tensor_key});
        if (expected_it !=
            expected_shard_ids_by_corresponding_tensor_key_pair_.end()) {
          if (shards.size() >= expected_it->second.size()) {
            ready = true;
          }
        } else {
          ready = true;
        }
      }

      if (!ready) {
        all_completed = false;
        continue;
      }

      RETURN_IF_ERROR(ProcessCompletedShards(optimized_tensor_position,
                                             original_tensor_key, original_info,
                                             shards));
      completed_tensor_keys_.insert(original_tensor_key);
    }
  }

  if (all_completed) {
    received_tensor_shards_by_optimized_key_.erase(optimized_tensor_position);
  }

  return absl::OkStatus();
}

absl::Status HloOriginalValueReconstructor::ProcessCompletedShards(
    const AbsoluteScopedTensorKey& optimized_tensor_position,
    const AbsoluteScopedTensorKey& original_tensor_key,
    const HloOriginalValueAnalysis::OriginalTensorInfo& original_tensor_info,
    const std::vector<ShardTensor>& shards) {
  auto dims_it = analysis_->optimized_tensor_dimensions().find(
      optimized_tensor_position.tensor_key);
  CHECK(dims_it != analysis_->optimized_tensor_dimensions().end());

  auto sharding_it = analysis_->optimized_tensor_sharding().find(
      optimized_tensor_position.tensor_key);

  auto it = analysis_->requested_original_arrays().find(
      original_tensor_info.original_array);
  std::vector<HloModule::DebugAttributes> debug_attributes;
  if (it != analysis_->requested_original_arrays().end()) {
    debug_attributes = it->second;
  }

  std::vector<HloModule::DebugAttributes> attrs_with_partitioned_false;
  for (const auto& attr : debug_attributes) {
    if (!attr.partitioned) {
      attrs_with_partitioned_false.push_back(attr);
    }
  }

  if (sharding_it != analysis_->optimized_tensor_sharding().end()) {
    CHECK(!shards.empty());
    ASSIGN_OR_RETURN(ManualShardingInfo manual_info,
                     FactorManualSharding(shards, sharding_it->second));

    // Path 1: partitioned = false
    if (!attrs_with_partitioned_false.empty() || debug_attributes.empty()) {
      std::vector<int64_t> sorted_manual_ids;
      sorted_manual_ids.reserve(manual_info.manual_shard_groups.size());
      for (const auto& [manual_id, _] : manual_info.manual_shard_groups) {
        sorted_manual_ids.push_back(manual_id);
      }
      absl::c_sort(sorted_manual_ids);
      for (int64_t manual_id : sorted_manual_ids) {
        const auto& group_shards =
            manual_info.manual_shard_groups.at(manual_id);
        std::vector<ShardTensor> current_shards;
        current_shards.reserve(group_shards.size());
        for (const auto& s : group_shards) {
          current_shards.push_back(
              {.logical_shard_id = s.logical_shard_id,
               .data = std::make_shared<Literal>(s.data->Clone())});
        }

        auto unshard_current_shards =
            [&](const Shape& base_shape) -> absl::Status {
          Shape manual_shard_shape;
          if (!manual_info.has_manual_sharding) {
            manual_shard_shape = base_shape;
          } else {
            manual_shard_shape = current_shards[0].data->shape();
            for (int i = 0; i < manual_info.unshard_sharding.num_dimensions();
                 ++i) {
              manual_shard_shape.set_dimensions(
                  i, current_shards[0].data->shape().dimensions(i) *
                         manual_info.unshard_sharding.dimension(i));
            }
          }

          ASSIGN_OR_RETURN(
              Literal combined,
              UnshardLiteral(current_shards, manual_info.unshard_sharding,
                             manual_shard_shape));

          current_shards.clear();
          current_shards.push_back(
              {.logical_shard_id = -1,
               .data = std::make_shared<Literal>(std::move(combined))});
          return absl::OkStatus();
        };

        bool already_unsharded = false;
        for (const auto& module : original_tensor_info.recovery_modules) {
          if (GetShardingFromUnshardRecoveryModule(*module).has_value()) {
            if (already_unsharded) {
              continue;
            }

            RETURN_IF_ERROR(unshard_current_shards(
                module->entry_computation()->root_instruction()->shape()));
            already_unsharded = true;
          } else {
            for (auto& shard : current_shards) {
              HloEvaluator evaluator;
              ASSIGN_OR_RETURN(*shard.data,
                               evaluator.Evaluate(*module, {shard.data.get()}));
            }
          }
        }

        if (!already_unsharded) {
          Shape base_shape = ShapeUtil::MakeShape(
              current_shards[0].data->shape().element_type(), dims_it->second);
          RETURN_IF_ERROR(unshard_current_shards(base_shape));
          already_unsharded = true;
        }

        CHECK_EQ(current_shards.size(), 1);
        for (const auto& attr : attrs_with_partitioned_false) {
          on_original_tensor_ready_(original_tensor_key,
                                    original_tensor_info.original_array,
                                    current_shards[0].data, {attr}, manual_id);
        }
        if (debug_attributes.empty()) {
          on_original_tensor_ready_(original_tensor_key,
                                    original_tensor_info.original_array,
                                    current_shards[0].data, {}, manual_id);
        }
      }
    }
  } else {
    CHECK_EQ(shards.size(), 1);
    Literal recovered_literal = shards[0].data->Clone();
    for (const auto& module : original_tensor_info.recovery_modules) {
      if (GetShardingFromUnshardRecoveryModule(*module).has_value()) {
        continue;
      }
      HloEvaluator evaluator;
      ASSIGN_OR_RETURN(recovered_literal,
                       evaluator.Evaluate(*module, {&recovered_literal}));
    }

    if (!attrs_with_partitioned_false.empty() || debug_attributes.empty()) {
      auto shared_recovered =
          std::make_shared<Literal>(std::move(recovered_literal));
      for (const auto& attr : attrs_with_partitioned_false) {
        on_original_tensor_ready_(original_tensor_key,
                                  original_tensor_info.original_array,
                                  shared_recovered, {attr}, 0);
      }
      if (debug_attributes.empty()) {
        on_original_tensor_ready_(original_tensor_key,
                                  original_tensor_info.original_array,
                                  shared_recovered, {}, 0);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace xla
