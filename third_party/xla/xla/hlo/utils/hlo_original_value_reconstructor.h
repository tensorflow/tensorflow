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

#ifndef XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_RECONSTRUCTOR_H_
#define XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_RECONSTRUCTOR_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/utils/hlo_original_value_analysis.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"
#include "xla/hlo/utils/hlo_sharding_reconstruction_util.h"
#include "xla/literal.h"

namespace xla {

class DeviceAssignment;

class HloOriginalValueReconstructor {
 public:
  // Callback invoked when a full original tensor is reconstructed.
  using OriginalTensorReadyCallback = std::function<void(
      const AbsoluteScopedTensorKey& original_tensor_key,
      const OriginalArray& original_tensor,
      std::shared_ptr<Literal> recovered_data,
      const std::vector<HloModule::DebugAttributes>& debug_attributes,
      int64_t partition_id)>;

  HloOriginalValueReconstructor(
      std::shared_ptr<const HloOriginalValueAnalysis> analysis,
      OriginalTensorReadyCallback on_original_tensor_ready,
      std::optional<absl::AnyInvocable<bool(int64_t)>>
          logical_device_is_addressable = std::nullopt,
      const HloModule* optimized_module = nullptr);

  // Processes a new tensor shard. Since shards can arrive at different times,
  // this is a stateful operation.
  // The `instruction_name` and `shape_index` identify the optimized tensor.
  // `optimized_root_scopes` represents the invocation context.
  absl::Status ProcessShardTensor(
      const AbsoluteScopedTensorKey& optimized_tensor_position,
      ShardTensor shard_tensor);

 private:
  absl::Status ProcessCompletedShards(
      const AbsoluteScopedTensorKey& optimized_tensor_position,
      const AbsoluteScopedTensorKey& original_tensor_key,
      const HloOriginalValueAnalysis::OriginalTensorInfo& original_tensor_info,
      const std::vector<ShardTensor>& shards);

  absl::StatusOr<AbsoluteScopedTensorKey> ConstructOriginalTensorKey(
      absl::Span<const ScopeInstruction> optimized_root_scopes,
      const RelativeScopedTensorKey& relative_original_key) const;

  void PopulateExpectedShardIds(const HloModule* optimized_module);

  std::optional<absl::AnyInvocable<bool(int64_t)>>
      logical_device_is_addressable_;
  absl::flat_hash_set<TensorKey> non_reconstructible_keys_;

  std::shared_ptr<const HloOriginalValueAnalysis> analysis_;
  OriginalTensorReadyCallback on_original_tensor_ready_;

  absl::flat_hash_map<std::pair<TensorKey, RelativeScopedTensorKey>,
                      absl::flat_hash_set<int64_t>>
      expected_shard_ids_by_corresponding_tensor_key_pair_;

  absl::flat_hash_map<AbsoluteScopedTensorKey, std::vector<ShardTensor>>
      received_tensor_shards_by_optimized_key_;

  absl::flat_hash_set<AbsoluteScopedTensorKey> completed_tensor_keys_;

  absl::flat_hash_map<AbsoluteScopedTensorKey, std::vector<bool>>
      completed_partitioned_shards_;
};

}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_RECONSTRUCTOR_H_
