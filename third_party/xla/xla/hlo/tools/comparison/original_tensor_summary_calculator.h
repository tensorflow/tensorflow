/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_CALCULATOR_H_
#define XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_CALCULATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/comparison_service.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"

namespace xla::numerics::comparison {

class OriginalTensorSummaryCalculator {
 public:
  struct ShardTensorSummary {
    // The logical shard ID of a device. This is after the DeviceAssignment is
    // applied.
    int64_t logical_shard_id;
    ::xla::comparison::FloatSummary summary;

    // Returns a human-readable string representation of the ShardTensorSummary.
    std::string ToDebugString() const {
      std::string result = "ShardTensorSummary{\n";
      absl::StrAppend(&result, "  logical_shard_id: ", logical_shard_id, "\n");
      absl::StrAppend(&result, "  summary:\n");

      absl::StrAppend(&result, "    split_spec:\n");
      for (const auto& spec : summary.split_spec) {
        absl::StrAppend(&result, "      {dim_index: ", spec.dim_index,
                        ", block_count: ", spec.block_count, "}\n");
      }

      absl::StrAppend(&result, "    block_summaries:\n");
      for (const auto& block : summary.block_summaries) {
        absl::StrAppend(&result, "      {block_indices: [",
                        absl::StrJoin(block.block_indices, ", "),
                        "], min: ", block.min, ", max: ", block.max,
                        ", mean: ", block.mean, ", stddev: ", block.stddev,
                        ", count: ", block.count, "}\n");
      }
      absl::StrAppend(&result, "}\n");
      return result;
    }
  };

  struct OriginalTensorInfo {
    // The key in the original module identifying the original tensor. This key
    // may not contain the full stack trace. Only the stack frames that are
    // inlined during optimization would be included here.
    RelativeScopedTensorKey original_scoped_tensor_key;
    // The recovering transformations that are applied to the optimized tensor
    // to recover the original tensor. If this is nullptr, it means there is no
    // recovering transformation needed. The summary can be used as is.
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        tensor_transformation;
  };

  struct CreationMetrics {
    // The total number of tensors in the optimized module. Note that this is
    // different from the number of instructions. For example, if an instruction
    // is a tuple of two tensors, it counts as two tensors.
    int64_t optimized_module_tensor_count = 0;
    // The number of tensors in the optimized module that have corresponding
    // original arrays in original value tracking.
    int64_t optimized_module_tensor_with_original_array_count = 0;
    // The number of tensors in the optimized module that have corresponding
    // original arrays in original value tracking, but the original arrays don't
    // lead to recovering any original tensor.
    int64_t optimized_module_tensor_with_dangling_original_array_count = 0;
    // The number of call-like instructions in the optimized module.
    int64_t optimized_module_call_like_instr_count = 0;
    // The number of call-like instructions in the optimized module that have
    // original values.
    int64_t optimized_module_call_like_instr_with_original_value_count = 0;
    // The total number of tensors that are recoverable in the original module
    // from the optimized module using original value tracking.
    int64_t original_module_recoverable_tensor_count = 0;
    // The tensor keys in the original module that are recoverable from the
    // optimized module using original value tracking.
    absl::flat_hash_set<TensorKey> recoverable_tensor_keys;
  };

  struct ProcessingMetrics {
    // The total number of optimized tensor shards received. That is, if a
    // tensor is sharded into 4 pieces, and 2 of them are received, this counter
    // would be incremented by 2.
    int64_t received_optimized_tensor_shard_count = 0;
    // The total number of original tensor shards that have been processed.
    // Here processed means the shard is known to correspond to some original
    // array or recovering computation. But it may still need other shards to be
    // combined to form the original tensor.
    int64_t processed_original_tensor_shard_count = 0;
    // The total number of optimized tensors that have been processed fully,
    // combining all the pieces from all the shards.
    int64_t completed_optimized_tensor_count = 0;
    // The total number of optimized tensors that have been processed partially,
    // meaning that not all the pieces from all the shards are received. This
    // is not expected and indicates an error.
    int64_t incomplete_optimized_tensor_count = 0;
    // The total number of original tensors that have been recovered fully,
    // meaning that all the pieces from all the shards are received.
    int64_t completed_original_tensor_count = 0;
    // The total number of original tensors that are not fully recovered,
    // meaning that not all the pieces from all the shards are received. This
    // is not expected and indicates an error.
    int64_t incomplete_original_tensor_count = 0;
  };

  static absl::StatusOr<std::pair<
      std::unique_ptr<OriginalTensorSummaryCalculator>, CreationMetrics>>
  Create(HloModule* optimized_module, HloModule* original_module,
         OriginalTensorSummaryCallback&& on_original_tensor_summary_ready,
         std::optional<::absl::Duration> log_interval = std::nullopt);

  OriginalTensorSummaryCalculator(
      std::shared_ptr<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>
          optimized_tensor_dimensions,
      std::shared_ptr<
          const absl::flat_hash_map<std::string, std::vector<ScopeInstruction>>>
          call_map,
      std::shared_ptr<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>
          original_tensor_by_optimized_tensor_key,
      OriginalTensorSummaryCallback&& on_original_tensor_summary_ready,
      std::optional<::absl::Duration> log_interval = std::nullopt);

  // Not copyable.
  OriginalTensorSummaryCalculator(const OriginalTensorSummaryCalculator&) =
      delete;
  OriginalTensorSummaryCalculator& operator=(
      const OriginalTensorSummaryCalculator&) = delete;

  // Movable.
  OriginalTensorSummaryCalculator(OriginalTensorSummaryCalculator&&) = default;
  OriginalTensorSummaryCalculator& operator=(
      OriginalTensorSummaryCalculator&&) = default;

  // Processes a shard summary for a given optimized tensor position. This
  // function should be called for each shard summary. When all the shard
  // summaries have been collected, this method would try to recover the
  // original tensor summaries based on the recovering transformations. For each
  // of the (partially) recovered original tensor summary, the callback
  // `on_original_tensor_summary_ready_` is then called.
  absl::Status ProcessShardSummary(
      const AbsoluteScopedTensorKey& optimized_tensor_position,
      const ShardTensorSummary& tensor_shard_summary);

  // Clones the calculator with a new callback. Note that the internal
  // states are not cloned. Only the HLO diff bimap is cloned.
  std::unique_ptr<OriginalTensorSummaryCalculator> CloneWithCallback(
      OriginalTensorSummaryCallback&& on_original_tensor_summary_ready) const;

  ProcessingMetrics GetProcessingMetrics() const;

  // Dumps the HLO-module-derived data in a clear and readable manner.
  std::string DumpHloDerivedData() const;

  const absl::flat_hash_map<TensorKey,
                            absl::InlinedVector<OriginalTensorInfo, 1>>&
  original_tensor_by_optimized_tensor_key() const {
    return *original_tensor_by_optimized_tensor_key_;
  }

 private:
  // Helper function to construct the original tensor key from the optimized
  // key and the relative original key, using the call map.
  absl::StatusOr<AbsoluteScopedTensorKey> ConstructOriginalTensorKey(
      absl::Span<const ScopeInstruction> optimized_root_scopes,
      const RelativeScopedTensorKey& relative_original_key) const;

  // Helper function to process the completed shards for a given original
  // tensor.
  absl::Status ProcessCompletedShards(
      const AbsoluteScopedTensorKey& optimized_tensor_position,
      const AbsoluteScopedTensorKey& original_tensor_key,
      const OriginalTensorInfo& original_tensor_info,
      const std::vector<ShardTensorSummary>& shard_summaries);

  void PopulateExpectedShardIds();

  // The dimensions of each optimized tensor in the optimized module.
  std::shared_ptr<const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>
      optimized_tensor_dimensions_;
  // The call map tracks corresponding call-like instructions in the original
  // module from the optimized module. The key is the instruction name of a
  // call-like instruction in the optimized module. The value is a vector of
  // ScopeInstruction, which is used to locate the corresponding call-like
  // instructions in the original module. The reason that there can be multiple
  // scoped instructions is due to call-inlining and loop unrolling, in which
  // case the call-like instruction in the optimized module may be an
  // instantiation of a called computation or a loop, which requires contextual
  // scope instructions to locate.
  std::shared_ptr<
      const absl::flat_hash_map<std::string, std::vector<ScopeInstruction>>>
      call_map_;
  // Key is the optimized tensor key. Value is a vector of original tensors
  // that can be recovered by applying the recovering transformations on the
  // tensor identified by the optimized tensor key. Note that both optimized
  // tensor key and original tensor key are relative. The parent scopes need to
  // be converted using the call map.
  std::shared_ptr<const absl::flat_hash_map<
      TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>
      original_tensor_by_optimized_tensor_key_;
  OriginalTensorSummaryCallback on_original_tensor_summary_ready_;

  // Set of instruction names that we have already warned about not being in
  // call_map_.
  mutable absl::flat_hash_set<std::string> missing_call_map_instr_names_;

  // Key is a pair of tensor keys. The first tensor key is the optimized tensor
  // key. The second tensor key is the relative original tensor key. Value is
  // the logical IDs of the shards that are needed to recover the original
  // tensor.
  //
  // This map is derived from `original_tensor_by_optimized_tensor_key_`.
  absl::flat_hash_map<std::pair<TensorKey, RelativeScopedTensorKey>,
                      absl::flat_hash_set<int64_t>>
      expected_shard_ids_by_corresponding_tensor_key_pair_;

  // Key is the optimized tensor key. Value is a vector of shard summaries. The
  // length of the vector starts from 0 and should be the expected shard count
  // after all the shards are collected.
  absl::flat_hash_map<AbsoluteScopedTensorKey, std::vector<ShardTensorSummary>>
      received_tensor_shards_by_optimized_key_;

  // Set of completed absolute original tensor keys.
  absl::flat_hash_set<AbsoluteScopedTensorKey> completed_tensor_keys_;

  ProcessingMetrics processing_metrics_;
  std::optional<::absl::Duration> log_interval_;
};
}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_CALCULATOR_H_
