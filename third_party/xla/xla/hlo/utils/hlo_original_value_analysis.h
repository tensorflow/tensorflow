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

#ifndef XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_ANALYSIS_H_
#define XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_ANALYSIS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"

namespace xla {

class HloOriginalValueAnalysis {
 public:
  struct OriginalTensorInfo {
    RelativeScopedTensorKey original_scoped_tensor_key;
    // The recovering transformations that are applied to the optimized tensor
    // to recover the original tensor.
    std::vector<std::shared_ptr<const HloModule>> recovery_modules;
    OriginalArray original_array;

    bool operator<(const OriginalTensorInfo& other) const {
      if (original_scoped_tensor_key < other.original_scoped_tensor_key) {
        return true;
      }
      if (other.original_scoped_tensor_key < original_scoped_tensor_key) {
        return false;
      }
      if (original_array.instruction_name !=
          other.original_array.instruction_name) {
        return original_array.instruction_name <
               other.original_array.instruction_name;
      }
      return original_array.shape_index < other.original_array.shape_index;
    }
  };

  // Creates the analysis by processing the original_value_recovery_table
  // of the optimized_module.
  static absl::StatusOr<std::unique_ptr<HloOriginalValueAnalysis>> Create(
      const HloModule* optimized_module,
      std::optional<absl::flat_hash_set<TensorKey>>
          logged_optimized_tensor_keys = std::nullopt);

  const absl::flat_hash_map<TensorKey, std::vector<int64_t>>&
  optimized_tensor_dimensions() const {
    return optimized_tensor_dimensions_;
  }
  const absl::flat_hash_map<std::string, std::vector<ScopeInstruction>>&
  call_map() const {
    return call_map_;
  }
  const absl::flat_hash_map<TensorKey,
                            absl::InlinedVector<OriginalTensorInfo, 1>>&
  original_tensor_by_optimized_tensor_key() const {
    return original_tensor_by_optimized_tensor_key_;
  }
  const absl::flat_hash_map<TensorKey, HloSharding>& optimized_tensor_sharding()
      const {
    return optimized_tensor_sharding_;
  }
  const absl::flat_hash_map<OriginalArray,
                            std::vector<HloModule::DebugAttributes>>&
  requested_original_arrays() const {
    return requested_original_arrays_;
  }
  const absl::flat_hash_map<std::string, std::vector<std::string>>&
  reverse_call_map() const {
    return call_instructions_by_computation_;
  }
  const absl::flat_hash_map<
      TensorKey, std::vector<std::pair<TensorKey, const OriginalTensorInfo*>>>&
  original_to_optimized_tensor_map() const {
    return original_to_optimized_tensor_map_;
  }
  const absl::flat_hash_map<std::string, std::string>&
  instruction_to_computation() const {
    return instruction_to_computation_;
  }

  // Returns true if the given original absolute tensor key can be recovered
  // via some recovery chain.
  bool IsOriginalAbsoluteTensorKeyRecoverable(
      const AbsoluteScopedTensorKey& original_key) const;

 private:
  HloOriginalValueAnalysis(
      absl::flat_hash_map<TensorKey, std::vector<int64_t>>
          optimized_tensor_dimensions,
      absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map,
      absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
          original_tensor_by_optimized_tensor_key,
      absl::flat_hash_map<TensorKey, HloSharding> optimized_tensor_sharding,
      absl::flat_hash_map<OriginalArray,
                          std::vector<HloModule::DebugAttributes>>
          requested_original_arrays,
      absl::flat_hash_map<std::string, std::vector<std::string>>
          reverse_call_map,
      absl::flat_hash_map<std::string, std::string> instruction_to_computation);

  // Maps an optimized tensor key (instruction name and shape index) to its
  // array dimensions.
  absl::flat_hash_map<TensorKey, std::vector<int64_t>>
      optimized_tensor_dimensions_;

  // Maps the name of an optimized call-like instruction to the sequence of
  // scope instructions that represent its context in the original program.
  // Used to expand scopes when constructing absolute tensor keys.
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map_;

  // Maps an optimized tensor to the original tensors it corresponds to, along
  // with the recovery modules needed to reconstruct the original value.
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      original_tensor_by_optimized_tensor_key_;

  // Maps an optimized tensor key to its sharding information.
  absl::flat_hash_map<TensorKey, HloSharding> optimized_tensor_sharding_;

  // Maps an original array to the debug attributes associated with it,
  // representing the arrays that were requested to be logged.
  absl::flat_hash_map<OriginalArray, std::vector<HloModule::DebugAttributes>>
      requested_original_arrays_;

  // Maps the name of an optimized computation to the names of instructions
  // that call it.
  absl::flat_hash_map<std::string, std::vector<std::string>>
      call_instructions_by_computation_;

  // Maps an original tensor key to the corresponding optimized candidates.
  absl::flat_hash_map<
      TensorKey, std::vector<std::pair<TensorKey, const OriginalTensorInfo*>>>
      original_to_optimized_tensor_map_;

  // Maps an optimized instruction name to the name of the computation
  // containing it.
  absl::flat_hash_map<std::string, std::string> instruction_to_computation_;
};

}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_ANALYSIS_H_
