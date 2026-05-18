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

#include "xla/hlo/utils/hlo_original_value_analysis.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

namespace {

constexpr int64_t kWildcardIndex = -1;
constexpr int64_t kPlaceholderIndex = -2;

struct State {
  std::string current_instruction;
  size_t matched_scopes;

  bool operator==(const State& other) const {
    return current_instruction == other.current_instruction &&
           matched_scopes == other.matched_scopes;
  }

  template <typename H>
  friend H AbslHashValue(H h, const State& s) {
    return H::combine(std::move(h), s.current_instruction, s.matched_scopes);
  }
};

bool IsCallLike(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kCall ||
         instr.opcode() == HloOpcode::kWhile ||
         instr.opcode() == HloOpcode::kConditional ||
         instr.opcode() == HloOpcode::kFusion;
}

void RecoverPlaceholderIndexWithHeuristics(
    const HloInstruction& optimized_instr,
    std::vector<ScopeInstruction>& scope_instructions) {
  if (optimized_instr.opcode() != HloOpcode::kWhile ||
      scope_instructions.empty()) {
    return;
  }

  // Only set an iteration_index to kPlaceholderIndex if there are no scope
  // instructions with kPlaceholderIndex iteration count already.
  if (absl::c_any_of(scope_instructions, [](const ScopeInstruction& scope) {
        return scope.iteration_index == kPlaceholderIndex;
      })) {
    return;
  }

  // Try to find the first scope instruction whose name starts with while and
  // set the iteration count to kPlaceholderIndex on this scope instruction.
  auto it =
      absl::c_find_if(scope_instructions, [](const ScopeInstruction& scope) {
        return absl::StartsWith(scope.instruction_name, "while");
      });
  if (it != scope_instructions.end()) {
    it->iteration_index = kPlaceholderIndex;
    return;
  }

  // If nothing is found, set it on the last instruction like now.
  if (scope_instructions.back().iteration_index == 0) {
    scope_instructions.back().iteration_index = kPlaceholderIndex;
  }
}

void PopulateCallMapForCallLikeInstruction(
    const HloInstruction& optimized_instr,
    absl::flat_hash_map<std::string, std::vector<ScopeInstruction>>& call_map) {
  if (optimized_instr.original_value()) {
    auto scope_instructions_str =
        optimized_instr.original_value()->GetOriginalCallLikeInstructions();
    if (scope_instructions_str.has_value()) {
      std::vector<std::string> scope_names =
          absl::StrSplit(*scope_instructions_str, '/');
      std::vector<ScopeInstruction> scope_instructions;
      scope_instructions.reserve(scope_names.size());
      for (const auto& scope_name : scope_names) {
        scope_instructions.push_back(ScopeInstruction::FromString(scope_name));
      }
      RecoverPlaceholderIndexWithHeuristics(optimized_instr,
                                            scope_instructions);
      call_map[optimized_instr.name()] = std::move(scope_instructions);
    }
  }
}

void BuildTransformationChain(
    const OriginalArray& placeholder,
    const absl::flat_hash_map<
        OriginalArray, std::vector<std::pair<OriginalArray, const HloModule*>>>&
        placeholder_to_recoverables,
    const TensorKey& optimized_key,
    absl::flat_hash_map<TensorKey, HloSharding>& optimized_tensor_sharding,
    absl::flat_hash_set<OriginalArray>& visited_placeholders,
    std::vector<HloOriginalValueAnalysis::OriginalTensorInfo>& results) {
  if (!visited_placeholders.insert(placeholder).second) {
    return;
  }
  auto it = placeholder_to_recoverables.find(placeholder);
  if (it == placeholder_to_recoverables.end()) {
    visited_placeholders.erase(placeholder);
    return;
  }
  for (const auto& [goal, rec_module] : it->second) {
    std::shared_ptr<const HloModule> rec_shared;
    if (rec_module) {
      if (auto sharding = GetShardingFromUnshardRecoveryModule(*rec_module)) {
        optimized_tensor_sharding.insert({optimized_key, *sharding});
      }
      rec_shared =
          std::shared_ptr<const HloModule>(rec_module, [](const HloModule*) {});
    }

    std::vector<HloOriginalValueAnalysis::OriginalTensorInfo>
        downstream_results;
    BuildTransformationChain(goal, placeholder_to_recoverables, optimized_key,
                             optimized_tensor_sharding, visited_placeholders,
                             downstream_results);

    if (!absl::StrContains(goal.instruction_name, "__ovp")) {
      std::vector<std::shared_ptr<const HloModule>> recovery_modules;
      if (rec_shared) {
        recovery_modules.push_back(rec_shared);
      }
      auto relative_key = RelativeScopedTensorKey::FromString(
          goal.instruction_name, goal.shape_index);
      HloOriginalValueAnalysis::OriginalTensorInfo info;
      info.original_scoped_tensor_key.tensor_key = relative_key.tensor_key;
      info.original_scoped_tensor_key.scope_instructions =
          relative_key.scope_instructions;
      info.recovery_modules = std::move(recovery_modules);
      info.original_array = goal;
      results.push_back(std::move(info));
    }

    for (auto& downstream_info : downstream_results) {
      if (rec_shared) {
        downstream_info.recovery_modules.insert(
            downstream_info.recovery_modules.begin(), rec_shared);
      }
      results.push_back(std::move(downstream_info));
    }
  }
  visited_placeholders.erase(placeholder);
}

}  // namespace

absl::StatusOr<std::unique_ptr<HloOriginalValueAnalysis>>
HloOriginalValueAnalysis::Create(const HloModule* optimized_module,
                                 std::optional<absl::flat_hash_set<TensorKey>>
                                     logged_optimized_tensor_keys) {
  if (logged_optimized_tensor_keys.has_value()) {
    LOG(INFO) << "logged_optimized_tensor_keys contains "
              << logged_optimized_tensor_keys->size() << " keys:";
    std::vector<TensorKey> sorted_keys(logged_optimized_tensor_keys->begin(),
                                       logged_optimized_tensor_keys->end());
    absl::c_sort(sorted_keys);
    for (const auto& key : sorted_keys) {
      LOG(INFO) << "  instruction_name: " << key.instruction_name
                << ", shape_index: " << key.shape_index;
    }
  } else {
    LOG(INFO) << "logged_optimized_tensor_keys is not provided.";
  }

  absl::flat_hash_map<TensorKey, std::vector<int64_t>>
      optimized_tensor_dimensions;
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      original_tensor_by_optimized_tensor_key;
  absl::flat_hash_map<TensorKey, HloSharding> optimized_tensor_sharding;
  absl::flat_hash_map<std::string, std::vector<std::string>> reverse_call_map;
  absl::flat_hash_map<std::string, std::string> instruction_to_computation;

  absl::flat_hash_map<OriginalArray,
                      std::vector<std::pair<OriginalArray, const HloModule*>>>
      placeholder_to_recoverables;
  for (const auto& [goal, recovery_pair] :
       optimized_module->original_value_recovery_table()) {
    placeholder_to_recoverables[recovery_pair.first].push_back(
        {goal, recovery_pair.second.get()});
  }

  for (const HloComputation* comp : optimized_module->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      instruction_to_computation[instr->name()] = comp->name();

      if (IsCallLike(*instr)) {
        PopulateCallMapForCallLikeInstruction(*instr, call_map);
        for (const HloComputation* called : instr->called_computations()) {
          reverse_call_map[called->name()].push_back(
              std::string(instr->name()));
        }
      }

      if (instr->original_value() == nullptr) {
        continue;
      }
      for (const auto& [shape_idx, original_array_opt] :
           instr->original_value()->original_arrays()) {
        if (!original_array_opt.has_value()) {
          continue;
        }
        const Shape& subshape =
            ShapeUtil::GetSubshape(instr->shape(), shape_idx);
        if (subshape.IsArray()) {
          TensorKey key = {std::string(instr->name()), shape_idx};
          optimized_tensor_dimensions[key] = std::vector<int64_t>(
              subshape.dimensions().begin(), subshape.dimensions().end());
        }

        const OriginalArray& oa = *original_array_opt;
        TensorKey optimized_key = {std::string(instr->name()), shape_idx};

        if (logged_optimized_tensor_keys.has_value() &&
            !logged_optimized_tensor_keys->contains(optimized_key)) {
          continue;
        }

        if (absl::StrContains(oa.instruction_name, "__ovp")) {
          std::vector<OriginalTensorInfo> results;
          absl::flat_hash_set<OriginalArray> visited_placeholders;
          BuildTransformationChain(oa, placeholder_to_recoverables,
                                   optimized_key, optimized_tensor_sharding,
                                   visited_placeholders, results);
          for (auto& info : results) {
            original_tensor_by_optimized_tensor_key[optimized_key].push_back(
                std::move(info));
          }
        } else {
          auto relative_key = RelativeScopedTensorKey::FromString(
              oa.instruction_name, oa.shape_index);
          HloOriginalValueAnalysis::OriginalTensorInfo info;
          info.original_scoped_tensor_key.tensor_key = relative_key.tensor_key;
          info.original_scoped_tensor_key.scope_instructions =
              relative_key.scope_instructions;
          info.recovery_modules = {};
          info.original_array.instruction_name =
              relative_key.tensor_key.instruction_name;
          info.original_array.shape_index = relative_key.tensor_key.shape_index;
          original_tensor_by_optimized_tensor_key[optimized_key].push_back(
              std::move(info));
        }
      }
    }
  }

  absl::flat_hash_map<OriginalArray, std::vector<HloModule::DebugAttributes>>
      requested_original_arrays;
  for (const auto& [original_array, debug_attributes_list] :
       optimized_module->debug_attributes()) {
    std::vector<HloModule::DebugAttributes> filtered;
    filtered.reserve(debug_attributes_list.size());
    for (const auto& debug_attributes : debug_attributes_list) {
      if (debug_attributes.callback_id != 0) {
        filtered.push_back(debug_attributes);
      }
    }
    if (!filtered.empty()) {
      requested_original_arrays.emplace(original_array, std::move(filtered));
    }
  }

  return std::unique_ptr<HloOriginalValueAnalysis>(new HloOriginalValueAnalysis(
      std::move(optimized_tensor_dimensions), std::move(call_map),
      std::move(original_tensor_by_optimized_tensor_key),
      std::move(optimized_tensor_sharding),
      std::move(requested_original_arrays), std::move(reverse_call_map),
      std::move(instruction_to_computation)));
}

HloOriginalValueAnalysis::HloOriginalValueAnalysis(
    absl::flat_hash_map<TensorKey, std::vector<int64_t>>
        optimized_tensor_dimensions,
    absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map,
    absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
        original_tensor_by_optimized_tensor_key,
    absl::flat_hash_map<TensorKey, HloSharding> optimized_tensor_sharding,
    absl::flat_hash_map<OriginalArray, std::vector<HloModule::DebugAttributes>>
        requested_original_arrays,
    absl::flat_hash_map<std::string, std::vector<std::string>> reverse_call_map,
    absl::flat_hash_map<std::string, std::string> instruction_to_computation)
    : optimized_tensor_dimensions_(std::move(optimized_tensor_dimensions)),
      call_map_(std::move(call_map)),
      original_tensor_by_optimized_tensor_key_(
          std::move(original_tensor_by_optimized_tensor_key)),
      optimized_tensor_sharding_(std::move(optimized_tensor_sharding)),
      requested_original_arrays_(std::move(requested_original_arrays)),
      call_instructions_by_computation_(std::move(reverse_call_map)),
      instruction_to_computation_(std::move(instruction_to_computation)) {
  std::vector<TensorKey> sorted_keys;
  sorted_keys.reserve(original_tensor_by_optimized_tensor_key_.size());
  for (const auto& [opt_key, _] : original_tensor_by_optimized_tensor_key_) {
    sorted_keys.push_back(opt_key);
  }
  absl::c_sort(sorted_keys);
  for (const auto& opt_key : sorted_keys) {
    const auto& infos = original_tensor_by_optimized_tensor_key_.at(opt_key);
    for (const auto& info : infos) {
      original_to_optimized_tensor_map_[info.original_scoped_tensor_key
                                            .tensor_key]
          .push_back({opt_key, &info});
    }
  }
}

bool HloOriginalValueAnalysis::IsOriginalAbsoluteTensorKeyRecoverable(
    const AbsoluteScopedTensorKey& original_key) const {
  auto it = original_to_optimized_tensor_map_.find(original_key.tensor_key);
  if (it == original_to_optimized_tensor_map_.end()) {
    return false;
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

  auto matches_suffix = [&](absl::Span<const ScopeInstruction> pattern,
                            size_t end_idx) {
    if (pattern.size() > end_idx) {
      return false;
    }
    for (size_t i = 0; i < pattern.size(); ++i) {
      if (!scopes_match(pattern[pattern.size() - 1 - i],
                        original_key.scope_instructions[end_idx - 1 - i])) {
        return false;
      }
    }
    return true;
  };

  std::vector<State> worklist;
  size_t total_query_scopes = original_key.scope_instructions.size();

  for (const auto& [opt_key, info_ptr] : it->second) {
    const auto& cand_scopes =
        info_ptr->original_scoped_tensor_key.scope_instructions;
    if (matches_suffix(cand_scopes, total_query_scopes)) {
      worklist.push_back({opt_key.instruction_name, cand_scopes.size()});
    }
  }

  if (worklist.empty()) {
    return false;
  }

  absl::flat_hash_set<State> visited;

  while (!worklist.empty()) {
    State current = worklist.back();
    worklist.pop_back();

    if (current.matched_scopes == total_query_scopes) {
      return true;
    }

    if (!visited.insert(current).second) {
      continue;
    }

    auto comp_it =
        instruction_to_computation_.find(current.current_instruction);
    if (comp_it == instruction_to_computation_.end()) {
      continue;
    }
    const std::string& comp_name = comp_it->second;

    auto caller_it = call_instructions_by_computation_.find(comp_name);
    if (caller_it == call_instructions_by_computation_.end()) {
      continue;
    }

    for (const std::string& caller_instr_name : caller_it->second) {
      auto call_map_it = call_map_.find(caller_instr_name);
      if (call_map_it == call_map_.end()) {
        continue;
      }
      const std::vector<ScopeInstruction>& call_scopes = call_map_it->second;

      size_t rem_scopes = total_query_scopes - current.matched_scopes;
      if (matches_suffix(call_scopes, rem_scopes)) {
        worklist.push_back(
            {caller_instr_name, current.matched_scopes + call_scopes.size()});
      }
    }
  }

  return false;
}

}  // namespace xla
