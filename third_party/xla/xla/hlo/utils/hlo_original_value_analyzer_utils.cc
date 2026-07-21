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

#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"

namespace xla {

namespace {

// Parses a string representation of a scope instruction (e.g., "loop#3") into
// a ScopeInstruction object.
absl::StatusOr<ScopeInstruction> ParseScopeInstruction(
    absl::string_view scope_str) {
  std::pair<absl::string_view, absl::string_view> parts =
      absl::StrSplit(scope_str, absl::MaxSplits('#', 1));
  int64_t iteration_index = 0;
  if (!parts.second.empty()) {
    if (parts.second == "*") {
      iteration_index = -1;
    } else if (parts.second == "$") {
      iteration_index = -2;
    } else if (!absl::SimpleAtoi(parts.second, &iteration_index)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid iteration index: ", parts.second));
    }
  }
  return ScopeInstruction::Create(parts.first, iteration_index);
}

}  // namespace

ScopeInstruction ScopeInstruction::FromString(absl::string_view scope_str) {
  auto parsed = ParseScopeInstruction(scope_str);
  CHECK(parsed.ok()) << "Failed to parse scope instruction: " << scope_str;
  return *parsed;
}

std::string ScopeInstruction::ToString() const {
  if (iteration_index == 0) {
    return instruction_name;
  }
  if (iteration_index == -1) {
    return absl::StrCat(instruction_name, "#*");
  }
  if (iteration_index == -2) {
    return absl::StrCat(instruction_name, "#$");
  }
  return absl::StrCat(instruction_name, "#", iteration_index);
}

std::string TensorKey::ToString() const {
  return absl::StrCat(instruction_name, " (", shape_index.ToString(), ")");
}

AbsoluteScopedTensorKey AbsoluteScopedTensorKey::Create(
    absl::Span<const ScopeInstruction> optimized_root_scopes,
    const RelativeScopedTensorKey& relative_key,
    const absl::flat_hash_map<std::string, std::vector<ScopeInstruction>>&
        call_map) {
  std::vector<ScopeInstruction> combined_scopes;

  // Translate optimized_root_scopes
  for (const auto& scope : optimized_root_scopes) {
    auto it = call_map.find(scope.instruction_name);
    if (it != call_map.end()) {
      const std::vector<ScopeInstruction>& mapped_original_scopes = it->second;
      combined_scopes.insert(combined_scopes.end(),
                             mapped_original_scopes.begin(),
                             mapped_original_scopes.end());
      if (mapped_original_scopes.empty()) {
        continue;
      }
      if (scope.iteration_index != 0) {
        if (mapped_original_scopes.size() == 1) {
          if (combined_scopes.back().iteration_index == -2) {
            combined_scopes.back().iteration_index = scope.iteration_index;
          }
          continue;
        }
        bool found_while_loop = false;
        for (int64_t i = 0; i < mapped_original_scopes.size(); ++i) {
          auto& original_scope =
              combined_scopes[combined_scopes.size() - i - 1];
          if (original_scope.iteration_index == -2) {
            original_scope.iteration_index = scope.iteration_index;
            found_while_loop = true;
            break;
          }
        }
        // If not found, we don't do fallbacks as requested by user.
      }
    } else {
      combined_scopes.push_back(scope);
    }
  }

  // Translate relative_key.scope_instructions
  for (const auto& scope : relative_key.scope_instructions) {
    auto it = call_map.find(scope.instruction_name);
    if (it != call_map.end()) {
      combined_scopes.insert(combined_scopes.end(), it->second.begin(),
                             it->second.end());
    } else {
      combined_scopes.push_back(scope);
    }
  }

  return AbsoluteScopedTensorKey::Create(relative_key.tensor_key,
                                         std::move(combined_scopes));
}

std::optional<HloSharding> GetShardingFromUnshardRecoveryModule(
    const HloModule& recovery_module) {
  const HloComputation* comp = recovery_module.entry_computation();
  if (comp->num_parameters() == 1) {
    const HloInstruction* param = comp->parameter_instruction(0);
    if (param->has_sharding()) {
      return param->sharding();
    }
  }
  return std::nullopt;
}

}  // namespace xla
