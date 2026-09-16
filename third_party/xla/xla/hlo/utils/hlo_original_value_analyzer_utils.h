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

#ifndef XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_ANALYZER_UTILS_H_
#define XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_ANALYZER_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/shape_util.h"

namespace xla {

class HloModule;

struct ScopeInstruction {
  // The name of the call or loop instruction
  std::string instruction_name;

  // Only set if this instruction is a loop instruction.
  //
  // -1 is reserved to
  // mean this matches any iteration index. This is useful when a loop
  // instruction is hoisted out of the loop body. In string format this is
  // encoded as `*`.
  //
  // The wildcard index (-1) is necessary to handle transformations like
  // loop-invariant code motion. When an expression within a loop is found to be
  // invariant across iterations, compilers can optimize by moving this
  // expression outside the loop. This means an instruction that originally
  // executed inside each iteration (producing a tensor per iteration) now
  // executes only once before the loop. To maintain the correspondence, the
  // single tensor produced by the hoisted instruction in the optimized version
  // needs to be matched with all the tensors produced by the original
  // instruction in each iteration of the unoptimized version. The -1 index
  // acts as a wildcard, indicating that this ScopeInstruction instance
  // represents the hoisted instruction and should match any iteration index of
  // its counterpart within the loop.
  //
  // For example, consider a loop that runs 5 times. If an instruction `A` is
  // inside the loop in the baseline, we will have 5 distinct tensors
  // (A#0, A#1, A#2, A#3, A#4). If, in the target, instruction `A` is hoisted
  // out of the loop, there will be only one tensor. We would represent the
  // hoisted tensor's scope with iteration_index = -1 (e.g., A#*). This allows
  // the single hoisted tensor (A#*) in the target to be correctly compared
  // against all 5 original tensors (A#0 through A#4) from the baseline.
  //
  // -2 is reserved to mean this iteration index should be replaced by the
  // actual iteration index at runtime. In string format this is encoded as `$`.
  // This is used by original call-like instruction tracking. For example,
  // consider the following program with nested loops written in pseudocode:
  //
  // ```
  // while.1 (i; i < 5; i++) {
  //   constant = 1
  //   while.2 (j; j < 3; j++) {
  //     constant += j
  //   }
  //   a = constant + i
  // }
  // ```
  // Here while.2 can be hoisted out of while.1 by invariant code motion. So
  // after optimiazation this becomes
  //
  // ```
  // constant' = 1
  // while.2' (j; j < 3; j++) {
  //   constant' += j
  // }
  // while.1' (i; i < 5; i++) {
  //   a' = constant' + i
  // }
  // ```
  //
  // Now, the original call-like instruction of `while.2'` should be
  // `whiile.1#*/while.2#$`. This indicates that at runtime when the scope
  // instruction is `while.2'#3`, the recovered original scope instruction
  // should be `while.1#*/while.2#3`, note `$` is replaced by the actual
  // iteration index 3.
  int64_t iteration_index = 0;

  // Returns true if this scope instruction matches any iteration index.
  bool MatchesAnyIteration() const { return iteration_index == -1; }

  static ScopeInstruction Create(absl::string_view instruction_name,
                                 int64_t iteration_index = 0) {
    return {std::string(instruction_name), iteration_index};
  }

  static ScopeInstruction FromString(absl::string_view scope_str);

  bool operator==(const ScopeInstruction& other) const {
    return instruction_name == other.instruction_name &&
           iteration_index == other.iteration_index;
  }

  bool operator<(const ScopeInstruction& other) const {
    if (instruction_name < other.instruction_name) {
      return true;
    }
    if (other.instruction_name < instruction_name) {
      return false;
    }
    return iteration_index < other.iteration_index;
  }

  template <typename H>
  friend H AbslHashValue(H h, const ScopeInstruction& scope_instr) {
    return H::combine(std::move(h), scope_instr.instruction_name,
                      scope_instr.iteration_index);
  }

  std::string ToString() const;
};

struct TensorKey {
  std::string instruction_name;
  ShapeIndex shape_index;

  static TensorKey Create(absl::string_view instruction_name,
                          const ShapeIndex& shape_index = {}) {
    return {std::string(instruction_name), shape_index};
  }

  bool operator==(const TensorKey& other) const {
    return instruction_name == other.instruction_name &&
           shape_index == other.shape_index;
  }

  bool operator<(const TensorKey& other) const {
    if (instruction_name < other.instruction_name) {
      return true;
    }
    if (other.instruction_name < instruction_name) {
      return false;
    }
    return shape_index < other.shape_index;
  }

  template <typename H>
  friend H AbslHashValue(H h, const TensorKey& key) {
    return H::combine(std::move(h), key.instruction_name, key.shape_index);
  }

  std::string ToString() const;
};

struct AbsoluteTag {};
struct RelativeTag {};

template <typename Tag>
struct ScopedTensorKey {
  TensorKey tensor_key;
  std::vector<ScopeInstruction> scope_instructions;

  bool operator==(const ScopedTensorKey& other) const {
    return tensor_key == other.tensor_key &&
           scope_instructions == other.scope_instructions;
  }

  bool operator<(const ScopedTensorKey& other) const {
    if (tensor_key < other.tensor_key) {
      return true;
    }
    if (other.tensor_key < tensor_key) {
      return false;
    }
    return scope_instructions < other.scope_instructions;
  }

  template <typename H>
  friend H AbslHashValue(H h, const ScopedTensorKey& key) {
    return H::combine(std::move(h), key.tensor_key, key.scope_instructions);
  }

  std::string ToString() const {
    std::vector<std::string> scope_strs;
    scope_strs.reserve(scope_instructions.size());
    for (const auto& scope : scope_instructions) {
      scope_strs.push_back(scope.ToString());
    }
    return absl::StrCat(absl::StrJoin(scope_strs, "/"),
                        scope_instructions.empty() ? "" : "/",
                        tensor_key.ToString());
  }
};

struct RelativeScopedTensorKey : public ScopedTensorKey<RelativeTag> {
  static RelativeScopedTensorKey Create(
      TensorKey key, std::vector<ScopeInstruction> instructions) {
    RelativeScopedTensorKey k;
    k.tensor_key = std::move(key);
    k.scope_instructions = std::move(instructions);
    return k;
  }

  static RelativeScopedTensorKey FromString(
      absl::string_view str, const ShapeIndex& shape_index = {}) {
    std::vector<absl::string_view> parts = absl::StrSplit(str, '/');
    std::vector<ScopeInstruction> scope_instructions;
    scope_instructions.reserve(parts.size() - 1);
    for (int i = 0; i < parts.size() - 1; ++i) {
      scope_instructions.push_back(ScopeInstruction::FromString(parts[i]));
    }
    return Create(TensorKey::Create(parts.back(), shape_index),
                  std::move(scope_instructions));
  }
};

struct AbsoluteScopedTensorKey : public ScopedTensorKey<AbsoluteTag> {
  static AbsoluteScopedTensorKey Create(
      TensorKey key, std::vector<ScopeInstruction> instructions) {
    AbsoluteScopedTensorKey k;
    k.tensor_key = std::move(key);
    k.scope_instructions = std::move(instructions);
    return k;
  }

  static AbsoluteScopedTensorKey FromString(
      absl::string_view str, const ShapeIndex& shape_index = {}) {
    std::vector<absl::string_view> parts = absl::StrSplit(str, '/');
    std::vector<ScopeInstruction> scope_instructions;
    scope_instructions.reserve(parts.size() - 1);
    for (int i = 0; i < parts.size() - 1; ++i) {
      scope_instructions.push_back(ScopeInstruction::FromString(parts[i]));
    }
    return Create(TensorKey::Create(parts.back(), shape_index),
                  std::move(scope_instructions));
  }

  static AbsoluteScopedTensorKey Create(
      absl::Span<const ScopeInstruction> optimized_root_scopes,
      const RelativeScopedTensorKey& relative_key,
      const absl::flat_hash_map<std::string, std::vector<ScopeInstruction>>&
          call_map);
};

std::optional<HloSharding> GetShardingFromUnshardRecoveryModule(
    const HloModule& recovery_module);

}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_ANALYZER_UTILS_H_
