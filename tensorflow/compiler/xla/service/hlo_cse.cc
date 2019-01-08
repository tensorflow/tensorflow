/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_cse.h"

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace xla {

namespace {

// Find and combine identical constants. Constants are identical if they have
// the same type and value.
StatusOr<bool> CombineConstants(HloComputation* computation,
                                bool is_layout_sensitive) {
  TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));
  // Map from ShortDebugString of the layoutless shape of the constant to the
  // set of constant instructions with that shape. Layoutless shape is used to
  // bin possible common constants together to reduce number of constant
  // comparisons. If we end up having too many constant comparisons, a more
  // precise binning might have to be used.
  std::multimap<string, HloInstruction*> constants;
  int64 combined = 0;
  auto inst_it = computation->instructions().begin();
  while (inst_it != computation->instructions().end()) {
    HloInstruction* instruction = *inst_it;

    // Advance list iterator before loop body because iterator may be
    // invalidated due to deletion.
    ++inst_it;

    if (instruction->opcode() == HloOpcode::kConstant) {
      Shape shape = instruction->shape();
      if (!is_layout_sensitive) {
        LayoutUtil::ClearLayout(&shape);
      }
      string shape_string = shape.ShortDebugString();

      // Compare against all constants with the same shape
      auto range = constants.equal_range(shape_string);
      HloInstruction* match = nullptr;
      for (auto it = range.first; it != range.second; ++it) {
        if (instruction->literal() == it->second->literal() &&
            domain_map->InSameDomain(it->second, instruction)) {
          match = it->second;
          break;
        }
      }
      if (match == nullptr) {
        constants.emplace(shape_string, instruction);
      } else {
        // Match found, replace this instruction with the one in the multimap.
        TF_CHECK_OK(instruction->ReplaceAllUsesWith(match));
        TF_CHECK_OK(computation->RemoveInstruction(instruction));
        ++combined;
      }
    }
  }
  VLOG(4) << "Combined " << combined << " constants in " << computation->name()
          << " computation";
  return combined > 0;
}

// An instruction is considered to be equivalent to another only if they
// share the exact same set of operands.
int64 CseHash(const HloInstruction* instruction) {
  int64 hash = std::hash<int64>()(static_cast<int64>(instruction->opcode()));
  hash = tensorflow::Hash64Combine(
      hash, instruction->opcode() == HloOpcode::kGetTupleElement
                ? instruction->tuple_index()
                : -1);
  for (auto operand : instruction->operands()) {
    hash = tensorflow::Hash64Combine(hash, operand->unique_id());
  }
  if (instruction->opcode() == HloOpcode::kConstant) {
    hash = tensorflow::Hash64Combine(hash, instruction->literal().Hash());
  }
  return hash;
}

}  // namespace

StatusOr<bool> HloCSE::Run(HloModule* module) {
  bool changed = false;
  const std::function<bool(const HloInstruction*, const HloInstruction*)>
      eq_instructions = std::equal_to<const HloInstruction*>();
  const std::function<bool(const HloComputation*, const HloComputation*)>
      eq_computations = [](const HloComputation* lhs,
                           const HloComputation* rhs) { return *lhs == *rhs; };

  auto cse_equal = [&](const HloInstruction* lhs, const HloInstruction* rhs) {
    return lhs->Identical(*rhs, eq_instructions, eq_computations,
                          is_layout_sensitive_);
  };

  for (auto* computation : module->computations()) {
    if (only_fusion_computations_ && !computation->IsFusionComputation()) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(bool combined,
                        CombineConstants(computation, is_layout_sensitive_));
    changed |= combined;

    // HLO instructions are grouped into equivalency classes by using the
    // cse_equal predicate defined above. This set holds a representative
    // instruction for each class.
    absl::flat_hash_set<HloInstruction*, decltype(&CseHash),
                        decltype(cse_equal)>
        representatives(/*N=*/computation->instruction_count() + 1, &CseHash,
                        cse_equal);
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      // If the instruction has zero operands (constants, parameters, etc.) skip
      // over it.
      if (instruction->operand_count() == 0) {
        continue;
      }
      // Skip instructions which have side effects.
      if (instruction->HasSideEffect()) {
        continue;
      }

      auto it = representatives.find(instruction);
      if (it != representatives.end()) {
        HloInstruction* equivalent_instruction = *it;
        TF_RETURN_IF_ERROR(
            instruction->ReplaceAllUsesWith(equivalent_instruction));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
        changed = true;
        continue;
      }
      representatives.insert(instruction);
    }
  }
  return changed;
}

}  // namespace xla
