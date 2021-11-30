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
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/hash.h"

namespace xla {

namespace {

// Find and combine identical constants. Constants are identical if they have
// the same type and value.
//
// While we're here, also combine identical iota instructions, since they need
// similar treatment.
StatusOr<bool> CombineConstants(HloComputation* computation,
                                bool is_layout_sensitive) {
  TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));
  // Map from the literal hash of a constant or the shape hash of an iota all
  // equivalent instructions. This avoids extreme quadratic behavior with many
  // scalar constants.
  std::unordered_multimap<int64_t, HloInstruction*> instrs;
  int64_t combined = 0;
  auto inst_it = computation->instructions().begin();
  while (inst_it != computation->instructions().end()) {
    HloInstruction* instruction = *inst_it;

    // Advance list iterator before loop body because iterator may be
    // invalidated due to deletion.
    ++inst_it;

    if (instruction->opcode() == HloOpcode::kConstant ||
        instruction->opcode() == HloOpcode::kIota) {
      Shape shape = instruction->shape();
      if (!is_layout_sensitive) {
        LayoutUtil::ClearLayout(&shape);
      }
      // Only the first element of a constant is comparable for layout sensitive
      // constants
      int64_t key =
          instruction->IsConstant()
              ? instruction->literal().Hash(
                    /*byte_limit=*/64)
              : (ShapeUtil::Hash(shape) *
                 (1 + Cast<HloIotaInstruction>(instruction)->iota_dimension()));

      // Compare against all iotas/constants with the same shape.
      HloInstruction* match = nullptr;
      auto range = instrs.equal_range(key);
      for (auto it = range.first; it != range.second; ++it) {
        if (instruction->opcode() == it->second->opcode() &&
            domain_map->InSameDomain(it->second, instruction) &&
            (is_layout_sensitive ? Shape::Equal()
                                 : Shape::Equal().IgnoreLayout())(
                instruction->shape(), it->second->shape()) &&
            ((instruction->opcode() == HloOpcode::kConstant &&
              instruction->literal() == it->second->literal()) ||
             (instruction->opcode() == HloOpcode::kIota &&
              Cast<HloIotaInstruction>(instruction)->iota_dimension() ==
                  Cast<HloIotaInstruction>(it->second)->iota_dimension()))) {
          match = it->second;
          break;
        }
      }
      if (match == nullptr) {
        instrs.emplace(key, instruction);
      } else {
        // Match found, replace this instruction with the one in the multimap.
        TF_CHECK_OK(instruction->ReplaceAllUsesWith(match));
        TF_CHECK_OK(computation->RemoveInstruction(instruction));
        ++combined;
      }
    }
  }
  VLOG(4) << "Combined " << combined << " constants and iotas in "
          << computation->name() << " computation";
  return combined > 0;
}

// An instruction is considered to be equivalent to another only if they
// share the exact same set of operands.
int64_t CseHash(const HloInstruction* instruction) {
  int64_t hash =
      std::hash<int64_t>()(static_cast<int64_t>(instruction->opcode()));
  auto c_hash = [](auto c) {
    return tensorflow::Hash64(reinterpret_cast<const char*>(c.data()),
                              c.size() * sizeof(c[0]));
  };
  auto proto_hash = [](auto proto) {
    return std::hash<int64_t>{}(proto.ByteSizeLong());
  };
  hash = tensorflow::Hash64Combine(
      hash, instruction->opcode() == HloOpcode::kGetTupleElement
                ? instruction->tuple_index()
                : c_hash(instruction->shape().dimensions()));
  for (auto operand : instruction->operands()) {
    hash = tensorflow::Hash64Combine(hash, operand->unique_id());
  }
  for (auto c : instruction->called_computations()) {
    hash = tensorflow::Hash64Combine(
        hash, std::hash<int64_t>()(
                  static_cast<int64_t>(c->root_instruction()->opcode())));
  }
  switch (instruction->opcode()) {
    case HloOpcode::kSlice:
      return tensorflow::Hash64Combine(
          tensorflow::Hash64Combine(hash, c_hash(instruction->slice_starts())),
          c_hash(instruction->slice_strides()));
    case HloOpcode::kPad:
      return tensorflow::Hash64Combine(
          hash, proto_hash(instruction->padding_config()));
    case HloOpcode::kDot:
      return tensorflow::Hash64Combine(
          hash, proto_hash(instruction->dot_dimension_numbers()));
    case HloOpcode::kConvolution:
      return tensorflow::Hash64Combine(
          tensorflow::Hash64Combine(
              hash, proto_hash(instruction->convolution_dimension_numbers())),
          proto_hash(instruction->window()));
    case HloOpcode::kReduceWindow:
      return tensorflow::Hash64Combine(hash, proto_hash(instruction->window()));
    case HloOpcode::kConcatenate:
    case HloOpcode::kBroadcast:
    case HloOpcode::kTranspose:
    case HloOpcode::kReduce:
      return tensorflow::Hash64Combine(hash, c_hash(instruction->dimensions()));
    default:
      return hash;
  }
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
      if (instruction->operand_count() == 0 &&
          instruction->opcode() != HloOpcode::kPartitionId &&
          instruction->opcode() != HloOpcode::kReplicaId) {
        continue;
      }
      // Skip instructions which have side effects.
      if (instruction->HasSideEffect()) {
        continue;
      }

      auto pair = representatives.insert(instruction);
      if (!pair.second) {
        HloInstruction* equivalent_instruction = *pair.first;
        TF_RETURN_IF_ERROR(
            instruction->ReplaceAllUsesWith(equivalent_instruction));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
        changed = true;
        continue;
      }
    }
  }
  return changed;
}

}  // namespace xla
