/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/analysis/alias_info.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {
bool IsDefaultInPlaceOperation(const HloInstruction* hlo) {
  HloOpcode opcode = hlo->opcode();
  return opcode == HloOpcode::kDynamicUpdateSlice ||
         opcode == HloOpcode::kScatter || opcode == HloOpcode::kAllReduceStart;
}
}  // namespace

// Returns in-place input/output pairs for the given fusion instruction,
// according to the aliasing rules for the corresponding fusion computation.
//
// `instruction` must be a fusion instruction.
std::vector<std::pair<HloOperandIndex, ShapeIndex>>
AliasInfo::GetFusionInstructionInPlaceInputOutputPairs(
    const HloFusionInstruction* fusion) const {
  std::vector<std::pair<HloOperandIndex, ShapeIndex>>
      in_place_input_output_pairs;

  // Each of these leaves represents one array output of the fusion that might
  // be aliased with one of the fusion computation's array inputs (both could be
  // nested arbitrarily deep inside tuples).
  ShapeUtil::ForEachLeafShape(fusion->shape(), [&](const Shape& sub_shape,
                                                   const ShapeIndex& index) {
    // Start from the root instruction of the fusion computation and follow
    // tuple indirection backwards to find the "output source", i.e. the
    // instruction that is the original source of the array output in
    // question. If there is no such indirection the "output source" will
    // just be the fusion root instruction itself.
    const HloInstruction* output_source_instruction =
        fusion->fused_expression_root();
    ShapeIndex output_source_index = index;
    std::tie(output_source_instruction, output_source_index) =
        FollowTupleIndirection(output_source_instruction, output_source_index);

    // The aliasing rules of the "output source" instruction determine the
    // aliasing rules for the entire fusion. If we can connect (following
    // tuple indirection) the input of an "in-place" pair to one of the
    // fusion's inputs, and the output of this "in-place" pair to the fusion
    // output in question, then this fusion input and output must alias.
    auto in_place_pairs = GetInPlaceInputOutputPairs(output_source_instruction);
    ShapeIndex in_place_input_index;
    const HloInstruction* in_place_input_source = nullptr;

    for (const auto& output_source_in_place_pair : in_place_pairs) {
      const HloOperandIndex& input = output_source_in_place_pair.first;
      const ShapeIndex& output_index = output_source_in_place_pair.second;
      if (output_index == output_source_index) {
        // It is not possible for the same output to alias multiple inputs.
        CHECK(in_place_input_source == nullptr);
        in_place_input_source =
            output_source_instruction->operand(input.operand_number);
        in_place_input_index = input.operand_index;
        // Follow tuple indirection backwards from the instruction input to
        // try to find a fusion parameter. If found, that parameter aliases
        // the current output. If not, the current output aliases no input.
        std::tie(in_place_input_source, in_place_input_index) =
            FollowTupleIndirection(in_place_input_source, in_place_input_index);
        if (in_place_input_source->opcode() == HloOpcode::kFusion) {
          // Nested fusions can have aliasing that allows us to peephole
          // through to their producer.
          auto nested_in_place_input_output_pairs =
              GetInPlaceInputOutputPairs(in_place_input_source);
          for (const auto& pair : nested_in_place_input_output_pairs) {
            if (pair.second == in_place_input_index) {
              // If the nested fusion has aliasing that matches the index of
              // this input for its output, then peephole to its input.
              in_place_input_source =
                  in_place_input_source->operand(pair.first.operand_number);
              in_place_input_index = pair.first.operand_index;
              std::tie(in_place_input_source, in_place_input_index) =
                  FollowTupleIndirection(in_place_input_source,
                                         in_place_input_index);
            }
          }
        }
      }
    }
    // Skip bitcast
    if (in_place_input_source != nullptr &&
        in_place_input_source->opcode() == HloOpcode::kBitcast) {
      in_place_input_source = in_place_input_source->operand(0);
    }
    if (in_place_input_source != nullptr &&
        in_place_input_source->opcode() == HloOpcode::kParameter) {
      in_place_input_output_pairs.emplace_back(
          HloOperandIndex{in_place_input_source->parameter_number(),
                          in_place_input_index},
          index);
    }
  });
  return in_place_input_output_pairs;
}

bool AliasInfo::MustAlias(const HloInstruction* operand,
                          const ShapeIndex& operand_index,
                          const HloInstruction* user,
                          const ShapeIndex& user_index) const {
  for (const auto& [hlo_operand_index, user_shape_index] :
       GetInPlaceInputOutputPairs(user)) {
    if (user->operand(hlo_operand_index.operand_number) == operand &&
        hlo_operand_index.operand_index == operand_index &&
        user_shape_index == user_index) {
      return true;
    }
  }
  return false;
}

std::vector<std::pair<HloOperandIndex, ShapeIndex>>
AliasInfo::GetInPlaceInputOutputPairs(const HloInstruction* user) const {
  if (std::optional<std::vector<std::pair<HloOperandIndex, ShapeIndex>>> hint =
          GetNonDefaultInPlaceInputOutputPairs(user)) {
    return *hint;
  }

  // TODO tixxx: nvshmem default one-shot allreduce algo requires
  // separate buffers for IO, remove this once nvshmem is upgraded to 3.3
  if (user->opcode() == HloOpcode::kAllReduceStart) {
    if (absl::StrContainsIgnoreCase(user->raw_backend_config_string(),
                                    "nvshmem")) {
      return {};
    }
  }
  if (IsDefaultInPlaceOperation(user)) {
    int64_t num_in_place_operands = user->operand_count();
    const HloScatterInstruction* scatter = DynCast<HloScatterInstruction>(user);
    if (scatter) {
      num_in_place_operands = scatter->scatter_operand_count();
    } else if (user->opcode() == HloOpcode::kDynamicUpdateSlice) {
      num_in_place_operands = 1;
    }
    // Default handling: one operand shares buffer with single output.
    if (num_in_place_operands == 1) {
      return {{HloOperandIndex{0, {}}, {}}};
    }
    // Default handling: operand i shares buffer with output i.
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    in_place_pairs.reserve(num_in_place_operands);
    for (int i = 0; i < num_in_place_operands; i++) {
      in_place_pairs.push_back({HloOperandIndex{i, {}}, {i}});
    }
    return in_place_pairs;
  }

  // Ops that require special handling.
  if (user->opcode() == HloOpcode::kCollectivePermute &&
      user->operands().size() == 4) {
    if (user->operand(1)->shape().IsTuple()) {
      std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs(
          {{HloOperandIndex{1, {}}, {}}});
      for (int i = 0; i < user->operand(1)->shape().tuple_shapes().size();
           i++) {
        in_place_pairs.push_back({HloOperandIndex{1, {i}}, {i}});
      }
      return in_place_pairs;
    }
    return {{HloOperandIndex{1, {}}, {}}};
  }
  if (user->opcode() == HloOpcode::kCollectivePermuteStart &&
      user->operands().size() == 4) {
    if (user->operand(1)->shape().IsTuple()) {
      std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs(
          {{HloOperandIndex{1, {}}, {1}}});
      for (int i = 0; i < user->operand(1)->shape().tuple_shapes().size();
           i++) {
        in_place_pairs.push_back({HloOperandIndex{1, {i}}, {1, i}});
      }
      return in_place_pairs;
    }
    return {{HloOperandIndex{1, {}}, {1}}};
  }
  if (user->opcode() == HloOpcode::kCustomCall) {
    // Custom Calls previously assumed that aliased operands were
    // forwarded, but now supports modification semantics.
    const auto& aliasing_pairs =
        Cast<HloCustomCallInstruction>(user)->output_to_operand_aliasing();
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    in_place_pairs.reserve(aliasing_pairs.size());
    for (const auto& pair : aliasing_pairs) {
      ShapeIndex output_shape_index = pair.first;
      int64_t operand_index = pair.second.first;
      ShapeIndex operand_shape_index = pair.second.second;
      in_place_pairs.push_back(
          {HloOperandIndex{operand_index, {operand_shape_index}},
           output_shape_index});
    }
    return in_place_pairs;
  }
  if (user->opcode() == HloOpcode::kFusion) {
    const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(user);
    const auto& aliasing_pairs = fusion->output_to_operand_aliasing();
    // WARNING: The users of fusion's output_to_operand_aliasing should be aware
    // that the annotated output-operand-aliasing pairs should not conflict with
    // those discovered by GetFusionInstructionInPlaceInputOutputPairs.
    // TODO (b/259460539): Make sure the annotated and discovered pairs do not
    // conflict (possibly through implementing a new pass)
    auto in_place_pairs = GetFusionInstructionInPlaceInputOutputPairs(fusion);
    if (!aliasing_pairs.empty()) {
      for (const auto& pair : aliasing_pairs) {
        ShapeIndex output_shape_index = pair.first;
        int64_t operand_index = pair.second.first;
        ShapeIndex operand_shape_index = pair.second.second;
        in_place_pairs.push_back(
            {HloOperandIndex{operand_index, {operand_shape_index}},
             output_shape_index});
      }
    }
    return in_place_pairs;
  }
  if (user->opcode() == HloOpcode::kAsyncStart) {
    // Custom Calls previously assumed that aliased operands were
    // forwarded, but now supports modification semantics.
    const auto& aliasing_pairs =
        Cast<HloAsyncStartInstruction>(user)->output_to_operand_aliasing();
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    in_place_pairs.reserve(aliasing_pairs.size());
    for (const auto& pair : aliasing_pairs) {
      ShapeIndex output_shape_index = pair.first;
      int64_t operand_index = pair.second.first;
      ShapeIndex operand_shape_index = pair.second.second;
      in_place_pairs.push_back(
          {HloOperandIndex{operand_index, {operand_shape_index}},
           output_shape_index});
    }
    return in_place_pairs;
  }
  if (user->opcode() == HloOpcode::kSetDimensionSize) {
    int64_t dimension = user->dimension();
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    if (user->shape().is_dynamic_dimension(dimension) ==
        user->shape().is_dynamic_dimension(dimension)) {
      in_place_pairs.push_back({HloOperandIndex{0, {}}, {}});
    }
    return in_place_pairs;
  }
  if (user->opcode() == HloOpcode::kRaggedAllToAll) {
    return {{HloOperandIndex{1, {}}, {}}};
  }
  return {};
}

std::pair<const HloInstruction*, ShapeIndex> FollowTupleIndirection(
    const HloInstruction* instruction, ShapeIndex operand_index) {
  while (instruction->opcode() == HloOpcode::kTuple && !operand_index.empty()) {
    instruction = instruction->operand(operand_index.front());
    operand_index.pop_front();
  }
  while (instruction->opcode() == HloOpcode::kGetTupleElement) {
    operand_index.push_front(instruction->tuple_index());
    instruction = instruction->operand(0);
  }

  return {instruction, operand_index};
}

}  // namespace xla
