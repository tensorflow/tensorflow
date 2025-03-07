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

#include "xla/service/buffer_to_attributes.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace {

constexpr absl::string_view kOutputToOperandAliasingAttr =
    "output_to_operand_aliasing";

const char* kNonCopyableAttr = "_xla_non_copyable_attribute";

// Represents the use of a buffer in an instruction.
struct BufferInfo {
  int64_t buffer_id;  // The ID of the buffer.
  // The shape index in the output shape with the buffer ID.
  ShapeIndex output_shape_index;
  int64_t operand_id;  // The operand that use the buffer.
  // The shape index in the operand shape with the buffer ID.
  ShapeIndex operand_shape_index;
};

// Converts output and operand with the same buffer id to
// _xla_non_copyable_attribute and output_to_operand_aliasing attribute.
//
// We add both attributes to kCustomCall. For kTuple and kWhile, we add
// _xla_non_copyable_attribute to an instruction that uses buffers to help
// discover a chain of instructions that use buffers.
//
// We don't need to add attributes to kParameter and kGetTupleElement as the
// buffer information for these instructions are represented in their
// corresponding call-sites and their tuple operands.
absl::Status ConvertToAttributes(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kCustomCall &&
          instr->opcode() != HloOpcode::kTuple &&
          instr->opcode() != HloOpcode::kWhile) {
        continue;
      }

      if (instr->IsCustomCall("allocateBuffer")) {
        // Custom-call to allocateBuffer is the only kind of instruction with
        // a non-copyable attribute without a output_to_operand_aliasing
        // attribute.
        instr->add_frontend_attribute(kNonCopyableAttr, "{}");
        continue;
      }
      // We will remove pin and unpin custom-calls.
      if (instr->IsCustomCall("pin") || instr->IsCustomCall("unpin")) {
        continue;
      }

      // Use a vector to collect output-to-operand pairs to preserve insertion
      // order.
      absl::InlinedVector<BufferInfo, 4> buffer_infos;
      auto add_dst_buffer = [&](int64_t buffer_id,
                                const ShapeIndex& output_shape_index) {
        buffer_infos.push_back(
            {buffer_id, output_shape_index, -1, ShapeIndex()});
      };
      auto add_src_buffer =
          [&](int64_t buffer_id, int64_t operand_id,
              const ShapeIndex& operand_shape_index) -> absl::Status {
        for (BufferInfo& buffer_info : buffer_infos) {
          if (buffer_info.buffer_id == buffer_id) {
            buffer_info.operand_id = operand_id;
            buffer_info.operand_shape_index = operand_shape_index;
            return absl::OkStatus();
          }
        }
        return absl::InternalError(
            absl::StrCat("Buffer id ", buffer_id, " not found in results."));
      };

      // Find all buffers in the result.
      const Shape& shape = instr->shape();
      if (shape.is_buffer()) {
        add_dst_buffer(shape.buffer_id(), ShapeIndex());
      } else if (shape.IsTuple()) {
        ShapeUtil::ForEachSubshape(
            shape, [&](const Shape& subshape, const ShapeIndex& index) {
              if (subshape.is_buffer()) {
                add_dst_buffer(subshape.buffer_id(), index);
              }
            });
      }

      if (buffer_infos.empty()) {
        continue;
      }

      auto it =
          instr->frontend_attributes().map().find(kOutputToOperandAliasingAttr);
      if (it != instr->frontend_attributes().map().end()) {
        return absl::InternalError(absl::StrCat(
            "Instruction ", instr->ToString(),
            " uses buffers shouldn't has output_to_operand_aliasing attribute.",
            kOutputToOperandAliasingAttr));
      }

      // Find all buffers in the operands.
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        const Shape& operand_shape = instr->operand(i)->shape();
        if (operand_shape.is_buffer()) {
          TF_RETURN_IF_ERROR(
              add_src_buffer(operand_shape.buffer_id(), i, ShapeIndex()));
        } else if (operand_shape.IsTuple()) {
          TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
              operand_shape,
              [&](const Shape& subshape,
                  const ShapeIndex& index) -> absl::Status {
                if (subshape.is_buffer()) {
                  TF_RETURN_IF_ERROR(
                      add_src_buffer(subshape.buffer_id(), i, index));
                }
                return absl::OkStatus();
              }));
        }
      }

      for (const BufferInfo& buffer_info : buffer_infos) {
        if (buffer_info.operand_id == -1) {
          return absl::InternalError(absl::StrCat(
              "Buffer id ", buffer_info.buffer_id, " not found in operands."));
        }
      }

      if (instr->opcode() == HloOpcode::kCustomCall) {
        std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
            aliasing;
        for (const BufferInfo& buffer_info : buffer_infos) {
          aliasing.push_back(
              std::make_pair(buffer_info.output_shape_index,
                             std::make_pair(buffer_info.operand_id,
                                            buffer_info.operand_shape_index)));
        }
        instr->set_output_to_operand_aliasing(aliasing);
      }
      instr->add_frontend_attribute(kNonCopyableAttr, "{}");
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ConvertBufferRepresentationToAttributes::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_RETURN_IF_ERROR(ConvertToAttributes(module, execution_threads));

  bool changed = false;
  // Clear buffer_id fields in shapes.
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      Shape* shape = instr->mutable_shape();
      if (shape->is_buffer()) {
        shape->clear_buffer_id();
        changed = true;
      } else if (shape->IsTuple()) {
        ShapeUtil::ForEachMutableSubshape(
            shape, [&](Shape* subshape, const ShapeIndex& index) {
              if (subshape->is_buffer()) {
                subshape->clear_buffer_id();
                changed = true;
              }
            });
      }
    }
  }

  if (!changed) {
    return false;
  }

  // Remove pin and unpin, leaving allocateBuffer as a no-op custom-call.
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->IsCustomCall("pin") || instr->IsCustomCall("unpin")) {
        TF_RETURN_IF_ERROR(
            instr->ReplaceAllUsesWith(instr->mutable_operand(0)));
        TF_RETURN_IF_ERROR(
            computation->RemoveInstructionAndUnusedOperands(instr));
      }
    }
  }

  return true;
}

}  // namespace xla
