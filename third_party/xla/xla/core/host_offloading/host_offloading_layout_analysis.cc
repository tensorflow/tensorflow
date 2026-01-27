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

#include "xla/core/host_offloading/host_offloading_layout_analysis.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/computation_layout.h"
#include "xla/shape_util.h"

namespace xla {

namespace {

bool ComputationNeedsLayoutConversion(HloComputation* computation);

bool InstructionNeedsLayoutConversion(const HloInstruction* instruction) {
  VLOG(3) << "opcode: " << instruction->opcode() << ", "
          << instruction->ToShortString();

  // A transpose copy is not pure elementwise and needs layout conversion.
  if (instruction->opcode() == HloOpcode::kCopy &&
      instruction->shape().layout().minor_to_major() !=
          instruction->operand(0)->shape().layout().minor_to_major()) {
    return true;
  }

  if (instruction->opcode() == HloOpcode::kTuple) {
    return false;
  }

  if ((instruction->opcode() == HloOpcode::kParameter ||
       instruction->IsElementwise()) &&
      // Only allow 32-bit element types for now. If there are mixed element
      // types, we need layout conversion to align elements since for
      // example bf16 can have bf16 packing which results in different
      // element order from f32.
      //
      // TODO(b/446667479): Support cases that don't have mixed element types.
      (!instruction->shape().IsArray() ||
       ShapeUtil::ElementSizeInBits(instruction->shape()) == 32)) {
    return false;
  }

  // Only allow scalar broadcasts for now.
  // If needed, we could allow row-major broadcasts without tiling.
  if (instruction->opcode() == HloOpcode::kBroadcast) {
    const HloInstruction* to_be_broadcast = instruction->operand(0);
    return !to_be_broadcast->shape().dimensions().empty();
  }
  // If we find a kCall, recurse.
  if (instruction->opcode() == HloOpcode::kCall) {
    for (HloComputation* callee : instruction->called_computations()) {
      if (ComputationNeedsLayoutConversion(callee)) {
        return true;
      }
    }
    return false;
  }
  return true;
}

bool ComputationNeedsLayoutConversion(HloComputation* computation) {
  for (HloInstruction* instruction : computation->instructions()) {
    if (InstructionNeedsLayoutConversion(instruction)) {
      return true;
    }
  }
  return false;
}

// Check that the program shape does not have padded shapes due to tiling.
bool ModuleHasPadding(const HloModule* module) {
  const ComputationLayout& entry_layout = module->entry_computation_layout();
  for (int i = 0; i < entry_layout.parameter_count(); ++i) {
    if (HostOffloadingLayoutAnalysis::ShapeHasPadding(
            entry_layout.parameter_shape(i))) {
      return true;
    }
  }
  return HostOffloadingLayoutAnalysis::ShapeHasPadding(
      entry_layout.result_shape());
}

}  // namespace

bool HostOffloadingLayoutAnalysis::ShapeHasPadding(const Shape& shape) {
  bool has_padding = false;
  ShapeUtil::ForEachSubshape(
      shape,
      [&has_padding, &shape](const Shape& subshape, const ShapeIndex& index) {
        if (!ShapeUtil::IsLeafIndex(shape, index)) {
          return;
        }
        int64_t array_size = ShapeUtil::ArraySize(subshape);
        int64_t elements_size = ShapeUtil::ByteSizeOfElements(subshape);
        VLOG(4) << "subshape: " << subshape.ToString(/*print_layout=*/true)
                << ", arraysize: " << array_size
                << ", elements_size: " << elements_size;
        if (array_size != elements_size) {
          has_padding = true;
        }
      });
  return has_padding;
}

absl::StatusOr<bool> HostOffloadingLayoutAnalysis::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // TODO(ecg): relax this by allowing padding to then operate on a modified
  // program that has no padding, but whose shapes have been expanded to
  // include the padding bytes in the original tiled tensor.
  if (ModuleHasPadding(module)) {
    return true;
  }
  for (HloComputation* computation : module->computations(execution_threads)) {
    if (ComputationNeedsLayoutConversion(computation)) {
      return true;
    }
  }
  return false;
}

}  // namespace xla
