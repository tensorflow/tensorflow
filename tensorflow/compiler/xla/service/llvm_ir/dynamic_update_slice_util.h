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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_DYNAMIC_UPDATE_SLICE_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_DYNAMIC_UPDATE_SLICE_UTIL_H_

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"

// Utilities related to emitting LLVM IR for various HLO ops.

namespace xla {
namespace llvm_ir {

// Checks if we can emit code for the given DynamicUpdateSlice node that updates
// its input in place.  Returns true if the dynamic-update-slice's
// array-to-be-updated and output share the same BufferAllocation::Slice.
//
// dynamic_update_slice must be a DynamicUpdateSlice op.
bool CanUpdateDynamicSliceInPlace(HloInstruction* dynamic_update_slice,
                                  const BufferAssignment& assignment);

// Checks if the given fusion node is amenable to being implemented by
// EmitFusedDynamicUpdateSliceInPlace.
inline bool CanEmitFusedDynamicUpdateSliceInPlace(
    HloInstruction* fusion, const BufferAssignment& assignment) {
  CHECK_EQ(fusion->opcode(), HloOpcode::kFusion);
  HloInstruction* fused_root = fusion->fused_expression_root();
  if (fused_root->opcode() != HloOpcode::kDynamicUpdateSlice ||
      fusion->fusion_kind() != HloInstruction::FusionKind::kLoop) {
    return false;
  }
  // Walk DynamicUpdateSlice operand(0) to fused parameter and get its
  // associated operand. See if it shares an allocation with this operand.
  HloInstruction* fusion_operand;
  ShapeIndex index;
  std::tie(fusion_operand, index) =
      fused_root->mutable_operand(0)->LatestNonGteAncestorAndIndex();
  if (fusion_operand->opcode() != HloOpcode::kParameter) {
    return false;
  }
  auto* operand = fusion->operand(fusion_operand->parameter_number());
  return assignment.HasAllocationAt(operand, index) &&
         assignment.HasAllocationAt(fusion, {}) &&
         assignment.SharesSliceAtIndex(fusion, {}, operand, index);
}

// Emits IR for running the given dynamic-update-slice op in-place -- that is,
// where the input and output buffers share the same slice, so we can simply
// modify the input/output buffer without touching any of the other elements.
Status EmitDynamicUpdateSliceInPlace(absl::Span<const IrArray> operand_arrays,
                                     const IrArray& output_array,
                                     absl::string_view name,
                                     llvm::IRBuilder<>* b);

// Given a loop-fusion node whose root is a dynamic-update-slice op whose
// array-to-be-updated and output share the same buffer slice, emits
// (sequential) code for a fusion node that does the dynamic-update-slice in
// place.
Status EmitFusedDynamicUpdateSliceInPlace(
    HloInstruction* fusion, absl::Span<const IrArray> fusion_operand_arrays,
    const IrArray& fusion_output_array, ElementalIrEmitter* elemental_emitter,
    llvm::IRBuilder<>* b);

// Same as EmitFusedDynamicUpdateSliceInPlace, except emits a parallel loop with
// the given launch dimensions.
Status EmitParallelFusedDynamicUpdateSliceInPlace(
    HloInstruction* fusion, absl::Span<const IrArray> fusion_operand_arrays,
    const IrArray& fusion_output_array, ElementalIrEmitter* elemental_emitter,
    const gpu::LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* b);

}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_DYNAMIC_UPDATE_SLICE_UTIL_H_
