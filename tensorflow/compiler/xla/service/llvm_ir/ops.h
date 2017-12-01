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

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_OPS_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_OPS_H_

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
  return fusion->fusion_kind() == HloInstruction::FusionKind::kLoop &&
         fusion->fused_expression_root()->opcode() ==
             HloOpcode::kDynamicUpdateSlice &&
         CanUpdateDynamicSliceInPlace(fusion->fused_expression_root(),
                                      assignment);
}

// Emits IR for running the given dynamic-update-slice op in-place -- that is,
// where the input and output buffers share the same slice, so we can simply
// modify the input/output buffer without touching any of the other elements.
Status EmitDynamicUpdateSliceInPlace(
    tensorflow::gtl::ArraySlice<IrArray> operand_arrays,
    const IrArray& output_array, tensorflow::StringPiece name,
    llvm::IRBuilder<>* ir_builder);

// Given a loop-fusion node whose root is a dynamic-update-slice op whose
// array-to-be-updated and output share the same buffer slice, emits
// (sequential) code for a fusion node that does the dynamic-update-slice in
// place.
Status EmitFusedDynamicUpdateSliceInPlace(
    HloInstruction* fusion,
    tensorflow::gtl::ArraySlice<IrArray> fusion_operand_arrays,
    const IrArray& fusion_output_array, ElementalIrEmitter* elemental_emitter,
    llvm::IRBuilder<>* ir_builder);

// Same as EmitFusedDynamicUpdateSliceInPlace, except emits a parallel loop with
// the given launch dimensions.
Status EmitParallelFusedDynamicUpdateSliceInPlace(
    HloInstruction* fusion,
    tensorflow::gtl::ArraySlice<IrArray> fusion_operand_arrays,
    const IrArray& fusion_output_array, ElementalIrEmitter* elemental_emitter,
    const gpu::LaunchDimensions& launch_dimensions,
    llvm::IRBuilder<>* ir_builder);

}  // namespace llvm_ir
}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_OPS_H_
