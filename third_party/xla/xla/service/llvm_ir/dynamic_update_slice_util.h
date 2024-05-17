/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LLVM_IR_DYNAMIC_UPDATE_SLICE_UTIL_H_
#define XLA_SERVICE_LLVM_IR_DYNAMIC_UPDATE_SLICE_UTIL_H_
#include <utility>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"

// Utilities related to emitting LLVM IR for various HLO ops.

namespace xla {
namespace llvm_ir {

using GeneratorForOperandIrArrays =
    std::function<std::vector<llvm_ir::IrArray>()>;

// Determines whether the given instruction might be implemented as an
// in-place dynamic-update-slice after we have a buffer assignment.
//
// If this returns false, then CanUpdateDynamicSliceInPlace and
// CanEmitFusedDynamicUpdateSliceInPlace will also return false.
//
// This is useful if you want to check whether an instruction might be an
// in-place DUS during an HLO pass, at which point you don't have a buffer
// assignment.
//
// Note that simplifications to the HLO graph might change this function from
// returning false to returning true.  Specifically, simplifying the contents of
// fusion nodes might cause a false->true transition.  In general this isn't a
// problem by the time you're calling this function, but beware.
bool MayBeImplementedAsInPlaceDynamicUpdateSlice(const HloInstruction* instr);

// Checks if we can emit code for the given DynamicUpdateSlice node that updates
// its input in place.  Returns true if the dynamic-update-slice's
// array-to-be-updated and output share the same BufferAllocation::Slice.
//
// dynamic_update_slice must be a DynamicUpdateSlice op.
bool CanUpdateDynamicSliceInPlace(HloInstruction* dynamic_update_slice,
                                  const BufferAssignment& assignment);

// Checks if the given fusion node is amenable to being implemented by
// EmitFusedDynamicUpdateSliceInPlace.
bool CanEmitFusedDynamicUpdateSliceInPlace(HloInstruction* fusion,
                                           const BufferAssignment& assignment);

// Emits IR for running the given dynamic-update-slice op in-place -- that is,
// where the input and output buffers share the same slice, so we can simply
// modify the input/output buffer without touching any of the other elements.
absl::Status EmitDynamicUpdateSliceInPlace(
    absl::Span<const IrArray> operand_arrays, const IrArray& output_array,
    absl::string_view name, llvm::IRBuilder<>* b);

// Given a loop-fusion node whose root is a dynamic-update-slice op whose
// array-to-be-updated and output share the same buffer slice, emits
// (sequential) code for a fusion node that does the dynamic-update-slice in
// place.
absl::Status EmitFusedDynamicUpdateSliceInPlace(
    HloInstruction* fusion, const IrArray& fusion_output_array,
    FusedIrEmitter* fused_emitter, llvm::IRBuilder<>* b);

// Same as EmitFusedDynamicUpdateSliceInPlace, except emits a parallel loop with
// the given launch dimensions for arbitrarily many independent dynamic slice
// updates.
absl::Status EmitParallelFusedDynamicUpdateSliceInPlace(
    const HloComputation* fusion,
    const std::vector<std::pair<const HloInstruction*, const IrArray>>&
        dus_and_output_array,
    FusedIrEmitter* fused_emitter,
    const gpu::LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* b);

}  // namespace llvm_ir
}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_DYNAMIC_UPDATE_SLICE_UTIL_H_
