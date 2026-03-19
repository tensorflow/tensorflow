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

#ifndef XLA_CODEGEN_IR_EMISSION_UTILS_H_
#define XLA_CODEGEN_IR_EMISSION_UTILS_H_

#include <functional>
#include <optional>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/codegen/hlo_fusion_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Checks if the instruction is elementwise.
bool IsIntermediate(const HloInstruction* instr, int allowed_operand_count = 1);

// Find the first gero that statises the given predicate.
std::optional<HloInstructionAdaptor> FindHero(
    const HloInstructionAdaptor& root,
    absl::AnyInvocable<bool(const HloInstruction&)> predicate);

// Should the given fusion be emitted using the DUS emitter.
bool IsDynamicUpdateSliceFusion(const HloFusionSpec& fusion_spec);

// Returns the dynamic-update-slice instructions defining the results of a
// fusion node. A dynamic slice update is said to be "defining" of a result if
// that result is the output of a dynamic slice update, or if that result is the
// output of a bitcast of a dynamic slice update---since such bitcast may be
// handled as a no-op.
std::vector<HloInstructionAdaptor> GetOutputDefiningDynamicUpdateSlices(
    absl::Span<HloInstructionAdaptor const> roots);

// Returns whether the fusion represented by 'fusion_adaptor' can be emitted
// with the dynamic update slice in-place emitter. If 'fusion_adaptor'
// represents a single fusion computation, 'fusion' should provide the fusion
// instruction corresponding to that fusion computation. 'get_allocation_slice'
// is a callback for getting the allocated buffer slice, given an instruction
// and a shape index. This is ignored in case 'fusion' is a nullptr.
absl::StatusOr<bool> CanEmitFusedDynamicUpdateSliceInPlace(
    const HloFusionAdaptor& fusion_adaptor,
    std::function<absl::StatusOr<BufferAllocation::Slice>(
        const HloInstruction* instr, const ShapeIndex& index)>
        get_allocation_slice,
    const HloInstruction* fusion = nullptr);

// Same as above, but uses the buffer assignment to get the allocated buffer
// slices.
absl::StatusOr<bool> CanEmitFusedDynamicUpdateSliceInPlace(
    const HloFusionAdaptor& fusion_adaptor,
    const BufferAssignment* buffer_assignment, const HloInstruction* fusion);

}  // namespace xla

#endif  // XLA_CODEGEN_IR_EMISSION_UTILS_H_
