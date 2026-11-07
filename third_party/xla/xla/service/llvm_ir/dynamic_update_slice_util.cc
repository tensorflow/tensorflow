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

#include "xla/service/llvm_ir/dynamic_update_slice_util.h"

#include <tuple>

#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/shape_util.h"

namespace xla {
namespace llvm_ir {

bool MayBeImplementedAsInPlaceDynamicUpdateSlice(const HloInstruction* instr) {
  // Today we can't emit a dynamic-update-slice if the DUS node is parallelized;
  // the emitter will not emit correct code.  It's possible to change this, but
  // then ParallelTaskAssigner would have to somehow know whether a node *will*
  // be emitted as an in-place DUS, and it can't, because it doesn't have a
  // buffer assignment when it runs.
  auto cpu_backend_config_or = instr->backend_config<xla::cpu::BackendConfig>();
  if (cpu_backend_config_or.ok() &&
      !cpu_backend_config_or->outer_dimension_partitions().empty()) {
    return false;
  }

  // Until we know the final buffer assignment, any unfused dynamic-update-slice
  // might be implementable as an in-place DUS.
  if (instr->opcode() == HloOpcode::kDynamicUpdateSlice) {
    return true;
  }

  // A fusion may be implementable as an in-place dynamic update slice if
  //  - it's a loop fusion,
  //  - dynamic-update-slice is the root of the fusion, and
  //  - operand 0 of the dynamic-update-slice is a parameter to the fusion
  //    (ignoring any get-tuple-element operations in the way).
  if (instr->IsLoopFusion()) {
    const HloInstruction* fused_root = instr->fused_expression_root();
    return fused_root->opcode() == HloOpcode::kDynamicUpdateSlice &&
           fused_root->operand(0)->LatestNonGteAncestor()->opcode() ==
               HloOpcode::kParameter;
  }

  return false;
}

bool CanUpdateDynamicSliceInPlace(HloInstruction* dynamic_update_slice,
                                  const BufferAssignment& assignment) {
  CHECK_EQ(HloOpcode::kDynamicUpdateSlice, dynamic_update_slice->opcode());
  const HloInstruction* operand = dynamic_update_slice->operand(0);
  return assignment.HasTopLevelAllocation(dynamic_update_slice) &&
         assignment.HasTopLevelAllocation(operand) &&
         assignment.SharesTopLevelSlice(dynamic_update_slice, operand);
}

bool CanEmitFusedDynamicUpdateSliceInPlace(HloInstruction* fusion,
                                           const BufferAssignment& assignment) {
  CHECK_EQ(fusion->opcode(), HloOpcode::kFusion);
  if (!MayBeImplementedAsInPlaceDynamicUpdateSlice(fusion)) {
    return false;
  }

  // Walk DynamicUpdateSlice operand(0) to fused parameter and get its
  // associated operand. See if it shares an allocation with this operand.
  HloInstruction* fused_root = fusion->fused_expression_root();
  HloInstruction* fusion_operand;
  ShapeIndex index;
  std::tie(fusion_operand, index) =
      fused_root->mutable_operand(0)->LatestNonGteAncestorAndIndex();
  // MayBeImplementedAsInPlaceDynamicUpdateSlice should have ensured that
  // fusion_operand is a parameter.
  CHECK_EQ(fusion_operand->opcode(), HloOpcode::kParameter);
  auto* operand = fusion->operand(fusion_operand->parameter_number());
  return assignment.HasAllocationAt(operand, index) &&
         assignment.HasAllocationAt(fusion, {}) &&
         assignment.SharesSliceAtIndex(fusion, {}, operand, index);
}

}  // namespace llvm_ir
}  // namespace xla
