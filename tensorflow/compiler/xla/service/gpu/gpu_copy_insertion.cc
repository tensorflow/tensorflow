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

#include "tensorflow/compiler/xla/service/gpu/gpu_copy_insertion.h"

#include <memory>
#include <set>
#include <vector>

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace gpu {

StatusOr<HloInstruction*> GpuCopyInsertion::FindOrInsertCopy(
    HloInstruction* hlo) {
  HloInstruction*& copy = hlo_to_copy_map_[hlo];
  if (copy == nullptr) {
    TF_ASSIGN_OR_RETURN(copy, hlo->parent()->DeepCopyInstruction(hlo));
  }
  return copy;
}

StatusOr<bool> GpuCopyInsertion::Run(HloModule* module) {
  CopyInsertion generic_copy_insertion;

  TF_ASSIGN_OR_RETURN(bool changed, generic_copy_insertion.Run(module));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloDataflowAnalysis> dataflow,
                      HloDataflowAnalysis::Run(*module));

  // Make sure all operands of a library call are in memory instead of constants
  // in IR.
  for (HloInstruction* hlo :
       module->entry_computation()->MakeInstructionPostOrder()) {
    // Inserts a copy of hlo->operand(n) if it's a constant.
    auto copy_operand_if_constant = [&](int64 n) -> Status {
      HloInstruction* operand = hlo->mutable_operand(n);
      TF_RET_CHECK(ShapeUtil::IsArray(operand->shape()));
      const auto& values = dataflow->GetValueSet(operand).values();
      if (std::any_of(values.begin(), values.end(), [](const HloValue* value) {
            return value->defining_instruction()->opcode() ==
                   HloOpcode::kConstant;
          })) {
        TF_ASSIGN_OR_RETURN(HloInstruction * copy, FindOrInsertCopy(operand));
        TF_RETURN_IF_ERROR(hlo->ReplaceOperandWith(n, copy));
        changed = true;
      }
      return Status::OK();
    };

    if (IsCustomCallToDnnBatchNorm(*hlo)) {
      // The epsilon and feature_index operands to a CUDNN batchnorm op don't
      // need to be materialized in memory -- in fact, they must be constants.
      // These are the last two operands of all three batchnorm ops.
      for (int64 i = 0; i < hlo->operand_count() - 2; ++i) {
        TF_RETURN_IF_ERROR(copy_operand_if_constant(i));
      }
    } else if (IsCustomCallToDnnConvolution(*hlo)) {
      // The last two arguments to a CUDNN convolution are two HLO constants for
      // cudnn algorithm and tensor_ops_enabled flag, which shouldn't be copied.
      for (int64 i = 0; i < hlo->operand_count() - 2; ++i) {
        TF_RETURN_IF_ERROR(copy_operand_if_constant(i));
      }
    } else if (ImplementedAsLibraryCall(*hlo)) {
      // For all other library calls, materialize all the operands into memory.
      for (int64 i = 0; i < hlo->operand_count(); ++i) {
        TF_RETURN_IF_ERROR(copy_operand_if_constant(i));
      }
    }
  }

  // Init values of while and conditional nodes cannot be constants. Insert
  // copies for any constants found at the operands of these nodes.
  tensorflow::gtl::FlatSet<HloInstruction*> inserted_copies;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kWhile &&
          instruction->opcode() != HloOpcode::kConditional) {
        continue;
      }
      for (auto operand : instruction->operands()) {
        // Skip the operands that have already been replaced with a copy in a
        // previous iteration (which is possible when a constant is used as an
        // operand in multiple places).
        if (ContainsKey(inserted_copies, operand)) {
          continue;
        }
        for (auto& pair : dataflow->GetInstructionValueSet(operand)) {
          const HloValueSet& value_set = pair.second;
          for (const HloValue* value : value_set.values()) {
            if (value->defining_instruction()->IsConstant() &&
                !ContainsKey(hlo_to_copy_map_, value->defining_instruction())) {
              HloInstruction* constant = value->defining_instruction();
              TF_ASSIGN_OR_RETURN(HloInstruction * copy,
                                  FindOrInsertCopy(constant));
              TF_RETURN_IF_ERROR(constant->ReplaceAllUsesWith(copy));
              inserted_copies.insert(copy);
              changed = true;
            }
          }
        }
      }
    }
  }

  // The GPU backend needs additional copies added due to deficiencies in
  // buffer assignment.
  TF_ASSIGN_OR_RETURN(bool buffer_assignment_changed,
                      CopyInsertion::AddCopiesForBufferAssignment(module));

  return changed || buffer_assignment_changed;
}

}  // namespace gpu
}  // namespace xla
