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

#include "tensorflow/compiler/xla/service/gpu/copy_insertion.h"

#include <memory>
#include <set>
#include <vector>

#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

StatusOr<bool> GpuCopyInsertion::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(bool changed, CopyInsertion::Run(module));

  TF_ASSIGN_OR_RETURN(auto points_to_analysis,
                      TuplePointsToAnalysis::Run(module));

  // Make sure all operands of a library call are in memory instead of constants
  // in IR. The top-level (index {}) of the points-to set of each operand
  // indicates the source(s) of the array buffer. If any of these are constant,
  // then add a copy to materialize the array.
  HloComputation* computation = module->entry_computation();
  for (HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
    if (ImplementedAsLibraryCall(*hlo)) {
      for (int64 i = 0; i < hlo->operand_count(); ++i) {
        HloInstruction* operand = hlo->mutable_operand(i);
        const PointsToSet& points_to =
            points_to_analysis->GetPointsToSet(operand);
        const auto& element = points_to.element(/*index=*/{});
        if (std::any_of(element.begin(), element.end(),
                        [](const LogicalBuffer* buffer_source) {
                          return buffer_source->instruction()->opcode() ==
                                 HloOpcode::kConstant;
                        })) {
          TF_ASSIGN_OR_RETURN(HloInstruction * copy,
                              CopyInsertion::FindOrInsertCopy(operand));
          TF_RETURN_IF_ERROR(hlo->ReplaceOperandWith(i, copy));
          changed = true;
        }
      }
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
