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

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace gpu {

StatusOr<bool> GpuCopyInsertion::Run(HloModule* module) {
  CopyInsertion generic_copy_insertion;

  TF_ASSIGN_OR_RETURN(bool changed, generic_copy_insertion.Run(module));

  // Check the assumption that the epsilon and feature_index constants of the
  // CUDNN batchnorm op are not shared with other ops where we would replace
  // them with a copy. These custom op calls are generated with the
  // CudnnBatchNormRewriter, so this would only happen if HloCSE merges them.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* hlo : computation->instructions()) {
      if (!IsCustomCallToDnnBatchNorm(*hlo)) {
        continue;
      }
      for (int64 i = hlo->operand_count() - 2; i < hlo->operand_count(); ++i) {
        CHECK_EQ(hlo->operand(i)->opcode(), HloOpcode::kConstant);
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
