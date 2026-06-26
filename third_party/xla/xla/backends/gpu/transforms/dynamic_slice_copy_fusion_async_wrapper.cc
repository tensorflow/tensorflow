/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/dynamic_slice_copy_fusion_async_wrapper.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/transforms/dynamic_slice_copy.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

absl::StatusOr<bool> DynamicSliceCopyFusionAsyncWrapper::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Offset verification runs the dynamic-slice fusion thunk in a runtime
  // checking mode that compares host-computed offsets with device-computed
  // offsets, so leave those fusions synchronous instead of wrapping them in
  // async-start/done.
  if (module->config()
          .debug_options()
          .xla_gpu_experimental_dynamic_slice_fusion_verify_offsets()) {
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (computation->IsAsyncComputation()) {
      continue;
    }

    std::vector<HloInstruction*> instructions =
        computation->MakeInstructionPostOrder();
    for (HloInstruction* instruction : instructions) {
      if (!IsDynamicSliceCopyFusion(instruction)) {
        continue;
      }

      // Use the same async-start context shape as the existing generic async
      // wrapper. LHS classifies this pair as async memcpy, not async compute.
      RETURN_IF_ERROR(computation
                          ->CreateAsyncInstructions(
                              instruction, {ShapeUtil::MakeScalarShape(U32)})
                          .status());
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla::gpu
