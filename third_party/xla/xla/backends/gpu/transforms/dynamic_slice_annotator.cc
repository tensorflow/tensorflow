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

#include "xla/backends/gpu/transforms/dynamic_slice_annotator.h"

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/transforms/dynamic_slice_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"

namespace xla::gpu {

absl::StatusOr<bool> DynamicSliceAnnotator::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool has_changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kDynamicSlice &&
          instr->opcode() != HloOpcode::kDynamicUpdateSlice) {
        continue;
      }

      ASSIGN_OR_RETURN(auto descriptor, AnalyzeDynamicSlice(instr));
      if (!descriptor) {
        continue;
      }

      ASSIGN_OR_RETURN(auto backend_config,
                       instr->backend_config<GpuBackendConfig>());
      auto* ds_config = backend_config.mutable_dynamic_slice_config();
      if (descriptor->loop_index.has_value()) {
        ds_config->set_loop_index(*descriptor->loop_index);
      }
      ds_config->set_byte_offset(descriptor->byte_offset);
      ds_config->set_byte_stride(descriptor->byte_stride);
      RETURN_IF_ERROR(instr->set_backend_config(backend_config));

      VLOG(2) << "Annotated " << instr->name() << " with DynamicSliceConfig:"
              << " loop_index="
              << (descriptor->loop_index.has_value()
                      ? std::to_string(*descriptor->loop_index)
                      : "none")
              << ", offset=" << descriptor->byte_offset
              << ", stride=" << descriptor->byte_stride;
      has_changed = true;
    }
  }

  return has_changed;
}

}  // namespace xla::gpu
