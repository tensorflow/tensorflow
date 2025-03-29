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

#include "xla/service/gpu/transforms/collectives/collective_combiner_annotator.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/gpu_collective_combiner_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

absl::StatusOr<bool> CollectiveCombinerAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(
      absl::flat_hash_set<HloInstruction*> sync_collectives,
      SynchronousCollectives(*module, pointer_size_, device_info_));
  if (sync_collectives.empty()) {
    return false;
  }

  bool changed = false;
  for (HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (!sync_collectives.contains(instr)) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(GpuBackendConfig config,
                          instr->backend_config<GpuBackendConfig>());
      config.mutable_collective_backend_config()
          ->set_is_sync_combiner_candidate(true);
      TF_RETURN_IF_ERROR(instr->set_backend_config(config));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla::gpu
