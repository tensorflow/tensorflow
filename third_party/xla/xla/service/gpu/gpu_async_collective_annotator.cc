/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_async_collective_annotator.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> GpuAsyncCollectiveAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (!hlo_query::IsAsyncCollectiveStartOp(instruction)) {
        continue;
      }
      CollectiveBackendConfig config;
      config.set_is_sync(!is_collective_async_(instruction));
      TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          instruction->backend_config<GpuBackendConfig>());
      *gpu_config.mutable_collective_backend_config() = config;
      TF_RETURN_IF_ERROR(instruction->set_backend_config(gpu_config));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
