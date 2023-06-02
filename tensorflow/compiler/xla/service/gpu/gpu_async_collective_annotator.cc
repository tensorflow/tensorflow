/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_async_collective_annotator.h"

#include "tensorflow/compiler/xla/hlo/utils/hlo_query.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"

namespace xla {
namespace gpu {

StatusOr<bool> GpuAsyncCollectiveAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (!hlo_query::IsAsyncCollectiveStartOp(instruction->opcode())) {
        continue;
      }
      CollectiveBackendConfig config;
      config.set_is_sync(!is_collective_async_(instruction));
      TF_RETURN_IF_ERROR(instruction->set_backend_config(config));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
