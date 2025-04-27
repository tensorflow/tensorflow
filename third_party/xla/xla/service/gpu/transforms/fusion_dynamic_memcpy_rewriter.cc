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

#include "xla/service/gpu/transforms/fusion_dynamic_memcpy_rewriter.h"

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/copy.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> FusionDynamicMemcpyRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool has_changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      continue;
    }

    HloFusionInstruction* fusion =
        ::xla::Cast<HloFusionInstruction>(computation->FusionInstruction());
    if (DynamicMemcpyFusion::GetMemcpyDescriptorForFusion(*fusion)) {
      TF_ASSIGN_OR_RETURN(auto backend_config,
                          fusion->backend_config<GpuBackendConfig>());
      backend_config.mutable_fusion_backend_config()->set_kind(
          std::string(kDynamicMemcpyFusionKind));
      TF_RETURN_IF_ERROR(fusion->set_backend_config(backend_config));
      has_changed = true;
    }
  }

  return has_changed;
}

}  // namespace gpu
}  // namespace xla
