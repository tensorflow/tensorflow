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

#include "xla/backends/gpu/transforms/collectives/convert_async_collectives_to_sync.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/collectives/convert_async_collectives_to_sync.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

GpuConvertAsyncCollectivesToSync::GpuConvertAsyncCollectivesToSync()
    : ConvertAsyncCollectivesToSync(/*is_nop=*/
                                    HloPredicateIsOp<
                                        HloOpcode::kParameter,
                                        HloOpcode::kConstant,
                                        HloOpcode::kBitcast,
                                        HloOpcode::kGetTupleElement>) {}

absl::Status GpuConvertAsyncCollectivesToSync::ConvertAsyncInstructionsToSync(
    HloComputation* computation,
    absl::Span<const std::pair<HloInstruction*, HloInstruction*>> async_pairs)
    const {
  for (auto& [async_start, async_done] : async_pairs) {
    // Tag the async start with is_sync = true.
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                     async_start->backend_config<GpuBackendConfig>());
    gpu_config.mutable_collective_backend_config()->set_is_sync(true);
    ABSL_RETURN_IF_ERROR(async_start->set_backend_config(gpu_config));
  }
  return ReplaceAsyncInstructionsWithSync(computation, async_pairs);
}

absl::StatusOr<bool> GpuConvertAsyncCollectivesToSync::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // The pre-scheduling annotator marks collectives that must execute
  // synchronously. After scheduling, restore their canonical synchronous form
  // independently of the overlap analysis below.
  if (module->has_schedule()) {
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      if (!module->schedule().is_computation_scheduled(computation)) {
        continue;
      }

      std::vector<std::pair<HloInstruction*, HloInstruction*>> async_pairs;
      for (HloInstruction* instruction : computation->instructions()) {
        if (!hlo_query::IsAsyncCollectiveStartOp(instruction) ||
            !IsGPUSyncCollective(*instruction)) {
          continue;
        }
        if (instruction->user_count() != 1 ||
            !hlo_query::IsAsyncCollectiveDoneOp(instruction->users()[0])) {
          continue;
        }
        HloInstruction* async_done = instruction->users()[0];
        async_pairs.push_back({instruction, async_done});
      }

      if (!async_pairs.empty()) {
        ABSL_RETURN_IF_ERROR(
            ReplaceAsyncInstructionsWithSync(computation, async_pairs));
        changed = true;
      }
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      bool converted_unoverlapped,
      ConvertAsyncCollectivesToSync::RunImpl(module, execution_threads));
  return changed || converted_unoverlapped;
}

}  // namespace xla::gpu
