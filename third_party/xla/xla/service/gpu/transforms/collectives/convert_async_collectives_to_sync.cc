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

#include "xla/service/gpu/transforms/collectives/convert_async_collectives_to_sync.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/transforms/collectives/convert_async_collectives_to_sync.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

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
  absl::flat_hash_map<HloInstruction*, HloInstruction*> replaced_ops;
  for (auto& [async_start, async_done] : async_pairs) {
    // Tag the async start with is_sync = true.
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        async_start->backend_config<GpuBackendConfig>());
    gpu_config.mutable_collective_backend_config()->set_is_sync(true);
    TF_RETURN_IF_ERROR(async_start->set_backend_config(gpu_config));
    replaced_ops[async_start] = nullptr;
    replaced_ops[async_done] = async_start;
  }

  // Update schedule.
  HloModule* module = computation->parent();
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  std::vector<HloInstruction*> new_sequence;
  new_sequence.reserve(sequence.size());
  for (HloInstruction* instr : sequence.instructions()) {
    auto it = replaced_ops.find(instr);
    // If its not a start or done, add it to new schedule.
    if (it == replaced_ops.end()) {
      new_sequence.push_back(instr);
      continue;
    }

    // If its a start op, do not add it to the schedule yet.
    if (it->second == nullptr) {
      continue;
    }

    // Its a done op. First add the start and then the done.
    new_sequence.push_back(it->second);
    new_sequence.push_back(instr);
  }
  module->schedule().set_sequence(computation, new_sequence);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
