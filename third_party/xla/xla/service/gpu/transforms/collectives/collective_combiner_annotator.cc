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

#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/gpu/transforms/collectives/convert_async_collectives_to_sync.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

static constexpr const char kCollectiveIdAttr[] = "collective_id";
static constexpr const char kCollectiveSyncAttr[] = "sync_collective";

static std::string CollectiveId(const HloInstruction* instr) {
  return absl::StrCat(instr->unique_id());
}

// Annotate all collective instructions with a unique identifier that will be
// preserved after async collective conversion.
static void AnnotateCollectives(HloModule* module) {
  HloPredicate is_collective = [](const HloInstruction* instr) {
    return hlo_query::IsCollectiveCommunicationOp(instr->opcode());
  };
  hlo_query::ForEachInstructionWithPred(
      *module, is_collective, [](HloInstruction* instr) {
        instr->add_frontend_attribute(kCollectiveIdAttr, CollectiveId(instr));
      });
}

static absl::Status AnnotateSyncCollectives(HloModule* module) {
  HloPassPipeline pipeline("annotate-sync-collectives");
  pipeline.AddPass<GpuConvertAsyncCollectivesToSync>();
  return pipeline.Run(module).status();
}

static absl::flat_hash_set<std::string> SyncCollectiveIds(
    const HloModule& module) {
  absl::flat_hash_set<std::string> sync_collective_ids;
  HloPredicate is_sync_collective = [](const HloInstruction* instr) {
    return IsGPUSyncCollective(*instr);
  };
  hlo_query::ForEachInstructionWithPred(
      module, is_sync_collective,
      [&sync_collective_ids](const HloInstruction* instr) {
        sync_collective_ids.insert(
            *instr->get_frontend_attribute(kCollectiveIdAttr));
      });
  return sync_collective_ids;
}

// Return the set of collective instructions that are synchronous post
// scheduling.
static absl::StatusOr<absl::flat_hash_set<std::string>> SynchronousCollectives(
    const HloModule& module, int64_t pointer_size,
    const se::DeviceDescription& device_info) {
  std::unique_ptr<HloModule> cloned_module = module.Clone();
  AnnotateCollectives(cloned_module.get());
  TF_RETURN_IF_ERROR(RunAsyncCollectivesConversionPasses(cloned_module.get()));
  TF_RETURN_IF_ERROR(
      ScheduleGpuModule(cloned_module.get(), pointer_size, device_info)
          .status());
  TF_RETURN_IF_ERROR(AnnotateSyncCollectives(cloned_module.get()));
  return SyncCollectiveIds(*cloned_module);
}

absl::StatusOr<bool> CollectiveCombinerAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(
      absl::flat_hash_set<std::string> sync_collectives,
      SynchronousCollectives(*module, pointer_size_, device_info_));
  if (sync_collectives.empty()) {
    return false;
  }

  bool changed = false;
  for (HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (!sync_collectives.contains(CollectiveId(instr))) {
        continue;
      }
      instr->add_frontend_attribute(kCollectiveSyncAttr, "true");
      changed = true;
    }
  }

  return changed;
}

bool IsCombinableSyncCollective(const HloInstruction& instr) {
  return instr.get_frontend_attribute(kCollectiveSyncAttr).value_or("false") ==
         "true";
}

bool ContainsCombinableSyncCollective(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (IsCombinableSyncCollective(*instr)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace xla::gpu
