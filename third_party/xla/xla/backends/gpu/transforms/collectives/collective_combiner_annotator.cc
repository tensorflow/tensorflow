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

#include "xla/backends/gpu/transforms/collectives/collective_combiner_annotator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/backends/gpu/transforms/collectives/convert_async_collectives_to_sync.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

constexpr const char kCollectiveIdAttr[] = "collective_id";
constexpr const char kCollectiveSyncAttr[] = "sync_collective";
constexpr const char kSuggestedCombinerThresholdAttr[] =
    "suggested_combiner_threshold";

std::string CollectiveId(const HloInstruction* instr) {
  return absl::StrCat(instr->unique_id());
}

// Annotate all collective instructions with a unique identifier that will be
// preserved after async collective conversion.
void AnnotateCollectives(HloModule* module) {
  HloPredicate is_collective = [](const HloInstruction* instr) {
    return hlo_query::IsCollectiveCommunicationOp(instr->opcode());
  };
  hlo_query::ForEachInstructionWithPred(
      *module, is_collective, [](HloInstruction* instr) {
        instr->add_frontend_attribute(kCollectiveIdAttr, CollectiveId(instr));
      });
}

absl::Status AnnotateSyncCollectives(HloModule* module) {
  HloPassPipeline pipeline("annotate-sync-collectives");
  pipeline.AddPass<GpuConvertAsyncCollectivesToSync>();
  return pipeline.Run(module).status();
}

absl::flat_hash_set<std::string> SyncCollectiveIds(const HloModule& module) {
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

struct Metadata {
  int64_t peak_memory_bytes;
  absl::flat_hash_set<std::string> sync_collective_ids;
};

// Collect the following metadata by dry-running the scheduler:
// - peak_memory_bytes: the peak memory usage of the module.
// - sync_collective_ids: the set of collective instructions that are
//   synchronous post scheduling.
absl::StatusOr<Metadata> GetSchedulingMetadata(
    const HloModule& module, int64_t pointer_size,
    const se::DeviceDescription& device_info, mlir::MLIRContext* mlir_context,
    const GpuAliasInfo* alias_info) {
  std::unique_ptr<HloModule> cloned_module = module.Clone();
  AnnotateCollectives(cloned_module.get());
  TF_RETURN_IF_ERROR(RunAsyncCollectivesConversionPasses(cloned_module.get()));
  TF_ASSIGN_OR_RETURN(ScheduleMetadata schedule_metadata,
                      ScheduleGpuModule(cloned_module.get(), pointer_size,
                                        device_info, mlir_context, alias_info));
  TF_RETURN_IF_ERROR(AnnotateSyncCollectives(cloned_module.get()));
  return Metadata{schedule_metadata.peak_memory_usage,
                  SyncCollectiveIds(*cloned_module)};
}

int64_t MaxAvailableMemory(const HloModule& module,
                           const se::DeviceDescription& device_info) {
  int64_t base_limit = module.config().device_memory_size() != 0
                           ? module.config().device_memory_size()
                           : device_info.device_memory_size();
  int32_t slop_factor =
      module.config().debug_options().xla_gpu_memory_limit_slop_factor();
  return base_limit * slop_factor / 100;
}

}  // namespace

absl::StatusOr<bool> CollectiveCombinerAnnotator::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(
      Metadata metadata,
      GetSchedulingMetadata(*module, pointer_size_, device_info_, mlir_context_,
                            alias_info_));
  int64_t combiner_threshold =
      MaxAvailableMemory(*module, device_info_) - metadata.peak_memory_bytes;
  if (combiner_threshold <= 0) {
    LOG(ERROR) << "Computed combiner threshold " << combiner_threshold
               << " is <= 0.";
    return false;
  }

  AnnotateWithSuggestedCombinerThreshold(module, combiner_threshold);
  if (metadata.sync_collective_ids.empty()) {
    return true;
  }

  for (HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (!metadata.sync_collective_ids.contains(CollectiveId(instr))) {
        continue;
      }
      instr->add_frontend_attribute(kCollectiveSyncAttr, "true");
    }
  }

  return true;
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

std::optional<int64_t> SuggestedCombinerThreshold(const HloModule& module) {
  auto it =
      module.frontend_attributes().map().find(kSuggestedCombinerThresholdAttr);
  if (it == module.frontend_attributes().map().end()) {
    return std::nullopt;
  }
  int64_t combiner_threshold;
  if (!absl::SimpleAtoi(it->second, &combiner_threshold)) {
    return std::nullopt;
  }
  return combiner_threshold;
}

void AnnotateWithSuggestedCombinerThreshold(HloModule* module,
                                            int64_t combiner_threshold) {
  module->add_frontend_attribute(kSuggestedCombinerThresholdAttr,
                                 absl::StrCat(combiner_threshold));
}

}  // namespace xla::gpu
