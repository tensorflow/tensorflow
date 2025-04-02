/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/collectives/all_reduce_combiner.h"

#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/collectives/all_reduce_combiner.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_combiner_annotator.h"
#include "xla/service/gpu/transforms/collectives/gpu_collective_combiner_utils.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

std::optional<AllReduceCombiner::GroupKey> PipelinedCombinerKey(
    const HloInstruction* instruction, const HloDomainMap& domain_map) {
  auto backend_config = instruction->backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    return std::nullopt;
  }
  if (!backend_config->collective_backend_config().is_pipelined()) {
    return std::nullopt;
  }
  return AllReduceCombiner::CombineKey(instruction, domain_map);
}

std::optional<AllReduceCombiner::GroupKey> SynchronousCombinerKey(
    const HloInstruction* instruction, const HloDomainMap& domain_map) {
  if (!IsCombinableSyncCollective(*instruction)) {
    return std::nullopt;
  }
  return AllReduceCombiner::CombineKey(instruction, domain_map);
}

}  // namespace

absl::StatusOr<bool> GpuAllReduceCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Combiner threshold is specified. Running parent pass code.
  if (combine_threshold_in_bytes_ != default_combine_threshold_in_bytes_) {
    return AllReduceCombiner::Run(module, execution_threads);
  }

  // Combiner threshold is not specified. We use heuristics.
  // We sequentially combine synchronous collectives then pipelined collectives
  // and finally the rest. Note that collectives can be both synchronous and
  // pipelined. Hence, we combine them in two steps.

  bool changed = false;

  // Combine as much as possible for synchronous collectives.
  if (ContainsCombinableSyncCollective(*module)) {
    combine_threshold_in_bytes_ = MaxAvailableMemory(*module, device_info_);
    TF_ASSIGN_OR_RETURN(
        bool combined,
        RunWithKeyCombiner(module, execution_threads, SynchronousCombinerKey));
    changed |= combined;
  }

  // If there are no pipelined instructions in the IR, the optimizations below
  // do not kick in anyway.
  if (ContainsPipelinedInstruction(*module)) {
    // Combine as much as possible for pipelined collectives.
    combine_threshold_in_bytes_ = ComputeSuggestedCombinerThreshold(
        *module, device_info_, HloOpcode::kAllReduce, pointer_size_);
    TF_ASSIGN_OR_RETURN(
        bool combined,
        RunWithKeyCombiner(module, execution_threads, PipelinedCombinerKey));
    changed |= combined;
  }

  // Use default combiner thresholds after we combine pipelined collectives.
  // The rest is combined by the parent pass code.
  combine_threshold_in_bytes_ = default_combine_threshold_in_bytes_;
  TF_ASSIGN_OR_RETURN(bool combined_rest,
                      AllReduceCombiner::Run(module, execution_threads));
  changed |= combined_rest;
  return changed;
}

}  // namespace xla::gpu
