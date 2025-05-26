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

#include "xla/service/gpu/transforms/collectives/all_gather_combiner.h"

#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/collectives/all_gather_combiner.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_combiner_annotator.h"
#include "xla/service/gpu/transforms/collectives/gpu_collective_combiner_utils.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

std::optional<AllGatherCombiner::GroupKey> DefaultCombinerKey(
    const HloInstruction* instruction, const HloDomainMap& domain_map,
    bool combine_by_dim, bool combine_different_dtypes) {
  std::optional<AllGatherCombiner::GroupKey> key =
      AllGatherCombiner::CombineKey(instruction, domain_map, combine_by_dim,
                                    combine_different_dtypes);
  if (!key.has_value()) {
    return std::nullopt;
  }
  // Don't combine pipelined and non-pipelined collectives.
  if (IsPipelinedCollective(*instruction)) {
    absl::StrAppend(&AllGatherCombiner::GetGroupKeyExtraArgs(*key),
                    " pipelined=true");
  }
  return key;
}

std::optional<AllGatherCombiner::GroupKey> PipelinedCombinerKey(
    const HloInstruction* instruction, const HloDomainMap& domain_map,
    bool combine_by_dim, bool combine_different_dtypes) {
  if (!IsPipelinedCollective(*instruction)) {
    return std::nullopt;
  }
  return AllGatherCombiner::CombineKey(instruction, domain_map, combine_by_dim,
                                       combine_different_dtypes);
}

std::optional<AllGatherCombiner::GroupKey> SynchronousCombinerKey(
    const HloInstruction* instruction, const HloDomainMap& domain_map,
    bool combine_by_dim, bool combine_different_dtypes) {
  if (!IsCombinableSyncCollective(*instruction)) {
    return std::nullopt;
  }
  // Exclude pipelined collectives.
  if (IsPipelinedCollective(*instruction)) {
    return std::nullopt;
  }
  return AllGatherCombiner::CombineKey(instruction, domain_map, combine_by_dim,
                                       combine_different_dtypes);
}

}  // namespace

absl::StatusOr<bool> GpuAllGatherCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Combiner threshold is specified. Running parent pass code.
  if (combine_threshold_in_bytes_ != default_combine_threshold_in_bytes_) {
    return RunWithKeyCombiner(module, execution_threads, DefaultCombinerKey);
  }

  // Combiner threshold is not specified. We use heuristics.
  // We sequentially combine pipelined collectives then synchronous collectives
  // and finally the rest.

  bool changed = false;

  // If there are no pipelined instructions in the IR, the optimizations below
  // do not kick in anyway.
  if (ContainsPipelinedInstruction(*module)) {
    // Combine as much as possible for pipelined collectives.
    combine_threshold_in_bytes_ = ComputeSuggestedCombinerThreshold(
        *module, device_info_, HloOpcode::kAllGather, pointer_size_);
    TF_ASSIGN_OR_RETURN(
        bool combined,
        RunWithKeyCombiner(module, execution_threads, PipelinedCombinerKey));
    changed |= combined;
  }

  // Combine as much as possible for synchronous collectives.
  if (ContainsCombinableSyncCollective(*module)) {
    combine_threshold_in_bytes_ = MaxAvailableMemory(*module, device_info_);
    TF_ASSIGN_OR_RETURN(
        bool combined,
        RunWithKeyCombiner(module, execution_threads, SynchronousCombinerKey));
    changed |= combined;
  }

  // Use default combiner thresholds after we combine pipelined collectives.
  // The rest is combined by the parent pass code.
  combine_threshold_in_bytes_ = default_combine_threshold_in_bytes_;
  TF_ASSIGN_OR_RETURN(
      bool combined,
      RunWithKeyCombiner(module, execution_threads, DefaultCombinerKey));
  changed |= combined;
  return changed;
}

}  // namespace xla::gpu
