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

#include "xla/backends/gpu/transforms/collectives/all_reduce_combiner.h"

#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/transforms/collectives/collective_combiner_annotator.h"
#include "xla/backends/gpu/transforms/collectives/gpu_collective_combiner_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/collectives/all_reduce_combiner.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

std::optional<AllReduceCombiner::GroupKey> DefaultCombinerKey(
    const HloInstruction* instruction, const HloDomainMap& domain_map) {
  std::optional<AllReduceCombiner::GroupKey> key =
      AllReduceCombiner::CombineKey(instruction, domain_map);
  if (!key.has_value()) {
    return std::nullopt;
  }
  // Don't combine pipelined and non-pipelined collectives.
  if (IsPipelinedCollective(*instruction)) {
    absl::StrAppend(&AllReduceCombiner::GetGroupKeyExtraArgs(*key),
                    " pipelined=true");
  }
  AppendCombinerKeyFromFrontendAttr(
      instruction, AllReduceCombiner::GetGroupKeyExtraArgs(*key));
  return key;
}

std::optional<AllReduceCombiner::GroupKey> CustomCombinerKey(
    const HloInstruction* instruction, const HloDomainMap& domain_map) {
  std::optional<AllReduceCombiner::GroupKey> key =
      AllReduceCombiner::CombineKey(instruction, domain_map);
  if (!key.has_value()) {
    return std::nullopt;
  }
  if (!key.has_value()) {
    return std::nullopt;
  }
  if (IsPipelinedCollective(*instruction)) {
    absl::StrAppend(&AllReduceCombiner::GetGroupKeyExtraArgs(*key),
                    " pipelined=true");
    AppendCombinerKeyFromFrontendAttr(
        instruction, AllReduceCombiner::GetGroupKeyExtraArgs(*key));
    return key;
  }
  if (IsCombinableSyncCollective(*instruction)) {
    absl::StrAppend(&AllReduceCombiner::GetGroupKeyExtraArgs(*key),
                    " sync=true");
    AppendCombinerKeyFromFrontendAttr(
        instruction, AllReduceCombiner::GetGroupKeyExtraArgs(*key));
    return key;
  }
  return std::nullopt;
}

}  // namespace

absl::StatusOr<bool> GpuAllReduceCombiner::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto post_combine = [](absl::Span<HloInstruction* const> to_combine,
                         HloInstruction* combined) {
    return MergeCollectiveBackendConfig(to_combine, combined);
  };

  // Combiner threshold is specified. Running parent pass code.
  if (combine_threshold_in_bytes_ != default_combine_threshold_in_bytes_) {
    return RunWithKeyCombiner(module, execution_threads, DefaultCombinerKey,
                              post_combine);
  }

  // Combiner threshold is not specified. We use heuristics.
  // We sequentially combine pipelined collectives and synchronous collectives
  // and finally the rest.

  bool changed = false;

  if (auto suggested_threshold = SuggestedCombinerThreshold(*module)) {
    combine_threshold_in_bytes_ = *suggested_threshold;
    TF_ASSIGN_OR_RETURN(bool combined,
                        RunWithKeyCombiner(module, execution_threads,
                                           CustomCombinerKey, post_combine));
    changed |= combined;
  }

  // Use the default combiner thresholds after we combined pipelined and
  // synchronous collectives.
  combine_threshold_in_bytes_ = default_combine_threshold_in_bytes_;
  TF_ASSIGN_OR_RETURN(bool combined,
                      RunWithKeyCombiner(module, execution_threads,
                                         DefaultCombinerKey, post_combine));
  changed |= combined;
  return changed;
}

}  // namespace xla::gpu
