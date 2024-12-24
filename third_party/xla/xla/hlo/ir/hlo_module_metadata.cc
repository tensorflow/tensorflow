/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_module_metadata.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/metrics.pb.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<HloPassMetadata*>
HloModuleMetadata::GetCurrentHloPassMetadata() {
  if (running_passes_.empty()) {
    return NotFound(
        "HloPassMetadata for currently running pass not found, either because "
        "the pass did not call RecordPassStart or because a pass is "
        "creating/switching modules without using "
        "HloModuleGroup::ReplaceModule.");
  }
  return running_passes_.back();
}

absl::Status HloModuleMetadata::MutateCurrentHloPassMetadata(
    absl::FunctionRef<void(HloPassMetadata*)> mutator) {
  TF_ASSIGN_OR_RETURN(HloPassMetadata * pass_metadata,
                      GetCurrentHloPassMetadata());
  mutator(pass_metadata);
  return absl::OkStatus();
}

void HloModuleMetadata::RecordPassStart() {
  HloPassMetadata* pass_metadata = module_metadata_.add_pass_metadata();
  pass_metadata->set_pass_id(next_pass_id_++);
  pass_metadata->set_start_timestamp_usec(env_->NowMicros());
  running_passes_.push_back(pass_metadata);
}

absl::Status HloModuleMetadata::RecordPassEnd() {
  TF_ASSIGN_OR_RETURN(HloPassMetadata * pass_metadata,
                      GetCurrentHloPassMetadata());
  pass_metadata->set_end_timestamp_usec(env_->NowMicros());
  running_passes_.pop_back();
  return absl::OkStatus();
}

void HloModuleMetadata::set_prepartitioning_metadata(
    const HloModuleMetadata& prepartitioning_metadata) {
  module_metadata_.set_original_module_id(
      prepartitioning_metadata.proto().canonical_module_id());
  prepartitioning_metadata_ = prepartitioning_metadata.proto();
  prepartitioning_metadata_->clear_pass_metadata();

  // Because HloPassMetadata represents the completion of a pass, metadata for
  // all currently running passes need to be moved over to the new module.
  absl::flat_hash_set<HloPassMetadata*> running_passes(
      prepartitioning_metadata.running_passes_.begin(),
      prepartitioning_metadata.running_passes_.end());
  for (const HloPassMetadata& pass_metadata :
       prepartitioning_metadata.proto().pass_metadata()) {
    if (running_passes.contains(&pass_metadata)) {
      HloPassMetadata* added_pass_metadata =
          module_metadata_.add_pass_metadata();
      *added_pass_metadata = pass_metadata;
      running_passes_.push_back(added_pass_metadata);
      next_pass_id_ =
          std::max(next_pass_id_,
                   static_cast<int64_t>(added_pass_metadata->pass_id()) + 1);
    } else {
      *prepartitioning_metadata_->add_pass_metadata() = pass_metadata;
    }
  }
}

absl::Status HloModuleMetadata::set_custom_metadata(
    const ::tsl::protobuf::Message& message) {
  TF_ASSIGN_OR_RETURN(HloPassMetadata * pass_metadata,
                      GetCurrentHloPassMetadata());
  if (!pass_metadata->mutable_custom_metadata()->PackFrom(message)) {
    LOG(WARNING) << "failed to pack custom metadata for "
                 << pass_metadata->pass_id();
    return Internal("failed to pack custom metadata");
  };
  return absl::OkStatus();
}

absl::Status HloModuleMetadata::set_key_value_metric(const std::string& key,
                                                     int64_t value) {
  TF_ASSIGN_OR_RETURN(HloPassMetadata * pass_metadata,
                      GetCurrentHloPassMetadata());
  auto* kv_metrics = pass_metadata->mutable_kv_metrics();
  // Iterating here since we expect only a few kv_metrics per pass ..
  for (auto& kv_metric : *kv_metrics) {
    if (kv_metric.key() == key) {
      kv_metric.set_value(value);

      return absl::OkStatus();
    }
  }

  // If none exists, add new one.
  KeyValueMetric* kv_metric = pass_metadata->add_kv_metrics();
  kv_metric->set_key(key);
  kv_metric->set_value(value);
  return absl::OkStatus();
}

}  // namespace xla
