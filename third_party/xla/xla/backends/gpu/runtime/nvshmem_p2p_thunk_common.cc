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

#include "xla/backends/gpu/runtime/nvshmem_p2p_thunk_common.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::Status NvshmemP2PExecutionCounters::Initialize(
    se::StreamExecutor* executor, RunId run_id) {
  absl::MutexLock lock(&mu_);
  CounterKey key = {executor, run_id};
  if (counters_.contains(key)) {
    return absl::OkStatus();
  }
  counters_.emplace(key, 0);
  return absl::OkStatus();
}

absl::StatusOr<int64_t*> NvshmemP2PExecutionCounters::GetCounter(
    se::StreamExecutor* executor, RunId run_id) {
  absl::MutexLock lock(&mu_);
  CounterKey key = {executor, run_id};
  auto counter = counters_.find(key);
  if (counter == counters_.end()) {
    return absl::InternalError("Execution counter not initialized");
  }

  return &counter->second;
}

absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>>
NvshmemP2PGetSourceTargetPairs(mlir::DictionaryAttr frontend_attributes) {
  VLOG(3) << "Extracting source/target pairs from frontend attributes";
  mlir::StringAttr src_dst_string = frontend_attributes.getAs<mlir::StringAttr>(
      kSendRecvSourceTargetPairsAttr);
  if (!src_dst_string) {
    VLOG(3) << "No source/target pairs found in frontend attributes";
    return absl::AbortedError(
        absl::StrCat("expecting send/recv op with string attribute ",
                     kSendRecvSourceTargetPairsAttr));
  }
  VLOG(3) << "Found source/target pairs string: " << src_dst_string.str();
  TF_ASSIGN_OR_RETURN(std::vector<ReplicaGroup> replica_groups,
                      ParseReplicaGroupsOnly(src_dst_string.str()));
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs;
  source_target_pairs.reserve(replica_groups.size());
  for (const ReplicaGroup& replica_group : replica_groups) {
    TF_RET_CHECK(replica_group.replica_ids_size() == 2);
    source_target_pairs.emplace_back(replica_group.replica_ids(0),
                                     replica_group.replica_ids(1));
    VLOG(3) << "Added source/target pair: " << replica_group.replica_ids(0)
            << " -> " << replica_group.replica_ids(1);
  }
  return source_target_pairs;
}

NvshmemP2PConfig GetNvshmemP2PConfigForPutGet(
    const HloSendRecvInstruction* instr, const Shape& shape,
    int64_t replica_count, int64_t partition_count) {
  VLOG(3) << "Creating P2P config for instruction: " << instr->name();
  NvshmemP2PConfig p2p_config;
  auto& config = p2p_config.config;

  config.operand_count = 1;
  config.operand_element_type.push_back(shape.element_type());
  config.SetCollectiveOpKindAndID(instr);
  config.group_mode = GetCollectiveOpGroupMode(
                          instr->channel_id().value_or(0) > 0, std::nullopt)
                          .value();
  VLOG(3) << "Group mode: " << CollectiveOpGroupModeToString(config.group_mode);

  const int64_t num_participants =
      config.group_mode == CollectiveOpGroupMode::kCrossReplica
          ? replica_count
          : partition_count;
  VLOG(3) << "Number of participants: " << num_participants;
  config.replica_groups.emplace_back();
  ReplicaGroup& replica_group = config.replica_groups.front();
  for (int i = 0; i < num_participants; ++i) {
    replica_group.add_replica_ids(i);
  }

  std::optional<std::string> source_target_pairs_string =
      instr->frontend_attributes().map().at(kSendRecvSourceTargetPairsAttr);

  if (!source_target_pairs_string.has_value()) {
    VLOG(3) << "No source/target pairs string found in frontend attributes";
    return p2p_config;
  }
  VLOG(3) << "Found source/target pairs string: "
          << *source_target_pairs_string;
  auto statusor = ParseReplicaGroupsOnly(*source_target_pairs_string);
  if (!statusor.ok()) {
    VLOG(3) << "Failed to parse replica groups: " << statusor.status();
    return p2p_config;
  }

  std::vector<ReplicaGroup> replica_groups = statusor.value();
  auto validation_it =
      instr->frontend_attributes().map().find(kSendRecvValidationAttr);
  NvshmemP2PConfig::ValidationKind validation_kind =
      NvshmemP2PConfig::ValidationKind::kValid;
  std::vector<ReplicaGroup> bounds;
  if (validation_it != instr->frontend_attributes().map().end()) {
    VLOG(3) << "Found validation attribute: " << validation_it->second;
    if (validation_it->second == "invalid") {
      validation_kind = NvshmemP2PConfig::ValidationKind::kInvalid;
      VLOG(3) << "Setting validation kind to kInvalid";
    } else {
      auto statusor_bounds = ParseReplicaGroupsOnly(validation_it->second);
      if (!statusor_bounds.ok() ||
          statusor_bounds.value().size() != replica_groups.size()) {
        VLOG(3) << "Failed to parse validation bounds or size mismatch";
        return p2p_config;
      }
      validation_kind = NvshmemP2PConfig::ValidationKind::kConditional;
      bounds = statusor_bounds.value();
      VLOG(3) << "Setting validation kind to kConditional";
    }
  }

  int i = 0;
  p2p_config.validation_kind = validation_kind;
  NvshmemP2PConfig::SourceTargetToBounds& source_target_to_bounds =
      p2p_config.source_target_to_bounds;
  for (const ReplicaGroup& replica_group : replica_groups) {
    int64_t source = replica_group.replica_ids(0);
    int64_t target = replica_group.replica_ids(1);

    p2p_config.id_to_source_target.insert({target, {}}).first->second.source =
        source;
    p2p_config.id_to_source_target.insert({source, {}}).first->second.target =
        target;
    VLOG(3) << "Added source/target mapping: " << source << " -> " << target;

    if (validation_kind == NvshmemP2PConfig::ValidationKind::kConditional) {
      const ReplicaGroup& bound = bounds[i];
      int64_t lower = bound.replica_ids(0);
      int64_t upper = bound.replica_ids(1);
      source_target_to_bounds[std::make_pair(source, target)] =
          std::make_pair(lower, upper);
      VLOG(3) << "Added conditional bounds for " << source << " -> " << target
              << ": [" << lower << ", " << upper << "]";
      i++;
    }
  }

  return p2p_config;
}

}  // namespace gpu
}  // namespace xla
