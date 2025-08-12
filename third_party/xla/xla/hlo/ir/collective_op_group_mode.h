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

#ifndef XLA_HLO_IR_COLLECTIVE_OP_GROUP_MODE_H_
#define XLA_HLO_IR_COLLECTIVE_OP_GROUP_MODE_H_

#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/xla_data.pb.h"

namespace xla {

// There are broadly 4 modes that collective communication ops use to describe
// which sets of devices are participating with a given device in the operation.
// These modes are determined by the values of channel_id (optional) and
// use_global_device_ids (optional). The modes are as follows:
//
// kCrossReplica:
//    implied by: no channel id, use_global_device_ids = false, or
//                no channel_id, no use_global_device_ids:
//    replica_groups contain replica_id, group contains all replicas for the
//    current partition
//
// kCrossPartition:
//    implied by: channel_id is set, no use_global_device_ids:
//    replica_groups contain partition_id, group contains all partitions for the
//    current replica.
//
// kCrossReplicaAndPartition:
//    implied by: channel_id is set, use_global_device_ids = false:
//    replica_groups contain replica_id, group contains all replicas for all
//    partitions (as opposed to just current partition).
//
// kFlattenedID:
//    implied by: channel_id is set, use_global_device_ids = true:
//    replica_groups contain flattened-ids, group contains devices that are
//    listed in the flattened-id list.
//
// Rest of the combinations are invalid.
//
// Since the actual value of channel_id does not matter, we use a bool argument
// `has_channel_id`, and optional<bool> for use_global_device_ids.
// Note that use_global_device_ids true requires channel_id to be set as well.
// Additionally, if use_global_device_ids = true, replica groups cannot be
// empty (verified in the HLO verifier).
enum class CollectiveOpGroupMode {
  kCrossReplica,
  kCrossPartition,
  kCrossReplicaAndPartition,
  kFlattenedID,
};

absl::string_view CollectiveOpGroupModeToString(
    CollectiveOpGroupMode group_mode);

absl::StatusOr<CollectiveOpGroupMode> StringToCollectiveOpGroupMode(
    absl::string_view name);

CollectiveOpGroupModeProto CollectiveOpGroupModeToProto(
    CollectiveOpGroupMode group_mode);

absl::StatusOr<CollectiveOpGroupMode> CollectiveOpGroupModeFromProto(
    CollectiveOpGroupModeProto proto);

// Returns the group formation mode implied by (a) whether the operation has
// channel_id and (b) if it has use_global_device_ids and if yes, its value.
absl::StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    bool has_channel_id, std::optional<bool> use_global_device_ids);

}  // namespace xla

#endif  // XLA_HLO_IR_COLLECTIVE_OP_GROUP_MODE_H_
