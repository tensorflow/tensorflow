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

#include "xla/hlo/ir/collective_op_group_mode.h"

#include <optional>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/util.h"

namespace xla {
namespace {

struct CollectiveOpGroupModeInfo {
  CollectiveOpGroupMode mode;
  absl::string_view name;
};

const CollectiveOpGroupModeInfo kGroupModeInfos[] = {
    {CollectiveOpGroupMode::kCrossReplica, "cross_replica"},
    {CollectiveOpGroupMode::kCrossPartition, "cross_partition"},
    {CollectiveOpGroupMode::kCrossReplicaAndPartition,
     "cross_replica_and_partition"},
    {CollectiveOpGroupMode::kFlattenedID, "flattened_id"},
};

}  // namespace

absl::string_view CollectiveOpGroupModeToString(
    CollectiveOpGroupMode group_mode) {
  for (const CollectiveOpGroupModeInfo& info : kGroupModeInfos) {
    if (info.mode == group_mode) {
      return info.name;
    }
  }
  CHECK(false) << "Unknown collective op group mode: "
               << static_cast<int>(group_mode);
}

absl::StatusOr<CollectiveOpGroupMode> StringToCollectiveOpGroupMode(
    absl::string_view name) {
  for (const CollectiveOpGroupModeInfo& info : kGroupModeInfos) {
    if (info.name == name) {
      return info.mode;
    }
  }
  return InvalidArgument("Invalid collective op group mode: %s", name);
}

// Returns the group formation mode implied by (a) whether the operation has
// channel_id and (b) if it has use_global_device_ids and if yes, its value.
absl::StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    bool has_channel_id, std::optional<bool> use_global_device_ids) {
  if (!has_channel_id) {
    if (use_global_device_ids.has_value() && *use_global_device_ids) {
      return InvalidArgument(
          "Cannot have use_global_device_ids=true without channel_id");
    }
    return CollectiveOpGroupMode::kCrossReplica;
  }
  if (!use_global_device_ids.has_value()) {
    return CollectiveOpGroupMode::kCrossPartition;
  }
  if (!*use_global_device_ids) {
    return CollectiveOpGroupMode::kCrossReplicaAndPartition;
  }
  return CollectiveOpGroupMode::kFlattenedID;
}

}  // namespace xla
