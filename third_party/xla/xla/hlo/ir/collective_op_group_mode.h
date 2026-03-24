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


absl::string_view CollectiveOpGroupModeToString(
    CollectiveOpGroupMode group_mode);

absl::StatusOr<CollectiveOpGroupMode> StringToCollectiveOpGroupMode(
    absl::string_view name);

// Returns the group formation mode implied by (a) whether the operation has
// channel_id and (b) if it has use_global_device_ids and if yes, its value.
absl::StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    bool has_channel_id, std::optional<bool> use_global_device_ids);

}  // namespace xla

#endif  // XLA_HLO_IR_COLLECTIVE_OP_GROUP_MODE_H_
