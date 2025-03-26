/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_TF2XLA_SHARDING_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_SHARDING_UTIL_H_

#include <string>

#include "xla/hlo/builder/sharding_builder.h"
#include "xla/status_macros.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Parses the op sharding from the 'replicated core' device_name <device_name>.
// Returns an error:
// - if the device name is invalid.
// - the core is parsed and is out of the range [0, num_cores_per_replica).
//
// Otherwise, returns either:
// - explicit_sharding if explicit_sharding.has_value()
// - a non-value if there is no assigned core or
// - a sharding set as per xla::sharding_builder::AssignDevice.
absl::StatusOr<std::optional<xla::OpSharding>> ParseShardingFromDevice(
    const string& device_name, int num_cores_per_replica,
    std::optional<xla::OpSharding> explicit_sharding = std::nullopt,
    std::optional<xla::OpMetadata> metadata = std::nullopt);

absl::StatusOr<std::optional<xla::OpSharding>> ParseShardingFromDevice(
    const Node& node, int num_cores_per_replica, bool add_metadata);

absl::StatusOr<std::optional<xla::OpSharding>> ParseShardingFromDevice(
    const NodeDef& node_def, int num_cores_per_replica, bool add_metadata);

absl::StatusOr<std::optional<xla::OpSharding>> ParseShardingFromEdgeSource(
    const Edge& edge, int num_cores_per_replica, bool add_metadata);

void SetShardingDeviceAssignmentFromNode(const Node& src, Node* dst);

// Get sharding inforamtion from node.
absl::StatusOr<std::optional<xla::OpSharding>> GetShardingFromNodeDef(
    const NodeDef& node_def, bool add_metadata);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_SHARDING_UTIL_H_
