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
#include "tensorflow/compiler/tf2xla/sharding_util.h"

#include "absl/strings/match.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {
const char kDeviceSuffixReplicatedCore[] = "REPLICATED_CORE";
const char kShardingAttribute[] = "_XlaSharding";
}  // namespace

namespace {
xla::StatusOr<absl::optional<xla::OpSharding>> GetShardingFromNodeDef(
    const NodeDef& node_def) {
  if (!HasNodeAttr(node_def, kShardingAttribute)) {
    return absl::optional<xla::OpSharding>();
  }
  string value;
  xla::OpSharding sharding;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, kShardingAttribute, &value));
  if (!sharding.ParseFromString(value)) {
    return xla::InvalidArgument(
        "Experimental _XlaSharding attribute was not a valid encoded "
        "xla::OpSharding proto.");
  }
  return absl::optional<xla::OpSharding>(sharding);
}

Status CoreOutOfRangeError(int core, int num_cores_per_replica) {
  return errors::InvalidArgument(
      "Invalid replicated core id: ", core,
      "; num_cores_per_replica=", num_cores_per_replica);
}
}  // namespace

xla::StatusOr<absl::optional<xla::OpSharding>> ParseShardingFromDevice(
    const string& device_name, int num_cores_per_replica,
    absl::optional<xla::OpSharding> explicit_sharding) {
  if (device_name.empty()) {
    return absl::optional<xla::OpSharding>();
  }
  DeviceNameUtils::ParsedName parsed_device;
  if (!DeviceNameUtils::ParseFullName(device_name, &parsed_device)) {
    return errors::InvalidArgument("Malformed assigned device '", device_name,
                                   "'");
  }

  if (explicit_sharding.has_value()) {
    return explicit_sharding;
  } else if (!parsed_device.has_type || !parsed_device.has_id ||
             !absl::StrContains(parsed_device.type,
                                kDeviceSuffixReplicatedCore)) {
    return absl::optional<xla::OpSharding>();
  } else {
    const int core = parsed_device.id;
    if (core < 0 || core >= num_cores_per_replica) {
      return CoreOutOfRangeError(core, num_cores_per_replica);
    }
    return absl::optional<xla::OpSharding>(
        xla::sharding_builder::AssignDevice(core));
  }
}

xla::StatusOr<absl::optional<xla::OpSharding>> ParseShardingFromDevice(
    const NodeDef& node_def, int num_cores_per_replica) {
  const string& device_name = node_def.device();
  TF_ASSIGN_OR_RETURN(absl::optional<xla::OpSharding> sharding,
                      GetShardingFromNodeDef(node_def));
  return ParseShardingFromDevice(device_name, num_cores_per_replica, sharding);
}

xla::StatusOr<absl::optional<xla::OpSharding>> ParseShardingFromDevice(
    const Node& node, int num_cores_per_replica) {
  string device_name = node.assigned_device_name();
  if (device_name.empty()) {
    device_name = node.requested_device();
  }
  TF_ASSIGN_OR_RETURN(absl::optional<xla::OpSharding> sharding,
                      GetShardingFromNodeDef(node.def()));
  return ParseShardingFromDevice(device_name, num_cores_per_replica, sharding);
}

void SetShardingDeviceAssignmentFromNode(const Node& src, Node* dst) {
  string device_name = src.assigned_device_name();
  if (device_name.empty()) {
    device_name = src.requested_device();
  }
  dst->set_assigned_device_name(device_name);
  if (const AttrValue* attr = src.attrs().Find(kShardingAttribute)) {
    dst->AddAttr(kShardingAttribute, *attr);
  }
}

}  // namespace tensorflow
