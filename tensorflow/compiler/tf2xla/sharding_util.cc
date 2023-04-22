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
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {
const char kDeviceSuffixReplicatedCore[] = "REPLICATED_CORE";
const char kShardingAttribute[] = "_XlaSharding";
const char kShardingOpAttribute[] = "sharding";
}  // namespace

namespace {
xla::OpMetadata CreateOpMetadata(const std::string& op_type,
                                 const std::string& op_name) {
  xla::OpMetadata metadata;
  metadata.set_op_type(op_type);
  metadata.set_op_name(op_name);
  return metadata;
}

void AssignOpMetadataToSharding(xla::OpSharding& sharding,
                                const string& op_type, const string& op_name) {
  auto metadata = CreateOpMetadata(op_type, op_name);
  if (sharding.type() == xla::OpSharding::TUPLE) {
    for (auto& sharding_element : *sharding.mutable_tuple_shardings()) {
      *sharding_element.add_metadata() = metadata;
    }
  } else {
    *sharding.add_metadata() = metadata;
  }
}

Status CoreOutOfRangeError(int core, int num_cores_per_replica) {
  return errors::InvalidArgument(
      "Invalid replicated core id: ", core,
      "; num_cores_per_replica=", num_cores_per_replica);
}
}  // namespace

StatusOr<absl::optional<xla::OpSharding>> ParseShardingFromDevice(
    const string& device_name, int num_cores_per_replica,
    absl::optional<xla::OpSharding> explicit_sharding,
    absl::optional<xla::OpMetadata> metadata) {
  if (device_name.empty()) {
    return explicit_sharding;
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
    auto sharding = xla::sharding_builder::AssignDevice(core);
    if (metadata.has_value()) {
      *sharding.add_metadata() = metadata.value();
    }
    return absl::optional<xla::OpSharding>(sharding);
  }
}

StatusOr<absl::optional<xla::OpSharding>> ParseShardingFromDevice(
    const NodeDef& node_def, int num_cores_per_replica, bool add_metadata) {
  const string& device_name = node_def.device();
  TF_ASSIGN_OR_RETURN(absl::optional<xla::OpSharding> sharding,
                      GetShardingFromNodeDef(node_def, add_metadata));
  return ParseShardingFromDevice(
      device_name, num_cores_per_replica, sharding,
      add_metadata ? absl::optional<xla::OpMetadata>(
                         CreateOpMetadata(node_def.op(), node_def.name()))
                   : absl::nullopt);
}

StatusOr<absl::optional<xla::OpSharding>> ParseShardingFromDevice(
    const Node& node, int num_cores_per_replica, bool add_metadata) {
  string device_name = node.assigned_device_name();
  if (device_name.empty()) {
    device_name = node.requested_device();
  }
  TF_ASSIGN_OR_RETURN(absl::optional<xla::OpSharding> sharding,
                      GetShardingFromNodeDef(node.def(), add_metadata));
  return ParseShardingFromDevice(
      device_name, num_cores_per_replica, sharding,
      add_metadata ? absl::optional<xla::OpMetadata>(
                         CreateOpMetadata(node.type_string(), node.name()))
                   : absl::nullopt);
}

StatusOr<absl::optional<xla::OpSharding>> ParseShardingFromEdgeSource(
    const Edge& edge, int num_cores_per_replica, bool add_metadata) {
  if (edge.src() == nullptr) {
    return tensorflow::errors::InvalidArgument(
        "Null src for ParseShardingFromEdgeSource edge=", edge.DebugString());
  }
  TF_ASSIGN_OR_RETURN(absl::optional<xla::OpSharding> sharding,
                      ParseShardingFromDevice(
                          *edge.src(), num_cores_per_replica, add_metadata));
  if (sharding.has_value() &&
      sharding.value().type() == xla::OpSharding::TUPLE) {
    if (edge.src_output() < 0 ||
        edge.src_output() >= sharding.value().tuple_shardings_size()) {
      return tensorflow::errors::InvalidArgument(
          "Tuple index out of bound: edge=", edge.DebugString(),
          " sharding=", sharding->DebugString());
    }
    absl::optional<xla::OpSharding> subsharding =
        sharding.value().tuple_shardings(edge.src_output());
    return subsharding;
  }
  return sharding;
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

namespace {

StatusOr<absl::optional<xla::OpSharding>> GetShardingFromNodeDefInternal(
    const NodeDef& node_def, bool add_metadata, const char* attribute) {
  if (!HasNodeAttr(node_def, attribute)) {
    return absl::optional<xla::OpSharding>();
  }
  string value;
  xla::OpSharding sharding;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, attribute, &value));
  if (!sharding.ParseFromString(value)) {
    return xla::InvalidArgument(
        "Experimental %s attribute was not a valid encoded xla::OpSharding "
        "proto.",
        attribute);
  }
  if (add_metadata) {
    AssignOpMetadataToSharding(sharding, node_def.op(), node_def.name());
  }
  return absl::optional<xla::OpSharding>(sharding);
}

}  // namespace

xla::StatusOr<absl::optional<xla::OpSharding>> GetShardingFromNodeDef(
    const NodeDef& node_def, bool add_metadata) {
  if (node_def.op() == "XlaSharding") {
    TF_ASSIGN_OR_RETURN(auto sharding,
                        GetShardingFromNodeDefInternal(node_def, add_metadata,
                                                       kShardingOpAttribute));
    if (sharding.has_value()) {
      return sharding;
    }
  }
  return GetShardingFromNodeDefInternal(node_def, add_metadata,
                                        kShardingAttribute);
}

}  // namespace tensorflow
