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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

static const char DEVICE_SUFFIX_REPLICATED_CORE[] = "REPLICATED_CORE";

static Status CoreOutOfRangeError(int core, int num_cores_per_replica) {
  return errors::InvalidArgument(
      "Invalid replicated core id: ", core,
      "; num_cores_per_replica=", num_cores_per_replica);
}

xla::StatusOr<tensorflow::gtl::optional<xla::OpSharding>>
ParseShardingFromDevice(const string& device_name, int num_cores_per_replica) {
  if (device_name.empty()) {
    return tensorflow::gtl::optional<xla::OpSharding>();
  }

  DeviceNameUtils::ParsedName parsed_device;
  if (!DeviceNameUtils::ParseFullName(device_name, &parsed_device)) {
    return errors::InvalidArgument("Malformed assigned device '", device_name,
                                   "'");
  }
  if (!parsed_device.has_type ||
      !StringPiece(parsed_device.type)
           .ends_with(DEVICE_SUFFIX_REPLICATED_CORE)) {
    return tensorflow::gtl::optional<xla::OpSharding>();
  } else {
    const int core = parsed_device.id;
    if (core < 0 || core >= num_cores_per_replica) {
      return CoreOutOfRangeError(core, num_cores_per_replica);
    }
    return tensorflow::gtl::optional<xla::OpSharding>(
        xla::ShardingBuilder::AssignDevice(core));
  }
}

xla::StatusOr<tensorflow::gtl::optional<xla::OpSharding>>
ParseShardingFromDevice(const Node& node, int num_cores_per_replica) {
  string device_name = node.assigned_device_name();
  if (device_name.empty()) {
    device_name = node.requested_device();
  }
  return ParseShardingFromDevice(device_name, num_cores_per_replica);
}
void SetShardingDeviceAssignmentFromNode(const Node& src, Node* dst) {
  string device_name = src.assigned_device_name();
  if (device_name.empty()) {
    device_name = src.requested_device();
  }
  dst->set_assigned_device_name(device_name);
}

}  // namespace tensorflow
