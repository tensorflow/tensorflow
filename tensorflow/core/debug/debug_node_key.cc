/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/debug/debug_node_key.h"

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

const char* const DebugNodeKey::kMetadataFilePrefix = "_tfdbg_";

const char* const DebugNodeKey::kDeviceTag = "device_";

DebugNodeKey::DebugNodeKey(const string& device_name, const string& node_name,
                           const int32 output_slot, const string& debug_op)
    : device_name(device_name),
      node_name(node_name),
      output_slot(output_slot),
      debug_op(debug_op),
      debug_node_name(
          strings::StrCat(node_name, ":", output_slot, ":", debug_op)),
      device_path(DeviceNameToDevicePath(device_name)) {}

bool DebugNodeKey::operator==(const DebugNodeKey& other) const {
  return (device_name == other.device_name && node_name == other.node_name &&
          output_slot == other.output_slot && debug_op == other.debug_op);
}

bool DebugNodeKey::operator!=(const DebugNodeKey& other) const {
  return !((*this) == other);
}

const string DebugNodeKey::DeviceNameToDevicePath(const string& device_name) {
  return strings::StrCat(
      kMetadataFilePrefix, kDeviceTag,
      str_util::StringReplace(
          str_util::StringReplace(device_name, ":", "_", true), "/", ",",
          true));
}

}  // namespace tensorflow
