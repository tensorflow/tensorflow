/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/autotune_maps/conv_map_wrapper.h"

#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "tensorflow/core/util/autotune_maps/autotune_map.pb.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.pb.h"

namespace tensorflow {

/*static*/ absl::StatusOr<ConvMapWrapper> ConvMapWrapper::FromKeyAndValue(
    OpaqueKey key, OpaqueValue value) {
  ConvMapProto::Entry key_proto;
  if (!key_proto.ParseFromString(key)) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Could not parse the provided key");
  }

  ConvMapProto::Entry value_proto;
  if (!value_proto.ParseFromString(value)) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Could not parse the provided value");
  }

  ConvMapProto::Entry full_entry;
  *full_entry.mutable_key() = key_proto.key();
  *full_entry.mutable_value() = value_proto.value();
  return ConvMapWrapper(full_entry);
}

ConvMapWrapper::OpaqueKey ConvMapWrapper::Key() const {
  ConvMapProto::Entry entry;
  *entry.mutable_key() = conv_map_entry_.key();
  OpaqueKey serialized;
  CHECK(tsl::SerializeToStringDeterministic(entry, &serialized));  // Crash OK
  return serialized;
}

ConvMapWrapper::OpaqueValue ConvMapWrapper::Value() const {
  ConvMapProto::Entry entry;
  *entry.mutable_value() = conv_map_entry_.value();
  OpaqueValue serialized;
  CHECK(tsl::SerializeToStringDeterministic(entry, &serialized));  // Crash OK
  return serialized;
}

/*static*/ std::vector<ConvMapWrapper> ConvMapWrapper::ConvMapToWrappers(
    const ConvMapProto& autotune_results) {
  std::vector<ConvMapWrapper> wrappers;
  wrappers.reserve(autotune_results.kv_pairs_size());
  for (const auto& entry : autotune_results.kv_pairs()) {
    wrappers.push_back(ConvMapWrapper(entry));
  }
  return wrappers;
}

/*static*/ absl::StatusOr<ConvMapProto> ConvMapWrapper::ConvMapFromWrappers(
    const std::vector<ConvMapWrapper>& wrappers) {
  ConvMapProto conv_map_proto;
  for (const auto& wrapper : wrappers) {
    *conv_map_proto.add_kv_pairs() = wrapper.conv_map_entry_;
  }
  return conv_map_proto;
}

}  // namespace tensorflow
