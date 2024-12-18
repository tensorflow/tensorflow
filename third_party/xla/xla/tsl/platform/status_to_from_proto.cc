/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/tsl/platform/status_to_from_proto.h"

#include <string>

#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/tsl/protobuf/status.pb.h"
#include "tsl/platform/status.h"

namespace tsl {

tensorflow::StatusProto StatusToProto(const absl::Status& s) {
  tensorflow::StatusProto status_proto;
  if (s.ok()) {
    return status_proto;
  }

  status_proto.set_code(static_cast<tsl::error::Code>(s.code()));
  if (!s.message().empty()) {
    status_proto.set_message(std::string(s.message()));
  }

  s.ForEachPayload(
      [&status_proto](absl::string_view type_url, absl::Cord value) {
        status_proto.mutable_payload()->insert(
            {std::string(type_url), std::string(value)});
      });
  return status_proto;
}

#if defined(PLATFORM_GOOGLE)
absl::Status StatusFromProto(const tensorflow::StatusProto& proto,
                             absl::SourceLocation loc) {
  if (proto.code() == tensorflow::error::OK) {
    return absl::OkStatus();
  }
  absl::Status s(static_cast<absl::StatusCode>(proto.code()), proto.message(),
                 loc);
  for (const auto& [key, payload] : proto.payload()) {
    s.SetPayload(key, absl::Cord(payload));
  }
  return s;
}
#else
Status StatusFromProto(const tensorflow::StatusProto& proto) {
  if (proto.code() == tensorflow::error::OK) {
    return OkStatus();
  }
  Status s(static_cast<absl::StatusCode>(proto.code()), proto.message());
  for (const auto& [key, payload] : proto.payload()) {
    s.SetPayload(key, absl::Cord(payload));
  }
  return s;
}
#endif

}  // namespace tsl
