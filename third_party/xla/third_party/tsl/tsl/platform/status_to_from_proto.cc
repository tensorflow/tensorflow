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
#include "tsl/platform/status_to_from_proto.h"

#include <string>

#include "tsl/platform/status.h"
#include "tsl/protobuf/error_codes.pb.h"
#include "tsl/protobuf/status.pb.h"

namespace tsl {

tensorflow::StatusProto StatusToProto(const Status& s) {
  tensorflow::StatusProto status_proto;
  if (s.ok()) {
    return status_proto;
  }

  status_proto.set_code(static_cast<tsl::error::Code>(s.code()));
  if (!s.message().empty()) {
    status_proto.set_message(std::string(s.message()));
  }
  return status_proto;
}

#if defined(PLATFORM_GOOGLE)
Status StatusFromProto(const tensorflow::StatusProto& proto,
                       absl::SourceLocation loc) {
  if (proto.code() == tensorflow::error::OK) {
    return OkStatus();
  }
  return Status(static_cast<absl::StatusCode>(proto.code()), proto.message(),
                loc);
}
#else
Status StatusFromProto(const tensorflow::StatusProto& proto) {
  if (proto.code() == tensorflow::error::OK) {
    return OkStatus();
  }
  return Status(static_cast<absl::StatusCode>(proto.code()), proto.message());
}
#endif

}  // namespace tsl
