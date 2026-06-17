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

#ifndef XLA_TSL_UTIL_PROTO_PARSE_TEXT_PROTO_H_
#define XLA_TSL_UTIL_PROTO_PARSE_TEXT_PROTO_H_

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

namespace tsl::proto_testing {

// Parses the given `text_proto` into a protobuf message of type `T`.
//
// `T` must be a protobuf message type.
//
// This is a test-only utility that is equivalent to the Google internal
// `ParseTextProtoOrDie`, but works in OSS. Note that you must explicitly
// specify the template argument, unlike in the internal version, where the type
// can be inferred.
//
// Usage: auto proto = ParseTextProtoOrDie<MyProto>(R"pb(...)pb");
template <typename T>
inline T ParseTextProtoOrDie(absl::string_view text_proto) {
  T proto;
  CHECK(google::protobuf::TextFormat::ParseFromString(text_proto, &proto))
      << "Failed to parse text proto";
  return proto;
}

}  // namespace tsl::proto_testing

#endif  // XLA_TSL_UTIL_PROTO_PARSE_TEXT_PROTO_H_
