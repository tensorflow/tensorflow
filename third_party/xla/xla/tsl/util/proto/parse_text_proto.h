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
namespace internal {

class ParseProtoHelper {
 public:
  explicit ParseProtoHelper(absl::string_view text_proto)
      : text_proto_(text_proto) {}

  template <typename T>
  operator T() const {  // NOLINT(google-explicit-constructor)
    T proto;
    CHECK(google::protobuf::TextFormat::ParseFromString(text_proto_, &proto))
        << "Failed to parse text proto";
    return proto;
  }

 private:
  absl::string_view text_proto_;
};

}  // namespace internal

// Parses the given `text_proto` into a protobuf message.
//
// When called without an explicit template argument, the return type is
// automatically inferred from the target type (e.g. `MyProto proto = ...` or
// when passed to a function expecting `MyProto` / `const MyProto&`).
//
// When the type cannot be inferred (e.g. with `auto`), explicitly specify the
// template argument or use `ParseTextOrDie<T>`.
//
// Usage:
//   MyProto proto = ParseTextProtoOrDie(R"pb(...)pb");
//   auto proto = ParseTextProtoOrDie<MyProto>(R"pb(...)pb");
//   auto proto = ParseTextOrDie<MyProto>(R"pb(...)pb");
template <typename T = internal::ParseProtoHelper>
inline T ParseTextProtoOrDie(absl::string_view text_proto) {
  return internal::ParseProtoHelper(text_proto);
}

// Parses the given `text_proto` into a protobuf message of type `T`.
//
// Usage: auto proto = ParseTextOrDie<MyProto>(R"pb(...)pb");
template <typename T>
inline T ParseTextOrDie(absl::string_view text_proto) {
  return ParseTextProtoOrDie<T>(text_proto);
}

}  // namespace tsl::proto_testing

#endif  // XLA_TSL_UTIL_PROTO_PARSE_TEXT_PROTO_H_
