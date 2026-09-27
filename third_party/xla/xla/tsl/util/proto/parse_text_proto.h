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

#include <type_traits>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

namespace tsl::proto_testing {
namespace internal {

// Helper class that automatically converts to the target protobuf message type.
class ParseTextProtoHelper {
 public:
  explicit ParseTextProtoHelper(absl::string_view text_proto)
      : text_proto_(text_proto) {}

  template <typename T,
            typename = std::enable_if_t<std::is_base_of_v<google::protobuf::Message, T> &&
                                        std::is_same_v<T, std::decay_t<T>>>>
  operator T() const {  // NOLINT(google-explicit-constructor)
    std::decay_t<T> proto;
    CHECK(google::protobuf::TextFormat::ParseFromString(text_proto_, &proto))
        << "Failed to parse text proto:\n"
        << text_proto_;
    return proto;
  }

 private:
  absl::string_view text_proto_;
};

}  // namespace internal

// Parses the given `text_proto` into a protobuf message.
//
// This is a test-only utility that is equivalent to the Google internal
// `ParseTextProtoOrDie`, but works in OSS. Note that you must explicitly
// specify the template argument, unlike in the internal version, where the type
// can be inferred.
//
// The message type can be inferred from the context:
//   MyProto proto = ParseTextProtoOrDie(R"pb(...)pb");
//   AcceptsMyProto(ParseTextProtoOrDie(R"pb(...)pb"));
//
// Or explicitly specified if the context type is ambiguous (e.g. with `auto`):
//   auto proto = ParseTextProtoOrDie<MyProto>(R"pb(...)pb");
template <typename T = internal::ParseTextProtoHelper>
inline T ParseTextProtoOrDie(absl::string_view text_proto) {
  if constexpr (std::is_same_v<T, internal::ParseTextProtoHelper>) {
    return internal::ParseTextProtoHelper(text_proto);
  } else {
    std::decay_t<T> proto;
    CHECK(google::protobuf::TextFormat::ParseFromString(text_proto, &proto))
        << "Failed to parse text proto:\n"
        << text_proto;
    return proto;
  }
}

}  // namespace tsl::proto_testing

#endif  // XLA_TSL_UTIL_PROTO_PARSE_TEXT_PROTO_H_
