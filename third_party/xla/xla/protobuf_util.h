/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_PROTOBUF_UTIL_H_
#define XLA_PROTOBUF_UTIL_H_

#include <cstddef>
#include <functional>
#include <string>

#include "absl/status/status.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace protobuf_util {

// Returns true if m1 and m2 have the same serialization.
//
// WARNING: Protobuf serialization is not guaranteed to be stable. Use this ONLY
// IF you are SURE that you want this form of equality.
//
// In g3 tests, prefer matchers like ::testing::EqualsProto. In OSS tests,
// prefer ::tsl::proto_testing::EqualsProto. These have more precise semantics
// and will give far better error messages.
[[nodiscard]] bool HaveSameSerialization(const tsl::protobuf::Message& m1,
                                         const tsl::protobuf::Message& m2);

// Return the hash of the message "m", based on its serialization.
//
// WARNING: This uses the same serialization approach used by
// HaveSameSerialization, so the WARNING for that function applies here.
[[nodiscard]] size_t ProtobufHashBySerialization(
    const tsl::protobuf::Message& m);

// Wrappers for HaveSameSerialization() so that we can use protos in containers
// that require equality.
//
// WARNING: This uses the same serialization approach used by
// HaveSameSerialization, so the WARNING for that function applies here.
class HaveSameSerializationFunctor {
 public:
  [[nodiscard]] bool operator()(const tsl::protobuf::Message& m1,
                                const tsl::protobuf::Message& m2) const {
    return HaveSameSerialization(m1, m2);
  }
};

// Functor for hashing a protobuf message by its serialization.
//
// WARNING: This uses the same serialization approach used by
// HaveSameSerialization, so the WARNING for that function applies here.
class ProtobufHashBySerializationFunctor {
 public:
  [[nodiscard]] size_t operator()(const tsl::protobuf::Message& m) const {
    return ProtobufHashBySerialization(m);
  }
};

}  // namespace protobuf_util
}  // namespace xla

#endif  // XLA_PROTOBUF_UTIL_H_
