/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_UTIL_SPLIT_PROTO_PROTO_FIELD_SIZE_UTILS_H_
#define XLA_UTIL_SPLIT_PROTO_PROTO_FIELD_SIZE_UTILS_H_

#include <string>

#include "absl/status/status.h"
#include "tsl/platform/protobuf.h"

namespace xla {

// Returns a formatted human-readable summary of the top `top_k` largest fields
// by serialized byte size in `message`.
//
// If `message` does not support reflection (cannot be downcast to
// `tsl::protobuf::Message`), returns a summary with the total byte size.
std::string GetTopKProtoFieldSizes(const tsl::protobuf::MessageLite& message,
                                   int top_k = 10);

// If `status` is `absl::StatusCode::kResourceExhausted`, returns a new
// ResourceExhausted status with `GetTopKProtoFieldSizes(record, top_k)`
// appended to `status.message()`. Otherwise returns `status` unchanged.
absl::Status AnnotateResourceExhaustedError(
    const absl::Status& status, const tsl::protobuf::MessageLite& record,
    int top_k = 10);

}  // namespace xla

#endif  // XLA_UTIL_SPLIT_PROTO_PROTO_FIELD_SIZE_UTILS_H_
