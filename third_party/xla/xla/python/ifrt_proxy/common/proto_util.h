/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_COMMON_PROTO_UTIL_H_
#define XLA_PYTHON_IFRT_PROXY_COMMON_PROTO_UTIL_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"

namespace xla {
namespace ifrt {
namespace proxy {

// Makes an IfrtResponse proto with the given metadata.
std::unique_ptr<IfrtResponse> NewIfrtResponse(
    uint64_t op_id, absl::Status status = absl::OkStatus());

// Converts an `absl::string_view` into a type that is appropriate for doing
// `proto->set_string_field(...)`. This type can be absl::string_view in the
// newest versions of protobuf, but needs to be std::string for previous
// versions. (As of Feb 2024, OpenXLA uses an old version.)
#if defined(PLATFORM_GOOGLE)
inline absl::string_view AsProtoStringData(
    absl::string_view s ABSL_ATTRIBUTE_LIFETIME_BOUND) {
  return s;
}
#else
inline std::string AsProtoStringData(absl::string_view s) {
  LOG_FIRST_N(WARNING, 5) << "AsProtoStringData(): copying string_view->string";
  return std::string(s);
}
#endif

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_COMMON_PROTO_UTIL_H_
