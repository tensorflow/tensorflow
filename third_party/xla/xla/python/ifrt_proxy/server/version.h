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

#ifndef XLA_PYTHON_IFRT_PROXY_SERVER_VERSION_H_
#define XLA_PYTHON_IFRT_PROXY_SERVER_VERSION_H_

#include "absl/status/statusor.h"
#include "xla/python/ifrt/serdes_any_version_accessor.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt_proxy/common/versions.h"

namespace xla {
namespace ifrt {
namespace proxy {

// Returns a protocol version that both the client and the server support, or an
// error if there is no such a version.
absl::StatusOr<int> ChooseProtocolVersion(
    int client_min_version, int client_max_version,
    int server_min_version = protocol_version::kServerMin,
    int server_max_version = protocol_version::kServerMax);

// Returns an IFRT SerDes version that both the client and the server support,
// or an error if there is no such a version.
absl::StatusOr<SerDesVersionNumber> ChooseIfrtSerdesVersionNumber(
    SerDesVersionNumber client_min_version_number,
    SerDesVersionNumber client_max_version_number,
    SerDesVersionNumber server_min_version_number =
        SerDesAnyVersionAccessor::GetMinimum().version_number(),
    SerDesVersionNumber server_max_version_number =
        SerDesVersion::current().version_number());

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_VERSION_H_
