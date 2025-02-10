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

#include "xla/python/ifrt_proxy/server/version.h"

#include <algorithm>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace xla {
namespace ifrt {
namespace proxy {

absl::StatusOr<int> ChooseVersion(int client_min_version,
                                  int client_max_version,
                                  int server_min_version,
                                  int server_max_version) {
  const int version = std::min(server_max_version, client_max_version);

  LOG(INFO) << "IFRT proxy: ChooseVersion(client_min_version="
            << client_min_version
            << ", client_max_version=" << client_max_version
            << ", server_min_version=" << server_min_version
            << ", server_max_version=" << server_max_version << ") is "
            << version;

  if (version < server_min_version || version < client_min_version) {
    return absl::InvalidArgumentError(absl::StrCat(
        "IFRT Proxy client and server failed to agree on the "
        "protocol version; supported versions: client = [",
        client_min_version, ", ", client_max_version, "], server = [",
        server_min_version, ", ", server_max_version, "]"));
  }

  return version;
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
