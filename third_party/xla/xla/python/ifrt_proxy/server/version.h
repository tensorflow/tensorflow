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

namespace xla {
namespace ifrt {
namespace proxy {

// LINT.IfChange
// TODO(b/296144873): Document the version upgrade policy.
inline constexpr int kServerMinVersion = 1;
inline constexpr int kServerMaxVersion = 3;
// LINT.ThenChange(//tensorflow/compiler/xla/python/ifrt_proxy/common/VERSION.md)

// Returns a version that both the client and the server support, or an error if
// there is no such a version.
absl::StatusOr<int> ChooseVersion(int client_min_version,
                                  int client_max_version,
                                  int server_min_version = kServerMinVersion,
                                  int server_max_version = kServerMaxVersion);

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_VERSION_H_
