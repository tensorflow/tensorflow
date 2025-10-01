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

#ifndef XLA_PYTHON_IFRT_PROXY_COMMON_TRANSFER_UTIL_H_
#define XLA_PYTHON_IFRT_PROXY_COMMON_TRANSFER_UTIL_H_

#include <optional>
#include <string>

namespace xla {
namespace ifrt {
namespace proxy {

// Returns the file path that may be used by HostBuffer implementations for
// sending a buffer with `handle`. Returns `std::nullopt` if such file-paths
// have not been configured via startup options.
std::optional<std::string> LargeTransferFilePath(int handle);

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_COMMON_TRANSFER_UTIL_H_
