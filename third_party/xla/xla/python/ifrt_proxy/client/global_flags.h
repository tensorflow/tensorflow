/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_GLOBAL_FLAGS_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_GLOBAL_FLAGS_H_

#include <ostream>

namespace xla {
namespace ifrt {
namespace proxy {

// Flags that are set based on command-line options or environment variables.
// As of November 2024, the OSSed code does not actually have any mechanism
// to configure these flags (global_flags_oss.cc has default values that are
// compile-time constants); Google-internal code allows it to be configured from
// command-line options.
struct GlobalClientFlags {
  // Setting to true reverts to implementation from before Nov 2024, where
  // host buffer stores were issued synchronously and waited upon.
  // TODO(madthanu): Remove flag once there is confidence that the asynchronous
  // codepath works well.
  bool synchronous_host_buffer_store;

  // TODO(b/375021159): Implement faster is_delete without needing a hack.
  bool array_is_deleted_hack;
};

GlobalClientFlags* GetGlobalClientFlags();

inline std::ostream& operator<<(std::ostream& os, GlobalClientFlags flags) {
  return os << "xla::ifrt::proxy::GlobalClientFlags{"
            << "synchronous_host_buffer_store="
            << flags.synchronous_host_buffer_store << ","
            << "array_is_deleted_hack=" << flags.array_is_deleted_hack << "}";
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_GLOBAL_FLAGS_H_
