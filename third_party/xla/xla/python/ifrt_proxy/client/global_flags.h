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

#include <cstdint>
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
  // Zero or negative values are interpreted as no maximum.
  int grpc_max_ongoing_host_buffer_stores;
  int grpc_max_ongoing_host_buffer_lookups;

  int64_t grpc_large_transfer_optimization_threshold_bytes;
};

GlobalClientFlags* GetGlobalClientFlags();

inline std::ostream& operator<<(std::ostream& os, GlobalClientFlags flags) {
  return os << "xla::ifrt::proxy::GlobalClientFlags{"
            << "grpc_max_ongoing_host_buffer_stores="
            << flags.grpc_max_ongoing_host_buffer_stores << ","
            << "grpc_max_ongoing_host_buffer_lookups="
            << flags.grpc_max_ongoing_host_buffer_lookups << ","
            << "grpc_large_transfer_optimization_threshold_bytes="
            << flags.grpc_large_transfer_optimization_threshold_bytes << "}";
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_GLOBAL_FLAGS_H_
