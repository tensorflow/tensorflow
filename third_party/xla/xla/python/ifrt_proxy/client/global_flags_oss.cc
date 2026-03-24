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

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "xla/python/ifrt_proxy/client/global_flags.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

bool GetBoolFromEnv(const char* key, bool default_value) {
  if (const char* valptr = std::getenv(key)) {
    std::string val(valptr);
    bool result;
    QCHECK(absl::SimpleAtob(val, &result)) << " " << key << ": '" << val << "'";
    return result;
  }
  return default_value;
}

template <typename IntType>
IntType GetIntFromEnv(const char* key, IntType default_value) {
  if (const char* valptr = std::getenv(key)) {
    std::string val(valptr);
    IntType result;
    QCHECK(absl::SimpleAtoi(val, &result)) << " " << key << ": '" << val << "'";
    return result;
  }
  return default_value;
}

}  // namespace

static GlobalClientFlags DefaultGlobalClientFlags() {
  GlobalClientFlags result;
  result.grpc_max_ongoing_host_buffer_stores =
      GetIntFromEnv<int>("IFRT_PROXY_GRPC_MAX_ONGOING_HOST_BUFFER_STORES", 0);
  result.grpc_max_ongoing_host_buffer_lookups =
      GetIntFromEnv<int>("IFRT_PROXY_GRPC_MAX_ONGOING_HOST_BUFFER_LOOKUPS", 0);
  result.grpc_large_transfer_optimization_threshold_bytes =
      GetIntFromEnv<int64_t>(
          "IFRT_PROXY_GRPC_LARGE_TRANSFER_OPTIMIZATION_THRESHOLD_BYTES",
          std::numeric_limits<int64_t>::max());
  return result;
};

GlobalClientFlags* GetGlobalClientFlags() {
  static GlobalClientFlags* result =
      absl::IgnoreLeak(new GlobalClientFlags(DefaultGlobalClientFlags()));
  return result;
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
