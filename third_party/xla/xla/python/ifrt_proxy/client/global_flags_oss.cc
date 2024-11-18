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

#include <cstdlib>
#include <string>

#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "xla/python/ifrt_proxy/client/global_flags.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

bool GetBoolFromEnv(const char* key) {
  if (const char* valptr = std::getenv(key)) {
    std::string val(valptr);
    bool result;
    QCHECK(absl::SimpleAtob(val, &result)) << " " << key << ": '" << val << "'";
    return result;
  }
  return false;
}

}  // namespace

static GlobalClientFlags DefaultGlobalClientFlags() {
  GlobalClientFlags result;
  result.synchronous_host_buffer_store = false;
  result.array_is_deleted_hack =
      GetBoolFromEnv("IFRT_PROXY_ARRAY_IS_DELETED_HACK");
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
