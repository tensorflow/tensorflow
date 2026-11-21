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

#include "xla/backends/profiler/cpu/runtime_metadata.h"

#include <stdlib.h>
#include <unistd.h>

#include <string>

#include "absl/strings/match.h"
#include "xla/backends/profiler/util/metadata_registry.h"

namespace xla {
namespace profiler {

void CollectRuntimeMetadata() {
  if (const char* xla_flags = getenv("XLA_FLAGS")) {
    SetProfilerMetadata("xla_flags", xla_flags);
  }
  char** env_list = environ;
  for (char** env = env_list; *env != nullptr; ++env) {
    std::string env_str = *env;
    size_t eq_pos = env_str.find('=');
    if (eq_pos == std::string::npos) continue;
    std::string key = env_str.substr(0, eq_pos);
    if (absl::StartsWith(key, "JAX_") || absl::StartsWith(key, "NCCL_")) {
      SetProfilerMetadata(key, env_str.substr(eq_pos + 1));
    }
  }
}

}  // namespace profiler
}  // namespace xla
