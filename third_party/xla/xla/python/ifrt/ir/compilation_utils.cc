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

#include "xla/python/ifrt/ir/compilation_utils.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/client/executable_build_options.h"
#include "xla/pjrt/pjrt_executable.h"

namespace xla {
namespace ifrt {

void SetReservedHbmBytes(xla::ExecutableBuildOptions& exec_build_options,
                         int64_t reserved_hbm_bytes) {}

absl::Status SetStrictMemoryReservation(absl::string_view program_name,
                                        int64_t device_memory,
                                        xla::CompileOptions& options) {
  return absl::OkStatus();
}

}  // namespace ifrt
}  // namespace xla
