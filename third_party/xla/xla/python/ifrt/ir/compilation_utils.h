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

#ifndef XLA_PYTHON_IFRT_IR_COMPILATION_UTILS_H_
#define XLA_PYTHON_IFRT_IR_COMPILATION_UTILS_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/client/executable_build_options.h"
#include "xla/pjrt/pjrt_executable.h"

namespace xla {
namespace ifrt {

// Sets the reserved HBM bytes in the executable build options.
void SetReservedHbmBytes(xla::ExecutableBuildOptions& exec_build_options,
                         int64_t reserved_hbm_bytes);

// Sets the strict memory reservation in the compile options.
absl::Status SetStrictMemoryReservation(absl::string_view program_name,
                                        int64_t device_memory,
                                        xla::CompileOptions& options);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_COMPILATION_UTILS_H_
