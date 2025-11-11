/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_RUNTIME_INTRINSICS_H_
#define XLA_BACKENDS_GPU_RUNTIME_RUNTIME_INTRINSICS_H_

#include "absl/strings/string_view.h"

namespace xla {

inline constexpr absl::string_view kXlaGpuAssertCustomCallTag =
    "__xla_gpu_assert";

inline constexpr absl::string_view kXlaGpuDebugPrintCustomCallTag =
    "__xla_gpu_debug_print";

inline constexpr absl::string_view kXlaGpuAppendToFileCustomCallTag =
    "__xla_gpu_append_to_file";

}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_RUNTIME_INTRINSICS_H_
