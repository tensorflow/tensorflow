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

#include "xla/stream_executor/cuda/ptx_compiler_support.h"

#include "absl/strings/match.h"

namespace stream_executor {
bool IsLibNvPtxCompilerSupported() { return LIBNVPTXCOMPILER_SUPPORT; }

bool IsPtxRegisterAllocationError(absl::string_view str) {
  return absl::StrContains(str, "ptxas fatal") &&
         (absl::StrContains(str, "Register allocation failed") ||
          absl::StrContains(str, "Insufficient registers"));
}
}  // namespace stream_executor
