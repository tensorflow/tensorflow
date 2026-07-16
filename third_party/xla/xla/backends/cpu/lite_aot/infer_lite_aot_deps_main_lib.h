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

#ifndef XLA_BACKENDS_CPU_LITE_AOT_INFER_LITE_AOT_DEPS_MAIN_LIB_H_
#define XLA_BACKENDS_CPU_LITE_AOT_INFER_LITE_AOT_DEPS_MAIN_LIB_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/service/cpu/executable.pb.h"

namespace xla {
namespace cpu {

inline std::string SerDesPath(absl::string_view name) {
  return absl::StrCat(
      "//xla/backends/cpu/runtime/"
      "thunk_serdes:",
      name, "_thunk_serdes");
}

// Analyzes the provided CompilationResultProto and returns a list of Bazel
// targets (e.g., specific thunk SerDes libraries) that must be linked
// in order to successfully load and execute an XLA:CPU AOT compiled model.
absl::StatusOr<std::vector<std::string>> InferLiteAotDeps(
    const CompilationResultProto& compilation_result);

}  // namespace cpu
}  // namespace xla

#endif  // XLA_BACKENDS_CPU_LITE_AOT_INFER_LITE_AOT_DEPS_MAIN_LIB_H_
