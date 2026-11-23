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

#ifndef XLA_SERVICE_CPU_ORC_FUNCTION_LIBRARY_LOADER_H_
#define XLA_SERVICE_CPU_ORC_FUNCTION_LIBRARY_LOADER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/any_span.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/executable.pb.h"

namespace xla::cpu {

// Loads `obj_files` into the current process using LLVM ORC.
absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadObjectFilesWithOrc(
    absl::Span<const FunctionLibrary::Symbol> compiled_symbols,
    absl::AnySpan<const ObjFileProto> obj_files,
    absl::string_view data_layout_str);

// `FunctionLibraryLoader` implementation that loads `proto.object_files()`
// using LLVM ORC.
absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadFunctionLibraryWithOrc(
    const CompilationResultProto& proto);

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_ORC_FUNCTION_LIBRARY_LOADER_H_
