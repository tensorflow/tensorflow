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

#ifndef XLA_SERVICE_CPU_FUNCTION_LIBRARY_LOADER_H_
#define XLA_SERVICE_CPU_FUNCTION_LIBRARY_LOADER_H_

#include <memory>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/any_span.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/executable.pb.h"

namespace xla::cpu {

// Loads a `FunctionLibrary` from a serialized XLA:CPU AOT compilation result.
using FunctionLibraryLoader =
    absl::FunctionRef<absl::StatusOr<std::unique_ptr<FunctionLibrary>>(
        const CompilationResultProto& proto)>;

// Converts serialized `SymbolProto` entries into `FunctionLibrary::Symbol`s.
// Returns an `Internal` error if any entry has an unknown `function_type_id`.
absl::StatusOr<std::vector<FunctionLibrary::Symbol>>
GetCompiledSymbolsFromProto(
    absl::AnySpan<const SymbolProto> compiled_symbols_proto);

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_FUNCTION_LIBRARY_LOADER_H_
