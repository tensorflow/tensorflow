/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_DUMP_DUMP_H_
#define XLA_PJRT_DUMP_DUMP_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"

namespace pjrt {

// Resolves the dump path. If the provided `dump_to` is "sponge" or
// "test_undeclared_outputs_dir", it will attempt to get the test undeclared
// outputs directory.
absl::StatusOr<std::string> ResolveTestingDumpPath(absl::string_view dump_to);

// Returns the dump subdirectory path, including a timestamp.
absl::StatusOr<std::string> GetDumpSubdirPath(absl::string_view dump_to_path,
                                              absl::string_view module_name);

// Dumps the compile inputs (module, options, topology) to the specified path.
absl::Status DumpCompileInputs(absl::string_view path,
                               xla::CompileOptions options,
                               mlir::ModuleOp module,
                               const xla::PjRtTopologyDescription& topology);

// Dumps the compile inputs (module, options, topology) to the specified
// path if the compile options specify a dump path via `xla_dump_to`.
//
// Does nothing if the compile options does not set `xla_dump_to`.
absl::Status MaybeDumpCompileInputs(
    xla::CompileOptions compile_options, mlir::ModuleOp module,
    const xla::PjRtTopologyDescription& topology);

}  // namespace pjrt

#endif  // XLA_PJRT_DUMP_DUMP_H_
