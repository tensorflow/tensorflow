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

#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_IFRT_CREATE_COMPILE_OPTIONS_PASS_H_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_IFRT_CREATE_COMPILE_OPTIONS_PASS_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/pjrt/pjrt_executable.h"

namespace xla {
namespace ifrt {

// Compile environment options override is a vector of pairs of compile option
// flag and value strings.
using EnvOptionsOverride =
    std::vector<std::pair<std::string, xla::CompileOptions::OptionOverride>>;

// Mapping between a string identifier (e.g., mesh name) and XLA compile
// options.
using CompileOptionsMap = absl::flat_hash_map<std::string, xla::CompileOptions>;

// Gets the per atom program compile options from the IFRT IR module.
//
// `compile_options_overrides` is a mapping from mesh name to compile option
// overrides. The overrides apply to all the computations assigned to a mesh.
//
// `threshold_for_parameter_tupling` is the threshold for parameter tupling.
absl::StatusOr<CompileOptionsMap> GetCompileOptions(
    mlir::ModuleOp module,
    const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
        compile_options_overrides,
    int threshold_for_parameter_tupling);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_IFRT_CREATE_COMPILE_OPTIONS_PASS_H_
