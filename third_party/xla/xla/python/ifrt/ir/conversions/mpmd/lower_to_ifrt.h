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

#ifndef XLA_PYTHON_IFRT_IR_CONVERSIONS_MPMD_LOWER_TO_IFRT_H_
#define XLA_PYTHON_IFRT_IR_CONVERSIONS_MPMD_LOWER_TO_IFRT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/pjrt/pjrt_executable.h"

namespace xla::ifrt::mpmd {

// Compile environment options override is a vector of pairs of compile option
// flag and value strings.
using EnvOptionsOverride =
    std::vector<std::pair<std::string, xla::CompileOptions::OptionOverride>>;

// Mapping between a string identifier (e.g., mesh name) and XLA compile
// options.
using CompileOptionsMap = absl::flat_hash_map<std::string, xla::CompileOptions>;

// Name of StringAttr on ifrt.CallOp used to contain the mesh name the IFRT
// atom program will execute on. The mesh name is used as a key into a mapping
// of optional compile options provided by users per mesh. If no compile options
// are provided for a mesh, then the default compile options are used.
inline constexpr llvm::StringLiteral kIfrtMeshNameAttrName = "ifrt.mesh_name";

std::unique_ptr<mlir::Pass> CreateLowerToIfrtPass();

// Registers:
// -mpmd-lower-to-ifrt:
// -mpmd-ifrt-add-ctrl-dependencies
void RegisterLowerToIfrtPasses();

// Lowers a Shardy:MPMD module as an IFRT module.
// `add_control_dependencies` is used to add control dependencies between
// fragments. This will enforce strict fragment execution order and is useful
// for guaranteeing correct pipelining.
absl::Status LowerToIfrt(mlir::ModuleOp module,
                         bool add_control_dependencies = true);

// Gets the per fragment compile options from the IFRT IR module.
// `compile_options_overrides` is a mapping from mesh name to compile option
// overrides. The overrides apply to all the computations assigned to a mesh.
// `threshold_for_argument_tupling` is the threshold for argument tupling.
// `set_reserved_bytes` is a callback to inform the compiler of how much memory
// is expected to be already used by other programs, when compiling a given
// fragment. This is platform dependent.
//
absl::StatusOr<CompileOptionsMap> GetCompileOptions(
    mlir::ModuleOp module,
    const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
        compile_options_overrides,
    int threshold_for_argument_tupling = 2000,
    llvm::function_ref<void(xla::ExecutableBuildOptions&, int64_t)>
        set_reserved_bytes = [](xla::ExecutableBuildOptions&, int64_t) {});

}  // namespace xla::ifrt::mpmd

#endif  // XLA_PYTHON_IFRT_IR_CONVERSIONS_MPMD_LOWER_TO_IFRT_H_
