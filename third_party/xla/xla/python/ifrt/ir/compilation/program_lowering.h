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

#ifndef XLA_PYTHON_IFRT_IR_COMPILATION_PROGRAM_LOWERING_H_
#define XLA_PYTHON_IFRT_IR_COMPILATION_PROGRAM_LOWERING_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/ir/compilation/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/tsl/concurrency/executor.h"

namespace xla {
namespace ifrt {

// Intermediate lowered state of an `IfrtIRProgram` produced after MLIR
// compilation passes run and atom program compilations have been dispatched.
struct LoweredIfrtIrProgramState {
  // Name of the lowered IFRT IR module.
  std::string program_name;
  // Futures for bound and/or compiled-and-loaded atom program executables.
  absl_nonnull std::shared_ptr<AtomExecutableFutureMap>
      atom_executable_future_map;
  // Specifications of the program inputs (`layout` is unpopulated).
  std::vector<ArraySpec> in_specs;
  // Specifications of the program outputs (`layout` is unpopulated).
  std::vector<ArraySpec> out_specs;
  // Indices of donatable inputs in the lowered program.
  std::vector<int> donatable_input_indices;
  // Lowered IFRT IR program.
  absl_nonnull std::unique_ptr<IfrtIRProgram> ifrt_ir_program;
  // Target device list resolved from `compile_options->device_assignments`.
  DeviceListRef device_list;
  // Compile options used to lower the program.
  absl_nonnull std::shared_ptr<IfrtIRCompileOptions> compile_options;
};

// Returns a `tsl::Executor` backed by `tsl::Env::SchedClosure` for running
// compilation future continuations off fibers.
tsl::Executor& GetIfrtIrCompilationExecutor();

// Runs IFRT IR lowering passes, dispatches atom program compilations, and
// extracts top-level input/output `ArraySpec`s and donatable input indices.
absl::StatusOr<LoweredIfrtIrProgramState> LowerIfrtIrProgram(
    absl_nonnull std::unique_ptr<IfrtIRProgram> ifrt_ir_program,
    absl_nonnull std::unique_ptr<IfrtIRCompileOptions> ifrt_ir_compile_options,
    Client* absl_nonnull client,
    absl_nonnull std::shared_ptr<AtomProgramCompiler> atom_program_compiler);

// Populates `layout` on `in_specs` and `out_specs` from loaded atom executables
// and device defaults.
absl::Status PopulateIfrtIrProgramLayouts(
    mlir::ModuleOp mlir_module, Client* absl_nonnull client,
    const AtomExecutableMap& atom_program_executables,
    absl::Span<ArraySpec> in_specs, absl::Span<ArraySpec> out_specs,
    const DeviceListRef& device_list);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_COMPILATION_PROGRAM_LOWERING_H_
