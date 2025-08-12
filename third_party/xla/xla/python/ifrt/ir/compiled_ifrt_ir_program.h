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

#ifndef XLA_PYTHON_IFRT_IR_COMPILED_IFRT_IR_PROGRAM_H_
#define XLA_PYTHON_IFRT_IR_COMPILED_IFRT_IR_PROGRAM_H_
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"

namespace xla {
namespace ifrt {

// The result of compiling an IFRT IR module so that it can be interpreted.
struct CompiledIfrtIrProgram {
  // The name of the compiled program.
  std::string program_name;

  // A mapping from the calee name of LoadedExecutableOps to
  // xla::ifrt::LoadedExecutables.
  std::shared_ptr<xla::ifrt::AtomExecutableMap> atom_program_executables;

  // Specifications of the program inputs.
  std::vector<xla::ifrt::ArraySpec> in_specs;

  // Specifications of the program outputs.
  std::vector<xla::ifrt::ArraySpec> out_specs;

  // Hold the IfrtIRProgram to avoid the module from being destroyed, in the
  // case where the IfrtIRProgram owns the MLIR module.
  std::unique_ptr<xla::ifrt::IfrtIRProgram> program;

  // TODO(b/382761415): Remove this field once the layouts in the types are
  // populated.
  // Note: It is important for the mlir_module to be after the program because
  // the module is using the MLIR context of the program, and thus must be
  // destroyed before the program.
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;

  // Mapping from logical device ids in IFRT IR MLIR module to runtime device
  // ids obtained from IFRT client.
  std::vector<xla::ifrt::DeviceId> device_assignments;

  // Compiles an IFRT IR program.
  static absl::StatusOr<CompiledIfrtIrProgram> Create(
      std::unique_ptr<xla::ifrt::IfrtIRProgram> ifrt_ir_program,
      std::unique_ptr<xla::ifrt::IfrtIRCompileOptions> compile_options,
      xla::ifrt::Client* client,
      std::shared_ptr<xla::ifrt::AtomProgramCompiler> atom_program_compiler);
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_COMPILED_IFRT_IR_PROGRAM_H_
