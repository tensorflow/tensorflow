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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
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

  // Indicates whether the program supports querying input/output layout. If
  // this is OK, `in_specs` and `out_specs` will have `layout` field populated.
  // Otherwise, the layout field will be nullptr.
  absl::Status layout_status;

  // The indices of the donatable inputs in the program.
  std::vector<int> donatable_input_indices;

  // The input program.
  std::unique_ptr<xla::ifrt::IfrtIRProgram> program;

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
