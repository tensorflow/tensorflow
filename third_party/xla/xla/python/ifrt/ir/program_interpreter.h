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

#ifndef XLA_PYTHON_IFRT_IR_PROGRAM_INTERPRETER_H_
#define XLA_PYTHON_IFRT_IR_PROGRAM_INTERPRETER_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"

namespace xla {
namespace ifrt {

// Environment to keep track of live arrays.
struct Environment;

// Interpreter for an IFRT IR program.
//
// The program interpreter is responsible for executing an IFRT IR program. The
// interpreter works in two stages. First, when `BuildExecuteFn` is called, it
// traverses the program and builds a function that can be invoked to execute
// the program, which happens only once during compilation. Second, the returned
// execute function can be called multiple times to interpret the IFRT IR
// program.
//
// This two-stage design has two primary purposes:
//
// 1. It allows us to leverage the static information available in the program
//    as much as possible. For example, `RemapArraysOp` builds its remap plan
//    during the first stage and the plan is reused for all executions.
//
// 2. It avoids running any LLVM/MLIR code during execution. This is
//    particularly useful in environments where the use of LLVM/MLIR
//    synchronization primitives may cause deadlocks, e.g., cooperatively
//    scheduled fibers.
class ProgramInterpreter {
 public:
  using ExecuteFn = absl::AnyInvocable<
      absl::StatusOr<xla::ifrt::LoadedExecutable::ExecuteResult>(
          absl::Span<xla::ifrt::ArrayRef> arrays,
          const xla::ifrt::LoadedExecutable::ExecuteOptions& options,
          std::optional<xla::ifrt::DeviceListRef> devices)>;

  static absl::StatusOr<std::unique_ptr<ProgramInterpreter>> Create(
      xla::ifrt::Client* client, absl::string_view program_name,
      mlir::ModuleOp mlir_module,
      std::shared_ptr<xla::ifrt::AtomExecutableMap> atom_program_executables,
      xla::ifrt::DeviceListRef devices);

  absl::StatusOr<ExecuteFn> BuildExecuteFn();

 private:
  using OpFn = absl::AnyInvocable<absl::Status(Environment& env) const>;

  ProgramInterpreter(
      xla::ifrt::Client* client, absl::string_view program_name,
      mlir::ModuleOp mlir_module,
      std::shared_ptr<xla::ifrt::AtomExecutableMap> atom_program_executables,
      xla::ifrt::DeviceListRef devices, mlir::Liveness liveness)
      : client_(client),
        program_name_(program_name),
        mlir_module_(mlir_module),
        atom_program_executables_(std::move(atom_program_executables)),
        devices_(std::move(devices)),
        liveness_(std::move(liveness)) {}

  absl::StatusOr<OpFn> HandleOp(
      xla::ifrt::CallLoadedExecutableOp call_loaded_op);
  absl::StatusOr<OpFn> HandleOp(xla::ifrt::RemapArraysOp remap_op);
  absl::StatusOr<OpFn> HandleOp(xla::ifrt::CopyArraysOp copy_arrays_op);
  absl::StatusOr<OpFn> HandleOp(mlir::func::ReturnOp return_op);

  // Returns a pretty string representation of the op.
  std::string PrettyPrint(mlir::Operation* op);

  xla::ifrt::Client* client_;
  mlir::SymbolTableCollection symbol_table_;
  std::string program_name_;
  mlir::ModuleOp mlir_module_;
  std::shared_ptr<xla::ifrt::AtomExecutableMap> atom_program_executables_;

  // All the devices the program uses.
  xla::ifrt::DeviceListRef devices_;

  // Cached liveness analysis of the IFRT IR program.
  mlir::Liveness liveness_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_PROGRAM_INTERPRETER_H_
