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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/sharding.h"

namespace xla {
namespace ifrt {

// Environment to keep track of live arrays.
struct Environment;

// Interpreter for an IFRT IR program.
class ProgramInterpreter {
 public:
  static absl::StatusOr<std::unique_ptr<ProgramInterpreter>> Create(
      xla::ifrt::Client* client, std::shared_ptr<CompiledIfrtIrProgram> program,
      xla::ifrt::DeviceListRef devices);

  // Executes the IFRT IR program.
  absl::StatusOr<xla::ifrt::LoadedExecutable::ExecuteResult> Execute(
      absl::Span<xla::ifrt::ArrayRef> arrays,
      const xla::ifrt::LoadedExecutable::ExecuteOptions& options,
      std::optional<xla::ifrt::DeviceListRef> devices);

 private:
  ProgramInterpreter(
      xla::ifrt::Client* client, std::shared_ptr<CompiledIfrtIrProgram> program,
      xla::ifrt::DeviceListRef devices, mlir::Liveness liveness,
      llvm::DenseMap<xla::ifrt::IfrtArrayType, xla::ifrt::ShardingRef>
          array_type_to_sharding)
      : client_(client),
        program_(std::move(program)),
        devices_(std::move(devices)),
        liveness_(std::move(liveness)),
        array_type_to_sharding_(std::move(array_type_to_sharding)) {}

  absl::Status ExecuteOp(xla::ifrt::CallLoadedExecutableOp call_loaded_op,
                         Environment& env);
  absl::Status ExecuteOp(xla::ifrt::RemapArraysOp remap_op, Environment& env);
  absl::Status ExecuteOp(xla::ifrt::CopyArraysOp copy_arrays_op,
                         Environment& env);
  absl::Status ExecuteOp(mlir::func::ReturnOp return_op, Environment& env);

  // Returns a pretty string representation of the op.
  std::string PrettyPrint(mlir::Operation* op);

  xla::ifrt::Client* client_;
  mlir::SymbolTableCollection symbol_table_;
  std::shared_ptr<CompiledIfrtIrProgram> program_;

  // All the devices the program uses.
  xla::ifrt::DeviceListRef devices_;

  // Cached liveness analysis of the IFRT IR program.
  mlir::Liveness liveness_;

  // Mapping between IfrtArrayType and Sharding. This map is used to cache
  // the Shardings at IFRT IR program compilation time in order to avoid
  // overheads at execution time.
  llvm::DenseMap<xla::ifrt::IfrtArrayType, xla::ifrt::ShardingRef>
      array_type_to_sharding_;

  // Set of donated program arguments, which can be deleted after their last
  // use. Entries are removed upon deletion or if they are aliased.
  llvm::DenseSet<mlir::Value> deletable_program_arguments_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_PROGRAM_INTERPRETER_H_
