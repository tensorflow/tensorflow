/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_IFRT_IR_PROGRAM_H_
#define XLA_PYTHON_IFRT_IR_IFRT_IR_PROGRAM_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_compile_options.pb.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {

struct IfrtIRProgram : llvm::RTTIExtends<IfrtIRProgram, Program> {
  IfrtIRProgram() = default;
  explicit IfrtIRProgram(mlir::ModuleOp mlir_module)
      : mlir_module(std::move(mlir_module)) {}
  IfrtIRProgram(std::unique_ptr<mlir::MLIRContext> context,
                mlir::OwningOpRef<mlir::ModuleOp> module)
      : mlir_module(*module),
        mlir_context(std::move(context)),
        owning_mlir_module(std::move(module)) {}

  static absl::string_view type_name() { return "xla::ifrt::IfrtIRProgram"; }

  mlir::ModuleOp mlir_module;

  static char ID;  // NOLINT

 private:
  std::unique_ptr<mlir::MLIRContext> mlir_context;
  mlir::OwningOpRef<mlir::ModuleOp> owning_mlir_module;
};

// Options for serializing IFRT IR programs.
struct SerializeIfrtIRProgramOptions
    : llvm::RTTIExtends<SerializeIfrtIRProgramOptions, SerializeOptions> {
  explicit SerializeIfrtIRProgramOptions(std::string ifrt_version,
                                         std::string atom_program_version,
                                         bool version_in_place = true)
      : ifrt_version(std::move(ifrt_version)),
        atom_program_version(std::move(atom_program_version)),
        version_in_place(version_in_place) {}

  static char ID;  // NOLINT

  // String of the form "major.minor.patch", representing the IFRT IR version.
  // TODO(hyeontaek): Migrate `ifrt_version` to `SerializeOptions::version`.
  std::string ifrt_version;
  // String of the form "major.minor.patch", representing the atom program
  // version (currently VHLO version).
  std::string atom_program_version;
  // Whether to version the IFRT IR ModuleOp in-place.
  bool version_in_place;
};

// Options for deserializing IFRT IR programs.
// If `context` is not nullptr then deserialization will create a new MLIR
// context, which will be owned by the deserialized program. Otherwise, the
// deserialization will use the provided MLIR context and the returned program
// will not own a MLIR context.
struct DeserializeIfrtIRProgramOptions
    : llvm::RTTIExtends<DeserializeIfrtIRProgramOptions, DeserializeOptions> {
  explicit DeserializeIfrtIRProgramOptions(mlir::MLIRContext* context)
      : context(context) {}

  static char ID;  // NOLINT

  mlir::MLIRContext* context;
};

// CompileOptions for an IFRT IR program.
struct IfrtIRCompileOptions
    : llvm::RTTIExtends<IfrtIRCompileOptions, CompileOptions> {
  IfrtIRCompileOptions() = default;
  explicit IfrtIRCompileOptions(
      std::vector<DeviceId> device_assignments,
      absl::flat_hash_map<std::string, LoadedExecutableRef>
          loaded_exec_binding = {},
      std::shared_ptr<absl::flat_hash_map<
          std::string, std::unique_ptr<xla::ifrt::CompileOptions>>>
          compile_options_overrides = {},
      bool propagate_shardings = false)
      : device_assignments(std::move(device_assignments)),
        loaded_exec_binding(std::move(loaded_exec_binding)),
        compile_options_overrides(std::move(compile_options_overrides)),
        propagate_shardings(propagate_shardings) {}

  // Mapping from logical device ids in IFRT IR MLIR module to runtime device
  // ids obtained from IFRT client.
  std::vector<DeviceId> device_assignments;

  // Map from symbol names of LoadedExecutableOp in the IFRT IR MLIR module
  // to pre-compiled `LoadedExecutable` instance. The `LoadedExecutable`s must
  // outlive the `LoadedExecutable` of the IFRT IR program.
  absl::flat_hash_map<std::string, LoadedExecutableRef> loaded_exec_binding;

  // Mapping from values of `ifrt.compile_option_key` attribute of a `CallOp` to
  // compile options. If a `CallOp` does not have have the attribute set or does
  // not have an entry in this map then default compile options are used.
  std::shared_ptr<absl::flat_hash_map<
      std::string, std::unique_ptr<xla::ifrt::CompileOptions>>>
      compile_options_overrides;

  // Whether to propagate shardings from atom program executables for
  // unspecified shardings.
  bool propagate_shardings;

  // Constructs `IfrtIRCompileOptions` from `IfrtIrCompileOptionsProto`.
  static absl::StatusOr<std::unique_ptr<IfrtIRCompileOptions>> FromProto(
      const IfrtIrCompileOptionsProto& proto);

  // Returns a `IfrtIrCompileOptionsProto` representation.
  absl::StatusOr<IfrtIrCompileOptionsProto> ToProto(
      SerDesVersion version = SerDesVersion::current()) const;

  static char ID;  // NOLINT
};

// Gets `xla::ifrt::IfrtIRCompileOptions` from `xla::ifrt::CompileOptions`.
absl::StatusOr<std::unique_ptr<IfrtIRCompileOptions>> GetIfrtIRCompileOptions(
    std::unique_ptr<CompileOptions> options);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_IFRT_IR_PROGRAM_H_
