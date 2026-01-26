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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/executable_serdes.h"
#include "xla/python/ifrt/ir/ifrt_ir_compile_options.pb.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_default_version_accessor.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/tsl/platform/errors.h"

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

  // Returns true if the program exclusively owns the MLIR context.
  bool OwnsMlirContext() const { return mlir_context != nullptr; }

  static char ID;  // NOLINT

 private:
  std::unique_ptr<mlir::MLIRContext> mlir_context;
  mlir::OwningOpRef<mlir::ModuleOp> owning_mlir_module;
};

// Options for serializing IFRT IR programs.
struct SerializeIfrtIRProgramOptions
    : llvm::RTTIExtends<SerializeIfrtIRProgramOptions, SerializeOptions> {
  explicit SerializeIfrtIRProgramOptions(
      std::string ifrt_version, std::string atom_program_version,
      bool version_in_place = true,
      // Using a parameter name `serdes_version` avoids shadowing the base class
      // member variable `version`.
      SerDesVersion serdes_version = SerDesDefaultVersionAccessor::Get())
      : llvm::RTTIExtends<SerializeIfrtIRProgramOptions, SerializeOptions>(
            /*version=*/serdes_version),
        ifrt_version(std::move(ifrt_version)),
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
    : llvm::RTTIExtends<DeserializeIfrtIRProgramOptions,
                        DeserializeExecutableOptions> {
  explicit DeserializeIfrtIRProgramOptions(mlir::MLIRContext* context)
      : context(context) {}
  DeserializeIfrtIRProgramOptions(
      mlir::MLIRContext* context,
      std::optional<xla::ifrt::DeviceListRef> device_list)
      : llvm::RTTIExtends<DeserializeIfrtIRProgramOptions,
                          DeserializeExecutableOptions>(device_list),
        context(context) {}

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
      std::string mlir_dump_to = "", std::string mlir_dump_pass_re = "",
      std::string mlir_dump_func_re = ".*", bool mlir_enable_timing = false,
      std::string dot_graph_dump_to = "",
      int64_t dot_graph_min_executable_peak_memory_bytes = 0,
      float dot_graph_min_executable_flops = 0.0,
      int64_t dot_graph_min_per_device_transfer_size_bytes = 0)
      : device_assignments(std::move(device_assignments)),
        loaded_exec_binding(std::move(loaded_exec_binding)),
        compile_options_overrides(std::move(compile_options_overrides)),
        mlir_dump_to(std::move(mlir_dump_to)),
        mlir_dump_pass_re(std::move(mlir_dump_pass_re)),
        mlir_dump_func_re(std::move(mlir_dump_func_re)),
        mlir_enable_timing(mlir_enable_timing),
        dot_graph_dump_to(std::move(dot_graph_dump_to)),
        dot_graph_min_executable_peak_memory_bytes(
            dot_graph_min_executable_peak_memory_bytes),
        dot_graph_min_executable_flops(dot_graph_min_executable_flops),
        dot_graph_min_per_device_transfer_size_bytes(
            dot_graph_min_per_device_transfer_size_bytes) {}

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

  // Constructs `IfrtIRCompileOptions` from `IfrtIrCompileOptionsProto`.
  static absl::StatusOr<std::unique_ptr<IfrtIRCompileOptions>> FromProto(
      const IfrtIrCompileOptionsProto& proto);

  // Converts the compile options to a protobuf.
  absl::Status ToProto(
      IfrtIrCompileOptionsProto& proto,
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const;

  // Returns a `IfrtIrCompileOptionsProto` representation.
  absl::StatusOr<IfrtIrCompileOptionsProto> ToProto(
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const {
    IfrtIrCompileOptionsProto proto;
    TF_RETURN_IF_ERROR(ToProto(proto, version));
    return proto;
  }

  std::string mlir_dump_to;
  std::string mlir_dump_pass_re;
  std::string mlir_dump_func_re;
  bool mlir_enable_timing;
  std::string dot_graph_dump_to;
  int64_t dot_graph_min_executable_peak_memory_bytes;
  float dot_graph_min_executable_flops;
  int64_t dot_graph_min_per_device_transfer_size_bytes;

  static char ID;  // NOLINT
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const IfrtIRCompileOptions& options);

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              std::shared_ptr<IfrtIRCompileOptions> options);

// Gets `xla::ifrt::IfrtIRCompileOptions` from `xla::ifrt::CompileOptions`.
absl::StatusOr<std::unique_ptr<IfrtIRCompileOptions>> GetIfrtIRCompileOptions(
    std::unique_ptr<CompileOptions> options);

}  // namespace ifrt
}  // namespace xla

namespace llvm::cl {

extern template class basic_parser<
    std::shared_ptr<xla::ifrt::IfrtIRCompileOptions>>;

template <>
class parser<std::shared_ptr<xla::ifrt::IfrtIRCompileOptions>>
    : public basic_parser<std::shared_ptr<xla::ifrt::IfrtIRCompileOptions>> {
 public:
  explicit parser(Option& opt) : basic_parser(opt) {}
  bool parse(Option& opt, StringRef argName, StringRef arg,
             std::shared_ptr<xla::ifrt::IfrtIRCompileOptions>& value);
  StringRef getValueName() const override { return "ifrt-ir-compile-options"; }
  void printOptionDiff(
      const Option& opt,
      const std::shared_ptr<xla::ifrt::IfrtIRCompileOptions>& value,
      const OptVal& defaultValue, size_t globalWidth) const;
  void anchor() override;
};

}  // namespace llvm::cl

#endif  // XLA_PYTHON_IFRT_IR_IFRT_IR_PROGRAM_H_
