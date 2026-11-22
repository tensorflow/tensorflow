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

#ifndef XLA_PYTHON_IFRT_IR_IFRT_IR_EXECUTABLE_H_
#define XLA_PYTHON_IFRT_IR_IFRT_IR_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/compilation/atom_program_compiler.h"
#include "xla/python/ifrt/ir/compilation/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

// Compiled `Executable` produced by `IfrtIrProgramCompiler::Compile`, holding a
// lowered `IfrtIRProgram` and its compiled outlined atom programs prior to
// loading via `Client::Load`.
//
// `Client` affinity and lifetime: `IfrtIrExecutable` embeds `Device*` pointers
// and any bound `LoadedExecutable`s resolved on the compile-time `Client*`, so
// it is bound to (and must not outlive) that `Client` and can only be loaded
// via `CreateCompiledIfrtIrProgram()` on that same `Client*`.
//
// Thread-safety: `Executable` metadata getters, read-only accessors, and
// `CreateCompiledIfrtIrProgram()` (which parses a fresh `MLIRContext` from
// immutable pre-serialized MLIR bytecode) are safe to call concurrently.
// Callers inspecting `mlir_module()` (or `program()->mlir_module`) directly
// must treat the `mlir::ModuleOp` as read-only and externally synchronize any
// concurrent MLIR operations that mutate or lock its `MLIRContext`.
class IfrtIrExecutable final
    : public RTTIExtends<IfrtIrExecutable, Executable> {
 public:
  // Compiles an `IfrtIRProgram` and its outlined atom programs without loading
  // them onto devices.
  static tsl::Future<ExecutableRef> Create(
      absl_nonnull std::unique_ptr<IfrtIRProgram> ifrt_ir_program,
      absl_nonnull std::unique_ptr<IfrtIRCompileOptions> compile_options,
      Client* absl_nonnull client,
      absl_nonnull std::shared_ptr<AtomProgramCompiler> atom_program_compiler);

  ~IfrtIrExecutable() override = default;

  // Builds a `CompiledIfrtIrProgram` once all `compiled_atom_programs()` have
  // been loaded on `client` (which must match the `Client*` used to compile
  // this `IfrtIrExecutable`) and provided in `newly_loaded_atom_executables`.
  //
  // Thread-safe across concurrent calls on the same `IfrtIrExecutable`
  // instance (parses a fresh `MLIRContext` from pre-serialized MLIR bytecode).
  absl::StatusOr<absl_nonnull std::shared_ptr<CompiledIfrtIrProgram>>
  CreateCompiledIfrtIrProgram(
      Client* absl_nonnull client,
      AtomExecutableMap newly_loaded_atom_executables) const;

  // Returns the lowered `IfrtIRProgram`.
  const std::shared_ptr<const IfrtIRProgram>& program() const {
    return program_;
  }
  // Returns the lowered MLIR `ModuleOp` owned by `program()`. Callers must
  // treat the returned `ModuleOp` as read-only and externally synchronize any
  // concurrent MLIR operations that mutate or lock its `MLIRContext`.
  mlir::ModuleOp mlir_module() const { return program_->mlir_module; }
  // Returns the target device list resolved from `compile_options()`.
  const DeviceListRef& devices() const { return devices_; }
  // Returns the options used to compile this IFRT IR program.
  const IfrtIRCompileOptions& compile_options() const {
    return *compile_options_;
  }
  // Returns input `ArraySpec`s (`layout` is populated iff
  // `layout_status().ok()`).
  absl::Span<const ArraySpec> in_specs() const { return in_specs_; }
  // Returns output `ArraySpec`s (`layout` is populated iff
  // `layout_status().ok()`).
  absl::Span<const ArraySpec> out_specs() const { return out_specs_; }
  // Returns whether input and output layouts were populated on `in_specs()`
  // and `out_specs()`.
  const absl::Status& layout_status() const { return layout_status_; }
  // Returns indices of inputs marked donatable in the lowered IFRT IR module.
  absl::Span<const int> donatable_input_indices() const {
    return donatable_input_indices_;
  }
  // Returns externally bound `LoadedExecutable`s keyed by symbol name.
  const AtomExecutableMap& bound_atom_program_executables() const {
    return *bound_atom_program_executables_;
  }
  // Returns compiled-but-unloaded atom programs keyed by symbol name.
  const CompiledAtomProgramMap& compiled_atom_programs() const {
    return *compiled_atom_programs_;
  }

  // `Executable` implementation.

  absl::string_view name() const override { return program_name_; }

  absl::StatusOr<std::optional<std::string>> Fingerprint() const override {
    return std::nullopt;
  }

  absl::StatusOr<std::string> Serialize() const override;

  int num_devices() const override { return devices_->size(); }

  int64_t SizeOfGeneratedCodeInBytes() const override { return -1; }

  absl::StatusOr<xla::CompiledMemoryStats> GetCompiledMemoryStats()
      const override;

  std::optional<std::vector<xla::OpSharding>> GetParameterShardings()
      const override;

  std::optional<std::vector<xla::OpSharding>> GetOutputShardings()
      const override;

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const override;

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const override;

  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> GetHloModules()
      const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

  absl::StatusOr<AttributeMap> GetCostAnalysis() const override;

  static char ID;  // NOLINT

 private:
  IfrtIrExecutable(
      const Client* absl_nonnull client, std::string program_name,
      absl_nonnull std::shared_ptr<AtomExecutableMap>
          bound_atom_program_executables,
      absl_nonnull std::shared_ptr<CompiledAtomProgramMap>
          compiled_atom_programs,
      std::vector<ArraySpec> in_specs, std::vector<ArraySpec> out_specs,
      absl::Status layout_status, std::vector<int> donatable_input_indices,
      absl_nonnull std::shared_ptr<const IfrtIRProgram> program,
      std::string program_bytecode, DeviceListRef devices,
      absl_nonnull std::shared_ptr<IfrtIRCompileOptions> compile_options);

  const Client* absl_nonnull client_;
  std::string program_name_;
  absl_nonnull std::shared_ptr<AtomExecutableMap>
      bound_atom_program_executables_;
  absl_nonnull std::shared_ptr<CompiledAtomProgramMap> compiled_atom_programs_;
  std::vector<ArraySpec> in_specs_;
  std::vector<ArraySpec> out_specs_;
  absl::Status layout_status_;
  std::vector<int> donatable_input_indices_;
  absl_nonnull std::shared_ptr<const IfrtIRProgram> program_;
  std::string program_bytecode_;
  DeviceListRef devices_;
  absl_nonnull std::shared_ptr<IfrtIRCompileOptions> compile_options_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_IFRT_IR_EXECUTABLE_H_
