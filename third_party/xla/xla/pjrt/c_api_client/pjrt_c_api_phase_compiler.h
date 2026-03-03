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

#ifndef XLA_PJRT_C_API_CLIENT_PJRT_C_API_PHASE_COMPILER_H_
#define XLA_PJRT_C_API_CLIENT_PJRT_C_API_PHASE_COMPILER_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"

namespace xla {

// This class is a wrapper around the PJRT C API PhaseCompiler. It provides a
// C++ interface for the phase compiler.
class PjRtCApiPhaseCompiler : public PjRtPhaseCompiler {
 public:
  PjRtCApiPhaseCompiler(
      const PJRT_Api* api,
      const PJRT_PhaseCompile_Extension* phase_compile_extension,
      const PJRT_PhaseCompiler* phase_compiler);

  // Runs the compilation phase with the given phases 'phases_to_run' on the
  // input programs 'partial_programs_in' and returns the output programs.
  // This function first performs plugin-agnostic validation on the inputs and
  // phase names. Next, it triggers plugin-specific phase validators for input
  // compatibility before invoking the appropriate phase compilers.
  absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>> RunPhases(
      xla::CompileOptions options,
      const std::vector<xla::PjRtPartialProgramProto>& partial_programs_in,
      const xla::PjRtTopologyDescription& topology,
      const std::vector<std::string>& phases_to_run) override;

  // Returns the names of all the phases following the order of their
  // registration.
  absl::StatusOr<std::vector<std::string>> GetPhaseNames() override;

  // Returns the reference to the phase compiler.
  const PJRT_PhaseCompiler* GetPhaseCompiler() const;

  // Compiles the 'computation' and returns a 'PjRtExecutable'. The returned
  // PjRtExecutable must be loaded by a compatible client before execution.
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) override {
    return absl::UnimplementedError(
        "PjRtCApiPhaseCompiler::Compile is not implemented.");
  }

  // Variant of `Compile` that accepts an MLIR module.
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override {
    return Compile(options, MaybeOwningMlirModule(std::move(module)), topology,
                   client);
  }

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, MaybeOwningMlirModule module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override {
    return absl::UnimplementedError(
        "PjRtCApiPhaseCompiler::Compile is not implemented.");
  }

  // CApiPhaseCompiler does not need to provide a registration of phases, as it
  // is wrapping a plugin-initialized phase compiler.
  absl::Status RegisterAllPhases() override { return absl::OkStatus(); }

 private:
  const PJRT_Api* api_ = nullptr;  // Not owned.
  const PJRT_PhaseCompile_Extension* phase_compile_extension_ =
      nullptr;  // Not owned.
  std::unique_ptr<const PJRT_PhaseCompiler,
                  std::function<void(const PJRT_PhaseCompiler*)>>
      phase_compiler_;
};

}  // namespace xla

#endif  // XLA_PJRT_C_API_CLIENT_PJRT_C_API_PHASE_COMPILER_H_
