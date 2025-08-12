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

#ifndef XLA_PJRT_PJRT_PHASE_COMPILE_SAMPLE_PLUGIN_H_
#define XLA_PJRT_PJRT_PHASE_COMPILE_SAMPLE_PLUGIN_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"

namespace pjrt {
namespace phase_compile_sample_plugin {

// This file demonstrates the artifacts a plugin developer needs to provide
// to create a phase compile plugin. Specifically, it shows the declaration of
// `PJRT_PhaseCompile_Extension` which contains all the functions that the
// plugin needs to implement.

// Helper class for serializing and deserializing StableHLO MLIR modules.
// This is crucial for converting `PjRtPartialProgramProto` bytes to
// actual MLIR modules and vice-versa, allowing programs to be transferred
// between compilation phases.
class StablehloTypeSerialization {
 public:
  // Deserializes a StableHLO program from a string into an MLIR ModuleOp.
  // Returns an error if deserialization fails (e.g., invalid artifact).
  static absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Deserialize(
      const std::string& program, mlir::MLIRContext& context);

  // Serializes an MLIR ModuleOp into a StableHLO bytecode string.
  // Returns an error if serialization fails.
  static absl::StatusOr<std::string> Serialize(
      const mlir::OwningOpRef<mlir::ModuleOp>& module_op);

 private:
  StablehloTypeSerialization() = delete;
};

// The name of the phase that the sample plugin implements. This name is used
// to register the phase with the `PjRtPhaseCompiler` and to identify the
// phase in the `PjRtPartialProgramProto` objects.
constexpr absl::string_view kPhaseName = "stablehlo_to_optimized_stablehlo";

// This class demonstrates an example phase compiler that the plugin developer
// needs to implement.
class SamplePhaseCompiler : public xla::PjRtPhaseCompiler {
 public:
  absl::Status RegisterAllPhases() final;

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      xla::CompileOptions options, const xla::XlaComputation& computation,
      const xla::PjRtTopologyDescription& topology,
      xla::PjRtClient* client) override;

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      xla::CompileOptions options, mlir::ModuleOp module,
      const xla::PjRtTopologyDescription& topology,
      xla::PjRtClient* client) override;
};

// Creates a phase compile extension for the sample plugin.
PJRT_PhaseCompile_Extension CreateSamplePhaseCompileExtension();

}  // namespace phase_compile_sample_plugin
}  // namespace pjrt

#endif  // XLA_PJRT_PJRT_PHASE_COMPILE_SAMPLE_PLUGIN_H_
