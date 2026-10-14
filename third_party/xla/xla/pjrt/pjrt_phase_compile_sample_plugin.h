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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/pjrt_relocatable.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

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
  static absl::StatusOr<std::string> Serialize(mlir::ModuleOp module_op);
  static absl::StatusOr<std::string> Serialize(
      const mlir::OwningOpRef<mlir::ModuleOp>& module_op);

 private:
  StablehloTypeSerialization() = delete;
};

// The name of the phase that the sample plugin implements. This name is used
// to register the phase with the `PjRtPhaseCompiler` and to identify the
// phase in the `PjRtPartialProgramProto` objects.
constexpr absl::string_view kPhaseName = "stablehlo_to_optimized_stablehlo";

// Sample implementation of `PjRtRelocatable` that contains a mock-optimized
// StableHLO program inside its set of partial programs, along with any
// dependent StableHLO programs.
class SampleRelocatable : public xla::PjRtRelocatable {
 public:
  SampleRelocatable(
      absl::string_view name, absl::string_view build_id,
      xla::ProgramShape shape, absl::string_view call_backend_config,
      std::vector<std::pair<int64_t, int64_t>> param_output_aliases,
      bool has_side_effects,
      std::vector<xla::PjRtPartialProgramProto> partial_programs)
      : name_(name),
        build_id_(build_id),
        shape_(std::move(shape)),
        call_backend_config_(call_backend_config),
        param_output_aliases_(std::move(param_output_aliases)),
        has_side_effects_(has_side_effects),
        partial_programs_(std::move(partial_programs)) {}

  ~SampleRelocatable() override = default;

  static absl::StatusOr<std::unique_ptr<SampleRelocatable>> Create(
      std::vector<xla::PjRtPartialProgramProto> partial_programs);

  absl::string_view name() const override { return name_; }
  absl::string_view build_id() const override { return build_id_; }
  absl::string_view call_backend_config() const override {
    return call_backend_config_;
  }
  bool has_side_effects() const override { return has_side_effects_; }
  const xla::ProgramShape& shape() const { return shape_; }
  const std::vector<xla::PjRtPartialProgramProto>& partial_programs() const {
    return partial_programs_;
  }

  absl::StatusOr<std::string> Serialize() const override;
  static absl::StatusOr<std::unique_ptr<SampleRelocatable>> Deserialize(
      absl::string_view serialized);

  absl::StatusOr<std::vector<xla::PrimitiveType>> GetParameterElementTypes()
      const override;
  absl::StatusOr<std::vector<xla::PrimitiveType>> GetOutputElementTypes()
      const override;
  absl::StatusOr<std::vector<xla::DimensionVector>> GetParameterDimensions()
      const override;
  absl::StatusOr<std::vector<xla::DimensionVector>> GetOutputDimensions()
      const override;
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const override;
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const override;
  absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>>
  GetParameterOutputAliases() const override {
    return param_output_aliases_;
  }

 private:
  std::string name_;
  std::string build_id_;
  xla::ProgramShape shape_;
  std::string call_backend_config_;
  std::vector<std::pair<int64_t, int64_t>> param_output_aliases_;
  bool has_side_effects_;
  std::vector<xla::PjRtPartialProgramProto> partial_programs_;
};

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
      xla::CompileOptions options, xla::MaybeOwningMlirModule module,
      const xla::PjRtTopologyDescription& topology,
      xla::PjRtClient* client) override;

  absl::StatusOr<std::unique_ptr<xla::PjRtRelocatable>> CompileToRelocatable(
      xla::CompileOptions options, xla::MaybeOwningMlirModule module,
      const xla::PjRtTopologyDescription& topology, absl::string_view name,
      bool is_entrypoint, xla::PjRtClient* client) override;

  absl::StatusOr<std::unique_ptr<xla::PjRtRelocatable>> DeserializeRelocatable(
      absl::string_view serialized) const override;
};

// Creates a phase compile extension for the sample plugin.
PJRT_PhaseCompile_Extension CreateSamplePhaseCompileExtension();

const PJRT_Api* GetSamplePhaseCompilePjrtApi();

}  // namespace phase_compile_sample_plugin
}  // namespace pjrt

#endif  // XLA_PJRT_PJRT_PHASE_COMPILE_SAMPLE_PLUGIN_H_
