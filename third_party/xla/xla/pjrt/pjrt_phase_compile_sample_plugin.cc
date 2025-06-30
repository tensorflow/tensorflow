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

#include "xla/pjrt/pjrt_phase_compile_sample_plugin.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/api/PortableApi.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"

namespace pjrt {
namespace phase_compile_sample_plugin {

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
StablehloTypeSerialization::Deserialize(const std::string& program,
                                        mlir::MLIRContext& context) {
  tsl::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      mlir::stablehlo::deserializePortableArtifact(program, &context);
  absl::Status diagnostic_status = diagnostic_handler.consumeStatus();

  if (!module_op) {
    if (!diagnostic_status.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "SHLOFormat deserialization failed: ", diagnostic_status.message()));
    }
    return absl::InvalidArgumentError(
        "SHLOFormat deserialization failed: No specific MLIR diagnostic "
        "available");
  }

  return module_op;
}

absl::StatusOr<std::string> StablehloTypeSerialization::Serialize(
    const mlir::OwningOpRef<mlir::ModuleOp>& module_op) {
  auto version = mlir::stablehlo::getCurrentVersion();
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  if (failed(mlir::stablehlo::serializePortableArtifact(*module_op, version, os,
                                                        true))) {
    return absl::InvalidArgumentError(
        "SHLOFormat serialization failed: Could not serialize MLIR module");
  }
  return bytecode;
}

namespace {

enum SamplePartialProgramFormat { kStablehloBytecode = 0, kUnknown = -1 };

constexpr absl::string_view kNextPhaseName = "some_next_phase";

absl::Status PhaseValidator(
    const std::vector<xla::PjRtPartialProgramProto>& input_programs) {
  if (input_programs.empty()) {
    return absl::InvalidArgumentError("Input partial programs cannot be empty");
  }

  for (const auto& input_program : input_programs) {
    if (input_program.program_format() !=
        SamplePartialProgramFormat::kStablehloBytecode) {
      return absl::InvalidArgumentError(
          "Input programs are not in expected format.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>> PhaseCompiler(
    xla::CompileOptions compile_options,
    const std::vector<xla::PjRtPartialProgramProto>& input_programs,
    const xla::PjRtTopologyDescription& topology) {
  std::vector<xla::PjRtPartialProgramProto> serialized_output_objects;
  mlir::MLIRContext context;

  for (const auto& input_program : input_programs) {
    // Deserialize from PjRtPartialProgramProto to StableHLO module
    absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
        deserialized_input_status = StablehloTypeSerialization::Deserialize(
            input_program.program(), context);
    if (!deserialized_input_status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input deserialization failed: ",
                       deserialized_input_status.status().message()));
    }

    mlir::OwningOpRef<mlir::ModuleOp> current_module =
        std::move(deserialized_input_status.value());

    // Convert stablehlo to optimized stablehlo
    mlir::PassManager pm(current_module->getContext());
    mlir::GreedyRewriteConfig config;
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::stablehlo::createStablehloAggressiveSimplificationPass({},
                                                                     config));
    if (failed(pm.run(current_module.get()))) {
      return absl::InvalidArgumentError("Failed to simplify StableHLO module");
    }

    // Serialize to PjRtPartialProgramProto
    absl::StatusOr<std::string> serialized_output_status =
        StablehloTypeSerialization::Serialize(
            current_module);  // Pass OwningOpRef directly
    if (!serialized_output_status.ok()) {
      return absl::InternalError(
          absl::StrCat("Output serialization failed: ",
                       serialized_output_status.status().message()));
    }

    xla::PjRtPartialProgramProto serialized_output_object;
    serialized_output_object.set_program(serialized_output_status.value());
    serialized_output_object.set_program_format(
        static_cast<size_t>(SamplePartialProgramFormat::kStablehloBytecode));
    serialized_output_object.set_generating_phase(kPhaseName);
    serialized_output_object.add_next_phases({std::string(kNextPhaseName)});
    serialized_output_object.set_version("1.0");

    serialized_output_objects.push_back(std::move(serialized_output_object));
  }

  return serialized_output_objects;
}

}  // namespace

absl::Status SamplePhaseCompiler::RegisterAllPhases() {
  xla::CompilationPhaseFunctions phase_functions;
  phase_functions.compiler = PhaseCompiler;
  phase_functions.validator = PhaseValidator;
  return RegisterPhase(std::string(kPhaseName), std::move(phase_functions));
}

absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>>
SamplePhaseCompiler::Compile(xla::CompileOptions options,
                             const xla::XlaComputation& computation,
                             const xla::PjRtTopologyDescription& topology,
                             xla::PjRtClient* client) {
  return absl::UnimplementedError(
      "Compile with XlaComputation is not implemented for sample phase "
      "compiler.");
}

absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>>
SamplePhaseCompiler::Compile(xla::CompileOptions options, mlir::ModuleOp module,
                             const xla::PjRtTopologyDescription& topology,
                             xla::PjRtClient* client) {
  return absl::UnimplementedError(
      "Compile with MLIR module is not implemented for sample phase "
      "compiler.");
}

PJRT_Error* PJRT_PhaseCompile_Get_Compiler(
    PJRT_PhaseCompile_Get_Compiler_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Get_Compiler_Args",
      PJRT_PhaseCompile_Get_Compiler_Args_STRUCT_SIZE, args->struct_size));

  auto phase_compiler = std::make_unique<SamplePhaseCompiler>();
  auto status = phase_compiler->RegisterAllPhases();
  if (!status.ok()) {
    return new PJRT_Error{status};
  }

  args->phase_compiler = new PJRT_PhaseCompiler{std::move(phase_compiler)};
  return nullptr;
}

void PJRT_PhaseCompile_Destroy_Compiler(
    PJRT_PhaseCompile_Destroy_Compiler_Args* args) {
  delete args->phase_compiler;
}

PJRT_PhaseCompile_Extension CreateSamplePhaseCompileExtension() {
  return pjrt::CreatePhaseCompileExtension(nullptr,
                                           PJRT_PhaseCompile_Get_Compiler,
                                           PJRT_PhaseCompile_Destroy_Compiler);
}

}  // namespace phase_compile_sample_plugin
}  // namespace pjrt
