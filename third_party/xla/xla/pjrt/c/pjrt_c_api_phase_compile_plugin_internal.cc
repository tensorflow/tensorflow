/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/c/pjrt_c_api_phase_compile_plugin_internal.h"

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
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"

namespace pjrt {
namespace phase_compile_cpu_plugin {

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
StableHLOTypeSerialization::deserialize(const std::string& program,
                                        mlir::MLIRContext& context) {
  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      mlir::stablehlo::deserializePortableArtifact(program, &context);

  if (!module_op) {
    return absl::InvalidArgumentError(
        "SHLOFormat deserialization failed: Invalid StableHLO artifact");
  }
  return module_op;
}

absl::StatusOr<std::string> StableHLOTypeSerialization::serialize(
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

enum class PjRtPartialProgramFormat { kStablehloBytecode = 0, kUnknown = -1 };
constexpr absl::string_view kPhaseName = "stablehlo_to_optimized_stablehlo";
constexpr absl::string_view kNextPhaseName = "some_next_phase";

absl::Status PhaseValidator(
    const std::vector<xla::PjRtPartialProgramProto>& input_programs) {
  for (const auto& input_program : input_programs) {
    if (input_program.program_format() !=
        static_cast<size_t>(PjRtPartialProgramFormat::kStablehloBytecode)) {
      return absl::InvalidArgumentError(
          "Input programs are not in SHLO format.");
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
        deserialized_input_status = StableHLOTypeSerialization::deserialize(
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
        StableHLOTypeSerialization::serialize(
            current_module);  // Pass OwningOpRef directly
    if (!serialized_output_status.ok()) {
      return absl::InternalError(
          absl::StrCat("Output serialization failed: ",
                       serialized_output_status.status().message()));
    }

    xla::PjRtPartialProgramProto serialized_output_object;
    serialized_output_object.set_program(serialized_output_status.value());
    serialized_output_object.set_program_format(
        static_cast<size_t>(PjRtPartialProgramFormat::kStablehloBytecode));
    serialized_output_object.set_generating_phase(std::string(kPhaseName));
    serialized_output_object.add_next_phases({std::string(kNextPhaseName)});
    serialized_output_object.set_version("1.0");

    serialized_output_objects.push_back(std::move(serialized_output_object));
  }

  return serialized_output_objects;
}

class PhaseCompileCpuCompiler : public xla::PjRtPhaseCompiler {
 public:
  static PhaseCompileCpuCompiler* Create() {
    auto phase_compiler = new PhaseCompileCpuCompiler();
    auto status = phase_compiler->RegisterPhase(std::string(kPhaseName),
                                                PhaseCompiler, PhaseValidator);
    if (!status.ok()) {
      return nullptr;
    }

    return phase_compiler;
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      xla::CompileOptions options, const xla::XlaComputation& computation,
      const xla::PjRtTopologyDescription& topology,
      xla::PjRtClient* client) override {
    return absl::UnimplementedError(
        "Compile with XlaComputation is not implemented for phase compile "
        "CPU.");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      xla::CompileOptions options, mlir::ModuleOp module,
      const xla::PjRtTopologyDescription& topology,
      xla::PjRtClient* client) override {
    return absl::UnimplementedError(
        "Compile with MLIR module is not implemented for phase compile CPU.");
  }

 private:
  PhaseCompileCpuCompiler() = default;
};

PJRT_Error* PJRT_Cpu_Topology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_TopologyDescription_Create_Args",
      PJRT_TopologyDescription_Create_Args_STRUCT_SIZE, args->struct_size));

  std::vector<std::string> machine_attributes;
  machine_attributes.push_back("abc");
  auto cpu_devices = std::vector<xla::CpuTopology::CpuDevice>();
  auto topology_description = std::make_unique<xla::CpuTopologyDescription>(
      xla::CpuId(), xla::CpuName(), "<unknown>", cpu_devices,
      machine_attributes);
  args->topology =
      pjrt::CreateWrapperDeviceTopology(std::move(topology_description));
  return nullptr;
}

PJRT_Error* PJRT_PhaseCompile_Get_Compiler(
    PJRT_PhaseCompile_Get_Compiler_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Get_Compiler_Args",
      PJRT_PhaseCompile_Get_Compiler_Args_STRUCT_SIZE, args->struct_size));

  static auto* phase_compiler = PhaseCompileCpuCompiler::Create();
  if (phase_compiler == nullptr) {
    return new PJRT_Error{
        absl::InternalError("Failed to create phase compiler.")};
  }
  args->phase_compiler = new PJRT_PhaseCompiler{phase_compiler};
  return nullptr;
}

}  // namespace

// Helper to register the phase compile CPU compiler and create a PJRT API
// with the phase compile extension.
const PJRT_Api* GetPhaseCompilePjrtApi() {
  static PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::CreatePhaseCompileExtension(nullptr,
                                        PJRT_PhaseCompile_Get_Compiler);

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      /*create_fn=*/nullptr, /*execute_context_create_fn=*/nullptr,
      PJRT_Cpu_Topology_Create, pjrt::PJRT_Plugin_Initialize_NoOp,
      &phase_compile_extension.base, pjrt::PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}

}  // namespace phase_compile_cpu_plugin
}  // namespace pjrt
