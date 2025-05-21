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

#include "xla/pjrt/c/pjrt_c_api_phase_compile_cpu_internal.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_partial_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_partial_compile_internal.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_partial_program.h"

namespace pjrt {
namespace phase_compile_cpu_plugin {

namespace {

enum class PjRtPartialProgramFormat { kStablehloBytecode = 0, kUnknown = -1 };

constexpr absl::string_view kPhaseName = "stablehlo_to_optimized_stablehlo";

absl::Status PhaseValidator(
    const std::vector<xla::PjRtPartialProgram>& input_programs) {
  for (const auto& input_program : input_programs) {
    if (input_program.GetFormat() !=
        static_cast<size_t>(PjRtPartialProgramFormat::kStablehloBytecode)) {
      return absl::InvalidArgumentError(
          "Input programs are not in SHLO format.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<xla::PjRtPartialProgram>> PhaseCompiler(
    xla::CompileOptions compile_options,
    const std::vector<xla::PjRtPartialProgram>& input_programs) {
  std::vector<xla::PjRtPartialProgram> serialized_output_objects;
  mlir::MLIRContext context;

  for (const auto& input_program : input_programs) {
    // Deserialize from PjRtPartialProgram to StableHLO module
    absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
        deserialized_input_status = StableHLOTypeSerialization::deserialize(
            input_program.GetProgram(), context);
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

    // Serialize to PjRtPartialProgram
    absl::StatusOr<std::string> serialized_output_status =
        StableHLOTypeSerialization::serialize(
            current_module);  // Pass OwningOpRef directly
    if (!serialized_output_status.ok()) {
      return absl::InternalError(
          absl::StrCat("Output serialization failed: ",
                       serialized_output_status.status().message()));
    }

    xla::PjRtPartialProgram serialized_output_object;
    serialized_output_object.SetProgram(serialized_output_status.value());
    serialized_output_object.SetFormat(
        static_cast<size_t>(PjRtPartialProgramFormat::kStablehloBytecode));
    serialized_output_object.SetGeneratingPhase(std::string(kPhaseName));
    serialized_output_object.SetNextPhases({"stablehlo_to_hlo"});
    serialized_output_object.SetVersion("1.0");

    serialized_output_objects.push_back(std::move(serialized_output_object));
  }

  return serialized_output_objects;
}

}  // namespace

PJRT_Error* PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "ExecuteContext not implemented for phase compile CPU.")};
}

PJRT_Error* PJRT_DeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "Topology not implemented for phase compile CPU.")};
}

const PJRT_Api* GetPhaseCompileForCpuPjrtApi() {
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(nullptr);

  // Create phases
  auto compiler = std::make_unique<xla::PjRtPhaseCompiler>();
  auto status = compiler->RegisterPhase(std::string(kPhaseName), PhaseCompiler,
                                        PhaseValidator);
  if (!status.ok()) {
    absl::FPrintF(stderr, "Failed to register partial compiler: %s\n",
                  status.message());
    return nullptr;
  }
  xla::PjRtRegisterCompiler("partial_compile", std::move(compiler));

  // Create partial compile extension
  static PJRT_PhaseCompile_Extension partial_compile_extension =
      pjrt::CreatePhaseCompileExtension(&layouts_extension.base);

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      nullptr, PJRT_ExecuteContext_Create, PJRT_DeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp, &partial_compile_extension.base,
      pjrt::PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}

}  // namespace phase_compile_cpu_plugin
}  // namespace pjrt
