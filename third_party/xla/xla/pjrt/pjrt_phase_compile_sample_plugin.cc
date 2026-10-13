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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/messages/parse_message.h"
#include "riegeli/messages/serialize_message.h"
#include "stablehlo/api/PortableApi.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/layout_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/pjrt_relocatable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

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
    mlir::ModuleOp module_op) {
  if (!module_op) {
    return absl::InvalidArgumentError(
        "SHLOFormat serialization failed: MLIR module is null");
  }
  auto version = mlir::stablehlo::getCurrentVersion();
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  if (failed(mlir::stablehlo::serializePortableArtifact(module_op, version, os,
                                                        true))) {
    return absl::InvalidArgumentError(
        "SHLOFormat serialization failed: Could not serialize MLIR module");
  }
  return bytecode;
}

absl::StatusOr<std::string> StablehloTypeSerialization::Serialize(
    const mlir::OwningOpRef<mlir::ModuleOp>& module_op) {
  return Serialize(*module_op);
}

namespace {

constexpr absl::string_view kStablehloBytecodeFormat = "bytecode";

constexpr absl::string_view kNextPhaseName = "some_next_phase";

absl::Status PhaseValidator(
    xla::CompileOptions compile_options,
    const std::vector<xla::PjRtPartialProgramProto>& input_programs) {
  if (input_programs.empty()) {
    return absl::InvalidArgumentError("Input partial programs cannot be empty");
  }

  for (const auto& input_program : input_programs) {
    if (input_program.program_format() != kStablehloBytecodeFormat) {
      return absl::InvalidArgumentError(
          "Input programs are not in expected format.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>> PhaseCompiler(
    xla::CompileOptions compile_options,
    std::vector<xla::PjRtPartialProgramProto>&& input_programs,
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
    serialized_output_object.set_program_format(kStablehloBytecodeFormat);
    serialized_output_object.set_producer_phase(kPhaseName);
    serialized_output_object.add_consumer_phases({std::string(kNextPhaseName)});
    serialized_output_object.set_version("1.0");
    if (!input_program.program_name().empty()) {
      serialized_output_object.set_program_name(input_program.program_name());
    }

    serialized_output_objects.push_back(std::move(serialized_output_object));
  }
  return serialized_output_objects;
}

}  // namespace

/*static*/
absl::StatusOr<std::unique_ptr<SampleRelocatable>> SampleRelocatable::Create(
    std::vector<xla::PjRtPartialProgramProto> partial_programs) {
  if (partial_programs.empty()) {
    return absl::InvalidArgumentError(
        "SampleRelocatable requires at least one partial program.");
  }

  mlir::MLIRContext context;
  ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module_op,
                   StablehloTypeSerialization::Deserialize(
                       partial_programs[0].program(), context));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloModule> hlo_module,
                   xla::ConvertStablehloToHlo(*module_op));

  xla::ProgramShape shape =
      hlo_module->entry_computation()->ComputeProgramShape();
  if (!xla::LayoutUtil::HasLayout(shape)) {
    xla::LayoutUtil::SetToDefaultLayout(&shape);
  }

  std::vector<std::pair<int64_t, int64_t>> aliases;
  ABSL_RETURN_IF_ERROR(
      hlo_module->input_output_alias_config().ForEachAliasWithStatus(
          [&](const xla::ShapeIndex& output_index,
              const xla::HloInputOutputAliasConfig::Alias& alias)
              -> absl::Status {
            TF_RET_CHECK(output_index.size() <= 1)
                << "Nested tuple outputs are unsupported.";
            TF_RET_CHECK(alias.parameter_index.empty())
                << "Tuple inputs are unsupported.";
            int64_t out_idx = output_index.empty() ? 0 : output_index[0];
            aliases.emplace_back(alias.parameter_number, out_idx);
            return absl::OkStatus();
          }));

  bool has_side_effects = false;
  for (const xla::HloComputation* computation : hlo_module->computations()) {
    has_side_effects = has_side_effects || computation->HasSideEffect();
    if (has_side_effects) {
      break;
    }
  }

  uint64_t fp = 0;
  for (const xla::PjRtPartialProgramProto& prog : partial_programs) {
    fp = tsl::FingerprintCat64(fp, tsl::Fingerprint64(prog.program()));
  }
  std::string name = partial_programs[0].program_name();
  std::string build_id = absl::StrCat(absl::Hex(fp, absl::kZeroPad16));
  std::string call_backend_config =
      absl::StrCat("sample_backend_config:", build_id);

  return std::make_unique<SampleRelocatable>(
      name, build_id, std::move(shape), call_backend_config, std::move(aliases),
      has_side_effects, std::move(partial_programs));
}

absl::StatusOr<std::string> SampleRelocatable::Serialize() const {
  std::string serialized;
  riegeli::StringWriter<> writer(&serialized);
  for (const xla::PjRtPartialProgramProto& program : partial_programs_) {
    ABSL_RETURN_IF_ERROR(riegeli::SerializeLengthPrefixedMessage(
        program, writer,
        riegeli::SerializeMessageOptions().set_deterministic(true)));
  }
  if (!writer.Close()) {
    return writer.status();
  }
  return serialized;
}

/*static*/
absl::StatusOr<std::unique_ptr<SampleRelocatable>>
SampleRelocatable::Deserialize(absl::string_view serialized) {
  if (serialized.empty()) {
    return absl::InvalidArgumentError(
        "Serialized SampleRelocatable cannot be empty.");
  }
  riegeli::StringReader<> reader(serialized);
  std::vector<xla::PjRtPartialProgramProto> partial_programs;
  while (reader.Pull()) {
    xla::PjRtPartialProgramProto program;
    ABSL_RETURN_IF_ERROR(riegeli::ParseLengthPrefixedMessage(reader, program));
    partial_programs.push_back(std::move(program));
  }
  if (!reader.Close()) {
    return reader.status();
  }
  return Create(std::move(partial_programs));
}

absl::StatusOr<std::vector<xla::PrimitiveType>>
SampleRelocatable::GetParameterElementTypes() const {
  std::vector<xla::PrimitiveType> element_types;
  element_types.reserve(shape_.parameters().size());
  for (const xla::Shape& shape : shape_.parameters()) {
    TF_RET_CHECK(!shape.IsTuple())
        << "Relocatable has unflattened program shape.";
    element_types.push_back(shape.element_type());
  }
  return element_types;
}

absl::StatusOr<std::vector<xla::PrimitiveType>>
SampleRelocatable::GetOutputElementTypes() const {
  std::vector<xla::PrimitiveType> element_types;
  if (!shape_.result().IsTuple()) {
    element_types.push_back(shape_.result().element_type());
    return element_types;
  }
  element_types.reserve(shape_.result().tuple_shapes().size());
  for (const xla::Shape& shape : shape_.result().tuple_shapes()) {
    TF_RET_CHECK(!shape.IsTuple())
        << "Relocatable has unflattened program shape.";
    element_types.push_back(shape.element_type());
  }
  return element_types;
}

absl::StatusOr<std::vector<xla::DimensionVector>>
SampleRelocatable::GetParameterDimensions() const {
  std::vector<xla::DimensionVector> dimensions;
  dimensions.reserve(shape_.parameters().size());
  for (const xla::Shape& shape : shape_.parameters()) {
    TF_RET_CHECK(!shape.IsTuple())
        << "Relocatable has unflattened program shape.";
    absl::Span<const int64_t> shape_dims = shape.dimensions();
    dimensions.emplace_back(shape_dims.begin(), shape_dims.end());
  }
  return dimensions;
}

absl::StatusOr<std::vector<xla::DimensionVector>>
SampleRelocatable::GetOutputDimensions() const {
  std::vector<xla::DimensionVector> dimensions;
  if (!shape_.result().IsTuple()) {
    absl::Span<const int64_t> shape_dims = shape_.result().dimensions();
    dimensions.emplace_back(shape_dims.begin(), shape_dims.end());
    return dimensions;
  }
  dimensions.reserve(shape_.result().tuple_shapes().size());
  for (const xla::Shape& shape : shape_.result().tuple_shapes()) {
    TF_RET_CHECK(!shape.IsTuple())
        << "Relocatable has unflattened program shape.";
    absl::Span<const int64_t> shape_dims = shape.dimensions();
    dimensions.emplace_back(shape_dims.begin(), shape_dims.end());
  }
  return dimensions;
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
SampleRelocatable::GetParameterLayouts() const {
  std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
  layouts.reserve(shape_.parameters().size());
  for (const xla::Shape& shape : shape_.parameters()) {
    TF_RET_CHECK(!shape.IsTuple())
        << "Relocatable has unflattened program shape.";
    TF_RET_CHECK(shape.has_layout()) << "Relocatable is missing layouts.";
    layouts.push_back(std::make_shared<const xla::PjRtLayout>(shape.layout()));
  }
  return layouts;
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
SampleRelocatable::GetOutputLayouts() const {
  std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
  if (!shape_.result().IsTuple()) {
    TF_RET_CHECK(shape_.result().has_layout())
        << "Relocatable is missing layouts.";
    layouts.push_back(
        std::make_shared<const xla::PjRtLayout>(shape_.result().layout()));
    return layouts;
  }
  layouts.reserve(shape_.result().tuple_shapes().size());
  for (const xla::Shape& shape : shape_.result().tuple_shapes()) {
    TF_RET_CHECK(!shape.IsTuple())
        << "Relocatable has unflattened program shape.";
    TF_RET_CHECK(shape.has_layout()) << "Relocatable is missing layouts.";
    layouts.push_back(std::make_shared<const xla::PjRtLayout>(shape.layout()));
  }
  return layouts;
}

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
SamplePhaseCompiler::Compile(xla::CompileOptions options,
                             xla::MaybeOwningMlirModule module,
                             const xla::PjRtTopologyDescription& topology,
                             xla::PjRtClient* client) {
  return absl::UnimplementedError(
      "Compile with MLIR module is not implemented for sample phase "
      "compiler.");
}

absl::StatusOr<std::unique_ptr<xla::PjRtRelocatable>>
SamplePhaseCompiler::CompileToRelocatable(
    xla::CompileOptions options, xla::MaybeOwningMlirModule module,
    const xla::PjRtTopologyDescription& topology, absl::string_view name,
    bool is_entrypoint, xla::PjRtClient* client) {
  ABSL_ASSIGN_OR_RETURN(std::string serialized_module,
                   StablehloTypeSerialization::Serialize(module.mlir_module()));

  xla::PjRtPartialProgramProto partial_program;
  partial_program.set_program(std::move(serialized_module));
  partial_program.set_program_format(kStablehloBytecodeFormat);
  partial_program.set_producer_phase("n/a");
  partial_program.add_consumer_phases({std::string(kPhaseName)});
  partial_program.set_version("1.0");
  partial_program.set_program_name(name);

  ABSL_ASSIGN_OR_RETURN(std::vector<xla::PjRtPartialProgramProto> programs,
                   RunPhases(std::move(options), {std::move(partial_program)},
                             topology, {std::string(kPhaseName)}));

  return SampleRelocatable::Create(std::move(programs));
}

absl::StatusOr<std::unique_ptr<xla::PjRtRelocatable>>
SamplePhaseCompiler::DeserializeRelocatable(
    absl::string_view serialized) const {
  return SampleRelocatable::Deserialize(serialized);
}

PJRT_Error* PJRT_PhaseCompile_Get_Compiler(
    PJRT_PhaseCompile_Get_Compiler_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Get_Compiler_Args",
      PJRT_PhaseCompile_Get_Compiler_Args_STRUCT_SIZE, args->struct_size));

  auto phase_compiler = std::make_unique<SamplePhaseCompiler>();
  auto status = phase_compiler->RegisterAllPhases();
  if (!status.ok()) {
    return StatusToPjRtError(status);
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

const PJRT_Api* GetSamplePhaseCompilePjrtApi();

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  xla::CpuClientOptions options;
  options.cpu_device_count = 4;

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetXlaPjrtCpuClient(std::move(options)));
  args->client = pjrt::CreateWrapperClient(GetSamplePhaseCompilePjrtApi(),
                                           std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_CpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return StatusToPjRtError(
      absl::UnimplementedError("Topology not supported for CPU compilation."));
}

const PJRT_Api* GetSamplePhaseCompilePjrtApi() {
  static PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::phase_compile_sample_plugin::CreateSamplePhaseCompileExtension();

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      PJRT_Client_Create, nullptr, PJRT_CpuDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp, &phase_compile_extension.base,
      pjrt::PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}

}  // namespace phase_compile_sample_plugin
}  // namespace pjrt
