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

#include "xla/pjrt/c_api_client/pjrt_c_api_phase_compiler.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/partial_program_utils.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/pjrt_relocatable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/pjrt/string_utils.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

// Checks if the input partial programs are compatible with the first phase in
// the list of phases to run.
absl::Status ValidatePhases(
    const std::vector<xla::PjRtPartialProgramProto>& partial_programs_in,
    const std::vector<std::string>& phases_to_run) {
  if (phases_to_run.empty()) {
    return absl::InvalidArgumentError("Phases to run cannot be empty");
  }

  for (const auto& partial_program : partial_programs_in) {
    if (!absl::c_contains(partial_program.consumer_phases(),
                          phases_to_run[0])) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input partial programs cannot be compiled by a phase with name \"",
          phases_to_run[0], "\""));
    }
  }
  return absl::OkStatus();
}

// Cleans up C buffers allocated by the caller-side code and passed to the
// plugin-side code. Example buffers include those used to propagate
// input programs, names of phases to run.
void CleanUpCallerDefinedCBuffers(absl::Span<const char* const> char_buffers) {
  for (const char* buffer : char_buffers) {
    delete[] buffer;
  }
}

// Cleans up C buffers allocated by the plugin-side code and passed to the
// caller-side code. Example buffers include those used to propagate the
// compiled output programs.
void CleanUpPluginDefinedCBuffers(
    const char** char_buffers, const size_t* char_buffer_sizes,
    size_t num_char_buffers,
    const PJRT_PhaseCompile_Extension* phase_compile_extension) {
  PJRT_PhaseCompile_C_Buffers_Destroy_Args destroy_args;
  destroy_args.struct_size =
      PJRT_PhaseCompile_C_Buffers_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.char_buffers = char_buffers;
  destroy_args.char_buffer_sizes = char_buffer_sizes;
  destroy_args.num_char_buffers = num_char_buffers;
  phase_compile_extension->phase_compile_c_buffers_destroy(&destroy_args);
}

}  // namespace

PjRtCApiPhaseCompiler::PjRtCApiPhaseCompiler(
    const PJRT_Api* api,
    const PJRT_PhaseCompile_Extension* phase_compile_extension,
    const PJRT_PhaseCompiler* phase_compiler)
    : api_(api),
      phase_compile_extension_(phase_compile_extension),
      phase_compiler_(
          phase_compiler,
          [phase_compile_extension](const PJRT_PhaseCompiler* p_compiler) {
            PJRT_PhaseCompile_Destroy_Compiler_Args destroy_args;
            destroy_args.struct_size =
                PJRT_PhaseCompile_Destroy_Compiler_Args_STRUCT_SIZE;
            destroy_args.extension_start = nullptr;
            destroy_args.phase_compiler = p_compiler;
            phase_compile_extension->phase_compile_destroy_compiler(
                &destroy_args);
          }) {}

// Returns the reference to the phase compiler.
const PJRT_PhaseCompiler* PjRtCApiPhaseCompiler::GetPhaseCompiler() const {
  return phase_compiler_.get();
}

absl::StatusOr<std::vector<std::string>>
PjRtCApiPhaseCompiler::GetPhaseNames() {
  PJRT_PhaseCompile_Get_PhaseNames_Args args;
  args.struct_size = PJRT_PhaseCompile_Get_PhaseNames_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.phase_compiler = GetPhaseCompiler();
  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->phase_compile_get_phase_names(&args), api_);

  std::vector<std::string> phase_names = xla::ConvertCharBuffersToCppStrings(
      absl::MakeSpan(args.phase_names, args.num_phase_names),
      absl::MakeConstSpan(args.phase_names_sizes, args.num_phase_names));
  CleanUpPluginDefinedCBuffers(args.phase_names, args.phase_names_sizes,
                               args.num_phase_names, phase_compile_extension_);

  return phase_names;
}

absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>>
PjRtCApiPhaseCompiler::RunPhases(
    xla::CompileOptions options,
    std::vector<xla::PjRtPartialProgramProto>&& partial_programs_in,
    const xla::PjRtTopologyDescription& topology,
    const std::vector<std::string>& phases_to_run) {
  // Plugin-agnostic validation of the input programs and phases.
  ABSL_RETURN_IF_ERROR(ValidatePhases(partial_programs_in, phases_to_run));

  PJRT_TopologyDescription* topology_description =
      absl::down_cast<const PjRtCApiTopologyDescription*>(&topology)
          ->c_topology();

  const size_t* programs_in_buffer_sizes;
  ABSL_ASSIGN_OR_RETURN(
      const char** programs_in_buffers,
      xla::ConvertPjRtPartialProgramProtosToCharBuffers(
          absl::MakeSpan(partial_programs_in), programs_in_buffer_sizes));
  size_t num_programs_in = partial_programs_in.size();

  // We no longer need this form of the input, so free it to save memory.
  partial_programs_in = std::vector<xla::PjRtPartialProgramProto>();

  absl::Cleanup cleanup_programs_in_buffers =
      [programs_in_buffers, programs_in_buffer_sizes, num_programs_in] {
        CleanUpCallerDefinedCBuffers(
            absl::MakeConstSpan(programs_in_buffers, num_programs_in));
        delete[] programs_in_buffer_sizes;
        delete[] programs_in_buffers;
      };

  ABSL_ASSIGN_OR_RETURN(const xla::CompileOptionsProto options_proto,
                   options.ToProto());
  std::string options_str = options_proto.SerializeAsString();

  const size_t* phases_to_run_buffer_sizes;
  size_t num_phases_to_run = phases_to_run.size();
  const char** phases_to_run_buffers = xla::ConvertCppStringsToCharBuffers(
      phases_to_run, phases_to_run_buffer_sizes);
  absl::Cleanup cleanup_phases_to_run_buffers =
      [phases_to_run_buffers, phases_to_run_buffer_sizes, num_phases_to_run]() {
        CleanUpCallerDefinedCBuffers(
            absl::MakeConstSpan(phases_to_run_buffers, num_phases_to_run));
        delete[] phases_to_run_buffer_sizes;
        delete[] phases_to_run_buffers;
      };

  PJRT_PhaseCompile_Run_Phase_Args run_args;
  run_args.struct_size = PJRT_PhaseCompile_Run_Phase_Args_STRUCT_SIZE;
  run_args.extension_start = nullptr;
  run_args.phase_compiler = GetPhaseCompiler();
  run_args.input_programs = programs_in_buffers;
  run_args.input_programs_sizes = programs_in_buffer_sizes;
  run_args.num_input_programs = num_programs_in;
  run_args.phases_to_run = phases_to_run_buffers;
  run_args.phases_to_run_sizes = phases_to_run_buffer_sizes;
  run_args.num_phases_to_run = num_phases_to_run;
  run_args.compile_options = options_str.c_str();
  run_args.compile_options_size = options_str.size();
  run_args.topology = topology_description;

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->phase_compile_run_phases(&run_args), api_);

  // Free caller-allocated input buffers before deserializing the output buffers
  // to avoid holding both input and output C buffers simultaneously in memory.
  std::move(cleanup_programs_in_buffers).Invoke();
  std::move(cleanup_phases_to_run_buffers).Invoke();

  absl::Cleanup cleanup_output_buffers = [this, &run_args] {
    CleanUpPluginDefinedCBuffers(
        run_args.output_programs, run_args.output_programs_sizes,
        run_args.num_output_programs, phase_compile_extension_);
  };
  ABSL_ASSIGN_OR_RETURN(std::vector<xla::PjRtPartialProgramProto> output_programs,
                   xla::ConvertCharBuffersToPjRtPartialProgramProtos(
                       absl::MakeSpan(run_args.output_programs,
                                      run_args.num_output_programs),
                       absl::MakeConstSpan(run_args.output_programs_sizes,
                                           run_args.num_output_programs)));

  return output_programs;
}

absl::StatusOr<std::unique_ptr<PjRtRelocatable>>
PjRtCApiPhaseCompiler::CompileToRelocatable(
    CompileOptions options, MaybeOwningMlirModule module,
    const PjRtTopologyDescription& topology, absl::string_view name,
    bool is_entrypoint, PjRtClient* client) {
  if (!module.mlir_module()) {
    return absl::InvalidArgumentError(
        "PjRtCApiPhaseCompiler::CompileToRelocatable: MLIR module is null");
  }

  PJRT_TopologyDescription* topology_description =
      absl::down_cast<const PjRtCApiTopologyDescription*>(&topology)
          ->c_topology();

  ABSL_ASSIGN_OR_RETURN(const CompileOptionsProto options_proto, options.ToProto());
  std::string options_str = options_proto.SerializeAsString();

  std::string target_version = GetPluginStablehloVersionOrDefault(api_, client);
  std::string sdy_version = GetPluginSdyVersionOrDefault(api_, client);
  ABSL_ASSIGN_OR_RETURN(
      std::string code,
      xla::Serialize(module.mlir_module(), target_version, sdy_version,
                     /*inplace=*/options.allow_in_place_mlir_modification));
  std::string format(pjrt::kMlirFormat);

  PJRT_Program program;
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.extension_start = nullptr;
  program.code = code.data();
  program.code_size = code.size();
  program.format = format.c_str();
  program.format_size = format.size();

  PJRT_PhaseCompile_CompileToRelocatable_Args args;
  args.struct_size = PJRT_PhaseCompile_CompileToRelocatable_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.phase_compiler = GetPhaseCompiler();
  args.topology = topology_description;
  args.program = &program;
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();
  args.name = name.data();
  args.name_size = name.size();
  args.is_entrypoint = is_entrypoint;
  if (client == nullptr) {
    args.client = nullptr;
  } else {
    args.client = absl::down_cast<PjRtCApiClient*>(client)->pjrt_c_client();
  }
  args.relocatable = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->phase_compile_compile_to_relocatable(&args),
      api_);
  return std::make_unique<PjRtCApiRelocatable>(api_, phase_compile_extension_,
                                               args.relocatable);
}

absl::StatusOr<std::unique_ptr<PjRtRelocatable>>
PjRtCApiPhaseCompiler::DeserializeRelocatable(
    absl::string_view serialized) const {
  PJRT_PhaseCompile_DeserializeRelocatable_Args args;
  args.struct_size = PJRT_PhaseCompile_DeserializeRelocatable_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.phase_compiler = GetPhaseCompiler();
  args.serialized_relocatable = serialized.data();
  args.serialized_relocatable_size = serialized.size();
  args.relocatable = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->phase_compile_deserialize_relocatable(&args),
      api_);
  return std::make_unique<PjRtCApiRelocatable>(api_, phase_compile_extension_,
                                               args.relocatable);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
PjRtCApiPhaseCompiler::LinkRelocatables(
    CompileOptions options, const PjRtTopologyDescription& topology,
    std::shared_ptr<const PjRtRelocatable> entrypoint,
    std::vector<std::shared_ptr<const PjRtRelocatable>> dependencies) {
  PJRT_TopologyDescription* topology_description =
      absl::down_cast<const PjRtCApiTopologyDescription*>(&topology)
          ->c_topology();

  ABSL_ASSIGN_OR_RETURN(const CompileOptionsProto options_proto, options.ToProto());
  std::string options_str = options_proto.SerializeAsString();

  const PJRT_Relocatable* c_entrypoint = nullptr;
  if (entrypoint != nullptr) {
    c_entrypoint = absl::down_cast<const PjRtCApiRelocatable*>(entrypoint.get())
                       ->c_relocatable();
  }

  std::vector<const PJRT_Relocatable*> c_dependencies;
  c_dependencies.reserve(dependencies.size());
  for (const auto& dep : dependencies) {
    if (dep != nullptr) {
      c_dependencies.push_back(
          absl::down_cast<const PjRtCApiRelocatable*>(dep.get())
              ->c_relocatable());
    } else {
      c_dependencies.push_back(nullptr);
    }
  }

  PJRT_PhaseCompile_LinkRelocatables_Args args;
  args.struct_size = PJRT_PhaseCompile_LinkRelocatables_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.phase_compiler = GetPhaseCompiler();
  args.topology = topology_description;
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();
  args.entrypoint = c_entrypoint;
  args.dependencies = c_dependencies.data();
  args.num_dependencies = c_dependencies.size();
  args.executable = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->phase_compile_link_relocatables(&args), api_);
  return std::make_unique<PjRtCApiExecutable>(api_, args.executable);
}

// ------------------------------- Relocatables --------------------------------

PjRtCApiRelocatable::PjRtCApiRelocatable(
    const PJRT_Api* c_api,
    const PJRT_PhaseCompile_Extension* phase_compile_extension,
    PJRT_Relocatable* relocatable)
    : c_api_(c_api),
      phase_compile_extension_(phase_compile_extension),
      relocatable_(relocatable, [c_api, phase_compile_extension](
                                    PJRT_Relocatable* reloc) {
        PJRT_Relocatable_Destroy_Args destroy_args;
        destroy_args.struct_size = PJRT_Relocatable_Destroy_Args_STRUCT_SIZE;
        destroy_args.extension_start = nullptr;
        destroy_args.relocatable = reloc;
        pjrt::LogFatalIfPjrtError(
            phase_compile_extension->relocatable_destroy(&destroy_args), c_api);
      }) {}

absl::string_view PjRtCApiRelocatable::name() const {
  PJRT_Relocatable_Name_Args args;
  args.struct_size = PJRT_Relocatable_Name_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.relocatable = c_relocatable();
  pjrt::LogFatalIfPjrtError(phase_compile_extension_->relocatable_name(&args),
                            c_api_);
  return absl::string_view(args.relocatable_name, args.relocatable_name_size);
}

absl::string_view PjRtCApiRelocatable::build_id() const {
  PJRT_Relocatable_BuildId_Args args;
  args.struct_size = PJRT_Relocatable_BuildId_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.relocatable = c_relocatable();
  pjrt::LogFatalIfPjrtError(
      phase_compile_extension_->relocatable_build_id(&args), c_api_);
  return absl::string_view(args.build_id, args.build_id_size);
}

absl::string_view PjRtCApiRelocatable::call_backend_config() const {
  PJRT_Relocatable_CallBackendConfig_Args args;
  args.struct_size = PJRT_Relocatable_CallBackendConfig_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.relocatable = c_relocatable();
  pjrt::LogFatalIfPjrtError(
      phase_compile_extension_->relocatable_call_backend_config(&args), c_api_);
  return absl::string_view(args.call_backend_config,
                           args.call_backend_config_size);
}

bool PjRtCApiRelocatable::has_side_effects() const {
  PJRT_Relocatable_HasSideEffects_Args args;
  args.struct_size = PJRT_Relocatable_HasSideEffects_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.relocatable = c_relocatable();
  pjrt::LogFatalIfPjrtError(
      phase_compile_extension_->relocatable_has_side_effects(&args), c_api_);
  return args.has_side_effects;
}

absl::StatusOr<std::string> PjRtCApiRelocatable::Serialize() const {
  PJRT_Relocatable_Serialize_Args ser_args;
  ser_args.struct_size = PJRT_Relocatable_Serialize_Args_STRUCT_SIZE;
  ser_args.extension_start = nullptr;
  ser_args.relocatable = c_relocatable();
  ser_args.serialized_relocatable = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->relocatable_serialize(&ser_args), c_api_);
  absl::Cleanup cleanup = [&ser_args] {
    ser_args.serialized_relocatable_deleter(ser_args.serialized_relocatable);
  };
  return std::string(ser_args.serialized_bytes, ser_args.serialized_bytes_size);
}

absl::StatusOr<std::vector<PrimitiveType>>
PjRtCApiRelocatable::GetParameterElementTypes() const {
  PJRT_Relocatable_ParameterElementTypes_Args args;
  args.struct_size = PJRT_Relocatable_ParameterElementTypes_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.relocatable = c_relocatable();

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->relocatable_parameter_element_types(&args),
      c_api_);

  std::vector<PrimitiveType> out;
  out.reserve(args.num_parameter_types);
  for (size_t i = 0; i < args.num_parameter_types; ++i) {
    out.push_back(pjrt::ConvertFromPjRtBufferType(args.parameter_types[i]));
  }
  return out;
}

absl::StatusOr<std::vector<PrimitiveType>>
PjRtCApiRelocatable::GetOutputElementTypes() const {
  PJRT_Relocatable_OutputElementTypes_Args args;
  args.struct_size = PJRT_Relocatable_OutputElementTypes_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.relocatable = c_relocatable();

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->relocatable_output_element_types(&args),
      c_api_);

  std::vector<PrimitiveType> out;
  out.reserve(args.num_output_types);
  for (size_t i = 0; i < args.num_output_types; ++i) {
    out.push_back(pjrt::ConvertFromPjRtBufferType(args.output_types[i]));
  }
  return out;
}

absl::StatusOr<std::vector<DimensionVector>>
PjRtCApiRelocatable::GetParameterDimensions() const {
  PJRT_Relocatable_ParameterDimensions_Args args;
  args.struct_size = PJRT_Relocatable_ParameterDimensions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.relocatable = c_relocatable();

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->relocatable_parameter_dimensions(&args),
      c_api_);

  std::vector<DimensionVector> out;
  out.reserve(args.num_parameters);
  size_t index = 0;
  for (size_t i = 0; i < args.num_parameters; ++i) {
    DimensionVector dimensions;
    dimensions.reserve(args.dim_sizes[i]);
    for (size_t j = 0; j < args.dim_sizes[i]; ++j) {
      dimensions.push_back(args.dims[index++]);
    }
    out.push_back(std::move(dimensions));
  }
  return out;
}

absl::StatusOr<std::vector<DimensionVector>>
PjRtCApiRelocatable::GetOutputDimensions() const {
  PJRT_Relocatable_OutputDimensions_Args args;
  args.struct_size = PJRT_Relocatable_OutputDimensions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.relocatable = c_relocatable();

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->relocatable_output_dimensions(&args), c_api_);

  std::vector<DimensionVector> out;
  out.reserve(args.num_outputs);
  size_t index = 0;
  for (size_t i = 0; i < args.num_outputs; ++i) {
    DimensionVector dimensions;
    dimensions.reserve(args.dim_sizes[i]);
    for (size_t j = 0; j < args.dim_sizes[i]; ++j) {
      dimensions.push_back(args.dims[index++]);
    }
    out.push_back(std::move(dimensions));
  }
  return out;
}

absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
PjRtCApiRelocatable::GetParameterLayouts() const {
  PJRT_Layouts_Extension* layouts_extension =
      pjrt::FindExtension<PJRT_Layouts_Extension>(
          c_api_, PJRT_Extension_Type::PJRT_Extension_Type_Layouts);
  if (layouts_extension == nullptr ||
      layouts_extension->PJRT_Layouts_MemoryLayout_Serialize == nullptr) {
    return absl::UnimplementedError(
        "PJRT_Layouts_Extension is required to serialize relocatable "
        "parameter layouts.");
  }

  PJRT_Relocatable_ParameterLayouts_Args args;
  args.struct_size = PJRT_Relocatable_ParameterLayouts_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.relocatable = c_relocatable();
  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->relocatable_parameter_layouts(&args), c_api_);

  std::vector<std::shared_ptr<const PjRtLayout>> layouts;
  layouts.reserve(args.num_parameters);
  for (size_t i = 0; i < args.num_parameters; ++i) {
    PJRT_Layouts_MemoryLayout_Serialize_Args serialize_args;
    serialize_args.struct_size =
        PJRT_Layouts_MemoryLayout_Serialize_Args_STRUCT_SIZE;
    serialize_args.extension_start = nullptr;
    serialize_args.layout = args.layouts[i];
    pjrt::LogFatalIfPjrtError(
        layouts_extension->PJRT_Layouts_MemoryLayout_Serialize(&serialize_args),
        c_api_);

    absl::Cleanup cleanup = [&serialize_args] {
      serialize_args.serialized_layout_deleter(
          serialize_args.serialized_layout);
    };

    std::string serialized_layout(serialize_args.serialized_bytes,
                                  serialize_args.serialized_bytes_size);
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<const PjRtLayout> pjrt_layout,
                     PjRtLayout::Deserialize(serialized_layout));
    layouts.push_back(std::move(pjrt_layout));
  }
  return layouts;
}

absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
PjRtCApiRelocatable::GetOutputLayouts() const {
  PJRT_Layouts_Extension* layouts_extension =
      pjrt::FindExtension<PJRT_Layouts_Extension>(
          c_api_, PJRT_Extension_Type::PJRT_Extension_Type_Layouts);
  if (layouts_extension == nullptr ||
      layouts_extension->PJRT_Layouts_MemoryLayout_Serialize == nullptr) {
    return absl::UnimplementedError(
        "PJRT_Layouts_Extension is required to serialize relocatable "
        "output layouts.");
  }

  PJRT_Relocatable_OutputLayouts_Args args;
  args.struct_size = PJRT_Relocatable_OutputLayouts_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.relocatable = c_relocatable();
  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->relocatable_output_layouts(&args), c_api_);

  std::vector<std::shared_ptr<const PjRtLayout>> layouts;
  layouts.reserve(args.num_outputs);
  for (size_t i = 0; i < args.num_outputs; ++i) {
    PJRT_Layouts_MemoryLayout_Serialize_Args serialize_args;
    serialize_args.struct_size =
        PJRT_Layouts_MemoryLayout_Serialize_Args_STRUCT_SIZE;
    serialize_args.extension_start = nullptr;
    serialize_args.layout = args.layouts[i];
    pjrt::LogFatalIfPjrtError(
        layouts_extension->PJRT_Layouts_MemoryLayout_Serialize(&serialize_args),
        c_api_);

    absl::Cleanup cleanup = [&serialize_args] {
      serialize_args.serialized_layout_deleter(
          serialize_args.serialized_layout);
    };

    std::string serialized_layout(serialize_args.serialized_bytes,
                                  serialize_args.serialized_bytes_size);
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<const PjRtLayout> pjrt_layout,
                     PjRtLayout::Deserialize(serialized_layout));
    layouts.push_back(std::move(pjrt_layout));
  }
  return layouts;
}

absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>>
PjRtCApiRelocatable::GetParameterOutputAliases() const {
  PJRT_Relocatable_ParameterOutputAliases_Args args;
  args.struct_size = PJRT_Relocatable_ParameterOutputAliases_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.relocatable = c_relocatable();

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->relocatable_parameter_output_aliases(&args),
      c_api_);

  std::vector<std::pair<int64_t, int64_t>> out;
  out.reserve(args.num_aliases);
  for (size_t i = 0; i < args.num_aliases; ++i) {
    out.emplace_back(args.parameter_indices[i], args.output_indices[i]);
  }
  return out;
}

}  // namespace xla
