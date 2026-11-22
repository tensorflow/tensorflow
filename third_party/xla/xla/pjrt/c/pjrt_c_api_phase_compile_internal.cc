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

#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
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

struct PJRT_Relocatable {
  // Must be shared_ptr so that we can share with LinkRelocatables.
  std::shared_ptr<xla::PjRtRelocatable> shared_relocatable;
  xla::PjRtRelocatable* relocatable;

  mutable absl::Mutex mutex;

  bool parameter_types_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<PJRT_Buffer_Type> parameter_types;

  bool out_types_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<PJRT_Buffer_Type> out_types;

  bool parameter_dimensions_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<int64_t> parameter_dimensions;
  std::vector<size_t> parameter_dimension_sizes;

  bool out_dimensions_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<int64_t> out_dimensions;
  std::vector<size_t> out_dimension_sizes;

  bool parameter_layouts_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<PJRT_Layouts_MemoryLayout> parameter_layouts;
  std::vector<PJRT_Layouts_MemoryLayout*> parameter_layouts_pointers;

  bool out_layouts_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<PJRT_Layouts_MemoryLayout> out_layouts;
  std::vector<PJRT_Layouts_MemoryLayout*> out_layouts_pointers;

  bool parameter_output_aliases_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<int64_t> alias_parameter_indices;
  std::vector<int64_t> alias_output_indices;

  explicit PJRT_Relocatable(
      std::shared_ptr<xla::PjRtRelocatable> shared_relocatable)
      : shared_relocatable(std::move(shared_relocatable)),
        relocatable(this->shared_relocatable.get()) {}

  const xla::PjRtRelocatable* get() const { return relocatable; }
  xla::PjRtRelocatable* get() { return relocatable; }
};

struct PJRT_SerializedRelocatable {
  std::string serialized;
};

namespace pjrt {

namespace {

absl::StatusOr<xla::CompileOptions> ParseCompileOptions(
    absl::string_view options_str) {
  xla::CompileOptionsProto options_proto;
  if (!options_proto.ParseFromString(options_str)) {
    return absl::InvalidArgumentError(
        "PJRT_Client_Compile: failed to deserialize CompileOptionsProto");
  }
  return xla::CompileOptions::FromProto(options_proto);
}

absl::StatusOr<xla::MaybeOwningMlirModule> ParseMlirProgram(
    const PJRT_Program* program) {
  auto format_str = absl::string_view(program->format, program->format_size);
  auto module_str = absl::string_view(program->code, program->code_size);
  if (format_str != pjrt::kMlirFormat) {
    return absl::InvalidArgumentError(ProgramFormatErrorMsg(format_str));
  }
  auto context = std::make_unique<mlir::MLIRContext>();
  ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                   xla::ParseMlirModuleString(module_str, *context));
  return xla::MaybeOwningMlirModule(std::move(context), std::move(module));
}

absl::Status PopulateRelocatableParameterElementTypes(
    PJRT_Relocatable* relocatable) {
  ABSL_ASSIGN_OR_RETURN(std::vector<xla::PrimitiveType> parameter_types,
                   relocatable->get()->GetParameterElementTypes());
  std::vector<PJRT_Buffer_Type>& param_types = relocatable->parameter_types;
  param_types.reserve(parameter_types.size());
  for (xla::PrimitiveType element_type : parameter_types) {
    param_types.push_back(ConvertToPjRtBufferType(element_type));
  }
  return absl::OkStatus();
}

absl::Status PopulateRelocatableOutputElementTypes(
    PJRT_Relocatable* relocatable) {
  ABSL_ASSIGN_OR_RETURN(std::vector<xla::PrimitiveType> output_types,
                   relocatable->get()->GetOutputElementTypes());
  std::vector<PJRT_Buffer_Type>& out_types = relocatable->out_types;
  out_types.reserve(output_types.size());
  for (xla::PrimitiveType element_type : output_types) {
    out_types.push_back(ConvertToPjRtBufferType(element_type));
  }
  return absl::OkStatus();
}

absl::Status PopulateRelocatableParameterDimensions(
    PJRT_Relocatable* relocatable) {
  ABSL_ASSIGN_OR_RETURN(std::vector<xla::DimensionVector> parameter_dims,
                   relocatable->get()->GetParameterDimensions());
  std::vector<size_t>& parameter_dimension_sizes =
      relocatable->parameter_dimension_sizes;
  parameter_dimension_sizes.reserve(parameter_dims.size());
  size_t total_size = 0;
  for (const auto& dimension_vector : parameter_dims) {
    parameter_dimension_sizes.push_back(dimension_vector.size());
    total_size += dimension_vector.size();
  }
  std::vector<int64_t>& parameter_dimensions =
      relocatable->parameter_dimensions;
  parameter_dimensions.reserve(total_size);
  for (const auto& dimension_vector : parameter_dims) {
    for (int64_t dim : dimension_vector) {
      parameter_dimensions.push_back(dim);
    }
  }
  return absl::OkStatus();
}

absl::Status PopulateRelocatableOutputDimensions(
    PJRT_Relocatable* relocatable) {
  ABSL_ASSIGN_OR_RETURN(std::vector<xla::DimensionVector> output_dims,
                   relocatable->get()->GetOutputDimensions());
  std::vector<size_t>& out_dimension_sizes = relocatable->out_dimension_sizes;
  out_dimension_sizes.reserve(output_dims.size());
  size_t total_size = 0;
  for (const auto& dimension_vector : output_dims) {
    out_dimension_sizes.push_back(dimension_vector.size());
    total_size += dimension_vector.size();
  }
  std::vector<int64_t>& out_dimensions = relocatable->out_dimensions;
  out_dimensions.reserve(total_size);
  for (const auto& dimension_vector : output_dims) {
    for (int64_t dim : dimension_vector) {
      out_dimensions.push_back(dim);
    }
  }
  return absl::OkStatus();
}

absl::Status PopulateRelocatableParameterLayouts(
    PJRT_Relocatable* relocatable) {
  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> cpp_parameter_layouts,
      relocatable->get()->GetParameterLayouts());
  relocatable->parameter_layouts.reserve(cpp_parameter_layouts.size());
  relocatable->parameter_layouts_pointers.reserve(cpp_parameter_layouts.size());
  for (std::shared_ptr<const xla::PjRtLayout>& layout : cpp_parameter_layouts) {
    relocatable->parameter_layouts.push_back(
        PJRT_Layouts_MemoryLayout{std::move(layout)});
  }
  for (PJRT_Layouts_MemoryLayout& layout : relocatable->parameter_layouts) {
    relocatable->parameter_layouts_pointers.push_back(&layout);
  }
  return absl::OkStatus();
}

absl::Status PopulateRelocatableOutputLayouts(PJRT_Relocatable* relocatable) {
  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> cpp_out_layouts,
      relocatable->get()->GetOutputLayouts());
  relocatable->out_layouts.reserve(cpp_out_layouts.size());
  relocatable->out_layouts_pointers.reserve(cpp_out_layouts.size());
  for (std::shared_ptr<const xla::PjRtLayout>& layout : cpp_out_layouts) {
    relocatable->out_layouts.push_back(
        PJRT_Layouts_MemoryLayout{std::move(layout)});
  }
  for (PJRT_Layouts_MemoryLayout& layout : relocatable->out_layouts) {
    relocatable->out_layouts_pointers.push_back(&layout);
  }
  return absl::OkStatus();
}

absl::Status PopulateRelocatableParameterOutputAliases(
    PJRT_Relocatable* relocatable) {
  ABSL_ASSIGN_OR_RETURN(auto aliases,
                   relocatable->get()->GetParameterOutputAliases());
  relocatable->alias_parameter_indices.reserve(aliases.size());
  relocatable->alias_output_indices.reserve(aliases.size());
  for (const auto& [param_idx, out_idx] : aliases) {
    relocatable->alias_parameter_indices.push_back(param_idx);
    relocatable->alias_output_indices.push_back(out_idx);
  }
  return absl::OkStatus();
}

}  // namespace

static PJRT_Error* PJRT_PhaseCompile_Run_Phase(
    PJRT_PhaseCompile_Run_Phase_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Run_Phase_Args",
      PJRT_PhaseCompile_Run_Phase_Args_STRUCT_SIZE, args->struct_size));

  std::vector<std::string> phases_to_run = xla::ConvertCharBuffersToCppStrings(
      absl::MakeSpan(args->phases_to_run, args->num_phases_to_run),
      absl::MakeConstSpan(args->phases_to_run_sizes, args->num_phases_to_run));

  PJRT_ASSIGN_OR_RETURN(
      std::vector<xla::PjRtPartialProgramProto> programs_in_protos,
      xla::ConvertCharBuffersToPjRtPartialProgramProtos(
          absl::MakeSpan(args->input_programs, args->num_input_programs),
          absl::MakeConstSpan(args->input_programs_sizes,
                              args->num_input_programs)));

  PJRT_ASSIGN_OR_RETURN(
      xla::CompileOptions options,
      ParseCompileOptions(absl::string_view(args->compile_options,
                                            args->compile_options_size)));

  if (args->phase_compiler == nullptr) {
    return StatusToPjRtError(absl::InternalError(
        "PJRT_PhaseCompile_Run_Phase: phase compiler is null"));
  }
  PJRT_ASSIGN_OR_RETURN(std::vector<xla::PjRtPartialProgramProto> programs_out,
                        args->phase_compiler->compiler->RunPhases(
                            options, std::move(programs_in_protos),
                            *args->topology->topology, phases_to_run));

  args->num_output_programs = programs_out.size();
  PJRT_ASSIGN_OR_RETURN(
      args->output_programs,
      xla::ConvertPjRtPartialProgramProtosToCharBuffers(
          absl::MakeSpan(programs_out), args->output_programs_sizes));

  return nullptr;
}

static PJRT_Error* PJRT_PhaseCompile_Get_Phase_Names(
    PJRT_PhaseCompile_Get_PhaseNames_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Get_Phase_Names_Args",
      PJRT_PhaseCompile_Get_PhaseNames_Args_STRUCT_SIZE, args->struct_size));

  if (args->phase_compiler == nullptr) {
    return StatusToPjRtError(absl::InternalError(
        "PJRT_PhaseCompile_Get_Phase_Names: phase compiler is null"));
  }
  PJRT_ASSIGN_OR_RETURN(std::vector<std::string> phase_names,
                        args->phase_compiler->compiler->GetPhaseNames());

  args->phase_names =
      xla::ConvertCppStringsToCharBuffers(phase_names, args->phase_names_sizes);
  args->num_phase_names = phase_names.size();

  return nullptr;
}

static void PJRT_PhaseCompile_C_Buffers_Destroy(
    PJRT_PhaseCompile_C_Buffers_Destroy_Args* args) {
  if (args == nullptr) {
    return;
  }

  if (args->char_buffers != nullptr) {
    for (size_t i = 0; i < args->num_char_buffers; ++i) {
      delete[] args->char_buffers[i];
    }
  }

  delete[] args->char_buffers;
  delete[] args->char_buffer_sizes;
}

static PJRT_Error* PJRT_Relocatable_Destroy(
    PJRT_Relocatable_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_Destroy_Args",
      PJRT_Relocatable_Destroy_Args_STRUCT_SIZE, args->struct_size));
  delete args->relocatable;
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_Name(PJRT_Relocatable_Name_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_Name_Args", PJRT_Relocatable_Name_Args_STRUCT_SIZE,
      args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_Name: relocatable is null"));
  }
  absl::string_view relocatable_name = args->relocatable->get()->name();
  args->relocatable_name = relocatable_name.data();
  args->relocatable_name_size = relocatable_name.size();
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_BuildId(
    PJRT_Relocatable_BuildId_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_BuildId_Args",
      PJRT_Relocatable_BuildId_Args_STRUCT_SIZE, args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_BuildId: relocatable is null"));
  }
  absl::string_view build_id = args->relocatable->get()->build_id();
  args->build_id = build_id.data();
  args->build_id_size = build_id.size();
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_CallBackendConfig(
    PJRT_Relocatable_CallBackendConfig_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_CallBackendConfig_Args",
      PJRT_Relocatable_CallBackendConfig_Args_STRUCT_SIZE, args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_CallBackendConfig: relocatable is null"));
  }
  absl::string_view call_backend_config =
      args->relocatable->get()->call_backend_config();
  args->call_backend_config = call_backend_config.data();
  args->call_backend_config_size = call_backend_config.size();
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_HasSideEffects(
    PJRT_Relocatable_HasSideEffects_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_HasSideEffects_Args",
      PJRT_Relocatable_HasSideEffects_Args_STRUCT_SIZE, args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_HasSideEffects: relocatable is null"));
  }
  args->has_side_effects = args->relocatable->get()->has_side_effects();
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_Serialize(
    PJRT_Relocatable_Serialize_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_Serialize_Args",
      PJRT_Relocatable_Serialize_Args_STRUCT_SIZE, args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_Serialize: relocatable is null"));
  }
  PJRT_ASSIGN_OR_RETURN(std::string serialization,
                        args->relocatable->get()->Serialize());

  PJRT_SerializedRelocatable* serialized_reloc = new PJRT_SerializedRelocatable;
  serialized_reloc->serialized = std::move(serialization);
  args->serialized_relocatable = serialized_reloc;
  args->serialized_bytes = serialized_reloc->serialized.data();
  args->serialized_bytes_size = serialized_reloc->serialized.size();
  args->serialized_relocatable_deleter =
      +[](PJRT_SerializedRelocatable* serialized_relocatable) {
        delete serialized_relocatable;
      };
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_ParameterElementTypes(
    PJRT_Relocatable_ParameterElementTypes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_ParameterElementTypes_Args",
      PJRT_Relocatable_ParameterElementTypes_Args_STRUCT_SIZE,
      args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_ParameterElementTypes: relocatable is null"));
  }

  {
    absl::MutexLock lock(args->relocatable->mutex);
    if (!args->relocatable->parameter_types_ran) {
      PJRT_RETURN_IF_ERROR(
          PopulateRelocatableParameterElementTypes(args->relocatable));
      args->relocatable->parameter_types_ran = true;
    }
  }

  args->num_parameter_types = args->relocatable->parameter_types.size();
  args->parameter_types = args->relocatable->parameter_types.data();
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_OutputElementTypes(
    PJRT_Relocatable_OutputElementTypes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_OutputElementTypes_Args",
      PJRT_Relocatable_OutputElementTypes_Args_STRUCT_SIZE, args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_OutputElementTypes: relocatable is null"));
  }

  {
    absl::MutexLock lock(args->relocatable->mutex);
    if (!args->relocatable->out_types_ran) {
      PJRT_RETURN_IF_ERROR(
          PopulateRelocatableOutputElementTypes(args->relocatable));
      args->relocatable->out_types_ran = true;
    }
  }

  args->num_output_types = args->relocatable->out_types.size();
  args->output_types = args->relocatable->out_types.data();
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_ParameterDimensions(
    PJRT_Relocatable_ParameterDimensions_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_ParameterDimensions_Args",
      PJRT_Relocatable_ParameterDimensions_Args_STRUCT_SIZE,
      args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_ParameterDimensions: relocatable is null"));
  }

  {
    absl::MutexLock lock(args->relocatable->mutex);
    if (!args->relocatable->parameter_dimensions_ran) {
      PJRT_RETURN_IF_ERROR(
          PopulateRelocatableParameterDimensions(args->relocatable));
      args->relocatable->parameter_dimensions_ran = true;
    }
  }

  args->num_parameters = args->relocatable->parameter_dimension_sizes.size();
  args->dim_sizes = args->relocatable->parameter_dimension_sizes.data();
  args->dims = args->relocatable->parameter_dimensions.data();
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_OutputDimensions(
    PJRT_Relocatable_OutputDimensions_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_OutputDimensions_Args",
      PJRT_Relocatable_OutputDimensions_Args_STRUCT_SIZE, args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_OutputDimensions: relocatable is null"));
  }

  {
    absl::MutexLock lock(args->relocatable->mutex);
    if (!args->relocatable->out_dimensions_ran) {
      PJRT_RETURN_IF_ERROR(
          PopulateRelocatableOutputDimensions(args->relocatable));
      args->relocatable->out_dimensions_ran = true;
    }
  }

  args->num_outputs = args->relocatable->out_dimension_sizes.size();
  args->dim_sizes = args->relocatable->out_dimension_sizes.data();
  args->dims = args->relocatable->out_dimensions.data();
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_ParameterLayouts(
    PJRT_Relocatable_ParameterLayouts_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_ParameterLayouts_Args",
      PJRT_Relocatable_ParameterLayouts_Args_STRUCT_SIZE, args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_ParameterLayouts: relocatable is null"));
  }

  {
    absl::MutexLock lock(args->relocatable->mutex);
    if (!args->relocatable->parameter_layouts_ran) {
      PJRT_RETURN_IF_ERROR(
          PopulateRelocatableParameterLayouts(args->relocatable));
      args->relocatable->parameter_layouts_ran = true;
    }
  }

  args->num_parameters = args->relocatable->parameter_layouts_pointers.size();
  args->layouts = args->relocatable->parameter_layouts_pointers.data();
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_OutputLayouts(
    PJRT_Relocatable_OutputLayouts_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_OutputLayouts_Args",
      PJRT_Relocatable_OutputLayouts_Args_STRUCT_SIZE, args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_OutputLayouts: relocatable is null"));
  }

  {
    absl::MutexLock lock(args->relocatable->mutex);
    if (!args->relocatable->out_layouts_ran) {
      PJRT_RETURN_IF_ERROR(PopulateRelocatableOutputLayouts(args->relocatable));
      args->relocatable->out_layouts_ran = true;
    }
  }

  args->num_outputs = args->relocatable->out_layouts_pointers.size();
  args->layouts = args->relocatable->out_layouts_pointers.data();
  return nullptr;
}

static PJRT_Error* PJRT_Relocatable_ParameterOutputAliases(
    PJRT_Relocatable_ParameterOutputAliases_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Relocatable_ParameterOutputAliases_Args",
      PJRT_Relocatable_ParameterOutputAliases_Args_STRUCT_SIZE,
      args->struct_size));
  if (args->relocatable == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_Relocatable_ParameterOutputAliases: relocatable is null"));
  }

  {
    absl::MutexLock lock(args->relocatable->mutex);
    if (!args->relocatable->parameter_output_aliases_ran) {
      PJRT_RETURN_IF_ERROR(
          PopulateRelocatableParameterOutputAliases(args->relocatable));
      args->relocatable->parameter_output_aliases_ran = true;
    }
  }

  args->num_aliases = args->relocatable->alias_parameter_indices.size();
  args->parameter_indices = args->relocatable->alias_parameter_indices.data();
  args->output_indices = args->relocatable->alias_output_indices.data();
  return nullptr;
}

static PJRT_Error* PJRT_PhaseCompile_CompileToRelocatable(
    PJRT_PhaseCompile_CompileToRelocatable_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_CompileToRelocatable_Args",
      PJRT_PhaseCompile_CompileToRelocatable_Args_STRUCT_SIZE,
      args->struct_size));
  if (args->program == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_PhaseCompile_CompileToRelocatable: program is null"));
  }
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Program", PJRT_Program_STRUCT_SIZE, args->program->struct_size));
  if (args->phase_compiler == nullptr) {
    return StatusToPjRtError(absl::InternalError(
        "PJRT_PhaseCompile_CompileToRelocatable: phase compiler is null"));
  }
  if (args->topology == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_PhaseCompile_CompileToRelocatable: topology is null"));
  }

  xla::PjRtClient* client = nullptr;
  if (args->client != nullptr) {
    client = args->client->client.get();
  }
  PJRT_ASSIGN_OR_RETURN(
      xla::CompileOptions options,
      ParseCompileOptions(absl::string_view(args->compile_options,
                                            args->compile_options_size)));
  PJRT_ASSIGN_OR_RETURN(xla::MaybeOwningMlirModule module,
                        ParseMlirProgram(args->program));
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtRelocatable> relocatable,
      args->phase_compiler->compiler->CompileToRelocatable(
          std::move(options), std::move(module), *args->topology->topology,
          absl::string_view(args->name, args->name_size), args->is_entrypoint,
          client));
  args->relocatable = new PJRT_Relocatable(std::move(relocatable));
  return nullptr;
}

static PJRT_Error* PJRT_PhaseCompile_DeserializeRelocatable(
    PJRT_PhaseCompile_DeserializeRelocatable_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_DeserializeRelocatable_Args",
      PJRT_PhaseCompile_DeserializeRelocatable_Args_STRUCT_SIZE,
      args->struct_size));
  if (args->phase_compiler == nullptr) {
    return StatusToPjRtError(absl::InternalError(
        "PJRT_PhaseCompile_DeserializeRelocatable: phase compiler is null"));
  }
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtRelocatable> relocatable,
      args->phase_compiler->compiler->DeserializeRelocatable(absl::string_view(
          args->serialized_relocatable, args->serialized_relocatable_size)));
  args->relocatable = new PJRT_Relocatable(std::move(relocatable));
  return nullptr;
}

static PJRT_Error* PJRT_PhaseCompile_LinkRelocatables(
    PJRT_PhaseCompile_LinkRelocatables_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_LinkRelocatables_Args",
      PJRT_PhaseCompile_LinkRelocatables_Args_STRUCT_SIZE, args->struct_size));
  if (args->phase_compiler == nullptr) {
    return StatusToPjRtError(absl::InternalError(
        "PJRT_PhaseCompile_LinkRelocatables: phase compiler is null"));
  }
  if (args->topology == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_PhaseCompile_LinkRelocatables: topology is null"));
  }
  PJRT_ASSIGN_OR_RETURN(
      xla::CompileOptions options,
      ParseCompileOptions(absl::string_view(args->compile_options,
                                            args->compile_options_size)));

  std::shared_ptr<const xla::PjRtRelocatable> entrypoint;
  if (args->entrypoint != nullptr) {
    entrypoint = args->entrypoint->shared_relocatable;
  }

  std::vector<std::shared_ptr<const xla::PjRtRelocatable>> dependencies;
  dependencies.reserve(args->num_dependencies);
  for (size_t i = 0; i < args->num_dependencies; ++i) {
    if (args->dependencies[i] != nullptr) {
      dependencies.push_back(args->dependencies[i]->shared_relocatable);
    } else {
      dependencies.push_back(nullptr);
    }
  }

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtExecutable> executable,
                        args->phase_compiler->compiler->LinkRelocatables(
                            std::move(options), *args->topology->topology,
                            std::move(entrypoint), std::move(dependencies)));
  args->executable = new PJRT_Executable(std::move(executable));
  return nullptr;
}

PJRT_PhaseCompile_Extension CreatePhaseCompileExtension(
    PJRT_Extension_Base* next, PJRT_PhaseCompile_Get_Compiler get_compiler,
    PJRT_PhaseCompile_Destroy_Compiler destroy_compiler) {
  return {
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_PhaseCompile_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_PhaseCompile,
          /*next=*/next,
      },
      /*phase_compile_get_compiler=*/get_compiler,
      /*phase_compile_destroy_compiler=*/destroy_compiler,
      /*phase_compile_run_phases=*/PJRT_PhaseCompile_Run_Phase,
      /*phase_compile_get_phase_names=*/
      PJRT_PhaseCompile_Get_Phase_Names,
      /*phase_compile_c_buffers_destroy=*/PJRT_PhaseCompile_C_Buffers_Destroy,
      /*relocatable_destroy=*/PJRT_Relocatable_Destroy,
      /*relocatable_name=*/PJRT_Relocatable_Name,
      /*relocatable_build_id=*/PJRT_Relocatable_BuildId,
      /*relocatable_call_backend_config=*/PJRT_Relocatable_CallBackendConfig,
      /*relocatable_has_side_effects=*/PJRT_Relocatable_HasSideEffects,
      /*relocatable_serialize=*/PJRT_Relocatable_Serialize,
      /*relocatable_parameter_element_types=*/
      PJRT_Relocatable_ParameterElementTypes,
      /*relocatable_output_element_types=*/
      PJRT_Relocatable_OutputElementTypes,
      /*relocatable_parameter_dimensions=*/PJRT_Relocatable_ParameterDimensions,
      /*relocatable_output_dimensions=*/PJRT_Relocatable_OutputDimensions,
      /*relocatable_parameter_layouts=*/PJRT_Relocatable_ParameterLayouts,
      /*relocatable_output_layouts=*/PJRT_Relocatable_OutputLayouts,
      /*relocatable_parameter_output_aliases=*/
      PJRT_Relocatable_ParameterOutputAliases,
      /*phase_compile_compile_to_relocatable=*/
      PJRT_PhaseCompile_CompileToRelocatable,
      /*phase_compile_deserialize_relocatable=*/
      PJRT_PhaseCompile_DeserializeRelocatable,
      /*phase_compile_link_relocatables=*/PJRT_PhaseCompile_LinkRelocatables,
  };
}
}  // namespace pjrt
