/* Copyright 2024 The OpenXLA Authors.

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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace pjrt {

namespace {
absl::StatusOr<xla::CompileOptions> ParseCompileOptions(
    absl::string_view options_str) {
  xla::CompileOptionsProto options_proto;
  if (!options_proto.ParseFromArray(options_str.data(), options_str.size())) {
    return absl::InvalidArgumentError(
        "PJRT_Client_Compile: failed to deserialize CompileOptionsProto");
  }
  return xla::CompileOptions::FromProto(options_proto);
}

absl::Status ValidatePhases(
    const std::vector<xla::PjRtPartialProgramProto>& partial_programs_in,
    const std::vector<std::string>& phases_to_run) {
  if (partial_programs_in.empty()) {
    return absl::InvalidArgumentError("Input partial programs cannot be empty");
  }

  if (phases_to_run.empty()) {
    return absl::InvalidArgumentError("Phases to run cannot be empty");
  }

  if (phases_to_run[0].empty()) {
    return absl::InvalidArgumentError("Phase name cannot be empty");
  }

  for (const auto& partial_program : partial_programs_in) {
    auto next_phases = partial_program.next_phases();
    if (std::find(next_phases.begin(), next_phases.end(), phases_to_run[0]) ==
        next_phases.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input partial program cannot be consumed by a phase with name ",
          phases_to_run[0]));
    }
  }
  return absl::OkStatus();
}

const PJRT_PhaseCompile_Extension* GetPhaseCompileExtension(
    const PJRT_Api* api) {
  return pjrt::FindExtension<PJRT_PhaseCompile_Extension>(
      api, PJRT_Extension_Type::PJRT_Extension_Type_PhaseCompile);
}

}  // namespace

static PJRT_Error* PJRT_PhaseCompile_Run_Phase(
    PJRT_PhaseCompile_Run_Phase_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Run_Phase_Args",
      PJRT_PhaseCompile_Run_Phase_Args_STRUCT_SIZE, args->struct_size));

  // Extract the phases to run from the input buffer.
  std::vector<std::string> phases_to_run = ConvertCharBufferToCppStrings(
      args->phases_to_run_buffer, /*char_buffer_sizes=*/nullptr,
      args->num_phases_to_run);

  // Extract the input programs from the input buffer.
  auto programs_in = ConvertCharBufferToCppStrings(
      args->programs_in_buffer, args->programs_in_buffer_sizes,
      args->num_programs_in);
  std::vector<xla::PjRtPartialProgramProto> programs_in_protos;
  for (const auto& program_in : programs_in) {
    xla::PjRtPartialProgramProto partial_program;
    partial_program.ParseFromString(program_in);
    programs_in_protos.push_back(partial_program);
  }

  // Parse the compile options.
  PJRT_ASSIGN_OR_RETURN(
      xla::CompileOptions options,
      ParseCompileOptions(absl::string_view(args->compile_options,
                                            args->compile_options_size)));

  // Run the partial compile phase.
  PJRT_ASSIGN_OR_RETURN(
      absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>> programs_out,
      xla::PjRtPhaseCompile(options, programs_in_protos,
                            *args->topology->topology, phases_to_run));
  if (!programs_out.ok()) {
    return new PJRT_Error(programs_out.status());
  }

  // Combine the output programs into a single output buffer.
  std::vector<std::string> serialized_programs_out;
  serialized_programs_out.reserve(programs_out.value().size());
  for (const auto& partial_program : programs_out.value()) {
    serialized_programs_out.push_back(partial_program.SerializeAsString());
  }

  ConvertCppStringsToCharBuffer(
      serialized_programs_out, &args->programs_out_buffer,
      &args->programs_out_buffer_sizes, &args->num_programs_out,
      /*is_null_terminated=*/false);

  return nullptr;
}

static PJRT_Error* PJRT_PhaseCompile_Get_Phase_Names(
    PJRT_PhaseCompile_Get_PhaseNames_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Get_Phase_Names_Args",
      PJRT_PhaseCompile_Get_PhaseNames_Args_STRUCT_SIZE, args->struct_size));

  std::string platform_name = args->platform_name;
  // Get the phase names from the compiler.
  PJRT_ASSIGN_OR_RETURN(absl::StatusOr<std::vector<std::string>> phase_names,
                        xla::PjRtGetPhaseNames(platform_name));
  if (!phase_names.ok()) {
    return new PJRT_Error(phase_names.status());
  }

  // Copy the phase names to the output buffer.
  const size_t* phase_names_buffer_sizes = nullptr;
  ConvertCppStringsToCharBuffer(phase_names.value(), &args->phase_names_buffer,
                                &phase_names_buffer_sizes, &args->num_phases);
  return nullptr;
}

// C++ convenience wrappers

absl::StatusOr<std::vector<std::string>> GetPhaseNames(const PJRT_Api* api) {
  // Get the phase compile extension.
  const PJRT_PhaseCompile_Extension* phase_compile_extension =
      GetPhaseCompileExtension(api);
  if (phase_compile_extension == nullptr) {
    return absl::InternalError("Phase compile extension not found");
  }

  // Get the topology description from PJRT api.
  PJRT_TopologyDescription_Create_Args topology_description_create_args = {
      .struct_size = PJRT_TopologyDescription_Create_Args_STRUCT_SIZE,
      .extension_start = nullptr,
  };

  RETURN_STATUS_IF_PJRT_ERROR(
      api->PJRT_TopologyDescription_Create(&topology_description_create_args),
      api);

  std::unique_ptr<PJRT_TopologyDescription, PJRT_TopologyDescriptionDeleter>
      topology_ptr = std::unique_ptr<PJRT_TopologyDescription,
                                     PJRT_TopologyDescriptionDeleter>(
          topology_description_create_args.topology,
          MakeTopologyDescriptionDeleter(api));

  // Get the platform name from the topology description.
  std::string platform_name(topology_ptr->topology->platform_name());

  // Get the phase names from the compiler.
  PJRT_PhaseCompile_Get_PhaseNames_Args args = {
      .struct_size = PJRT_PhaseCompile_Get_PhaseNames_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .platform_name = platform_name.c_str(),
      .num_phases = 0,
  };
  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension->phase_compile_get_phase_names(&args), api);

  // Convert the phase names from the output buffer to C++ strings.
  std::vector<std::string> phase_names = ConvertCharBufferToCppStrings(
      args.phase_names_buffer, /*char_buffer_sizes=*/nullptr, args.num_phases);

  return phase_names;
}

absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>> RunPhases(
    const PJRT_Api* api, xla::CompileOptions options,
    const std::vector<xla::PjRtPartialProgramProto>& partial_programs_in,
    const std::vector<std::string>& phases_to_run) {
  // Plugin-agnostic validation of the input programs and phases.
  auto phase_validation_status =
      ValidatePhases(partial_programs_in, phases_to_run);
  if (!phase_validation_status.ok()) {
    return phase_validation_status;
  }

  // Get the phase compile extension.
  const PJRT_PhaseCompile_Extension* phase_compile_extension =
      GetPhaseCompileExtension(api);
  if (phase_compile_extension == nullptr) {
    return absl::InternalError("Phase compile extension not found");
  }

  // Get the topology description from PJRT api.
  PJRT_TopologyDescription_Create_Args topology_description_create_args = {
      .struct_size = PJRT_TopologyDescription_Create_Args_STRUCT_SIZE,
      .extension_start = nullptr,
  };

  RETURN_STATUS_IF_PJRT_ERROR(
      api->PJRT_TopologyDescription_Create(&topology_description_create_args),
      api);

  std::unique_ptr<PJRT_TopologyDescription, PJRT_TopologyDescriptionDeleter>
      topology_ptr = std::unique_ptr<PJRT_TopologyDescription,
                                     PJRT_TopologyDescriptionDeleter>(
          topology_description_create_args.topology,
          MakeTopologyDescriptionDeleter(api));

  // Convert the input proto programs to C++ strings.
  std::vector<std::string> serialized_programs_in;
  serialized_programs_in.reserve(partial_programs_in.size());
  for (const auto& partial_program : partial_programs_in) {
    serialized_programs_in.push_back(partial_program.SerializeAsString());
  }

  // Convert compile options to a string.
  TF_ASSIGN_OR_RETURN(const xla::CompileOptionsProto options_proto,
                      options.ToProto());
  std::string options_str = options_proto.SerializeAsString();

  // Convert the input programs to a C buffer.
  const char** programs_in_buffer = nullptr;
  const size_t* programs_in_buffer_sizes = nullptr;
  size_t num_programs_in = 0;
  ConvertCppStringsToCharBuffer(serialized_programs_in, &programs_in_buffer,
                                &programs_in_buffer_sizes, &num_programs_in,
                                /*is_null_terminated=*/false);

  // Convert the phases to run to a C buffer.
  const char** phases_to_run_buffer = nullptr;
  const size_t* phases_to_run_buffer_sizes = nullptr;
  size_t num_phases_to_run = 0;
  ConvertCppStringsToCharBuffer(phases_to_run, &phases_to_run_buffer,
                                &phases_to_run_buffer_sizes,
                                &num_phases_to_run);

  // Run the phases.
  PJRT_PhaseCompile_Run_Phase_Args run_args = {
      .struct_size = PJRT_PhaseCompile_Run_Phase_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .programs_in_buffer = programs_in_buffer,
      .programs_in_buffer_sizes = programs_in_buffer_sizes,
      .num_programs_in = num_programs_in,
      .phases_to_run_buffer = phases_to_run_buffer,
      .num_phases_to_run = num_phases_to_run,
      .compile_options = options_str.c_str(),
      .compile_options_size = options_str.size(),
      .topology = topology_ptr.get(),
      .programs_out_buffer = nullptr,
      .programs_out_buffer_sizes = nullptr,
      .num_programs_out = 0,
  };

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension->phase_compile_run_phase(&run_args), api);

  // Convert the output programs from C buffers to C++ strings.
  auto serialized_programs_out = ConvertCharBufferToCppStrings(
      run_args.programs_out_buffer, run_args.programs_out_buffer_sizes,
      run_args.num_programs_out);

  // Convert the output programs from C++ strings to protos.
  std::vector<xla::PjRtPartialProgramProto> programs_out;
  for (const auto& serialized_program : serialized_programs_out) {
    xla::PjRtPartialProgramProto partial_program;
    partial_program.ParseFromString(serialized_program);
    programs_out.push_back(partial_program);
  }

  return programs_out;
}

PJRT_PhaseCompile_Extension CreatePhaseCompileExtension(
    PJRT_Extension_Base* next) {
  return {
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_PhaseCompile_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_PhaseCompile,
          /*next=*/next,  // Chain this extension in the list
      },
      /*phase_compile_run_phase=*/PJRT_PhaseCompile_Run_Phase,
      /*phase_compile_get_phase_names=*/
      PJRT_PhaseCompile_Get_Phase_Names,
  };
}
}  // namespace pjrt
