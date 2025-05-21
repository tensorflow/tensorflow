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

#include "xla/pjrt/c/pjrt_c_api_partial_compile_internal.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_partial_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_partial_compile_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_partial_program.h"
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
    const std::vector<xla::PjRtPartialProgram>& partial_programs_in,
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
    auto next_phases = partial_program.GetNextPhases();
    if (std::find(next_phases.begin(), next_phases.end(), phases_to_run[0]) ==
        next_phases.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input partial program cannot be consumed by a phase with name ",
          phases_to_run[0]));
    }
  }
  return absl::OkStatus();
}

}  // namespace

static PJRT_Error* PJRT_PhaseCompile_Run_Phase(
    PJRT_PhaseCompile_Run_Phase_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Run_Phase_Args",
      PJRT_PhaseCompile_Run_Phase_Args_STRUCT_SIZE, args->struct_size));

  // Extract the phases to run from the input buffer.
  std::vector<std::string> phases_to_run = ConvertCharBufferToCppStrings(
      args->phases_to_run_buffer, args->num_phases_to_run);

  // Extract the input programs from the input buffer.
  auto programs_in = ConvertCPartialProgramsToCppPartialPrograms(
      args->programs_in, args->num_programs_in);

  // Parse the compile options.
  PJRT_ASSIGN_OR_RETURN(
      xla::CompileOptions options,
      ParseCompileOptions(absl::string_view(args->compile_options,
                                            args->compile_options_size)));

  // Run the partial compile phase.
  PJRT_ASSIGN_OR_RETURN(
      absl::StatusOr<std::vector<xla::PjRtPartialProgram>> programs_out,
      xla::PjRtPhaseCompile(options, programs_in, *args->topology->topology,
                            phases_to_run));
  if (!programs_out.ok()) {
    return new PJRT_Error(programs_out.status());
  }

  // Combine the output programs into a single output buffer.
  ConvertCppPartialProgramsToCPartialPrograms(
      programs_out.value(), &args->programs_out, &args->num_programs_out);

  return nullptr;
}

static PJRT_Error* PJRT_PhaseCompile_Get_Phase_Names(
    PJRT_PhaseCompile_Get_PhaseNames_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Get_Phase_Names_Args",
      PJRT_PhaseCompile_Get_PhaseNames_Args_STRUCT_SIZE, args->struct_size));

  // Get the phase names from the compiler.
  PJRT_ASSIGN_OR_RETURN(absl::StatusOr<std::vector<std::string>> phase_names,
                        xla::PjRtGetPhaseNames());
  if (!phase_names.ok()) {
    return new PJRT_Error(phase_names.status());
  }

  // Copy the phase names to the output buffer.
  ConvertCppStringsToCharBuffer(phase_names.value(), &args->phase_names,
                                &args->num_phases);
  return nullptr;
}

// C++ convenience wrappers

absl::StatusOr<std::vector<std::string>> GetPhaseNames(
    const PJRT_PhaseCompile_Extension* partial_compile_extension) {
  PJRT_PhaseCompile_Get_PhaseNames_Args args = {
      .struct_size = PJRT_PhaseCompile_Get_PhaseNames_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .phase_names = nullptr,
      .num_phases = 0,
  };
  PJRT_Error* error =
      partial_compile_extension->partial_compile_get_phase_names(&args);
  if (error != nullptr) {
    return absl::InternalError(error->status.message());
  }
  std::vector<std::string> phase_names =
      ConvertCharBufferToCppStrings(args.phase_names, args.num_phases);

  return phase_names;
}

absl::StatusOr<std::vector<xla::PjRtPartialProgram>> RunPhases(
    const PJRT_PhaseCompile_Extension* partial_compile_extension,
    xla::CompileOptions options, const xla::PjRtTopologyDescription& topology,
    const std::vector<xla::PjRtPartialProgram>& partial_programs_in,
    const std::vector<std::string>& phases_to_run) {
  // Plugin-agnostic validation of the input programs and phases.
  auto phase_validation_status =
      ValidatePhases(partial_programs_in, phases_to_run);
  if (!phase_validation_status.ok()) {
    return phase_validation_status;
  }

  // Convert the input programs to C buffers.
  PJRT_PartialProgram* programs_in = nullptr;
  size_t num_programs_in = 0;
  ConvertCppPartialProgramsToCPartialPrograms(partial_programs_in, &programs_in,
                                              &num_programs_in);

  // Convert compile options to a string.
  TF_ASSIGN_OR_RETURN(const xla::CompileOptionsProto options_proto,
                      options.ToProto());
  std::string options_str = options_proto.SerializeAsString();

  // Convert topology to a C buffer.
  PJRT_TopologyDescription* topology_description =
      CreateWrapperDeviceTopology(&topology);

  // Convert the phases to run to a C buffer.
  const char** phases_to_run_buffer;
  size_t num_phases_to_run = 0;
  ConvertCppStringsToCharBuffer(phases_to_run, &phases_to_run_buffer,
                                &num_phases_to_run);

  PJRT_PhaseCompile_Run_Phase_Args run_args = {
      .struct_size = PJRT_PhaseCompile_Run_Phase_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .phases_to_run_buffer = phases_to_run_buffer,
      .num_phases_to_run = num_phases_to_run,
      .programs_in = programs_in,
      .num_programs_in = num_programs_in,
      .compile_options = options_str.c_str(),
      .compile_options_size = options_str.size(),
      .topology = topology_description,
      .num_programs_out = 0,
  };
  PJRT_Error* error =
      partial_compile_extension->partial_compile_run_phase(&run_args);
  if (error != nullptr) {
    return absl::InternalError(error->status.message());
  }

  auto programs_out = ConvertCPartialProgramsToCppPartialPrograms(
      run_args.programs_out, run_args.num_programs_out);

  delete topology_description;
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
      /*partial_compile_run_phase=*/PJRT_PhaseCompile_Run_Phase,
      /*partial_compile_get_phase_names=*/
      PJRT_PhaseCompile_Get_Phase_Names,
  };
}
}  // namespace pjrt
