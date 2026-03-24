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
#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/partial_program_utils.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/pjrt/string_utils.h"

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
    return new PJRT_Error{absl::InternalError(
        "PJRT_PhaseCompile_Run_Phase: phase compiler is null")};
  }
  PJRT_ASSIGN_OR_RETURN(std::vector<xla::PjRtPartialProgramProto> programs_out,
                        args->phase_compiler->compiler->RunPhases(
                            options, programs_in_protos,
                            *args->topology->topology, phases_to_run));

  PJRT_ASSIGN_OR_RETURN(args->output_programs,
                        xla::ConvertPjRtPartialProgramProtosToCharBuffers(
                            programs_out, args->output_programs_sizes));
  args->num_output_programs = programs_out.size();

  return nullptr;
}

static PJRT_Error* PJRT_PhaseCompile_Get_Phase_Names(
    PJRT_PhaseCompile_Get_PhaseNames_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Get_Phase_Names_Args",
      PJRT_PhaseCompile_Get_PhaseNames_Args_STRUCT_SIZE, args->struct_size));

  if (args->phase_compiler == nullptr) {
    return new PJRT_Error{absl::InternalError(
        "PJRT_PhaseCompile_Get_Phase_Names: phase compiler is null")};
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
      /*phase_compile_run_phase=*/PJRT_PhaseCompile_Run_Phase,
      /*phase_compile_get_phase_names=*/
      PJRT_PhaseCompile_Get_Phase_Names,
      /*phase_compile_c_buffers_destroy=*/PJRT_PhaseCompile_C_Buffers_Destroy,
  };
}
}  // namespace pjrt
