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

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/partial_program_utils.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/pjrt/string_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

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
    auto consumer_phases = partial_program.consumer_phases();
    if (std::find(consumer_phases.begin(), consumer_phases.end(),
                  phases_to_run[0]) == consumer_phases.end()) {
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
    const std::vector<xla::PjRtPartialProgramProto>& partial_programs_in,
    const xla::PjRtTopologyDescription& topology,
    const std::vector<std::string>& phases_to_run) {
  // Plugin-agnostic validation of the input programs and phases.
  TF_RETURN_IF_ERROR(ValidatePhases(partial_programs_in, phases_to_run));

  PJRT_TopologyDescription* topology_description =
      tensorflow::down_cast<const xla::PjRtCApiTopologyDescription*>(&topology)
          ->c_topology();

  const size_t* programs_in_buffer_sizes;
  TF_ASSIGN_OR_RETURN(const char** programs_in_buffers,
                      xla::ConvertPjRtPartialProgramProtosToCharBuffers(
                          partial_programs_in, programs_in_buffer_sizes));
  size_t num_programs_in = partial_programs_in.size();
  absl::Cleanup cleanup_programs_in_buffers =
      [programs_in_buffers, programs_in_buffer_sizes, num_programs_in] {
        CleanUpCallerDefinedCBuffers(
            absl::MakeConstSpan(programs_in_buffers, num_programs_in));
        delete[] programs_in_buffer_sizes;
        delete[] programs_in_buffers;
      };

  TF_ASSIGN_OR_RETURN(const xla::CompileOptionsProto options_proto,
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

  TF_ASSIGN_OR_RETURN(std::vector<xla::PjRtPartialProgramProto> output_programs,
                      xla::ConvertCharBuffersToPjRtPartialProgramProtos(
                          absl::MakeSpan(run_args.output_programs,
                                         run_args.num_output_programs),
                          absl::MakeConstSpan(run_args.output_programs_sizes,
                                              run_args.num_output_programs)));
  CleanUpPluginDefinedCBuffers(
      run_args.output_programs, run_args.output_programs_sizes,
      run_args.num_output_programs, phase_compile_extension_);

  return output_programs;
}

}  // namespace xla
