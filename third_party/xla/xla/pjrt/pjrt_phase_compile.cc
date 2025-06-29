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

#include "xla/pjrt/pjrt_phase_compile.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace pjrt {

namespace {
absl::Status ValidatePhases(
    const std::vector<xla::PjRtPartialProgramProto>& partial_programs_in,
    const std::vector<std::string>& phases_to_run) {
  if (phases_to_run.empty()) {
    return absl::InvalidArgumentError("Phases to run cannot be empty");
  }

  for (const auto& partial_program : partial_programs_in) {
    auto next_phases = partial_program.next_phases();
    if (std::find(next_phases.begin(), next_phases.end(), phases_to_run[0]) ==
        next_phases.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input partial programs cannot be compiled by a phase with name \"",
          phases_to_run[0], "\""));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<PhaseCompileExtensionWrapper>>
PhaseCompileExtensionWrapper::Create(const PJRT_Api* api) {
  // Extract the phase compile extension. Return an error if the extension is
  // not found or if the required callbacks phase_compile_get_compiler and
  // phase_compile_destroy_compiler are not set.
  auto phase_compile_extension =
      pjrt::FindExtension<PJRT_PhaseCompile_Extension>(
          api, PJRT_Extension_Type::PJRT_Extension_Type_PhaseCompile);
  if (phase_compile_extension == nullptr) {
    return absl::InternalError("Phase compile extension not found");
  }

  if (phase_compile_extension->phase_compile_get_compiler == nullptr ||
      phase_compile_extension->phase_compile_destroy_compiler == nullptr) {
    return absl::InternalError(
        "phase_compile_get_compiler or phase_compile_destroy_compiler "
        "callbacks are not set to the phase compile "
        "extension");
  }

  // Get the phase compiler.
  PJRT_PhaseCompile_Get_Compiler_Args get_compiler_args;
  get_compiler_args.struct_size =
      PJRT_PhaseCompile_Get_Compiler_Args_STRUCT_SIZE;
  get_compiler_args.extension_start = nullptr;
  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension->phase_compile_get_compiler(&get_compiler_args),
      api);

  return std::unique_ptr<PhaseCompileExtensionWrapper>(
      new PhaseCompileExtensionWrapper(api, phase_compile_extension,
                                       get_compiler_args.phase_compiler));
}

absl::StatusOr<std::vector<std::string>>
PhaseCompileExtensionWrapper::GetPhaseNames() {
  PJRT_PhaseCompile_Get_PhaseNames_Args args;
  args.struct_size = PJRT_PhaseCompile_Get_PhaseNames_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.phase_compiler = phase_compiler_;
  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->phase_compile_get_phase_names(&args), api_);

  auto phase_names_buffer_ptr = std::make_unique<PhaseCompileCBuffersWrapper>(
      args.phase_names, args.phase_names_sizes, args.num_phase_names,
      /*is_locally_owned=*/false, phase_compile_extension_);

  // Convert the phase names from the output buffer to C++ strings.
  std::vector<std::string> phase_names =
      ConvertCharBufferToCppStrings(phase_names_buffer_ptr->char_buffers,
                                    phase_names_buffer_ptr->char_buffer_sizes,
                                    phase_names_buffer_ptr->num_char_buffers);

  return phase_names;
}

absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>>
PhaseCompileExtensionWrapper::RunPhases(
    xla::CompileOptions options,
    const std::vector<xla::PjRtPartialProgramProto>& partial_programs_in,
    const xla::PjRtTopologyDescription& topology,
    const std::vector<std::string>& phases_to_run) {
  // Plugin-agnostic validation of the input programs and phases.
  TF_RETURN_IF_ERROR(ValidatePhases(partial_programs_in, phases_to_run));

  // Create a C wrapper for the topology description.
  PJRT_TopologyDescription* topology_description =
      CreateWrapperDeviceTopology(&topology);
  auto topology_ptr =
      std::unique_ptr<PJRT_TopologyDescription>(topology_description);

  // Convert the input proto programs to C buffers.
  std::vector<std::string> serialized_programs_in;
  serialized_programs_in.reserve(partial_programs_in.size());
  for (const auto& partial_program : partial_programs_in) {
    serialized_programs_in.push_back(partial_program.SerializeAsString());
  }

  const char** programs_in_buffer = nullptr;
  const size_t* programs_in_buffer_sizes = nullptr;
  size_t num_programs_in = 0;
  ConvertCppStringsToCharBuffer(serialized_programs_in, &programs_in_buffer,
                                &programs_in_buffer_sizes, &num_programs_in);
  auto programs_in_buffer_ptr = std::make_unique<PhaseCompileCBuffersWrapper>(
      programs_in_buffer, programs_in_buffer_sizes, num_programs_in,
      /*is_locally_owned=*/true, phase_compile_extension_);

  // Convert compile options to a string.
  TF_ASSIGN_OR_RETURN(const xla::CompileOptionsProto options_proto,
                      options.ToProto());
  std::string options_str = options_proto.SerializeAsString();

  // Convert the "phases to run" to a C buffer.
  const char** phases_to_run_buffer = nullptr;
  const size_t* phases_to_run_buffer_sizes = nullptr;
  size_t num_phases_to_run = 0;
  ConvertCppStringsToCharBuffer(phases_to_run, &phases_to_run_buffer,
                                &phases_to_run_buffer_sizes,
                                &num_phases_to_run);
  auto phases_to_run_buffer_ptr = std::make_unique<PhaseCompileCBuffersWrapper>(
      phases_to_run_buffer, phases_to_run_buffer_sizes, num_phases_to_run,
      /*is_locally_owned=*/true, phase_compile_extension_);

  // Run the phases.
  PJRT_PhaseCompile_Run_Phase_Args run_args;
  run_args.struct_size = PJRT_PhaseCompile_Run_Phase_Args_STRUCT_SIZE;
  run_args.extension_start = nullptr;
  run_args.phase_compiler = phase_compiler_;
  run_args.input_programs = programs_in_buffer_ptr->char_buffers;
  run_args.input_programs_sizes = programs_in_buffer_ptr->char_buffer_sizes;
  run_args.num_input_programs = programs_in_buffer_ptr->num_char_buffers;
  run_args.phases_to_run = phases_to_run_buffer_ptr->char_buffers;
  run_args.phases_to_run_sizes = phases_to_run_buffer_ptr->char_buffer_sizes;
  run_args.num_phases_to_run = phases_to_run_buffer_ptr->num_char_buffers;
  run_args.compile_options = options_str.c_str();
  run_args.compile_options_size = options_str.size();
  run_args.topology = topology_ptr.get();

  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension_->phase_compile_run_phases(&run_args), api_);

  auto output_programs_buffer_ptr =
      std::make_unique<PhaseCompileCBuffersWrapper>(
          run_args.output_programs, run_args.output_programs_sizes,
          run_args.num_output_programs, /*is_locally_owned=*/false,
          phase_compile_extension_);

  // Convert the output programs from C buffers to protos.
  auto serialized_programs_out = ConvertCharBufferToCppStrings(
      output_programs_buffer_ptr->char_buffers,
      output_programs_buffer_ptr->char_buffer_sizes,
      output_programs_buffer_ptr->num_char_buffers);

  std::vector<xla::PjRtPartialProgramProto> programs_out;
  for (const auto& serialized_program : serialized_programs_out) {
    xla::PjRtPartialProgramProto partial_program;
    partial_program.ParseFromString(serialized_program);
    programs_out.push_back(partial_program);
  }

  return programs_out;
}

PhaseCompileExtensionWrapper::~PhaseCompileExtensionWrapper() {
  PJRT_PhaseCompile_Destroy_Compiler_Args destroy_compiler_args;
  destroy_compiler_args.struct_size =
      PJRT_PhaseCompile_Destroy_Compiler_Args_STRUCT_SIZE;
  destroy_compiler_args.extension_start = nullptr;
  destroy_compiler_args.phase_compiler = phase_compiler_;
  phase_compile_extension_->phase_compile_destroy_compiler(
      &destroy_compiler_args);
}

}  // namespace pjrt
