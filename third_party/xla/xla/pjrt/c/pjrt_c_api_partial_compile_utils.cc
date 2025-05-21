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

#include "xla/pjrt/c/pjrt_c_api_partial_compile_utils.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api_partial_compile_extension.h"
#include "xla/pjrt/pjrt_partial_program.h"

std::vector<std::string> ConvertCharBufferToCppStrings(const char** char_buffer,
                                                       size_t num_strings) {
  assert(char_buffer != nullptr);

  std::vector<std::string> cpp_strings;
  cpp_strings.reserve(num_strings);
  for (size_t i = 0; i < num_strings; ++i) {
    cpp_strings.push_back(std::string(char_buffer[i]));
  }

  // Destroy the char buffers.
  for (size_t i = 0; i < num_strings; ++i) {
    delete[] char_buffer[i];
  }
  delete[] char_buffer;

  return cpp_strings;
}

void ConvertCppStringsToCharBuffer(const std::vector<std::string>& strings,
                                   const char*** char_buffer,
                                   size_t* num_strings) {
  *num_strings = strings.size();
  const char** buffer_of_pointers = new const char*[*num_strings];
  for (size_t i = 0; i < *num_strings; ++i) {
    char* single_string_buffer = new char[strings[i].size() + 1];
    memcpy(single_string_buffer, strings[i].c_str(), strings[i].size() + 1);
    buffer_of_pointers[i] = single_string_buffer;
  }
  *char_buffer = buffer_of_pointers;
}

std::vector<xla::PjRtPartialProgram>
ConvertCPartialProgramsToCppPartialPrograms(const PJRT_PartialProgram* programs,
                                            size_t num_programs) {
  assert(programs != nullptr);
  std::vector<xla::PjRtPartialProgram> programs_out;
  programs_out.reserve(num_programs);
  for (size_t i = 0; i < num_programs; ++i) {
    xla::PjRtPartialProgram cpp_program;
    cpp_program.SetProgramBuffer(programs[i].program);  // Zero copy
    cpp_program.SetProgramBufferSize(programs[i].program_size);
    cpp_program.SetGeneratingPhase(std::string(programs[i].generating_phase));
    cpp_program.SetNextPhases(ConvertCharBufferToCppStrings(
        programs[i].next_phases, programs[i].num_next_phases));
    cpp_program.SetVersion(std::string(programs[i].version));
    cpp_program.SetFormat(programs[i].format);
    programs_out.push_back(std::move(cpp_program));

    delete[] programs[i].generating_phase;
    delete[] programs[i].version;
  }

  delete[] programs;
  return programs_out;
}

void ConvertCppPartialProgramsToCPartialPrograms(
    const std::vector<xla::PjRtPartialProgram>& programs,
    PJRT_PartialProgram** programs_out, size_t* num_programs_out) {
  size_t num_programs = programs.size();
  *num_programs_out = num_programs;
  *programs_out = new PJRT_PartialProgram[num_programs];

  for (size_t i = 0; i < num_programs; ++i) {
    PJRT_PartialProgram& program_out = (*programs_out)[i];

    // copy 'program' pointer
    program_out.program = programs[i].GetProgramBuffer();  // Zero copy
    program_out.program_size = programs[i].GetProgramBufferSize();

    // Deep copy 'generating_phase'
    const std::string& generating_phase_str = programs[i].GetGeneratingPhase();
    program_out.generating_phase = new char[generating_phase_str.size() + 1];
    memcpy(program_out.generating_phase, generating_phase_str.c_str(),
           generating_phase_str.size() + 1);

    // Deep copy 'next_phases'
    ConvertCppStringsToCharBuffer(programs[i].GetNextPhases(),
                                  &(program_out.next_phases),
                                  &(program_out.num_next_phases));

    // Deep copy 'version'
    const std::string& version_str = programs[i].GetVersion();
    program_out.version = new char[version_str.size() + 1];
    memcpy(program_out.version, version_str.c_str(), version_str.size() + 1);

    // Assign 'format' directly as it's a value type
    program_out.format = programs[i].GetFormat();
  }
}
