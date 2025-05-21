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

#ifndef XLA_PJRT_C_PJRT_C_API_PARTIAL_COMPILE_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_PARTIAL_COMPILE_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_PARTIAL_COMPILE_EXTENSION_VERSION 1

// -------------------------------- Data types ---------------------------------
typedef struct PJRT_PartialProgram PJRT_PartialProgram;

// This struct contains the phase name to run, input program, compile options,
// and an output buffer for the result.
struct PJRT_PhaseCompile_Run_Phase_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char** phases_to_run_buffer;
  size_t num_phases_to_run;
  const PJRT_PartialProgram* programs_in;
  size_t num_programs_in;
  const char* compile_options;
  size_t compile_options_size;
  PJRT_TopologyDescription* topology;
  PJRT_PartialProgram*
      programs_out;  // out: Buffer to store the output programs. The callee
                     // allocates the memory, and the caller is responsible
                     // for deallocating it using delete[].
  size_t num_programs_out;  // out: Number of output programs.
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_Run_Phase_Args, num_programs_out);

typedef PJRT_Error* PJRT_PhaseCompile_Run_Phase(
    PJRT_PhaseCompile_Run_Phase_Args* args);

// Used to retrieve the names of all registered partial compilation phases.
struct PJRT_PhaseCompile_Get_PhaseNames_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char** phase_names;
  size_t num_phases;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_Get_PhaseNames_Args, num_phases);
typedef PJRT_Error* PJRT_PhaseCompile_Get_PhaseNames(
    PJRT_PhaseCompile_Get_PhaseNames_Args* args);

// --------------------------- Extension entrypoint ----------------------------

// The main struct for the PJRT Partial Compile Extension.
typedef struct PJRT_PhaseCompile_Extension {
  PJRT_Extension_Base base;
  PJRT_PhaseCompile_Run_Phase* partial_compile_run_phase;
  PJRT_PhaseCompile_Get_PhaseNames* partial_compile_get_phase_names;
} PJRT_PhaseCompile_Extension;

PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_Extension,
                          partial_compile_get_phase_names);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_PARTIAL_COMPILE_EXTENSION_H_
