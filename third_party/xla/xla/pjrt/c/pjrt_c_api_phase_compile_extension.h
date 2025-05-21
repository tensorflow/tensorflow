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

#ifndef XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// PJRT PhaseCompile extension provides the ability to interact with individual
// compilation phases. This is essential for optimizing compilation time through
// techniques such as Forge caching, and for facilitating debugging.
#define PJRT_API_PHASE_COMPILE_EXTENSION_VERSION 1

// This struct containing arguments for running a compilation phase.
// The arguments include the names of the phases to run, input programs,
// compile options, topology and an output buffer for the results.
struct PJRT_PhaseCompile_Run_Phase_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char**
      programs_in_buffer;  // Input: Array of pointers to input serialized
                           // xla::PjRtPartialProgramProto programs. The caller
                           // is responsible for freeing these buffers.
  const size_t*
      programs_in_buffer_sizes;  // Input: Array of sizes of the above input
                                 // programs. The caller is responsible for
                                 // freeing this buffer.
  size_t num_programs_in;        // Input: Number of input programs.
  const char**
      phases_to_run_buffer;  // Input: Array of pointers to the null-terminated
                             // names of the phases to run. The caller is
                             // responsible for freeing these buffers.
  size_t num_phases_to_run;  // Input: Number of phases to run.
  const char* compile_options;  // Input: Serialized CompileOptionsProto.
  size_t compile_options_size;  // Input: Size of the serialized
                                // CompileOptionsProto.
  PJRT_TopologyDescription*
      topology;  // Input: Description of the device topology.
  const char**
      programs_out_buffer;  // Output: Array of pointers to output serialized
                            // xla::PjRtPartialProgramProto. The caller is
                            // responsible for freeing these buffers.
  const size_t*
      programs_out_buffer_sizes;  // Output: Array of sizes of the above output
                                  // programs. The caller is responsible for
                                  // freeing this buffer.
  size_t num_programs_out;        // Output: Number of output programs.
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_Run_Phase_Args, num_programs_out);

typedef PJRT_Error* PJRT_PhaseCompile_Run_Phase(
    PJRT_PhaseCompile_Run_Phase_Args* args);

// Struct containing arguments for retrieving the names of all the phases
// in the order they are registered.
struct PJRT_PhaseCompile_Get_PhaseNames_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* platform_name;  // Input: Name of the platform to get the phase
                              // names for.
  const char** phase_names_buffer;  // Output: Array of pointers to
                                    // null-terminated names of registered
                                    // phases. The caller is responsible for
                                    // freeing these buffers.
  size_t num_phases;                // Output: Number of registered phases.
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_Get_PhaseNames_Args, num_phases);
typedef PJRT_Error* PJRT_PhaseCompile_Get_PhaseNames(
    PJRT_PhaseCompile_Get_PhaseNames_Args* args);

// --------------------------- Extension entrypoint ----------------------------

// This struct serves as the entry point for accessing the phase compilation
// functionalities provided by this extension.
typedef struct PJRT_PhaseCompile_Extension {
  PJRT_Extension_Base base;
  PJRT_PhaseCompile_Run_Phase* phase_compile_run_phase;
  PJRT_PhaseCompile_Get_PhaseNames* phase_compile_get_phase_names;
} PJRT_PhaseCompile_Extension;

PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_Extension,
                          phase_compile_get_phase_names);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_EXTENSION_H_
