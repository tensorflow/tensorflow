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

#ifndef XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_EXTENSION_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"

#ifdef __cplusplus
extern "C" {
#endif

// PJRT PhaseCompile extension provides the ability to interact with individual
// compilation phases. This is essential for caching intermediate artifacts and
// for facilitating debugging.
#define PJRT_API_PHASE_COMPILE_EXTENSION_VERSION 2

// -------------------------------- Data types ---------------------------------

typedef struct PJRT_Relocatable PJRT_Relocatable;
typedef struct PJRT_SerializedRelocatable PJRT_SerializedRelocatable;

// ---------------------------------- Methods ----------------------------------

// This struct contains arguments for retrieving the phase compiler.
struct PJRT_PhaseCompile_Get_Compiler_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_PhaseCompiler*
      phase_compiler;  // Output: Pointer to the phase compiler.
                       // The caller is responsible for freeing
                       // this pointer using PJRT_PhaseCompile_Destroy_Compiler.
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_Get_Compiler_Args, phase_compiler);
typedef PJRT_Error* PJRT_PhaseCompile_Get_Compiler(
    PJRT_PhaseCompile_Get_Compiler_Args* args);

// This struct contains arguments for destroying a phase compiler.
struct PJRT_PhaseCompile_Destroy_Compiler_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_PhaseCompiler* phase_compiler;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_Destroy_Compiler_Args,
                          phase_compiler);
typedef void PJRT_PhaseCompile_Destroy_Compiler(
    PJRT_PhaseCompile_Destroy_Compiler_Args* args);

// This struct contains arguments for running a compilation phase.
// The arguments include the names of the phases to run, input programs,
// compile options, topology and an output buffer for the results.
struct PJRT_PhaseCompile_Run_Phase_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_PhaseCompiler*
      phase_compiler;  // Input: Pointer to the phase compiler.
  const char**
      input_programs;  // Input: Array of pointers to input serialized
                       // xla::PjRtPartialProgramProto programs. The caller
                       // is responsible for freeing these buffers.
  const size_t* input_programs_sizes;  // Input: Array of sizes of the above
                                       // input programs. The caller is
                                       // responsible for freeing this buffer.
  size_t num_input_programs;           // Input: Number of input programs.
  const char**
      phases_to_run;  // Input: Array of pointers to the null-terminated
                      // names of the phases to run. The caller is
                      // responsible for freeing these buffers.
  const size_t* phases_to_run_sizes;  // Input: Array of sizes of the above
                                      // phases to run. The caller is
                                      // responsible for freeing this buffer.
  size_t num_phases_to_run;           // Input: Number of phases to run.
  const char* compile_options;        // Input: Serialized CompileOptionsProto.
  size_t compile_options_size;        // Input: Size of the serialized
                                      // CompileOptionsProto.
  PJRT_TopologyDescription*
      topology;  // Input: Description of the device topology.
  const char**
      output_programs;  // Output: Array of pointers to output serialized
                        // xla::PjRtPartialProgramProto. The caller is
                        // responsible for freeing these buffers.
  const size_t* output_programs_sizes;  // Output: Array of sizes of the above
                                        // output programs. The caller is
                                        // responsible for freeing this buffer.
  size_t num_output_programs;           // Output: Number of output programs.
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_Run_Phase_Args,
                          num_output_programs);

typedef PJRT_Error* PJRT_PhaseCompile_Run_Phase(
    PJRT_PhaseCompile_Run_Phase_Args* args);

// Struct containing arguments for retrieving the names of all the phases
// in the order they are registered.
struct PJRT_PhaseCompile_Get_PhaseNames_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_PhaseCompiler*
      phase_compiler;        // Input: Pointer to the phase compiler.
  const char** phase_names;  // Output: Array of pointers to
                             // null-terminated names of registered
                             // phases. The caller is responsible for
                             // freeing these buffers.
  const size_t*
      phase_names_sizes;   // Output: Array of sizes of the above phase names.
                           // The caller is responsible for freeing this buffer.
  size_t num_phase_names;  // Output: Number of registered phases.
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_Get_PhaseNames_Args,
                          num_phase_names);
typedef PJRT_Error* PJRT_PhaseCompile_Get_PhaseNames(
    PJRT_PhaseCompile_Get_PhaseNames_Args* args);

// Arguments for destroying buffers allocated during PJRT_PhaseCompile_Run_Phase
// and PJRT_PhaseCompile_Get_PhaseNames.
struct PJRT_PhaseCompile_C_Buffers_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char** char_buffers;        // Input: Buffers to be freed by the plugin.
  const size_t* char_buffer_sizes;  // Input: Buffer sizes to be freed by the
                                    // plugin.
  size_t num_char_buffers;          // Input: Number of buffers.
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_C_Buffers_Destroy_Args,
                          num_char_buffers);
typedef void PJRT_PhaseCompile_C_Buffers_Destroy(
    PJRT_PhaseCompile_C_Buffers_Destroy_Args* args);

// -------------------------------- Relocatable --------------------------------

struct PJRT_Relocatable_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_Destroy_Args, relocatable);

// Frees `relocatable`. `relocatable` can be nullptr.
typedef PJRT_Error* PJRT_Relocatable_Destroy(
    PJRT_Relocatable_Destroy_Args* args);

struct PJRT_Relocatable_Name_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
  // `relocatable_name` has the same lifetime as `relocatable`. It is owned by
  // `relocatable`.
  const char* relocatable_name;  // out
  size_t relocatable_name_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_Name_Args, relocatable_name_size);

// Returns a string that identifies the relocatable.
typedef PJRT_Error* PJRT_Relocatable_Name(PJRT_Relocatable_Name_Args* args);

struct PJRT_Relocatable_BuildId_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
  // `build_id` has the same lifetime as `relocatable`. It is owned by
  // `relocatable`.
  const char* build_id;  // out
  size_t build_id_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_BuildId_Args, build_id_size);

// Returns the build ID of `relocatable`, derived from the inputs that produced
// it.
typedef PJRT_Error* PJRT_Relocatable_BuildId(
    PJRT_Relocatable_BuildId_Args* args);

struct PJRT_Relocatable_CallBackendConfig_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
  // `call_backend_config` has the same lifetime as `relocatable`. It is owned
  // by `relocatable`.
  const char* call_backend_config;  // out
  size_t call_backend_config_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_CallBackendConfig_Args,
                          call_backend_config_size);

// Returns the backend config to apply to any call sites for `relocatable`.
typedef PJRT_Error* PJRT_Relocatable_CallBackendConfig(
    PJRT_Relocatable_CallBackendConfig_Args* args);

struct PJRT_Relocatable_HasSideEffects_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
  bool has_side_effects;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_HasSideEffects_Args,
                          has_side_effects);

// Returns whether `relocatable` has side effects.
typedef PJRT_Error* PJRT_Relocatable_HasSideEffects(
    PJRT_Relocatable_HasSideEffects_Args* args);

struct PJRT_Relocatable_Serialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_Relocatable* relocatable;

  // Lives only as long as serialized_relocatable
  const char* serialized_bytes;  // out
  size_t serialized_bytes_size;  // out

  PJRT_SerializedRelocatable*
      serialized_relocatable;  // backs serialized_bytes.
  // cleanup fn must be called to free the backing memory for serialized_bytes.
  // Should only be called once on serialized_relocatable.
  void (*serialized_relocatable_deleter)(
      PJRT_SerializedRelocatable* reloc);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_Serialize_Args,
                          serialized_relocatable_deleter);

// Returns a platform-specific serialization of `relocatable`. The serialization
// is not guaranteed to be stable over time.
typedef PJRT_Error* PJRT_Relocatable_Serialize(
    PJRT_Relocatable_Serialize_Args* args);

struct PJRT_Relocatable_ParameterElementTypes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
  PJRT_Buffer_Type* parameter_types;  // out
  size_t num_parameter_types;         // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_ParameterElementTypes_Args,
                          num_parameter_types);

// Returns a list of element types for parameters.
typedef PJRT_Error* PJRT_Relocatable_ParameterElementTypes(
    PJRT_Relocatable_ParameterElementTypes_Args* args);

struct PJRT_Relocatable_OutputElementTypes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
  PJRT_Buffer_Type* output_types;  // out
  size_t num_output_types;         // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_OutputElementTypes_Args,
                          num_output_types);

// Returns a list of element types for outputs.
typedef PJRT_Error* PJRT_Relocatable_OutputElementTypes(
    PJRT_Relocatable_OutputElementTypes_Args* args);

struct PJRT_Relocatable_ParameterDimensions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
  size_t num_parameters;
  // Has length: sum of all elements in the list `dim_sizes`.
  const int64_t* dims;  // out
  // Has length `num_parameters`.
  const size_t* dim_sizes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_ParameterDimensions_Args, dim_sizes);

// Returns a list of dimensions for parameters. Each parameter has an array
// shape, which is represented by a list of dimensions. The array shapes of all
// parameters are concatenated into a single list of dimensions.
typedef PJRT_Error* PJRT_Relocatable_ParameterDimensions(
    PJRT_Relocatable_ParameterDimensions_Args* args);

struct PJRT_Relocatable_OutputDimensions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
  size_t num_outputs;
  // Has length: sum of all elements in the list `dim_sizes`.
  const int64_t* dims;  // out
  // Has length `num_outputs`.
  const size_t* dim_sizes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_OutputDimensions_Args, dim_sizes);

// Returns a list of dimensions for outputs. Each output has an array shape,
// which is represented by a list of dimensions. The array shapes of all outputs
// are concatenated into a single list of dimensions.
typedef PJRT_Error* PJRT_Relocatable_OutputDimensions(
    PJRT_Relocatable_OutputDimensions_Args* args);

struct PJRT_Relocatable_ParameterLayouts_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
  size_t num_parameters;  // out
  // Layout of relocatable parameters. Should be of length `num_parameters`.
  // `layouts` and the data pointed by `layouts` are owned by `relocatable`, so
  // callers shouldn't call `PJRT_Layouts_MemoryLayout_Destroy` on them.
  PJRT_Layouts_MemoryLayout** layouts;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_ParameterLayouts_Args, layouts);

// Returns a list of layouts for relocatable parameters.
typedef PJRT_Error* PJRT_Relocatable_ParameterLayouts(
    PJRT_Relocatable_ParameterLayouts_Args* args);

struct PJRT_Relocatable_OutputLayouts_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
  size_t num_outputs;  // out
  // Layout of relocatable outputs. Should be of length `num_outputs`. `layouts`
  // and the data pointed by `layouts` are owned by `relocatable`, so callers
  // shouldn't call `PJRT_Layouts_MemoryLayout_Destroy` on them.
  PJRT_Layouts_MemoryLayout** layouts;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_OutputLayouts_Args, layouts);

// Returns a list of layouts for relocatable outputs.
typedef PJRT_Error* PJRT_Relocatable_OutputLayouts(
    PJRT_Relocatable_OutputLayouts_Args* args);

struct PJRT_Relocatable_ParameterOutputAliases_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Relocatable* relocatable;
  size_t num_aliases;  // out
  // Has length `num_aliases`. Owned by `relocatable`.
  const int64_t* parameter_indices;  // out
  // Has length `num_aliases`. Owned by `relocatable`.
  const int64_t* output_indices;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Relocatable_ParameterOutputAliases_Args,
                          output_indices);

// Returns which pairs of parameters and outputs may alias.
typedef PJRT_Error* PJRT_Relocatable_ParameterOutputAliases(
    PJRT_Relocatable_ParameterOutputAliases_Args* args);

struct PJRT_PhaseCompile_CompileToRelocatable_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_PhaseCompiler* phase_compiler;
  const PJRT_TopologyDescription* topology;
  // Only needs to stay alive for the duration of the CompileToRelocatable call.
  // `program->format` and `program->format_size` are owned by the caller.
  const PJRT_Program* program;
  // Serialized CompileOptionsProto.
  const char* compile_options;
  size_t compile_options_size;
  const char* name;
  size_t name_size;
  bool is_entrypoint;
  // Optionally provided for performance-guided optimizations.
  PJRT_Client* client;
  PJRT_Relocatable* relocatable;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_CompileToRelocatable_Args,
                          relocatable);

typedef PJRT_Error* PJRT_PhaseCompile_CompileToRelocatable(
    PJRT_PhaseCompile_CompileToRelocatable_Args* args);

struct PJRT_PhaseCompile_DeserializeRelocatable_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_PhaseCompiler* phase_compiler;
  const char* serialized_relocatable;
  size_t serialized_relocatable_size;
  PJRT_Relocatable* relocatable;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_DeserializeRelocatable_Args,
                          relocatable);

typedef PJRT_Error* PJRT_PhaseCompile_DeserializeRelocatable(
    PJRT_PhaseCompile_DeserializeRelocatable_Args* args);

struct PJRT_PhaseCompile_LinkRelocatables_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_PhaseCompiler* phase_compiler;
  const PJRT_TopologyDescription* topology;
  // Serialized CompileOptionsProto.
  const char* compile_options;
  size_t compile_options_size;
  const PJRT_Relocatable* entrypoint;
  const PJRT_Relocatable* const* dependencies;
  size_t num_dependencies;
  PJRT_Executable* executable;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_LinkRelocatables_Args, executable);

typedef PJRT_Error* PJRT_PhaseCompile_LinkRelocatables(
    PJRT_PhaseCompile_LinkRelocatables_Args* args);

// --------------------------- Extension entrypoint ----------------------------

// This struct serves as the entry point for accessing the phase compilation
// functionalities provided by this extension.
typedef struct PJRT_PhaseCompile_Extension {
  PJRT_Extension_Base base;
  PJRT_NO_DISCARD PJRT_PhaseCompile_Get_Compiler* phase_compile_get_compiler;
  PJRT_PhaseCompile_Destroy_Compiler* phase_compile_destroy_compiler;
  PJRT_NO_DISCARD PJRT_PhaseCompile_Run_Phase* phase_compile_run_phases;
  PJRT_NO_DISCARD PJRT_PhaseCompile_Get_PhaseNames*
      phase_compile_get_phase_names;
  PJRT_PhaseCompile_C_Buffers_Destroy* phase_compile_c_buffers_destroy;
  PJRT_NO_DISCARD PJRT_Relocatable_Destroy* relocatable_destroy;
  PJRT_NO_DISCARD PJRT_Relocatable_Name* relocatable_name;
  PJRT_NO_DISCARD PJRT_Relocatable_BuildId* relocatable_build_id;
  PJRT_NO_DISCARD PJRT_Relocatable_CallBackendConfig*
      relocatable_call_backend_config;
  PJRT_NO_DISCARD PJRT_Relocatable_HasSideEffects* relocatable_has_side_effects;
  PJRT_NO_DISCARD PJRT_Relocatable_Serialize* relocatable_serialize;
  PJRT_NO_DISCARD PJRT_Relocatable_ParameterElementTypes*
      relocatable_parameter_element_types;
  PJRT_NO_DISCARD PJRT_Relocatable_OutputElementTypes*
      relocatable_output_element_types;
  PJRT_NO_DISCARD PJRT_Relocatable_ParameterDimensions*
      relocatable_parameter_dimensions;
  PJRT_NO_DISCARD PJRT_Relocatable_OutputDimensions*
      relocatable_output_dimensions;
  PJRT_NO_DISCARD PJRT_Relocatable_ParameterLayouts*
      relocatable_parameter_layouts;
  PJRT_NO_DISCARD PJRT_Relocatable_OutputLayouts* relocatable_output_layouts;
  PJRT_NO_DISCARD PJRT_Relocatable_ParameterOutputAliases*
      relocatable_parameter_output_aliases;
  PJRT_NO_DISCARD PJRT_PhaseCompile_CompileToRelocatable*
      phase_compile_compile_to_relocatable;
  PJRT_NO_DISCARD PJRT_PhaseCompile_DeserializeRelocatable*
      phase_compile_deserialize_relocatable;
  PJRT_NO_DISCARD PJRT_PhaseCompile_LinkRelocatables*
      phase_compile_link_relocatables;
} PJRT_PhaseCompile_Extension;

PJRT_DEFINE_STRUCT_TRAITS(PJRT_PhaseCompile_Extension,
                          phase_compile_link_relocatables);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_EXTENSION_H_
