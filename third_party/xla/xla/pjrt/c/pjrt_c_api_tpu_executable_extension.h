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

#ifndef XLA_PJRT_C_PJRT_C_API_TPU_EXECUTABLE_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_TPU_EXECUTABLE_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides functionality related to TPU executable.

#define PJRT_API_TPU_EXECUTABLE_EXTENSION_VERSION 1

typedef struct PJRT_TpuExecutable_TargetArguments
    PJRT_TpuExecutable_TargetArguments;

typedef struct PJRT_TpuExecutable_CompiledMemoryStats
    PJRT_TpuExecutable_CompiledMemoryStats;

typedef struct PJRT_TpuExecutable_CompilationEnvironments
    PJRT_TpuExecutable_CompilationEnvironments;

typedef struct PJRT_TpuExecutable_CostAnalysisResult
    PJRT_TpuExecutable_CostAnalysisResult;

struct PJRT_TpuExecutable_GetTargetArguments_Args {
  size_t struct_size;
  const char* serialized_executable;
  size_t serialized_executable_size;

  const char* target_arguments;                              // out
  size_t target_arguments_size;                              // out
  PJRT_TpuExecutable_TargetArguments* target_arguments_ptr;  // out
  void (*target_arguments_deleter)(
      PJRT_TpuExecutable_TargetArguments* args);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuExecutable_GetTargetArguments_Args,
                          target_arguments_deleter);

typedef PJRT_Error* PJRT_TpuExecutable_GetTargetArguments(
    PJRT_TpuExecutable_GetTargetArguments_Args* args);

typedef struct PJRT_TpuExecutable_CoreProgramAbiVersion
    PJRT_TpuExecutable_CoreProgramAbiVersion;

struct PJRT_TpuExecutable_GetCoreProgramAbiVersion_Args {
  size_t struct_size;
  const char* serialized_executable;
  size_t serialized_executable_size;

  const char* abi_version;                                    // out
  size_t abi_version_size;                                    // out
  PJRT_TpuExecutable_CoreProgramAbiVersion* abi_version_ptr;  // out
  void (*abi_version_deleter)(
      PJRT_TpuExecutable_CoreProgramAbiVersion* args);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuExecutable_GetCoreProgramAbiVersion_Args,
                          abi_version_deleter);

typedef PJRT_Error* PJRT_TpuExecutable_GetCoreProgramAbiVersion(
    PJRT_TpuExecutable_GetCoreProgramAbiVersion_Args* args);

typedef struct PJRT_TpuExecutable_HloModuleWithConfig
    PJRT_TpuExecutable_HloModuleWithConfig;

struct PJRT_TpuExecutable_GetHloModuleWithConfig_Args {
  size_t struct_size;
  const char* serialized_executable;
  size_t serialized_executable_size;

  const char* hlo_module_with_config;                                  // out
  size_t hlo_module_with_config_size;                                  // out
  PJRT_TpuExecutable_HloModuleWithConfig* hlo_module_with_config_ptr;  // out
  void (*hlo_module_with_config_deleter)(
      PJRT_TpuExecutable_HloModuleWithConfig* module);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuExecutable_GetHloModuleWithConfig_Args,
                          hlo_module_with_config_deleter);

typedef PJRT_Error* PJRT_TpuExecutable_GetHloModuleWithConfig(
    PJRT_TpuExecutable_GetHloModuleWithConfig_Args* args);

struct PJRT_TpuExecutable_RunHloCostAnalysis_Args {
  size_t struct_size;

  const char* target_arguments;
  size_t target_arguments_size;

  // Serialized xla::HloModuleProtoWithConfig.
  const char** hlo_module;
  size_t* hlo_module_size;
  size_t num_hlo_modules;

  const PJRT_NamedValue* analysis_results;  // out
  size_t num_analysis_results;              // out

  PJRT_TpuExecutable_CostAnalysisResult* analysis_results_ptr;  // out
  void (*analysis_results_deleter)(
      PJRT_TpuExecutable_CostAnalysisResult* results);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuExecutable_RunHloCostAnalysis_Args,
                          analysis_results_deleter);

typedef PJRT_Error* PJRT_TpuExecutable_RunHloCostAnalysis(
    PJRT_TpuExecutable_RunHloCostAnalysis_Args* args);

struct PJRT_TpuExecutable_GetCompiledMemoryStats_Args {
  size_t struct_size;
  const char* serialized_executable;
  size_t serialized_executable_size;

  // Serialized xla::CompiledMemoryStatsProto.
  const char* compiled_memory_stats;                                  // out
  size_t compiled_memory_stats_size;                                  // out
  PJRT_TpuExecutable_CompiledMemoryStats* compiled_memory_stats_ptr;  // out
  void (*compiled_memory_stats_deleter)(
      PJRT_TpuExecutable_CompiledMemoryStats* stats);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuExecutable_GetCompiledMemoryStats_Args,
                          compiled_memory_stats_deleter);

typedef PJRT_Error* PJRT_TpuExecutable_GetCompiledMemoryStats(
    PJRT_TpuExecutable_GetCompiledMemoryStats_Args* args);

struct PJRT_TpuExecutable_SetTpuCompilationEnv_Args {
  size_t struct_size;

  // Serialized xla::CompilationEnvironmentsProto.
  const char* serialized_compilation_environments;
  size_t serialized_compilation_environments_size;

  // Serialized xla::CompilationEnvironmentsProto with TPU env set.
  const char* result_compilation_environments;  // out
  size_t result_compilation_environments_size;  // out

  PJRT_TpuExecutable_CompilationEnvironments*
      result_compilation_environments_ptr;  // out
  void (*result_compilation_environments_deleter)(
      PJRT_TpuExecutable_CompilationEnvironments* envs);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuExecutable_SetTpuCompilationEnv_Args,
                          result_compilation_environments_deleter);

typedef PJRT_Error* PJRT_TpuExecutable_SetTpuCompilationEnv(
    PJRT_TpuExecutable_SetTpuCompilationEnv_Args* args);

typedef struct PJRT_TpuExecutable_FieldValue PJRT_TpuExecutable_FieldValue;

struct PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString_Args {
  size_t struct_size;
  const char* serialized_compilation_environments;
  size_t serialized_compilation_environments_size;
  const char* field_name;
  size_t field_name_size;

  const char* field_value;                                            // out
  size_t field_value_size;                                            // out
  PJRT_TpuExecutable_FieldValue* field_value_ptr;                     // out
  void (*field_value_deleter)(PJRT_TpuExecutable_FieldValue* value);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(
    PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString_Args,
    field_value_deleter);

typedef PJRT_Error* PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString(
    PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString_Args* args);

typedef struct PJRT_TpuExecutable_Extension {
  PJRT_Extension_Base base;
  PJRT_TpuExecutable_GetTargetArguments* get_target_arguments;
  PJRT_TpuExecutable_GetHloModuleWithConfig* get_hlo_module_with_config;
  PJRT_TpuExecutable_GetCoreProgramAbiVersion* get_core_program_abi_version;
  PJRT_TpuExecutable_GetCompiledMemoryStats* get_compiled_memory_stats;
  PJRT_TpuExecutable_RunHloCostAnalysis* run_hlo_cost_analysis;
  PJRT_TpuExecutable_SetTpuCompilationEnv* set_tpu_compilation_env;
  PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString*
      get_tpu_compilation_env_field_as_string;
} PJRT_TpuExecutable_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuExecutable_Extension,
                          get_tpu_compilation_env_field_as_string);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_TPU_EXECUTABLE_EXTENSION_H_
