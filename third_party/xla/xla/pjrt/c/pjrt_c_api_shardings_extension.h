/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_SHARDINGS_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_SHARDINGS_EXTENSION_H_

#include <stddef.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides capabilities for retrieving parameter and output
// shardings of an executable.

#define PJRT_API_SHARDINGS_EXTENSION_VERSION 1

// -------------------------------- Data types ---------------------------------

// ---------------------------------- Methods ----------------------------------

struct PJRT_Shardings_PJRT_Executable_ParameterShardings_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_parameters;  // out
  // An array of serialized `xla.OpSharding` protos
  // (https://github.com/openxla/xla/blob/main/xla/xla_data.proto).
  // The array and its content are owned by and have the same lifetime as
  // `executable`.
  // If the plugin does not support shardings, this is set to `nullptr`.
  const char* const* shardings;  // out
  // The size of each serialized sharding in `shardings`. The array and its
  // content are owned by and have the same lifetime as `executable`.
  // If the plugin does not support shardings, this is set to `nullptr`.
  const size_t* sharding_sizes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(
    PJRT_Shardings_PJRT_Executable_ParameterShardings_Args, sharding_sizes);

// Returns a list of parameter shardings.
typedef PJRT_Error* PJRT_Shardings_PJRT_Executable_ParameterShardings(
    PJRT_Shardings_PJRT_Executable_ParameterShardings_Args* args);

struct PJRT_Shardings_PJRT_Executable_OutputShardings_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_outputs;  // out
  // An array of serialized `xla.OpSharding` protos
  // (https://github.com/openxla/xla/blob/main/xla/xla_data.proto).
  // The array and its content are owned by and have the same lifetime as
  // `executable`.
  // If the plugin does not support shardings, this is set to `nullptr`.
  const char* const* shardings;  // out
  // The size of each serialized sharding in `shardings`. The array and its
  // content are owned by and have the same lifetime as `executable`.
  // If the plugin does not support shardings, this is set to `nullptr`.
  const size_t* sharding_sizes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Shardings_PJRT_Executable_OutputShardings_Args,
                          sharding_sizes);

// Returns a list of output shardings.
typedef PJRT_Error* PJRT_Shardings_PJRT_Executable_OutputShardings(
    PJRT_Shardings_PJRT_Executable_OutputShardings_Args* args);

// ------------------------------ Methods --------------------------------------

typedef struct PJRT_Shardings_Extension {
  PJRT_Extension_Base base;
  PJRT_Shardings_PJRT_Executable_ParameterShardings*
      PJRT_Shardings_PJRT_Executable_ParameterShardings;
  PJRT_Shardings_PJRT_Executable_OutputShardings*
      PJRT_Shardings_PJRT_Executable_OutputShardings;
} PJRT_Shardings_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Shardings_Extension,
                          PJRT_Shardings_PJRT_Executable_OutputShardings);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_SHARDINGS_EXTENSION_H_
