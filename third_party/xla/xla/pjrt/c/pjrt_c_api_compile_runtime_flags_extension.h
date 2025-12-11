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

#ifndef XLA_PJRT_C_PJRT_C_API_COMPILE_RUNTIME_FLAGS_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_COMPILE_RUNTIME_FLAGS_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

// This extension provides functionality related to verifying that flags used at
// both compile- and run-time align.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define PJRT_API_COMPILE_RUNTIME_FLAGS_EXTENSION_VERSION 1

typedef struct PJRT_CompileRuntimeFlags_Validate_Args {
  size_t struct_size;
  const PJRT_TopologyDescription* topology;
  // Serialized CompileOptionsProto.
  const char* compile_options;
  size_t compile_options_size;
} PJRT_CompileRuntimeFlags_Validate_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CompileRuntimeFlags_Validate_Args,
                          compile_options_size);

// Returns whether the compile/runtime flags align.
typedef PJRT_Error* PJRT_CompileRuntimeFlags_Validate(
    PJRT_CompileRuntimeFlags_Validate_Args* args);

typedef struct PJRT_CompileRuntimeFlags_Extension {
  PJRT_Extension_Base base;
  PJRT_CompileRuntimeFlags_Validate* validate;
} PJRT_CompileRuntimeFlags_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CompileRuntimeFlags_Extension, validate);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // XLA_PJRT_C_PJRT_C_API_COMPILE_RUNTIME_FLAGS_EXTENSION_H_
